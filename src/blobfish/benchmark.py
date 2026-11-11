"""
Benchmarking for blobfish: play two players against each other and count wins/draws/losses.

pyffish (Fairy-Stockfish's rules) is the referee, so every game follows the real
Gardner rules -- the same rules Fairy-Stockfish will use when we benchmark against it.

A "player" is any function that takes (moves_so_far, rng) and returns a move string
like "b1c3". When the agent is ready, it just needs a function with that shape.

Run from src/blobfish:  PYTHON_GIL=0 FAIRY_STOCKFISH=/path/to/engine python benchmark.py
(Without FAIRY_STOCKFISH the Stockfish games are skipped and everything else still runs.)
"""

import atexit
import csv
import os
import random
import shutil
import subprocess
import time
import warnings
from collections import Counter

import pyffish as sf

from minichess.chess_helpers import get_initial_chess_object
from minichess.fastchess_utils import piece_matrix_to_legal_moves, chess_move_to_uci

# The minichess engine does bit tricks that make numpy print overflow warnings. Harmless.
warnings.filterwarnings("ignore", message="overflow encountered")

GAME = "gardner"
START = sf.start_fen(GAME)
MAX_MOVES = 200  # a game that hits this many moves counts as a draw


# PLAYERS

def random_player(moves_so_far, rng):
    """Picks any legal move."""
    return rng.choice(sf.legal_moves(GAME, START, moves_so_far))


def capture_player(moves_so_far, rng):
    """Takes a capture if there is one, otherwise plays randomly."""
    legal = sf.legal_moves(GAME, START, moves_so_far)
    captures = [m for m in legal if sf.is_capture(GAME, START, moves_so_far, m)]
    return rng.choice(captures if captures else legal)


# MINIMAX PLAYER
#
# Looks a few moves ahead and picks the move that leaves it with the most material,
# assuming the opponent also plays its best. Uses the minichess engine for the
# look-ahead because it's much faster than pyffish (pyffish is still the referee).

PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
MATE_SCORE = 1000

_START_BOARD = None


def engine_board_after(moves_so_far):
    """An engine board with the moves so far played on it."""
    global _START_BOARD
    if _START_BOARD is None:
        _START_BOARD = get_initial_chess_object("5x5gardner")
    board = _START_BOARD.copy()
    for move in moves_so_far:
        board_moves = engine_legal_moves(board)
        play_on_engine(board, board_moves[move])
    return board


def engine_legal_moves(board):
    """Dict of move string -> engine move, for the engine board's current position."""
    matrix, promotions = board.legal_moves()
    return {chess_move_to_uci(m, board.dims): m for m in piece_matrix_to_legal_moves(matrix, promotions)}


def play_on_engine(board, engine_move):
    (i, j), (dx, dy), promotion = engine_move
    board.make_move(i, j, dx, dy, promotion)


def material(board):
    """Material count from WHITE's point of view (white pieces minus black pieces)."""
    score = 0
    for ch in board.fen().split()[0]:
        if ch.lower() in PIECE_VALUES:
            value = PIECE_VALUES[ch.lower()]
            score += value if ch.isupper() else -value
    return score


def negamax(board, depth, alpha, beta, color):
    """
    Score of the position for the side to move (color = +1 white, -1 black).
    Alpha-beta pruning skips branches that can't change the answer.
    """
    moves = engine_legal_moves(board)
    if not moves:
        if board.any_checkers:
            return -MATE_SCORE - depth  # checkmated (sooner mates score worse)
        return 0  # stalemate
    if depth == 0:
        return color * material(board)

    best = -float("inf")
    for move in moves.values():
        child = board.copy()
        play_on_engine(child, move)
        score = -negamax(child, depth - 1, -beta, -alpha, -color)
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def make_minimax_player(depth):
    """Returns a player that searches `depth` half-moves ahead."""
    def minimax_player(moves_so_far, rng):
        board = engine_board_after(moves_so_far)
        color = 1 if len(moves_so_far) % 2 == 0 else -1
        moves = list(engine_legal_moves(board).items())
        rng.shuffle(moves)  # so ties are broken randomly and games aren't all identical

        best_score, best_move = -float("inf"), None
        for name, move in moves:
            child = board.copy()
            play_on_engine(child, move)
            score = -negamax(child, depth - 1, -float("inf"), float("inf"), -color)
            if score > best_score:
                best_score, best_move = score, name
        return best_move
    return minimax_player


# STOCKFISH PLAYER

# Runs the Fairy-Stockfish engine as a separate program and talks to it over UCI,
# a plain-text protocol: we send lines like "position startpos moves e2e3" and
# "go depth 1", and it answers "bestmove a4a3".

# Where the engine program is: set FAIRY_STOCKFISH to its path, or put a program
# called "fairy-stockfish" or "stockfish" on your PATH.

def find_stockfish():
    path = os.environ.get("FAIRY_STOCKFISH")
    if path:
        return path
    for name in ("fairy-stockfish", "stockfish"):
        if shutil.which(name):
            return shutil.which(name)
    return None


class StockfishPlayer:
    """
    A player backed by one running Fairy-Stockfish process.
    depth: how many half-moves it searches (1 = weakest).
    skill: Skill Level, -20 (weakest) to 20 (full strength). Lower skill makes it
           deliberately pick worse moves sometimes.
    """

    def __init__(self, depth=1, skill=20):
        path = find_stockfish()
        if path is None:
            raise RuntimeError("Fairy-Stockfish not found. Set FAIRY_STOCKFISH to the engine's path.")
        self.depth = depth
        self.engine = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        atexit.register(self.close)
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name UCI_Variant value {GAME}")
        self._send(f"setoption name Skill Level value {skill}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, line):
        self.engine.stdin.write(line + "\n")

    def _wait_for(self, prefix):
        while True:
            line = self.engine.stdout.readline()
            if not line:
                raise RuntimeError("Fairy-Stockfish stopped unexpectedly")
            if line.startswith(prefix):
                return line.strip()

    def __call__(self, moves_so_far, rng):
        position = "position startpos"
        if moves_so_far:
            position += " moves " + " ".join(moves_so_far)
        self._send(position)
        self._send(f"go depth {self.depth}")
        return self._wait_for("bestmove").split()[1]

    def close(self):
        if self.engine.poll() is None:
            self._send("quit")
            self.engine.wait(timeout=5)


def make_stockfish_player(depth, skill=20):
    """Starts the engine the first time this player is used, so the other players work without it."""
    player = None

    def stockfish_player(moves_so_far, rng):
        nonlocal player
        if player is None:
            player = StockfishPlayer(depth=depth, skill=skill)
        return player(moves_so_far, rng)
    return stockfish_player


PLAYERS = {
    "random": random_player,
    "capture": capture_player,
    "minimax1": make_minimax_player(1),
    "minimax2": make_minimax_player(2),
    # weakest Stockfish settings first; these are what Blobfish should try to beat
    "sf_d1_skill-20": make_stockfish_player(depth=1, skill=-20),
    "sf_d1": make_stockfish_player(depth=1),
    "sf_d3": make_stockfish_player(depth=3),
}


# ONE GAME

def game_over(moves_so_far):
    """
    Returns None if the game is still going,
    otherwise (result for WHITE: +1 win / 0 draw / -1 loss, reason).
    """
    white_to_move = len(moves_so_far) % 2 == 0
    legal = sf.legal_moves(GAME, START, moves_so_far)

    if not legal:
        # checkmate or stalemate. pyffish scores it for the side to move.
        score = sf.game_result(GAME, START, moves_so_far)
        if score == 0:
            return 0, "stalemate"
        mover_result = 1 if score > 0 else -1
        return (mover_result if white_to_move else -mover_result), "checkmate"

    ended, _ = sf.is_optional_game_end(GAME, START, moves_so_far)
    if ended:
        return 0, "draw rule (50 moves / repetition)"

    white_insufficient, black_insufficient = sf.has_insufficient_material(GAME, START, moves_so_far)
    if white_insufficient and black_insufficient:
        return 0, "insufficient material"

    if len(moves_so_far) >= MAX_MOVES:
        return 0, "move cap"

    return None


def play_game(white, black, rng):
    """Plays one game. Returns (result for white, reason, number of moves)."""
    moves_so_far = []
    while True:
        over = game_over(moves_so_far)
        if over is not None:
            result, reason = over
            return result, reason, len(moves_so_far)

        player = white if len(moves_so_far) % 2 == 0 else black
        moves_so_far.append(player(moves_so_far, rng))


# MATCH

def play_match(name_a, name_b, num_games, seed=0):
    """
    Plays num_games between A and B, swapping colors every game so neither
    side gets the white advantage. Returns a dict of results from A's point of view.
    """
    a, b = PLAYERS[name_a], PLAYERS[name_b]
    rng = random.Random(seed)
    wins = draws = losses = 0
    reasons = Counter()
    total_moves = 0

    for game in range(num_games):
        a_is_white = game % 2 == 0
        if a_is_white:
            result, reason, n = play_game(a, b, rng)
        else:
            result, reason, n = play_game(b, a, rng)
            result = -result  # flip so it's from A's point of view

        wins += result == 1
        draws += result == 0
        losses += result == -1
        reasons[reason] += 1
        total_moves += n

    return {
        "player": name_a,
        "opponent": name_b,
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round((wins + 0.5 * draws) / num_games, 3),  # 1.0 = won everything, 0.5 = even
        "avg_moves": round(total_moves / num_games, 1),
        "endings": dict(reasons),
    }


def save_results(rows, path="results/benchmark.csv"):
    """Appends results to a CSV so every run is kept for the report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    fields = ["timestamp", "player", "opponent", "games", "wins", "draws", "losses", "score", "avg_moves"]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M"), **row})


if __name__ == "__main__":
    NUM_GAMES = 100
    # A ladder of opponents from weakest to strongest. Blobfish's progress over
    # training = how far up this ladder it can climb.
    matchups = [
        ("random", "random"),            # sanity check: should be roughly even
        ("capture", "random"),           # greedy capturing beats random
        ("minimax2", "capture"),         # looking 2 moves ahead beats greedy
        ("sf_d1_skill-20", "random"),    # weakest Stockfish setting
        ("sf_d1_skill-20", "capture"),
        ("sf_d1", "minimax2"),           # real Stockfish at depth 1
        ("sf_d3", "minimax2"),
    ]

    if find_stockfish() is None:
        print("Fairy-Stockfish not found (set FAIRY_STOCKFISH), skipping Stockfish games\n")
        matchups = [m for m in matchups if not any(p.startswith("sf_") for p in m)]

    rows = []
    for a, b in matchups:
        start = time.time()
        r = play_match(a, b, NUM_GAMES)
        rows.append(r)
        print(f"{a} vs {b}: {r['wins']}W {r['draws']}D {r['losses']}L  "
              f"(score {r['score']}, avg {r['avg_moves']} moves, {time.time() - start:.0f}s)")
        print(f"   endings: {r['endings']}")

    save_results(rows)
    print("Saved to results/benchmark.csv")
