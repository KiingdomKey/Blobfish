"""
Checks the minichess engine's 5x5 Gardner rules against pyffish (Fairy-Stockfish),
which we treat as the answer key.

Run from src/blobfish:  PYTHON_GIL=0 python test_gardner_rules.py
"""

import random
import warnings

import pyffish as sf

from minichess.chess_helpers import get_initial_chess_object
from minichess.fastchess_utils import piece_matrix_to_legal_moves, chess_move_to_uci

# The engine does bit tricks that make numpy print overflow warnings. They're harmless.
warnings.filterwarnings("ignore", message="overflow encountered")

GAME = "gardner"
START = sf.start_fen(GAME)


def engine_moves(board):
    """
    Legal moves according to the minichess engine.
    Returns a dict: move string like 'b1c3' -> the engine's own move object
    (we need the engine's version to actually play the move on the board).
    """
    move_matrix, promotions = board.legal_moves()
    moves = piece_matrix_to_legal_moves(move_matrix, promotions)
    return {chess_move_to_uci(move, board.dims): move for move in moves}


def pyffish_moves(moves_so_far):
    """Legal moves according to pyffish, as a set of strings like 'b1c3'."""
    return set(sf.legal_moves(GAME, START, moves_so_far))


def compare(board, moves_so_far):
    """True if both agree on the legal moves. If not, prints what's different."""
    mine = set(engine_moves(board))
    ref = pyffish_moves(moves_so_far)

    if mine == ref:
        return True

    print("MISMATCH at position:", sf.get_fen(GAME, START, moves_so_far))
    print("  moves played:", " ".join(moves_so_far))
    print("  engine only: ", sorted(mine - ref))
    print("  pyffish only:", sorted(ref - mine))
    return False


def play_engine_move(board, engine_move):
    """Plays one of the engine's move objects on the engine board."""
    (i, j), (dx, dy), promotion = engine_move
    if promotion is None or promotion == -1:
        board.make_move(i, j, dx, dy)
    else:
        board.make_move(i, j, dx, dy, promotion)


def play_random_game(start_board, rng, max_moves=200):
    """
    Plays one random game, checking the engine against pyffish before every move.
    Returns (passed, positions_checked).
    """
    board = start_board.copy()
    moves_so_far = []

    for _ in range(max_moves):
        if not compare(board, moves_so_far):
            return False, len(moves_so_far) + 1

        legal = sorted(pyffish_moves(moves_so_far))

        # Game over: checkmate / stalemate, or a draw rule (50 moves, repetition, etc.)
        if not legal or sf.is_optional_game_end(GAME, START, moves_so_far)[0]:
            break

        move = rng.choice(legal)
        play_engine_move(board, engine_moves(board)[move])
        moves_so_far.append(move)

    return True, len(moves_so_far) + 1


if __name__ == "__main__":
    NUM_GAMES = 1000
    rng = random.Random(0)  # fixed seed so a failure can be reproduced exactly

    start_board = get_initial_chess_object("5x5gardner")
    print("Start position matches:", compare(start_board, []))

    total_positions = 0
    for game in range(NUM_GAMES):
        passed, checked = play_random_game(start_board, rng)
        total_positions += checked
        if not passed:
            print(f"FAILED in game {game + 1}")
            break
    else:
        print(f"PASSED: {NUM_GAMES} games, {total_positions} positions, all matched")