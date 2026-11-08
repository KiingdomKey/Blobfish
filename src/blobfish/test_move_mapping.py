"""
Tests for move_mapping.py, using pyffish to generate real legal moves.

Run from src/blobfish:  PYTHON_GIL=0 python test_move_mapping.py
"""

import random

import pyffish as sf

from move_mapping import (move_to_index, index_to_move, add_queen_promotion,
                          legal_mask, policy_size)

GAME = "gardner"
SIZE = 5
START = sf.start_fen(GAME)


def check_position(moves_so_far):
    """Checks every legal move in one position. Returns the legal moves (empty list = game over)."""
    legal = sf.legal_moves(GAME, START, moves_so_far)
    black_to_move = len(moves_so_far) % 2 == 1

    seen = {}
    for move in legal:
        index = move_to_index(move, SIZE, black_to_move)

        # 1. index is in range
        assert 0 <= index < policy_size(SIZE), f"{move} -> {index} is out of range"

        # 2. no two legal moves share a slot
        assert index not in seen, f"{move} and {seen[index]} both map to slot {index}"
        seen[index] = move

        # 3. round trip: move -> index -> move gives back the same move
        back = add_queen_promotion(index_to_move(index, SIZE, black_to_move), legal)
        assert back == move, f"{move} -> {index} -> {back}"

    # 4. mask has exactly one 1 per legal move
    mask = legal_mask(legal, SIZE, black_to_move)
    assert mask.sum() == len(legal), f"mask has {mask.sum()} ones but there are {len(legal)} legal moves"

    return legal


if __name__ == "__main__":
    assert policy_size(SIZE) == 1225

    # The example from the plan: white b1c3, and black's mirror image b5c3, share a slot
    assert move_to_index("b1c3", SIZE, black_to_move=False) == 81
    assert move_to_index("b5c3", SIZE, black_to_move=True) == 81
    print("Examples OK: b1c3 (white) and b5c3 (black) both -> slot 81")

    NUM_GAMES = 300
    rng = random.Random(0)
    positions = 0
    for game in range(NUM_GAMES):
        moves_so_far = []
        for _ in range(200):
            legal = check_position(moves_so_far)
            positions += 1
            if not legal or sf.is_optional_game_end(GAME, START, moves_so_far)[0]:
                break
            moves_so_far.append(rng.choice(legal))

    print(f"PASSED: {NUM_GAMES} games, {positions} positions, every legal move round-tripped")
