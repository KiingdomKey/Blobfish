"""
Move <-> network slot mapping for blobfish.

The network's policy head outputs one number per "slot". A slot is
(from-square, move type), the same layout AlphaZero uses, and its size matches
encoding.policy_size():

    slots = size * size * move_types
    move_types = 8 * (size - 1)   slides: 8 directions x distance 1..size-1
               + 8                knight jumps
               + 9                underpromotions: 3 directions x (knight, bishop, rook)

Everything is from the MOVER's point of view, matching encoding.encode():
on black's turn the board is flipped top-to-bottom (rank r -> rank size+1-r),
files stay the same. So black's b5c3 uses the same slot as white's b1c3.

Moves are strings like "b1c3" or "a4a5n" (from square, to square, optional promotion piece).
"""

import numpy as np

# Slide directions as (row change, col change). Row goes up the board, col goes right.
# The order here IS the numbering: direction 0 = up, 1 = up-right, ...
DIRECTIONS = [
    (1, 0),    # up
    (1, 1),    # up-right
    (0, 1),    # right
    (-1, 1),   # down-right
    (-1, 0),   # down
    (-1, -1),  # down-left
    (0, -1),   # left
    (1, -1),   # up-left
]

KNIGHT_JUMPS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

# Underpromotions: pawn moves forward-left, straight, or forward-right,
# and becomes a knight, bishop, or rook. (Queen promotions count as normal slides.)
UNDERPROMO_COLS = [-1, 0, 1]
UNDERPROMO_PIECES = ["n", "b", "r"]


def num_move_types(size):
    return 8 * (size - 1) + 8 + 9


def policy_size(size):
    """Total number of slots. Matches encoding.policy_size()."""
    return size * size * num_move_types(size)


# ---------- squares ----------

def square_to_row_col(square, size, flip):
    """'b1' -> (row, col) in the mover's view. Note: chess writes column first, arrays use row first."""
    col = ord(square[0]) - ord("a")
    row = int(square[1:]) - 1
    if flip:
        row = size - 1 - row
    return row, col


def row_col_to_square(row, col, size, flip):
    """(row, col) in the mover's view -> 'b1' on the real board."""
    if flip:
        row = size - 1 - row
    return chr(ord("a") + col) + str(row + 1)


# ---------- move -> index ----------

def move_to_index(move, size, black_to_move):
    """'b1c3' -> slot number in [0, policy_size(size))."""
    promotion = move[4:] if len(move) > 4 else ""
    from_row, from_col = square_to_row_col(move[0:2], size, black_to_move)
    to_row, to_col = square_to_row_col(move[2:4], size, black_to_move)
    d_row, d_col = to_row - from_row, to_col - from_col

    slides = 8 * (size - 1)

    if promotion in UNDERPROMO_PIECES:
        move_type = slides + 8 + UNDERPROMO_COLS.index(d_col) * 3 + UNDERPROMO_PIECES.index(promotion)
    elif (d_row, d_col) in KNIGHT_JUMPS:
        move_type = slides + KNIGHT_JUMPS.index((d_row, d_col))
    else:
        distance = max(abs(d_row), abs(d_col))
        direction = DIRECTIONS.index((d_row // distance, d_col // distance))
        move_type = direction * (size - 1) + (distance - 1)

    from_square = from_row * size + from_col
    return from_square * num_move_types(size) + move_type


# ---------- index -> move ----------

def index_to_move(index, size, black_to_move):
    """Slot number -> 'b1c3'. A pawn reaching the last rank by a slide gets 'q' added."""
    from_square, move_type = divmod(index, num_move_types(size))
    from_row, from_col = divmod(from_square, size)

    slides = 8 * (size - 1)
    promotion = ""

    if move_type < slides:
        direction, distance = divmod(move_type, size - 1)
        distance += 1
        d_row, d_col = DIRECTIONS[direction]
        d_row, d_col = d_row * distance, d_col * distance
    elif move_type < slides + 8:
        d_row, d_col = KNIGHT_JUMPS[move_type - slides]
    else:
        col_choice, piece = divmod(move_type - slides - 8, 3)
        d_row, d_col = 1, UNDERPROMO_COLS[col_choice]
        promotion = UNDERPROMO_PIECES[piece]

    to_row, to_col = from_row + d_row, from_col + d_col
    move = row_col_to_square(from_row, from_col, size, black_to_move) + \
           row_col_to_square(to_row, to_col, size, black_to_move) + promotion
    return move


def add_queen_promotion(move, legal_moves):
    """
    index_to_move can't tell a pawn from a queen by the slot alone, so a pawn
    slide to the last rank comes back without the 'q'. Pass the legal move list
    and this returns the version that's actually legal.
    """
    if move in legal_moves:
        return move
    if move + "q" in legal_moves:
        return move + "q"
    return move


# ---------- legal move mask ----------

def legal_mask(legal_moves, size, black_to_move):
    """1 for every legal slot, 0 everywhere else. Length = policy_size(size)."""
    mask = np.zeros(policy_size(size), dtype=np.float32)
    for move in legal_moves:
        mask[move_to_index(move, size, black_to_move)] = 1.0
    return mask