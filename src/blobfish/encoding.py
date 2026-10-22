"""
Encoding for blobfish

Turns a chess position (fen string) into numbers the network can read.
Everything that talks to the network should use this so they all agree on what it's getting.

AlphaZero trained on ~44 million games on a wack amount of ai chips. We have 3 laptops and a dream,
so we start small and build up: 5x5 -> 6x6 -> 8x8. The encoding is the same shape at every size,
so the weights trained on a smaller board can be the starting point for the next one huzzah!

in:  fen string, ex "rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1" (gardner 5x5 start)
out: tensor of shape (18, size, size)
"""

import torch


# pyffish variant for each board size (losalamos gotz no bishopz)
VARIANTS = {
    5: "gardner",
    6: "losalamos",
    8: "chess",
}

# 18 planes (size x size grids) at every board size:
#   0-11   pieces, 1 where that piece is (6 types x 2 colors)
#   12-15  castling (my kingside, my queenside, their kingside, their queenside), all 1s if allowed
#   16     en passant, 1 on the square that can be captured
#   17     50 move counter / 100
# small boards have no castling or en passant so 12-16 these guys just stay 0s
PLANES = 18
CASTLING_PLANE = 12
EN_PASSANT_PLANE = 16
FIFTY_MOVE_PLANE = 17

PIECE_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,     # white (uppercase)
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,   # black (lowercase)
}

# THE BIG FLIP is the deal so on blacks turn the board goes upside down and colors swap,
# so planes 0-5 are always "my pieces" and the network only learns chess from one side
# that's double the training per game >:)

# FEN looks like this  "rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1"
#   board (rows split by /, numbers = empty squares), turn, castling, en passant, 50 move, move number


def encode(fen):
    parts = fen.split()
    placement = parts[0]
    turn = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    en_passant = parts[3] if len(parts) > 3 else "-"
    fifty_move = int(parts[4]) if len(parts) > 4 else 0

    rows = placement.split("/")
    size = len(rows)
    flip = turn == "b"
    t = torch.zeros(PLANES, size, size)

    # pieces and row 0 is always the movers back rank
    for i, rank in enumerate(rows):
        row = i if flip else size - 1 - i
        col = 0
        num = ""
        for ch in rank:
            if ch.isdigit():
                num += ch
                continue
            if num:
                col += int(num)
                num = ""
            if ch in PIECE_PLANE:
                letter = ch.swapcase() if flip else ch
                t[PIECE_PLANE[letter], row, col] = 1.0
                col += 1


    # this stuff is just easier to do now

    # castling, mine first
    castle_order = "kqKQ" if flip else "KQkq"
    for j, letter in enumerate(castle_order):
        if letter in castling:
            t[CASTLING_PLANE + j] = 1.0

    # en passant
    if en_passant != "-":
        col = ord(en_passant[0]) - ord("a")
        rank_number = int(en_passant[1:])
        row = size - rank_number if flip else rank_number - 1
        t[EN_PASSANT_PLANE, row, col] = 1.0

    t[FIFTY_MOVE_PLANE] = min(fifty_move, 100) / 100
    return t


# this determines the number of possible moves for each board size
# most are illegal duh we just need the max for network's output size
def policy_size(board_size=8):
    move_types = 8 * (board_size - 1) + 8 + 9
    return board_size * board_size * move_types
