from nodes.base_node import Node

import minichess.fastchess
from minichess.chess_helpers import get_initial_chess_object
from minichess.move_utils import calculate_all_moves, index_to_move
from minichess.fastchess_utils import chess_move_to_uci, uci_move_to_native_move, visualize_board, piece_matrix_to_legal_moves

class MinichessNode(Node):
    def __init__(self,
                 state: Chess = None,
                 game: str = "8x8standard",
                 **kwargs):
        """
        Args:
            state:  current board state (FEN format)
            game:   name of variant
        """
        super().__init__(**kwargs)
        if state is None:
            state = get_initial_chess_object(game)
        self.state = state
        self.game = game

    def is_terminal(self) -> int | None:
        return self.state.game_result()


    def get_legal_moves(self):
        move_matrix, promotions = self.state.legal_moves()
        native_moves = piece_matrix_to_legal_moves(move_matrix, promotions)
        return native_moves

    def add_successors(self) -> None:
        successors = []
        for move in self.get_legal_moves():
            (i, j), (dx, dy), _ = move
            child_state = self.state.copy()
            child_state.make_move(i=i, j=j, dx=dx, dy=dy)
            child = MinichessNode(
                state=child_state,
                game=self.game,
                action=chess_move_to_uci(move, self.state.dims),
                parent=self,
                depth=self.depth+1
            )

            successors.append(child)
        self.children = successors

    def fen(self) -> str:
        return self.state.fen()




