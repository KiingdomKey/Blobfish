from nodes.base_node import Node

import pyffish as pf

class PyffishNode(Node):
    def __init__(self,
                 state: str,
                 game: str,
                 **kwargs):
        """
        Args:
            state:  current board state (FEN format)
            game:   name of variant
        """
        super().__init__(**kwargs)
        self.state = state
        self.game = game

    def copy(self) -> Node:
        c = PyffishNode(
            state=self.state,
            game=self.game,
        )
        return c

    def is_terminal(self) -> int | None:

        result = None
        if not pf.legal_moves(self.game, self.state, []):
            result = pf.game_result(self.game, self.state, []) / pf.VALUE_MATE
            result = result if self.white_to_move else -result
        elif pf.is_optional_game_end(self.game, self.state, [])[0]:
            result = 0



        return result if result is not None else None

    def add_successors(self) -> None:
        successors = []
        for action in pf.legal_moves(self.game, self.state, []):

            next_state = pf.get_fen(self.game, self.state, [action])
            child = PyffishNode(
                    state=next_state,
                    game=self.game,
                    action=action,
                    parent=self,
                    depth=self.depth+1,
                    white_to_move=not self.white_to_move,
                    )

            successors.append(child)
        
        self.children = successors

    def fen(self) -> str:
        return self.state





