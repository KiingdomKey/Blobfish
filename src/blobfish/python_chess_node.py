from base_node import Node

import chess

class PythonChessNode(Node):
    def __init__(self,
                 state: chess.Board,
                 player_color: chess.Color,
                 **kwargs):
        """
        Args:
            state:          current board state 
            player_color:   color the player starts as
        """
        super().__init__(**kwargs)
        self.state = state
        self.player_color = player_color

    def is_terminal(self) -> int | None:

        result = None
        if self.state.is_game_over():
            outcome = self.state.outcome()
            if not outcome:
                result = 0
            else:
                value = 1 if outcome.winner == self.player_color else -1

        return result

    def add_successors(self) -> None:
        successors = []
        for action in self.state.legal_moves:
            child = self.state.copy()
            child.push_san(str(action))
            successors.append(PythonChessNode(
                                              state=child,
                                              player_color=self.player_color,
                                              action=action,
                                              parent=self,
                                              depth=self.depth+1))

        self.children = successors

    def fen(self) -> str:
        return self.state.fen()




