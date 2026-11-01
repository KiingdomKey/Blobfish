import numpy as np
import chess

def UCB(T_child: float,
        c: float,
        n_parent: int,
        n_child: int) -> float:
    """
    Calculates the Upper Confidence Bound (UCB) value for the given inputs

    Args:
        T_child:    T value (the total sum of values of explored children)
        C:          exploration constant
        n_parent:   number of times the parent has been visited
        n_child:    number of times the child has been visited
    
    Returns:
        UCB value given the above arguments. Returns infinity if the child has never been explored
    """

    if n_child == 0:
        return float('inf')
    else:
        return (T_child / n_child) + c * np.sqrt(np.log(n_parent) / n_child) 

class Node:
    '''
    Class a node in the tree. Most members self-explanatory, but:
    
    Data members:
        state:              current board state
        action:             what move was made to get here from parent
        parent:             reference to parent node (we need to be able to backtrack up the tree for backpropagation)
        children:           list of all children nodes
        total_value:       total value of itself plus (explored) children
        visits:             number of visits to this node
        depth:              depth of node (not actually needed by algortihm, it's just here for testing purposes)   

    Child class should implement, at least:
        is_terminal
        add_successors
    '''
    def __init__(self,
                 action: str = None,
                 parent: "Node" = None,
                 children: list["Node"] = [], 
                 total_value: int = 0,
                 visits: int = 0,
                 depth: int = 0):
        self.action = action
        self.parent = parent
        self.children = children
        self.total_value = total_value
        self.visits = visits
        self.depth = depth
        self.expanded = False
    

    # Is here to make it easier to change out our MCTS metric (just have to replace the contents of this function)
    def mcts_metric(self):
        return UCB(T_child=self.total_value, c=2, n_parent=self.parent.visits, n_child=self.visits)

    # Here to be able to sort our nodes by metric value
    def __lt__(self, rhs):
        return self.mcts_metric() < rhs.mcts_metric()

    def __float__(self):
        return self.total_value / self.visits if self.visits > 0 else 0

    def __str__(self) -> str:
        board = chess.Board(fen=self.fen())
        return str(board)

    def is_terminal(self) -> float | None:
        """
        Returns:
            A float value if a terminal state was reached or None if no terminal state was reached
        """
        raise NotImplementedError

    def add_successors(self) -> None:
        """
        Adds all successors of self.state to self.children
        """

        raise NotImplementedError

    def fen(self) -> str:
        """
        Returns:
            The current board state in FEN format
        """

        raise NotImplementedError


