import chess
import numpy as np
import torch

from move_mapping import move_to_index, legal_mask




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

def PUCT(T_child: float,
         c: float,
         n_parent: int,
         n_child: int,
         p: float) -> float:
    """
    Calculates the Predictor Upper Confidence Bound applied to Trees (PUCT (I think this is what the acronym's short for?)) value for the given inputs.
    Note: The formula is a little different from UCB (it's not just adding an extra term)

    Args:
        T_child:    T value (the total sum of values of explored children)
        C:          exploration constant
        n_parent:   number of times the parent has been visited
        n_child:    number of times the child has been visited
        p:          Prior of the parent to the child
    
    Returns:
        UCB value given the above arguments. Returns infinity if the child has never been explored
    """
    if n_child == 0:
        Q = 0
    else:
        Q = (T_child / n_child)

    return Q + c * p * np.sqrt(n_parent) / (1 + n_child) 


class Node:
    '''
    Class a node in the tree. Most members self-explanatory, but:
    
    Data members:
        action:             what move was made to get here from parent
        parent:             reference to parent node (we need to be able to backtrack up the tree for backpropagation)
        children:           list of all children nodes
        total_value:        total value of itself plus (explored) children
        visits:             number of visits to this node
        depth:              depth of node (not actually needed by algortihm, it's just here for testing purposes)   
        prior_from_parent:  The prior from the parent state when predicting this state (i.e. the actor's estimated goodness from the parent state to this state)
        priors:             Move priors of this state 
        white_to_move:      Whether or not it's currently white's turn

    Child class should have the following data members:
        state
        board_len

    Child class should implement, at least:
        copy
        is_terminal
        add_successors
        fen
    '''
    def __init__(self,
                 action: str = None,
                 parent: "Node" = None,
                 children: list["Node"] = [], 
                 total_value: int = 0,
                 visits: int = 0,
                 depth: int = 0,
                 prior_from_parent: float = None,
                 priors: np.array = None,
                 white_to_move: bool = True,):
        self.action = action
        self.parent = parent
        self.children = children
        self.total_value = total_value
        self.visits = visits
        self.depth = depth
        self.expanded = False
        self.prior_from_parent = prior_from_parent
        self.priors = priors
        self.white_to_move = white_to_move
    

    # Is here to make it easier to change out our MCTS metric (just have to replace the contents of this function)
    def mcts_metric(self):
        """
        We have to make T_child negative as the child's perspective is always opposite to the parent's (if the child's value is -1 (e.g. bad for black), the parent should see that as a 1 (bad for black = good for white!))
        """
        if self.prior_from_parent is not None:
            score = PUCT(T_child=-self.total_value, c=2, n_parent=self.parent.visits, n_child=self.visits, p=self.prior_from_parent)
        else:
            score = UCB(T_child=-self.total_value, c=2, n_parent=self.parent.visits, n_child=self.visits)


        return score

    # Here to be able to sort our nodes by metric value
    def __lt__(self, rhs):
        return self.mcts_metric() < rhs.mcts_metric()

    def __float__(self):
        return self.total_value / self.visits if self.visits > 0 else 0

    # For pretty-printing
    def __str__(self) -> str:
        rows = self.fen().split()[0].split("/")
        size = len(rows)
        lines = []
        for i, row in enumerate(rows):
            squares = []
            for ch in row:
                squares += ["."] * int(ch) if ch.isdigit() else [ch]
            lines.append(f"{size - i} " + " ".join(squares))
        lines.append("  " + " ".join("abcdefgh"[:size]))
        return "\n".join(lines)

    def get_legal_move_mask(self):
        legal_moves = [c.action for c in self.children]
        mask = legal_mask(legal_moves=legal_moves, size=self.board_size, black_to_move=not self.white_to_move)
        return mask


    def add_priors_to_children_and_self(self, priors: np.array) -> None:
        """
        Gives priors from parent to each child respectively
        Args:
            priors:     a flattened NxN array of move priors
        """
        self.priors = priors
        if len(self.children) == 0:
            return

        for child in self.children:
            idx = move_to_index(child.action, self.board_size, black_to_move=not self.white_to_move)
            child.prior_from_parent = priors[idx]

    def get_child(self, action: str) -> Node | None:

        for child in self.children:
            if child.action == action:
                return child

        return None

    def copy(self) -> Node:
        """
        Returns: A deep copy of the object. Will *not* copy any children (we should never be in a situation where we do that anyways).
                 Basically, it returns a blank state node that starts from the state it was copied from.
        """
        raise NotImplementedError

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
