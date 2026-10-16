'''
Currently, the reward function is implemented as a modified Monte Carlo Tree Search.
The main change is that, instead of doing a rollout on a new state, we have the model predict the value of the state.
If the state is a terminal state, however, it does get the ground-truth value of the state (i.e. win/loss/draw)

Expect any of this to change...

Ideas for future:
    Found a paper on combining MCTS with Minimax, could help with not missing specific short-term moves (need to do more research)

TODO:
    -finish main MCTS
    -fix all the inevitable errors with said MCTS 
    
'''

from dataclasses import dataclass, field
import random

import chess
import numpy as np


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

@dataclass
class Node:
    '''
    Dataclass for a node in the tree. Most members self-explanatory, but:
    
    Data members:
        state:              current board state
        parent:             reference to parent node (we need to be able to backtrack up the tree for backpropagation)
        children:           list of all children nodes
        total_value:       total value of itself plus (explored) children
        visits:             number of visits to this node
        depth:              depth of node (not actually needed by algortihm, it's just here for testing purposes)   

    '''
    state: chess.Board
    parent: "Node" = None
    children: list["Node"] = field(default_factory=list) # Equivalent to a default value of [] (empty list). Can't do it normally in a dataclass
    total_value: int = 0
    visits: int = 0
    depth: int = 0

    # Is here to make it easier to change out our MCTS metric (just have to replace the contents of this function)
    def mcts_metric(self):
        return UCB(T_child=self.total_value, c=2, n_parent=self.parent.visits, n_child=self.visits)

    # Here to be able to sort our nodes by metric value
    def __lt__(self, rhs):
        return self.mcts_metric() < rhs.mcts_metric()

    def __float__(self):
        return self.total_value / self.visits if self.visits > 0 else 0

def backprop(node: Node, value) -> None:
    """
    Performs MCTS backpropagation from the current node.
    Note: this function modifies node and all parents 

    Args:
        node: Node we want to do backpropagation from
        value: value to update nodes along the way with
    Returns:
        None
    """

    cur = node
    #print("doing backprop")
    while cur:
        cur.total_value += value
        cur.visits += 1
        #print(f"-Node at depth={cur.depth} now has T={cur.total_value}. Avg. value = {cur.total_value / cur.visits}")
        cur = cur.parent



# purely temp class, just here to test
class Agent():

    def __init__(self,
                 temp_value: int) -> None:
                 self.val = temp_value
    # Every state just gets a random value estimation (again, just to test)
    def value(self, state):
        return random.randint(0, 30)



def mcts(player: Agent, 
         player_color: chess.Color,
         start_state: chess.Board, 
         num_runs: int,):

    root = Node(start_state)
    black_turn = False if player_color == chess.WHITE else True

    for i in range(0, num_runs):
        cur = root
        #print("\nAt the top")

        # Get to a leaf node (if we're not already at one)
        while len(cur.children) != 0:
         #   print("Looking for leaf")

            # Pick child with highest metric value
            cur.children.sort()
            cur = cur.children[-1]

            black_turn = not black_turn

        # We've been to this leaf at least once, so add its children and do backprop on its first child (we haven't evaluate the children, so it's ok to pick a random child)
        if cur.visits > 0:
            #print("Been to this leaf already, adding new states")
            for action in cur.state.legal_moves:

                child = cur.state.copy()
                child.push_san(str(action)) 
                cur.children.append(Node(state=child, parent=cur, depth=cur.depth + 1))

            cur = cur.children[random.randint(0, len(cur.children) - 1)]

        
        # If this state is a terminal state, get ground-truth value
        if cur.state.is_game_over():
            outcome = cur.state.Outcome()
            if not outcome:
                value = 0
            else:
                value = 1 if outcome.winner == player_color else -1
        # Non-terminal state, so get estimation of the state's value
        else:
            value = player.value(cur.state.mirror() if black_turn else cur.state)

        backprop(cur, value)

    # Find which action leads to the best immediate child
    idx = np.argmax(root.children)
    best_action = root.children[idx].state.peek()

    for child in root.children:
        print(child.total_value, child.visits, child.total_value / child.visits if child.visits != 0 else "", f"\tMove: {child.state.peek()}")


    return best_action, value


if __name__ == "__main__":
    board = chess.Board()
    p1 = Agent(50)
    best_action, value = mcts(p1, chess.WHITE, board, 25)

    print(best_action, value)


            

