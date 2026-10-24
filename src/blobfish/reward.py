'''
Currently, the reward function is implemented as a modified Monte Carlo Tree Search.
The main change is that, instead of doing a rollout on a new state, we have the model predict the value of the state.
If the state is a terminal state, however, it does get the ground-truth value of the state (i.e. win/loss/draw)

Note: all of this is subject to change and is me wanting to mess around with stuff.

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
import torch

from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Lock, Event

import sys
import time

print("GIL enabled: ", sys._is_gil_enabled())


NUM_THREADS = 64
BATCHING = True


batch = [None]*NUM_THREADS
batch_output = torch.rand(size=(NUM_THREADS, 1))


num_threads_alive_lock = Lock()
num_threads_alive = NUM_THREADS

# Keeps track of how many threads are done looking for a new node (and need it now processed)
num_threads_done_processing_lock = Lock()
num_threads_done_processing = 0

# Keeps track of how many threads are done waiting after the processing loop is finished
num_threads_done_waiting_lock = Lock()
num_threads_done_waiting = 0

# Makes sure all threads are synched up before touching anything related to the batch processing. Set to allow threads through the first time through.
preemptive = Event()
preemptive.set()

# Determines when to start processing a batch
ready_to_process = Event()

# Determines when threads can continue (after the batch is processed)
continue_search = Event()


def get_active_workers() -> int:
    """
    Returns:
        The number of active MCTS threads
    """
    return threading.active_count() - 2 # -2 is for the processing thread and the main thread


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
    # 
    def value(self, batch):
        return batch_output



def mcts(player_color: chess.Color,
         start_state: chess.Board, 
         num_runs: int,
         player: Agent = None,
         thread_num: int = None,
         batching=False) -> tuple[str, float]:
    """
    Performs MCTS with value estimation.

    Args:
        player_color:   whether the current player is black/white
        start_state:    Root position to start search at
        num_runs:       Max number of simulations
        player:         NN. Optional if batching
        thread_num:     Unique number for the current thread. Optional if not batching
        batching:       Whether or not to batch forward passes across all threads (required arguments change)
    """

    global num_threads_done_processing, num_threads_alive, num_threads_done_waiting

    root = Node(start_state)
    black_turn = False if player_color == chess.WHITE else True

    for i in range(0, num_runs):
        cur = root

        # Get to a leaf node (if we're not already at one)
        while len(cur.children) != 0:
            
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
            outcome = cur.state.outcome()
            if not outcome:
                value = 0
            else:
                value = 1 if outcome.winner == player_color else -1
        # Non-terminal state, so get estimation of the state's value
        else:
            value = get_value(thread_num, cur) if batching else 1


        backprop(cur, value)

    # Find which action leads to the best immediate child
    idx = np.argmax(root.children)
    best_action = root.children[idx].state.peek()

    # Clean up if batching
    if batching:
        with num_threads_alive_lock:
            num_threads_alive -= 1
            # If all other threads are already waiting to process but this one is going to get reaped, we need to let them process (this took forever to find)
            with num_threads_done_processing_lock:
                    # 2 Cases where we need to reset the processing loop
                    # 1. If all other threads are already waiting to process but this one is going to get reaped, we need to let them process (this took forever to find)
                    # 2. If all threads have ended their search, we have to make sure the processing loop doesn't get stuck if it's already waiting
                if (num_threads_alive == num_threads_done_processing) or (num_threads_alive == 0):
                    ready_to_process.set()
                    
    return best_action, value




def get_value(thread_num: int, state) -> float:
    """
    Adds a state to the batch to be processed. 

    Args:
        thread_num: Number associated with the calling thread. Used to know where to store / retrieve value
        state:      State to be processed.

    Returns:
        Estimated value of the state after processing.
    """

    global num_threads_done_processing, batch, num_threads_alive, num_threads_done_waiting, num_threads_alive

    batch[thread_num] = thread_num

    # Makes sure all threads are synched up. This isn't active the first time, because we want to make sure that no threads can reach the next part again while another thread is still in this function
    preemptive.wait()


    with num_threads_alive_lock:
        with num_threads_done_processing_lock:
            num_threads_done_processing += 1
            if num_threads_done_processing == num_threads_alive: # (If this is the last thread to check this)
                num_threads_done_processing = 0
                # Once the last thread is here, we reenable the previous checkpoint and start the processing
                preemptive.clear()
                ready_to_process.set()

    # All threads wait for the processing to finish
    continue_search.wait()

    with num_threads_alive_lock:
        with num_threads_done_waiting_lock:
            num_threads_done_waiting += 1
            if num_threads_done_waiting == num_threads_alive: # (If this is the last thread to check this)
                num_threads_done_waiting = 0
                # This is the last thread in this loop (doing anything relevant), so we reset the checkpoints
                continue_search.clear()
                preemptive.set() 

    return batch_output[thread_num, 0].item()



# Runs in a loop waiting to process a full batch
def process_batch_loop(player: Agent) -> None:
    """
    Loop that takes care of running forward pass batches on the network.
    Args:
        player: the network
    """

    global batch_output

    if not BATCHING:
        return

    while num_threads_alive > 0:
        # Wait until a new batch is ready (i.e. all threads are ready)
        ready_to_process.wait()
        ready_to_process.clear()

        if num_threads_alive < 1:
            break

        batch_output = torch.rand(size=(NUM_THREADS, 1)) # placeholder for forward pass

        # We're done processing so the threads can continue and read the data
        continue_search.set()
    


if __name__ == "__main__":
    board = chess.Board()
    p1 = Agent(50)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_THREADS + 1) as executor:
        executor.submit(process_batch_loop, p1)
        for i in range(NUM_THREADS):
            executor.submit(mcts, player_color=chess.WHITE, start_state=board.copy(), num_runs=500, thread_num=i, batching=BATCHING)


    print("time: ", time.time() - start_time)
            

