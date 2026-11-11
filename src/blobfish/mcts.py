'''
Currently, the reward function is implemented as a modified Monte Carlo Tree Search.
The main change is that, instead of doing a rollout on a new state, we have the model predict the value of the state.
If the state is a terminal state, however, it does get the ground-truth value of the state (i.e. win/loss/draw)

Note: all of this is subject to change and is me wanting to mess around with stuff.

Ideas for future:
    Found a paper on combining MCTS with Minimax, could help with not missing specific short-term moves (need to do more research)

TODO:
    Find out how to get different threads to actually report their errors
'''

from collections.abc import Callable
from dataclasses import dataclass, field
import random

import chess
import pyffish as pf
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor
import threading
import queue

import sys
import time

from nodes.pyffish_node import PyffishNode
from nodes.python_chess_node import PythonChessNode
from nodes.minichess_node import MinichessNode

import warnings

# What the processing loop reads from. In the format (<thread_id>, <data>, <queue to signal completion to>)
task_queue = queue.Queue()



def get_active_workers() -> int:
    """
    Returns:
        The number of active MCTS threads
    """
    return threading.active_count() - 2 # -2 is for the processing thread and the main thread



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
    while cur:
        cur.total_value += value
        cur.visits += 1
        cur = cur.parent



def mcts(root: Node,
         num_runs: int,
         value_func: Callable = None,
         thread_num: int = None,
         batching=False) -> tuple[str, float]:
    """
    Performs MCTS with value estimation.

    Args:
        start_state:    Root position to start search at
        num_runs:       Max number of simulations
        value_func:     Function to call to determine the value of a non-terminal state. Return value should be a float. Input type TBD. Optional if batching
        thread_num:     Unique number for the current thread. Optional if not batching
        batching:       Whether or not to batch forward passes across all threads (required arguments change)
    """
    # Note: this doesn't actually print anything if threading (as threads don't show errors by default)
    # TODO: figure out how to get threads to print errors (should've done this ages ago...)

    assert (batching and thread_num is not None) or value_func is not None, "Invalid combination of parameters."

    for i in range(0, num_runs):

        cur = root
        # Get to a leaf node (if we're not already at one)
        while len(cur.children) > 0:
            
            # Pick child with highest metric value
            cur.children.sort()
            cur = cur.children[-1]

        # We haven't been to this node yet, so add its children and do backprop on its first child (we haven't evaluate the children, so it's ok to pick a random child)
        if not cur.expanded and cur.is_terminal() is None:
            
            cur.add_successors()
            cur.expanded = True
            cur = cur.children[0]

        # If this state is a terminal state, get ground-truth value
        result = cur.is_terminal()
        if result is not None:
            value = result

        # Non-terminal state, so get estimation of the state's value
        else:

            value = get_value_batching(thread_num, cur) if batching else value_func(cur.state)

        backprop(cur, value)


    # Find which action leads to the best immediate child
    idx = np.argmax(root.children)
    best_action = root.children[idx].action

    # Clean up if batching
    if batching:
        task_queue.put(None)

    return best_action, value



def get_value_batching(thread_num: int, state) -> float:
    """
    Adds a state to the batch to be processed. 

    Args:
        thread_num: Number associated with the calling thread. Used to know where to store / retrieve value
        state:      State to be processed.

    Returns:
        Estimated value of the state after processing.
    """

    response_queue = queue.Queue()
    data = thread_num

    # Send data over to be processed
    task_queue.put((thread_num, data, response_queue))

    # Wait for data to be processed
    returned_thread_num, response = response_queue.get()

    assert returned_thread_num == thread_num # if this fails then each thread is not getting its own data back (bad)

    return response



# Runs in a loop waiting to process a full batch
def process_batch_loop(value_func: Callable) -> None:
    """
    Loop that takes care of running forward pass batches on the network.
    Args:
        value_func:     Function to call to determine the value of a non-terminal state. Input parameters TBD.
    """

    if get_active_workers() == 0:
        return

    finished_threads = 0
    batch_size = get_active_workers()
    id_list = []
    data_list = []
    response_queue_list = []

    while get_active_workers() > 0:

        process_batch = False

        # As we lose workers, we want to lower the batch size (so that we actually process batches)
        if get_active_workers() < batch_size:

            batch_size = get_active_workers()

        try:
            # Get data from a thread
            received = task_queue.get(timeout=0.05)
            if received is None: # i.e. a thread has ended/is ending
                finished_threads += 1

                continue

            # Add data to the batch
            thread_num, data, response_queue = received

            id_list.append(thread_num)
            data_list.append(data)
            response_queue_list.append(response_queue)

        # If the queue timed-out
        except queue.Empty:
            if len(data_list) > 0:
                process_batch = True
            pass

        # Batch is full, so process it
        if len(data_list) >= batch_size:
            process_batch = True

        if process_batch:

            for i, (thread_num, response_queue) in enumerate(zip(id_list, response_queue_list)):
                batch = torch.tensor(data_list)
                batch_output = value_func(batch)

                response_queue.put((thread_num, 0))
                response_queue.task_done() # Signal a thread that its data is processed

            id_list = []
            data_list = []
            response_queue_list = []


# main is just here for testing purposes
'''
if __name__ == "__main__":

        
    game = "gardner"
    start = pf.start_fen(game)

    def value_func(state):
        return random.random()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_THREADS + 1) as executor:

        for i in range(NUM_THREADS):

            #root = PythonChessNode(state=chess.Board(), player_color='WHITE')
            #root = PyffishNode(state=start, game=game)
            root = MinichessNode(game="5x5gardner")
            executor.submit(mcts, root=root, num_runs=500, thread_num=i, batching=BATCHING)
        if BATCHING:
            executor.submit(process_batch_loop, value_func)
    
    print("time: ", time.time() - start_time)
'''

            

