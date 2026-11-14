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

from encoding import encode

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
        value = -value



def mcts(root: Node,
         num_runs: int,
         value_func: Callable = None,
         thread_num: int = None,
         batching=False,
         sample_best_move: bool = True) -> tuple[str, np.array, dict[str, int]]:
    """
    Performs MCTS with value estimation.

    Args:
        start_state:        Root position to start search at
        num_runs:           Max number of simulations
        value_func:         Function to call to determine the value of a non-terminal state. Return value should be a float. Input type TBD. Optional if batching
        thread_num:         Unique number for the current thread. Optional if not batching
        batching:           Whether or not to batch forward passes across all threads (required arguments change)
        sample_best_move:    Should the final returned move be sampled proportional to each child's visit count, or should it just pick the child action with the highest visit count
    Returns:
        best_action:                The best move from the position given
        root_priors:                The priors of the root node
        action_visit_count_pairs    A dict mapping each action to how many times its associated child was visited
    """
    # Note: this doesn't actually print anything if threading (as threads don't show errors by default)
    # TODO: figure out how to get threads to print errors (should've done this ages ago...)

    assert (batching and thread_num is not None) or value_func is not None, "Invalid combination of parameters."

    while root.visits < num_runs:

        cur = root

        # Get to a leaf node (if we're not already at one)
        while len(cur.children) > 0:

            # Pick child with highest metric value
            best_idx, best_val = -1, float('-inf')
            for i in range(len(cur.children)):
                if cur.children[i].mcts_metric() > best_val:
                    best_idx = i
                    best_val = cur.children[i].mcts_metric()

            cur = cur.children[best_idx]
            
    
        # We haven't been to this node yet, so add its children and do backprop on its first child (we haven't evaluated the children, so it's ok to pick a random child)
        if not cur.expanded and cur.is_terminal() is None:

            cur.add_successors()
            cur.expanded = True

        # If this state is a terminal state, get ground-truth value
        result = cur.is_terminal()
        if result is not None:
            value = result if cur.white_to_move else -result

        # Non-terminal state, so get estimation of the state's value
        else:
            if not batching:
                state = encode(cur.fen()).unsqueeze(dim=0)
                mask = torch.tensor(cur.get_legal_move_mask()).unsqueeze(dim=0)
             
                policy_priors, value = value_func(state, mask)

                policy_priors = policy_priors.squeeze(0)
                value = value.item()
                
                cur.add_priors_to_children_and_self(policy_priors)

            else:

                policy_priors, value = get_value_batching(thread_num, cur)
                cur.add_priors_to_children_and_self(policy_priors)

        backprop(cur, value)

    visits = torch.tensor([c.visits for c in root.children]).to(torch.float32)

    sampling = torch.softmax(visits, dim=0)

    if sample_best_move:
        idx = torch.multinomial(sampling, num_samples=1).item()
        best_action = root.children[idx].action
    else:
        # Find which action leads to the best immediate child
        best_action, highest_visits = "", 0
        for child in root.children:
            if child.visits > highest_visits:
                highest_visits = child.visits
                best_action = child.action

    # Get visit counts
    action_visit_count_pairs = {}
    for child in root.children:
        action_visit_count_pairs[child.action] = child.visits

    # Clean up if batching
    if batching:
        task_queue.put(None)

    return best_action, root.priors, action_visit_count_pairs



def get_value_batching(thread_num: int, cur: Node) -> float:
    """
    Adds a state to the batch to be processed. 

    Args:
        thread_num: Number associated with the calling thread. Used to know where to store / retrieve value
        cur:        Current node

    Returns:
        Estimated value of the state after processing.
    """

    response_queue = queue.Queue()
    data = encode(cur.fen())

    # Get move mask
    mask = cur.get_legal_move_mask()

    # Send data over to be processed
    task_queue.put((thread_num, (data, mask), response_queue))

    # Wait for data to be processed
    returned_thread_num, response = response_queue.get()
    p, v = response

    assert returned_thread_num == thread_num # if this fails then each thread is not getting its own data back (bad)

    return p, v



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
    mask_list = []
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
            thread_num, (data, mask), response_queue = received

            id_list.append(thread_num)
            data_list.append(data)
            mask_list.append(torch.tensor(mask))
            response_queue_list.append(response_queue)

        # If the queue timed-out
        except queue.Empty:
            # If we have anything waiting to be processed, we might as well do it
            if len(data_list) > 0:
                process_batch = True
            pass

        # Batch is full, so process it
        if len(data_list) >= batch_size:
            process_batch = True

        if process_batch:

            for i, (thread_num, response_queue) in enumerate(zip(id_list, response_queue_list)):
                try:

                    batch = torch.stack(data_list)
                    masks = torch.stack(mask_list)

                    batch_output = value_func(batch, masks)

                    p = batch_output[0][i]
                    v = batch_output[1][i].item()

                    response_queue.put((thread_num, (p, v)))
                    response_queue.task_done() # Signal a thread that its data is processed
                except:
                    print("oh. something went wrong with the batching.")


            id_list = []
            data_list = []
            mask_list = []
            response_queue_list = []


            

