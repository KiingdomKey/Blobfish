import torch
import torchrl
import lightning.pytorch as pl
from torchrl.data import ReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer
from tensordict import TensorDict

import numpy as np

from concurrent.futures import ThreadPoolExecutor
import threading

from mcts import mcts, process_batch_loop

from nodes.minichess_node import MinichessNode
from nodes.python_chess_node import PythonChessNode

from networks.resnet import ResNet


from encoding import encode, policy_size

from move_mapping import move_to_index

import sys
import time

from torchinfo import summary

NUM_THREADS = 6
BATCHING = True # should not be off in this file
BATCH_SIZE = 16
RUNS = 20
NUM_MONTE_CARLO_SIMS=50

INPUT_SHAPE = (18, 5, 5)
OUTPUT_SHAPE = (policy_size(5),)

TEMP_OUTPUT_SHAPE = (NUM_THREADS, 18, 5, 5)

print(f"GIL enabled: {sys._is_gil_enabled()}")
print(f"Running {NUM_THREADS} threads")
print(f"Batching: {BATCHING}")



BUFFER_LOCK = threading.Lock()



def play_train_game(root_node: Node,
                    value_func = None,
                    num_simulations: int = 500,
                    thread_num: int = None,
                    buffer: ReplayBuffer = None) -> None:
    """
    Plays a full game until a terminal state is reached. Uses MCTS at each move
    Args:
        root_node:          The starting node for the game
        value_func:         How MCTS determines value if batching is off (please don't try and use this. it's just here because mcts needs that as a parameter...)
        num_simulations:    How many simulations should MCTS perform on each turn
        thread_num:         Thread id 
        buffer:             The replay buffer that gets added to after every turn

    """
    
    cur = root_node
    visited_states = []
    priors_list = []
    while cur.is_terminal() is None:

        best_action, value, priors = mcts(root=cur,
                                  value_func=value_func,
                                  num_runs=num_simulations,
                                  thread_num=thread_num,
                                  batching=True)

        # add one-hot encoding of picked move to prior data for the network to train off of
        a = torch.zeros_like(priors)
        idx = move_to_index(best_action, 5, black_to_move=not cur.white_to_move) 
        a[idx] = 1
        priors_list.append(a)

        visited_states.append(encode(cur.fen()))
        
        '''
        if buffer is not None:
            # Encode the state we just played from and add it and its estimated value from MCTS to the buffer
            cur_state = encode(cur.fen())
            data = TensorDict({
                "state": cur_state,
                "policy": a,
                "value": value
            })

            # Since multiple threads can access the buffer, we have to make sure there are no race conditions
            with BUFFER_LOCK:
                try:
                    buffer.add(data)
                except:
                    print("The replay buffer broke again") # if you see this print please dig me a nice hole to bury myself in before telling me
        '''

        cur = cur.get_child(best_action)
        
    states = torch.stack(visited_states)
    value = cur.is_terminal()
    if not root.white_to_move: # the terminal value is always from white's perspective, so flip it if playing black
        value *= -1
    values = np.full(states.shape[0], value)
    priors_tensor = torch.stack(priors_list)

    for state, value, priors in zip(states, values, priors_tensor):

 
        data = TensorDict({
                "state": state,
                "policy": priors,
                "value": value
            })
        # Since multiple threads can access the buffer, we have to make sure there are no race conditions
        with BUFFER_LOCK:
            try:
                buffer.add(data)
            except:
                print("The replay buffer broke again") # if you see this print please dig me a nice hole to bury myself in before telling me

    

def play_n_train_games(value_func,
                       root_node: Node,
                       num_simulations: int = 500,
                       num_games: int = 32,
                       buffer: ReplayBuffer = None) -> None:
    """
    Plays N training games in parallel until all games reach a terminal state
    Args:
        value_func:         what process_batch_loop calls to get the value for a state
        root_node:          Starting state for all games
        num_simulations:    How many simulations should MCTS perform on each turn (per game)
        num_games:          How many games to run at the same time
        buffer:             Replay buffer that each thread should add to
    """
            
    with ThreadPoolExecutor(max_workers=num_games + 1) as executor:

        for i in range(num_games):
            root = root_node.copy()
            executor.submit(play_train_game, root_node=root, num_simulations=num_simulations, buffer=buffer, thread_num=i)

        executor.submit(process_batch_loop, value_func)





    

if __name__ == "__main__":
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=1024)
    )


    node = MinichessNode(game="5x5gardner")

    # the following is to just get to a simplified endgame to test training
    all_moves = "c2c3 c4c3 d2e3 d4e3 d1d5 e5d5 b2b3 a4b3 a2b3 c3c2 c1a3 b4a3 a1a3 d5e5 a3b3 e5d5 c3c4 d5c4 b3b5 c4d5 b5a5 d5e5 a5c5 e5e3 c5b5 e4d4 b5e5 d4d3 e5e3 c4b5 e3c3 b5a5 e2e3 a5b5 e3e4 b5a5"
    moves = all_moves.split()
    for move in moves:
        node.add_successors()
        idx = -1
        for i in range(len(node.children)):
            if node.children[i].action == move:
                idx = i
        node = node.children[idx]

    root = node
    print(root)

    model = ResNet(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, value_weight=0.99)
    optimizer = model.configure_optimizers()
    
    start_time = time.time()

    for i in range(RUNS):

        play_n_train_games(value_func=model.evaluate,
                           root_node=root,
                           buffer=buffer,
                           num_simulations=NUM_MONTE_CARLO_SIMS,
                           num_games=NUM_THREADS)

        successes = 0
        fails = 0
        for _ in range(0, int((len(buffer)) / BATCH_SIZE)):

            sample = buffer.sample(BATCH_SIZE)
            y = sample["value"]
            successes += (y == 1).sum()
            fails += (y == -1).sum()

            optimizer.zero_grad()
            loss, _, _ = model.compute_loss(sample)
            loss.backward()
            optimizer.step()

        print(f"S: {successes}")
        print(f"F: {fails}")


        print(f"Run {i} done")





    print(f"Time taken: {time.time() - start_time}")

    torch.save(model.state_dict(), "temp.pt")



