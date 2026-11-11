import torch
import torchrl
from torchrl.data import ReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer
from tensordict import TensorDict

from concurrent.futures import ThreadPoolExecutor
import threading

from mcts import mcts, process_batch_loop

from nodes.minichess_node import MinichessNode
from nodes.python_chess_node import PythonChessNode


from encoding import encode

import sys
import time

print(f"GIL enabled: {sys._is_gil_enabled()}")


NUM_THREADS = 32
BATCHING = True
BATCH_SIZE = 16
RUNS = 1

TEMP_OUTPUT_SHAPE = (NUM_THREADS, 18, 5, 5)

print(f"GIL enabled: {sys._is_gil_enabled()}")
print(f"Running {NUM_THREADS}")
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
    while cur.is_terminal() is None:

        best_action, value = mcts(root=cur,
                                  value_func=value_func,
                                  num_runs=num_simulations,
                                  thread_num=thread_num,
                                  batching=True)


        if buffer is not None:
            # Encode the state we just played from and add it and its estimated value from MCTS to the buffer
            cur_state = encode(cur.fen())
            data = TensorDict({
                "state": cur_state,
                "reward": value
            })

            # Since multiple threads can access the buffer, we have to make sure there are no race conditions
            with BUFFER_LOCK:
                try:
                    buffer.add(data)
                except:
                    print("The replay buffer broke again") # if you see this print please dig me a nice hole to bury myself in before telling me


        cur = cur.get_child(best_action)

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
        storage=LazyTensorStorage(max_size=4048)
    )

    def get_dummy_data(x):
        return torch.rand(x.shape[0])

    root = MinichessNode(game="5x5gardner")
    
    start_time = time.time()

    for _ in range(RUNS):

        play_n_train_games(value_func=get_dummy_data,
                            root_node=root,
                            buffer=buffer,
                            num_simulations=50,
                            num_games=4)

        sample = buffer.sample(BATCH_SIZE)
        
        x = sample[:]["state"]
        y = sample[:]["reward"]

        print(x.shape)
        print(y)
        print(buffer)

    print(f"Time taken: {time.time() - start_time}")



