import torch
import torchrl
import lightning.pytorch as pl
from torchrl.data import ReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict
import tqdm

import numpy as np

from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
from dataclasses import dataclass

from mcts import mcts, process_batch_loop

from nodes.minichess_node import MinichessNode
from nodes.python_chess_node import PythonChessNode

from networks.resnet import ResNet

from encoding import encode, policy_size
from move_mapping import move_to_index

from network_player import load_network
from networks import NETWORKS

import sys
import os

BUFFER_LOCK = threading.Lock()



INPUT_SHAPE = (18, 5, 5)
OUTPUT_SHAPE = (policy_size(5),)



def play_train_game(root_node: Node,
                    value_func = None,
                    num_simulations: int = 500,
                    thread_num: int = None,
                    buffer: ReplayBuffer = None,
                    batching: bool = False,
                    sample_best_move: bool = True) -> None:
    """
    Plays a full game until a terminal state is reached. Uses MCTS at each move
    Args:
        root_node:          The starting node for the game
        value_func:         How MCTS determines value if batching is off
        num_simulations:    How many simulations should MCTS perform on each turn
        thread_num:         Thread id 
        buffer:             The replay buffer that gets added to after every turn
        batching:           Whether or not this thread should batch network forward operations
        sample
        sample_best_move:   Should MCTS pick its return move based on the distribution of visits or just return the most visited node's move
    """

    cur = root_node
    visited_states = []
    priors_list = []
    move_mask_list = []
    mcts_value_list = []

    while cur.is_terminal() is None:
        best_action, priors, action_visit_count_pairs = mcts(root=cur,
                                                             value_func=value_func,
                                                             num_runs=num_simulations,
                                                             thread_num=thread_num,
                                                             batching=batching,
                                                             sample_best_move=sample_best_move)
        # For the network to train, we give it in priors in the form of how many times MCTS visited that move
        # e.g.: if we had 4 moves, we visited the 2nd one twice and the 4th one once, then it would be [0, 2, 0, 1]
        move_visits = torch.zeros_like(priors)
        for action, visit_count in action_visit_count_pairs.items():
            idx = move_to_index(action, cur.board_size, black_to_move=not cur.white_to_move)
            move_visits[idx] = visit_count

        # Normalize move_visits
        move_visits = move_visits / (move_visits.sum())


        priors_list.append(move_visits)
        visited_states.append(encode(cur.fen()))
        move_mask_list.append(torch.tensor(cur.get_legal_move_mask())) # Network needs the move mask so we know what moves to ignore

        mcts_value_list.append(cur.total_value / cur.visits)

        cur = cur.get_child(best_action)
        cur.parent = None # We don't care about the parent anymore, so we set it to None so MCTS doesn't waste time doing backprop on it

    states = torch.stack(visited_states)
    value = cur.is_terminal()

    if not root.white_to_move: # the terminal value is always from white's perspective, so flip it if playing black (white winning is bad for black)
        value *= -1

    # Every move up to the end gets the value of the end of the game
    values = torch.tensor(np.full(states.shape[0], value))
    values[1::2] = -value # We want to train on this game from both white and black's perspective, so we need the value of a state to match the agent's perspective for that move

    mcts_values = torch.tensor(mcts_value_list)
    mcts_values = (values + mcts_values) / 2

    priors = torch.stack(priors_list)
    masks = torch.stack(move_mask_list)

    for state, value, priors, mask, mcts_value in zip(states, values, priors, masks, mcts_values):
        data = TensorDict({
                "state": state,
                "policy": priors,
                "mask": mask,
                "result": value,
                "value": mcts_value
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
                       buffer: ReplayBuffer = None,
                       batching: bool = False,
                       sample_best_move: bool = True) -> None:
    """
    Plays N training games in parallel until all games reach a terminal state
    Args:
        value_func:         what process_batch_loop calls to get the value for a state
        root_node:          Starting state for all games
        num_simulations:    How many simulations should MCTS perform on each turn (per game)
        num_games:          How many games to run at the same time
        buffer:             Replay buffer that each thread should add to
        batching:           Whether or not each thread should batch network forward operations
        sample_best_move:   Should MCTS pick its return move based on the distribution of visits or just return the most visited node's move
    """
            
    with ThreadPoolExecutor(max_workers=num_games + 1) as executor:

        for i in range(num_games):
            root = root_node.copy()
            executor.submit(play_train_game, root_node=root, num_simulations=num_simulations, buffer=buffer, thread_num=i, value_func=value_func, batching=batching, sample_best_move=sample_best_move)

        executor.submit(process_batch_loop, value_func, initial_threads=num_games)

    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--checkpoint", default=None, help="trained network to load, leave off for the untrained baseline")
    arg_parser.add_argument("--arch", default="resnet", choices=list(NETWORKS))
    arg_parser.add_argument("--channels", type=int, default=64)
    arg_parser.add_argument("--blocks", type=int, default=6)
    arg_parser.add_argument("--num-runs", type=int, default=100)
    arg_parser.add_argument("--games-per-run", type=int, default=16)
    arg_parser.add_argument("--mcts-sims", type=int, default=100)
    arg_parser.add_argument("--buffer-size", type=int, default=32768)
    arg_parser.add_argument("--checkpoint-every-n-runs", type=int, default=10, help="set to 0 to disable")
    arg_parser.add_argument("--batch-size", type=int, default=16)
    arg_parser.add_argument("--checkpoint-save-dir", type=str, default="./checkpoints")
    arg_parser.add_argument("--log-dir", type=str, default="./logs")
    arg_parser.add_argument("--no-sample", dest="sample", action="store_false", help="disable move sampling for mcts", default=True)
    arg_parser.add_argument("--no-save", dest="save", action="store_false", help="also append the results to results/benchmark.csv", default=True)
    arg_parser.add_argument("--no-multithreading", dest="multithreading", action="store_false",  default=True)
    arg_parser.add_argument("--no-benchmark", dest="benchmark", action="store_false", default=True)
    arg_parser.add_argument("--batching", dest="batching", action="store_true", default=False)

    args = arg_parser.parse_args()

    print(f"GIL enabled: {sys._is_gil_enabled()}")
    print(f"Running {args.games_per_run} threads")
    print(f"Batching: {args.batching}")

    i = 0
    run_folder_name = f"run_"
    while os.path.isdir(f"./{args.log_dir}/{run_folder_name}{i}"):
        i += 1
    run_folder_name += str(i)

    print("Writing to:")
    print(f"-{args.log_dir}/{run_folder_name}")
    print(f"-{args.checkpoint_save_dir}/{run_folder_name}")

    os.makedirs(f"./{args.log_dir}/{run_folder_name}")
    os.makedirs(f"./{args.checkpoint_save_dir}/{run_folder_name}")


    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size),
        batch_size=args.batch_size,
        sampler=SamplerWithoutReplacement()
    )

    root = MinichessNode(game="5x5gardner")

    model = load_network(checkpoint=args.checkpoint, arch=args.arch, channels=args.channels, blocks=args.blocks)
    optimizer = model.configure_optimizers()

    logger = pl.loggers.CSVLogger(save_dir=args.log_dir, name=run_folder_name) if args.benchmark else None

    for i in tqdm.tqdm(range(1, args.num_runs + 1)):

        play_n_train_games(value_func=model.evaluate,
                           root_node=root,
                           buffer=buffer,
                           num_simulations=args.mcts_sims,
                           num_games=args.games_per_run,
                           batching=args.batching)
        non_draws = 0
        draws = 0

        losses = []
        policy_losses = []
        value_losses = []


        for _, batch in enumerate(buffer):

            result = batch["result"]
            non_draws += (result == 1).sum() + (result == -1).sum() 
            draws += (result.shape[0] - (result == 1).sum() - (result == -1).sum())

            optimizer.zero_grad(set_to_none=True)
            loss, policy_loss, value_loss = model.compute_loss(batch)

            losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
  
            loss.backward()
            optimizer.step()


        if logger is not None:
            metrics = { "loss": np.mean(losses), 
                        "policy_loss": np.mean(policy_losses),
                        "value_loss": np.mean(value_losses),
                        "num_non_draws": non_draws,
                        "num_draws": draws}

            logger.log_metrics(metrics, step=i)
            logger.save()

        if args.checkpoint_every_n_runs > 0 and i % args.checkpoint_every_n_runs == 0:
            torch.save(model.state_dict(), f"./{args.checkpoint_save_dir}/{run_folder_name}/{args.blocks}_{args.channels}_{i}.pt")






