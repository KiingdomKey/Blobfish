"""
Network player for blobfish

Turns a network into a benchmark player so we can see how it does against the other players
in benchmark.py. No search, it just looks at the board once and plays the move the policy likes best,
so this measures what the network learned by itself without mcts helping it out

in:  a PolicyValueNet (untrained, or loaded from a checkpoint)
out: a player function with the same shape as the ones in benchmark.py, (moves_so_far, rng) -> "b1c3"

run from src/blobfish:
     python network_player.py                          # untrained network, the baseline
     python network_player.py --checkpoint net.ckpt    # a trained one
     python network_player.py --sample --games 50      # sample moves instead of always the top one
"""

import argparse
import time

import pyffish as sf
import torch

import benchmark
from encoding import PLANES, encode_with_mask, policy_size
from move_mapping import add_queen_promotion, index_to_move
from networks import NETWORKS, make_network


def make_network_player(net, sample=False):
    # gardner is the only board benchmark.py plays right now so the size is fixed at 5
    size = 5

    def network_player(moves_so_far, rng):
        # benchmark.py keeps games as a list of moves so we ask pyffish for the position and its legal moves
        fen = sf.get_fen(benchmark.GAME, benchmark.START, moves_so_far)
        legal = sf.legal_moves(benchmark.GAME, benchmark.START, moves_so_far)

        # one board in, move probabilities out, with illegal moves already zeroed
        # unsqueeze(0) makes it a batch of 1 since the network always wants a batch
        planes, mask = encode_with_mask(fen, legal)
        probs, _ = net.evaluate(planes.unsqueeze(0), mask.unsqueeze(0))
        probs = probs[0]

        # greedy takes the top move every time, sampling picks moves in proportion to their probability
        # so the games arent all identical
        if sample:
            index = rng.choices(range(len(probs)), weights=probs.tolist())[0]
        else:
            index = int(probs.argmax())

        # slot number back to a move string, flipped back to the real board if its blacks turn
        # a pawn reaching the last rank by a slide comes back without its q so we add it
        black_to_move = len(moves_so_far) % 2 == 1
        move = add_queen_promotion(index_to_move(index, size, black_to_move), legal)

        # if this ever fires the encoding, the mask and the move mapping disagree about the board
        # which is exactly the kind of bug we want to catch before training
        if move not in legal:
            raise RuntimeError(f"network picked {move} which isnt legal in {fen}, legal moves are {legal}")
        return move

    return network_player


def load_network(checkpoint=None, arch="resnet", channels=64, blocks=6, seed=0):
    # a checkpoint brings its own settings so we just load it
    if checkpoint:
        return NETWORKS[arch].load_from_checkpoint(checkpoint, map_location="cpu")

    # otherwise a fresh untrained one, seeded so the baseline is the same every run
    torch.manual_seed(seed)
    return make_network(arch,
                        input_shape=(PLANES, 5, 5),
                        output_shape=(policy_size(5),),
                        channels=channels,
                        blocks=blocks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="trained network to load, leave off for the untrained baseline")
    ap.add_argument("--arch", default="resnet", choices=list(NETWORKS))
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sample", action="store_true", help="sample moves instead of always playing the top one")
    ap.add_argument("--save", action="store_true", help="also append the results to results/benchmark.csv")
    args = ap.parse_args()

    net = load_network(args.checkpoint, args.arch, args.channels, args.blocks)
    name = "network" if args.checkpoint is None else f"network({args.checkpoint})"
    benchmark.PLAYERS[name] = make_network_player(net, sample=args.sample)
    print(f"{name}: {args.arch}, {args.blocks} blocks x {args.channels} channels, "
          f"{'sampling' if args.sample else 'greedy'}\n")

    # bottom of the ladder first, an untrained network should land around random
    matchups = [(name, "random"), (name, "capture"), (name, "minimax1")]

    rows = []
    for a, b in matchups:
        start = time.time()
        r = benchmark.play_match(a, b, args.games)
        rows.append(r)
        print(f"{a} vs {b}: {r['wins']}W {r['draws']}D {r['losses']}L  "
              f"(score {r['score']}, avg {r['avg_moves']} moves, {time.time() - start:.0f}s)")
        print(f"   endings: {r['endings']}")

    if args.save:
        benchmark.save_results(rows)
        print("Saved to results/benchmark.csv")
