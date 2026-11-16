"""
Evaluating blobfish checkpoints + drawing the learning curve

This is how we answer "is it actually learning?" for the report. The prof specifically
asked for a graph over training, and since blobfish won't beat real stockfish right away,
we measure it against the whole ladder from benchmark.py instead. If it's learning,
its score should climb against the easy opponents first, then the harder ones.

What it does:
    1. finds every saved checkpoint (a snapshot of the network at some training round)
    2. plays each one against the ladder (random, capture, minimax, weakest stockfish)
       AND against the untrained network (the prof's "beats older versions of itself")
    3. saves all the results to a csv
    4. draws the graph: score vs training round, one line per opponent

It also draws a second graph of the training losses if train.py logged them.

Run from src/blobfish:
    PYTHON_GIL=0 python evaluate.py                           # play everything, save csv, draw graphs
    PYTHON_GIL=0 python evaluate.py --plot-only               # skip the games, just redraw from the csv
    PYTHON_GIL=0 python evaluate.py --make-test-checkpoints   # fake checkpoints to test with

Checkpoints go in checkpoints/ with the round number in the name, like round_005.pt.
Works with .pt files (what train.py saves) and .ckpt files (lightning's format).

Add FAIRY_STOCKFISH=/path/to/stockfish in front of the command to include stockfish.
Without it the stockfish games just get skipped.

Loss graph: if train.py writes results/train_log.csv with the columns
    round, policy_loss, value_loss
(one row per round, extra columns are fine) you also get results/training_loss.png
"""

import argparse
import csv
import glob
import os
import re
import time

import torch

import benchmark
from encoding import PLANES, policy_size
from network_player import make_network_player
from networks import NETWORKS, make_network

# we're only training on 5x5 gardner right now
BOARD_SIZE = 5

# same seed every time so "untrained" is the exact same network on every run.
# otherwise round 0 would be a different random network each time and the graph
# wouldn't have a consistent starting point
UNTRAINED_SEED = 0

# the opponents, weakest to strongest. these names match the players in benchmark.py.
# (from my benchmark runs: random < weakest stockfish < capture < minimax, but i put
# stockfish last here so the "can it beat stockfish yet" line is easy to find)
LADDER = ["random", "capture", "minimax1", "sf_d1_skill-20"]

# every checkpoint also plays the untrained network. if a later checkpoint beats
# the untrained one more and more, that's the "better than its old self" proof
SELF_OPPONENT = "untrained"

# columns in results/eval.csv
CSV_FIELDS = ["checkpoint", "round", "opponent", "games", "wins", "draws", "losses", "score", "avg_moves"]


# ---------- loading networks ----------

def new_network(arch="resnet", channels=64, blocks=6):
    # a brand new network with the same shape train.py uses.
    # the shapes have to match exactly or the saved weights won't fit into it
    return make_network(arch,
                        input_shape=(PLANES, BOARD_SIZE, BOARD_SIZE),
                        output_shape=(policy_size(BOARD_SIZE),),
                        channels=channels,
                        blocks=blocks)


def load_any(path, arch="resnet", channels=64, blocks=6):
    # train.py saves a plain .pt file (just the weights) but emily's network_player
    # expects lightning's .ckpt format. so this handles both instead of making
    # anybody change how they save stuff
    if path.endswith(".ckpt"):
        # lightning files remember their own settings so we can load them directly
        net = NETWORKS[arch].load_from_checkpoint(path, map_location="cpu")
    else:
        # .pt is only the weights, so build an empty network first and pour them in
        net = new_network(arch, channels, blocks)
        state = torch.load(path, map_location="cpu")
        # some ways of saving wrap the weights in another dict, so unwrap it if so
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
    # eval mode = we're using it, not training it
    net.eval()
    return net


def untrained_network():
    # round 0, the baseline everything gets compared to
    torch.manual_seed(UNTRAINED_SEED)
    net = new_network()
    net.eval()
    return net


def round_number(path):
    # pulls the training round out of the file name, "checkpoints/round_015.pt" -> 15
    # uses the LAST number in the name in case there are other numbers in there
    numbers = re.findall(r"\d+", os.path.basename(path))
    if not numbers:
        raise ValueError(f"No round number in checkpoint name: {path} (try round_005.pt)")
    return int(numbers[-1])


def find_checkpoints(folder):
    # every .pt and .ckpt in the folder, sorted by round so the graph goes left to right
    paths = glob.glob(os.path.join(folder, "*.pt")) + glob.glob(os.path.join(folder, "*.ckpt"))
    return sorted(paths, key=round_number)


# ---------- playing ----------

def evaluate_one(name, net, opponents, games, sample):
    # turns this network into a benchmark player (emily's network_player does that part)
    # and plays it against each opponent. play_match from my benchmark.py handles the
    # actual games, swapping colors every game so nobody always gets white
    benchmark.PLAYERS[name] = make_network_player(net, sample=sample)
    rows = []
    for opponent in opponents:
        start = time.time()
        r = benchmark.play_match(name, opponent, games)
        rows.append(r)
        print(f"   vs {opponent}: {r['wins']}W {r['draws']}D {r['losses']}L  "
              f"(score {r['score']}, {time.time() - start:.0f}s)")
    # take it back out so the player list doesn't fill up with old checkpoints
    del benchmark.PLAYERS[name]
    return rows


def run_evaluation(args):
    checkpoints = find_checkpoints(args.checkpoints)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoints}/ "
              f"(try --make-test-checkpoints to make some fake ones)")
        return False

    # if stockfish isn't installed, drop it instead of crashing so
    # everybody can still run this without building stockfish
    opponents = list(LADDER)
    if benchmark.find_stockfish() is None:
        print("Fairy-Stockfish not found (set FAIRY_STOCKFISH), skipping the Stockfish rung\n")
        opponents = [o for o in opponents if not o.startswith("sf_")]

    # the untrained network is also the "old self" opponent.
    # it ALWAYS samples its moves (picks randomly weighted by how much it likes each move).
    # if both networks just played their top move every time, it'd be the exact same
    # game over and over, and every self game would come back a draw
    benchmark.PLAYERS[SELF_OPPONENT] = make_network_player(untrained_network(), sample=True)

    all_rows = []

    # round 0 = the untrained network against the ladder. this is the starting point
    # of every line on the graph. (no "vs untrained" here, it'd just be playing itself)
    print("round 0 (untrained)")
    for r in evaluate_one("round_0", untrained_network(), opponents, args.games, args.sample):
        all_rows.append({"checkpoint": "untrained", "round": 0, **r})

    # then every saved checkpoint, against the ladder + the untrained network
    for path in checkpoints:
        rnd = round_number(path)
        print(f"round {rnd} ({path})")
        net = load_any(path, args.arch, args.channels, args.blocks)
        for r in evaluate_one(f"round_{rnd}", net, opponents + [SELF_OPPONENT], args.games, args.sample):
            all_rows.append({"checkpoint": path, "round": rnd, **r})

    # save everything. this overwrites the csv every full run on purpose,
    # so the csv always matches the checkpoints you actually have
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved results to {args.csv}")
    return True


# ---------- the learning curve graph ----------

# one color + one marker shape per line. the colors are checked to be colorblind safe,
# and the different shapes mean you can still tell the lines apart without color
# (like if someone prints the report in black and white)
LINE_STYLES = [
    ("#2a78d6", "o"),   # blue circles
    ("#eb6834", "s"),   # orange squares
    ("#1baf7a", "^"),   # green triangles
    ("#eda100", "D"),   # yellow diamonds
    ("#e87ba4", "v"),   # pink upside-down triangles
]

# nicer names for the graph than the code names
PRETTY_NAMES = {
    "random": "vs random",
    "capture": "vs capture",
    "minimax1": "vs minimax (1 ply)",
    "minimax2": "vs minimax (2 ply)",
    "sf_d1_skill-20": "vs Stockfish (weakest)",
    "sf_d1": "vs Stockfish d1",
    SELF_OPPONENT: "vs untrained self",
}


def plot(csv_path, out_path):
    # only reads the csv, never plays games. that's why --plot-only is so fast,
    # you can mess with how the graph looks without replaying everything
    import matplotlib
    matplotlib.use("Agg")  # draw straight to a file, no popup window
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{csv_path} is empty, nothing to plot")
        return

    # keep opponents in ladder order so each one gets the same color every time.
    # if a run skips stockfish the other lines don't change color on you
    order = LADDER + [SELF_OPPONENT]
    opponents = [o for o in order if any(r["opponent"] == o for r in rows)]
    opponents += sorted({r["opponent"] for r in rows} - set(opponents))

    ink, ink_soft, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # one line per opponent: x = training round, y = score
    # score = (wins + half the draws) / games, so 1 = won everything, 0.5 = even, 0 = lost everything
    max_round = max(int(r["round"]) for r in rows)
    end_labels = []
    for i, opponent in enumerate(opponents):
        color, marker = LINE_STYLES[i % len(LINE_STYLES)]
        points = sorted((int(r["round"]), float(r["score"])) for r in rows if r["opponent"] == opponent)
        xs, ys = zip(*points)
        label = PRETTY_NAMES.get(opponent, f"vs {opponent}")
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
        end_labels.append([ys[-1], label])

    # label each line at its right end so you don't have to keep looking at the legend.
    # if two lines end at the same score their labels would print on top of each other,
    # so this scoots them apart a little
    min_gap = 0.045
    end_labels.sort()
    for k in range(1, len(end_labels)):
        if end_labels[k][0] - end_labels[k - 1][0] < min_gap:
            end_labels[k][0] = end_labels[k - 1][0] + min_gap
    for y, label in end_labels:
        ax.annotate(label, (max_round, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=ink_soft, annotation_clip=False)

    # dashed line at 0.5 = "even". above it means blobfish is beating that opponent.
    # it stops where the data stops so it doesn't cross through the end labels
    ax.hlines(0.5, 0, max(max_round, 1), color=ink_soft, linewidth=1, linestyle="--", zorder=0)
    ax.text(0, 0.51, "even (0.5)", fontsize=8, color=ink_soft, va="bottom")

    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0, right=max(max_round, 1) * 1.35)  # extra room on the right for the labels
    ax.set_xlabel("training round", color=ink_soft)
    ax.set_ylabel("score  (win = 1, draw = ½, loss = 0)", color=ink_soft)
    ax.set_title("Blobfish score against each opponent over training", color=ink, loc="left", fontsize=12)

    # light gridlines, no box around the graph, so the lines are what stands out
    ax.grid(axis="y", color=grid, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink_soft, labelsize=9)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # no "round 2.5"
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=ink_soft)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Saved graph to {out_path}")


# ---------- the training loss graph ----------

def plot_losses(log_path, out_path):
    # this one shows the network getting less wrong on its training data.
    # the learning curve above shows whether that actually turns into winning games.
    # you want both for the report: loss going down = it's learning SOMETHING,
    # score going up = what it's learning is actually useful
    if not os.path.exists(log_path):
        print(f"No training log at {log_path}, skipping the loss graph")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    with open(log_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("round") not in (None, "")]
    if not rows:
        print(f"{log_path} is empty, skipping the loss graph")
        return
    rounds = [int(float(r["round"])) for r in rows]

    # two separate panels stacked on top of each other instead of one graph.
    # policy loss is usually around 1-4 and value loss is way smaller (under 1),
    # so on the same axis the value line would just look flat at the bottom
    #
    # policy loss = log loss (cross-entropy) between what the network guessed
    #               and what MCTS actually ended up choosing (the visit counts)
    # value loss  = squared error between the network's guess of who's winning
    #               and how the game actually ended (+1 / 0 / -1)
    ink, ink_soft, grid, surface = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
    panels = [("policy_loss", "policy loss (log loss vs MCTS visits)", "#2a78d6"),
              ("value_loss", "value loss (squared error vs game result)", "#eb6834")]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), dpi=150, sharex=True)
    fig.patch.set_facecolor(surface)
    for ax, (column, title, color) in zip(axes, panels):
        ax.set_facecolor(surface)
        ys = [float(r[column]) for r in rows]
        ax.plot(rounds, ys, color=color, linewidth=2)
        ax.set_title(title, color=ink, loc="left", fontsize=10)
        ax.set_ylim(bottom=0)  # start at 0 so the drop doesn't look bigger than it is
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink_soft, labelsize=9)
    axes[-1].set_xlabel("training round", color=ink_soft)
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle("Blobfish training loss (lower is better)", color=ink, x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=surface)
    print(f"Saved loss graph to {out_path}")


# ---------- fake checkpoints for testing ----------

def make_test_checkpoints(folder):
    # makes 3 untrained networks with different seeds and saves them like real checkpoints.
    # they're NOT trained so the graph won't show improvement, this is just to check
    # the whole thing runs before we have real checkpoints from training
    os.makedirs(folder, exist_ok=True)
    for i, rnd in enumerate([5, 10, 15], start=1):
        torch.manual_seed(100 + i)
        path = os.path.join(folder, f"round_{rnd:03d}.pt")
        torch.save(new_network().state_dict(), path)
        print(f"made {path}")
    print("These are NOT trained, so expect scores around the untrained baseline.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints", help="folder of saved checkpoints")
    ap.add_argument("--games", type=int, default=50, help="games per matchup (more = more accurate but slower)")
    ap.add_argument("--csv", default="results/eval.csv")
    ap.add_argument("--plot", default="results/learning_curve.png")
    ap.add_argument("--train-log", default="results/train_log.csv", help="CSV of losses per round written by train.py")
    ap.add_argument("--loss-plot", default="results/training_loss.png")
    ap.add_argument("--plot-only", action="store_true", help="skip the games, just redraw the graphs from the CSVs")
    ap.add_argument("--sample", action="store_true", help="networks sample moves instead of always playing the top one")
    ap.add_argument("--make-test-checkpoints", action="store_true", help="make fake checkpoints in checkpoints/test")
    # these have to match whatever network train.py used, or the weights won't load
    ap.add_argument("--arch", default="resnet", choices=list(NETWORKS))
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    args = ap.parse_args()

    if args.make_test_checkpoints:
        make_test_checkpoints(os.path.join(args.checkpoints, "test"))
    elif args.plot_only:
        # just redraw. the learning curve needs eval.csv to exist,
        # the loss graph checks for train_log.csv on its own
        if os.path.exists(args.csv):
            plot(args.csv, args.plot)
        plot_losses(args.train_log, args.loss_plot)
    else:
        # full run: play everything, then draw both graphs
        if run_evaluation(args):
            plot(args.csv, args.plot)
        plot_losses(args.train_log, args.loss_plot)
