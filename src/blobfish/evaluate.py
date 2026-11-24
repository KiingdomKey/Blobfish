"""
Evaluate blobfish checkpoints and draw the learning curve.

For every saved checkpoint it plays a set of games against the benchmark ladder
(random, capture, minimax, weakest Stockfish) and against the untrained network,
saves everything to a CSV, and draws one graph: score vs training round, one line
per opponent. That graph is the "is it learning?" answer for the report.

Run from src/blobfish:
    PYTHON_GIL=0 python evaluate.py                       # play all checkpoints, save CSV, draw graph
    PYTHON_GIL=0 python evaluate.py --plot-only           # just redraw the graph from the saved CSV
    PYTHON_GIL=0 python evaluate.py --make-test-checkpoints   # make 3 fake checkpoints to try it out

Checkpoints: put them in checkpoints/ with the training round in the name,
e.g. round_005.pt, round_010.pt. Both .pt (torch.save of state_dict, what train.py
saves) and .ckpt (Lightning) files work.

Add FAIRY_STOCKFISH=/path/to/stockfish to include the Stockfish rung.
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

BOARD_SIZE = 5
UNTRAINED_SEED = 0

# Opponents in ladder order, weakest first. Stockfish is skipped if it isn't installed.
LADDER = ["random", "capture", "minimax1", "sf_d1_s0", "sf_d1", "sf_d3"]
# The "beats older versions of itself" line: every checkpoint also plays the untrained network.
SELF_OPPONENT = "untrained"

CSV_FIELDS = ["checkpoint", "round", "opponent", "games", "wins", "draws", "losses", "score", "avg_moves"]


# ---------- loading networks ----------

def new_network(arch="resnet", channels=64, blocks=6):
    """A fresh network with the same shape train.py uses."""
    return make_network(arch,
                        input_shape=(PLANES, BOARD_SIZE, BOARD_SIZE),
                        output_shape=(policy_size(BOARD_SIZE),),
                        channels=channels,
                        blocks=blocks)


def load_any(path, arch="resnet", channels=64, blocks=6):
    """Loads a .ckpt (Lightning) or a .pt (plain state_dict) checkpoint."""
    if path.endswith(".ckpt"):
        net = NETWORKS[arch].load_from_checkpoint(path, map_location="cpu")
    else:
        net = new_network(arch, channels, blocks)
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:  # some saves wrap it
            state = state["state_dict"]
        net.load_state_dict(state)
    net.eval()
    return net


def untrained_network():
    """The same untrained network every time (fixed seed), so round 0 is reproducible."""
    torch.manual_seed(UNTRAINED_SEED)
    net = new_network()
    net.eval()
    return net


def round_number(path):
    """'checkpoints/round_015.pt' -> 15. Uses the last number in the file name."""
    numbers = re.findall(r"\d+", os.path.basename(path))
    if not numbers:
        raise ValueError(f"No round number in checkpoint name: {path} (try round_005.pt)")
    return int(numbers[-1])


def find_checkpoints(folder):
    paths = glob.glob(os.path.join(folder, "*.pt")) + glob.glob(os.path.join(folder, "*.ckpt"))
    return sorted(paths, key=round_number)


# ---------- playing ----------

def evaluate_one(name, net, opponents, games, sample, max_mcts_sims):
    """Plays one network against each opponent. Returns a list of result dicts."""
    benchmark.PLAYERS[name] = make_network_player(net, sample=sample, max_mcts_sims=max_mcts_sims)
    rows = []
    for opponent in opponents:
        start = time.time()
        r = benchmark.play_match(name, opponent, games)
        rows.append(r)
        print(f"   vs {opponent}: {r['wins']}W {r['draws']}D {r['losses']}L  "
              f"(score {r['score']}, {time.time() - start:.0f}s)")
    del benchmark.PLAYERS[name]
    return rows


def run_evaluation(args):
    checkpoints = find_checkpoints(args.checkpoints)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoints}/ "
              f"(try --make-test-checkpoints to make some fake ones)")
        return False

    opponents = list(LADDER)
    if benchmark.find_stockfish() is None:
        print("Fairy-Stockfish not found (set FAIRY_STOCKFISH), skipping the Stockfish rung\n")
        opponents = [o for o in opponents if not o.startswith("sf_")]

    # The untrained network is both round 0 and the "older self" every checkpoint plays.
    # It always samples its moves: two networks that each play their top move would
    # just replay the same game over and over.
    benchmark.PLAYERS[SELF_OPPONENT] = make_network_player(untrained_network(), sample=True, max_mcts_sims=args.max_mcts_sims)

    all_rows = []

    print("round 0 (untrained)")
    for r in evaluate_one("round_0", untrained_network(), opponents, args.games, args.sample, args.max_mcts_sims):
        all_rows.append({"checkpoint": "untrained", "round": 0, **r})

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    for path in checkpoints:
        rnd = round_number(path)
        print(f"round {rnd} ({path})")
        net = load_any(path, args.arch, args.channels, args.blocks)
        for r in evaluate_one(f"round_{rnd}", net, opponents + [SELF_OPPONENT], args.games, args.sample, args.max_mcts_sims):
            all_rows.append({"checkpoint": path, "round": rnd, **r})

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved results to {args.csv}\n")

    return True


# ---------- the graph ----------

# Colors in a fixed order (checked for colorblind safety), plus a different marker
# shape per line so they can be told apart without color.
LINE_STYLES = [
    ("#2a78d6", "o"),
    ("#eb6834", "s"),
    ("#1baf7a", "^"),
    ("#eda100", "D"),
    ("#e87ba4", "v"),
    ("#000000", "o"),
    ("#8297b5", "s"),
]

PRETTY_NAMES = {
    "random": "vs random",
    "capture": "vs capture",
    "minimax1": "vs minimax (1 ply)",
    "minimax2": "vs minimax (2 ply)",
    "sf_d1_skill-20": "vs Stockfish (weakest)",
    "sf_d1_s0": "vs Stockfish d1 Skill 0 (intermediate)",
    "sf_d1": "vs Stockfish d1",
    "sf_d3": "vs Stockfish d3",
    SELF_OPPONENT: "vs untrained self",
}


def plot(csv_path, out_path):
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{csv_path} is empty, nothing to plot")
        return

    # keep opponents in ladder order so colors always mean the same thing
    order = LADDER + [SELF_OPPONENT]
    opponents = [o for o in order if any(r["opponent"] == o for r in rows)]
    opponents += sorted({r["opponent"] for r in rows} - set(opponents))

    ink, ink_soft, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

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

    # direct label at the end of each line, nudged apart so lines ending at the
    # same score don't print their labels on top of each other
    min_gap = 0.045
    end_labels.sort()
    for k in range(1, len(end_labels)):
        if end_labels[k][0] - end_labels[k - 1][0] < min_gap:
            end_labels[k][0] = end_labels[k - 1][0] + min_gap
    for y, label in end_labels:
        ax.annotate(label, (max_round, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=ink_soft, annotation_clip=False)

    # stop the "even" line where the data stops, so it doesn't run through the end labels
    ax.hlines(0.5, 0, max(max_round, 1), color=ink_soft, linewidth=1, linestyle="--", zorder=0)
    ax.text(0, 0.51, "even (0.5)", fontsize=8, color=ink_soft, va="bottom")

    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0, right=max(max_round, 1) * 1.35)  # room for the end labels
    ax.set_xlabel("training round", color=ink_soft)
    ax.set_ylabel("score  (win = 1, draw = ½, loss = 0)", color=ink_soft)
    ax.set_title("Blobfish score against each opponent over training", color=ink, loc="left", fontsize=12)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink_soft, labelsize=9)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # whole rounds only
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=ink_soft)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Saved graph to {out_path}")


# ---------- test checkpoints ----------

def make_test_checkpoints(folder):
    """Saves 3 untrained networks with different seeds, just to try the pipeline out."""
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
    ap.add_argument("--games", type=int, default=50, help="games per matchup")
    ap.add_argument("--csv", default="results/eval.csv")
    ap.add_argument("--plot", default="results/learning_curve.png")
    ap.add_argument("--plot-only", action="store_true", help="skip the games, just redraw the graph from the CSV")
    ap.add_argument("--sample", action="store_true", help="networks sample moves instead of always playing the top one")
    ap.add_argument("--make-test-checkpoints", action="store_true", help="make fake checkpoints in checkpoints/test")
    ap.add_argument("--arch", default="resnet", choices=list(NETWORKS))
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--max-mcts-sims", type=int, default=0)
    args = ap.parse_args()

    if args.make_test_checkpoints:
        make_test_checkpoints(os.path.join(args.checkpoints, "test"))
    elif args.plot_only:
        plot(args.csv, args.plot)
    else:
        if run_evaluation(args):
            plot(args.csv, args.plot)
