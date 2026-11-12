# Blobfish

A chess agent that learns entirely through self-play by combining **Monte Carlo Tree Search (MCTS)** with **Actor-Critic reinforcement learning**, similar in spirit to AlphaZero but at a much smaller scale.

A neural network suggests promising moves (the *actor*) and estimates how good positions are (the *critic*). MCTS uses those estimates to search selectively instead of by brute force. Progress is measured by playing Blobfish against Stockfish at increasing difficulty levels.

See [`4350_project_abstract.pdf`](docs/4350_project_abstract.pdf) for the full project proposal.

## How it works:

Each turn of self-play:

1. **Network**: the actor gives move probabilities π for the current board; the critic gives a value estimate V.
2. **MCTS**: repeated for a fixed number of simulations, with network parameters held fixed:
   - **Select**: descend the tree using a selection metric (UCB1 to start).
   - **Expand**: reach an unexplored state.
   - **Evaluate**: use the true result if the game is over, otherwise the critic's V(s) (no random rollouts).
   - **Back up**: pass the value up the search path.
3. **Move**: play the move MCTS selects.
4. **Update**: update the actor and critic using the advantage computed from the MCTS feedback.

## Board sizes

Training starts on small boards and moves up in size as we collect data:

| Size | Variant | Rules from |
|---|---|---|
| 5×5 | Gardner minichess | pyffish (`gardner`) |
| 6×6 | Los Alamos chess (no bishops) | pyffish (`losalamos`) |
| 8×8 | Standard chess | pyffish (`chess`) or python-chess |

The board encoding keeps the same 18 input planes at every size (12 for pieces, one per type and color, plus castling, en passant and the 50-move counter), so the network's convolutional trunk learned on a smaller board can be reused as a starting point for the next size up. The policy and value heads depend on the board size and are retrained.

## Libraries

| Library | What it does |
|---|---|
| [pyffish](https://pypi.org/project/pyffish/) | Python bindings for Fairy-Stockfish. Handles the rules for every board size: legal moves, starting positions and game-over detection, with positions as FEN strings. |
| [python-chess](https://python-chess.readthedocs.io/) | Rules for standard 8×8 chess, and talking to UCI engines like Stockfish for evaluation. |
| [PyTorch](https://pytorch.org/) | Tensors and the neural network. |
| [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) | Training loop on top of PyTorch. |

The board encoding takes a FEN string, so it works with positions from either pyffish or python-chess (`board.fen()`).

## Setup

Requires Python 3.10+.

```bash
python -m venv blobfish-env
source blobfish-env/bin/activate
pip install -e .
```

This installs the `blobfish` package (in `src/`) in editable mode along with its dependencies, so `import blobfish` works from anywhere and code changes take effect without reinstalling.

### Faster pyffish (recommended)

Our beloved Emily found a serious slowdown in the original Fairy-Stockfish code: the pyffish on PyPI spends about 3 ms on every single call rebuilding tables it already has, which makes benchmarks and anything that checks the rules a lot really slow. So she patched it! The fixed version is about 600x faster per call, gives the exact same answers as the original, and passes Fairy-Stockfish's own tests (the Python test suite, engine bench and move generation checks). It's also submitted upstream in [Fairy-Stockfish#1072](https://github.com/fairy-stockfish/Fairy-Stockfish/pull/1072).

To use it, run this inside `blobfish-env` after the setup above:

```bash
pip install --force-reinstall --no-deps --no-index --find-links https://github.com/et4v/Fairy-Stockfish/releases/expanded_assets/pyffish-0.0.90-fast pyffish
```

No compiler needed, pip picks the right file for your OS and Python version (Windows, macOS and Linux, Python 3.8 to 3.14). Check it worked with `pip show pyffish`, which should say `Version: 0.0.90.post1`.

To go back to the normal version: `pip install --force-reinstall pyffish`. Upgrading pyffish (`pip install --upgrade ...`) also swaps it back, so rerun the command above if that happens. Once the fix is released on PyPI this section can go away.

## Team Members

Each member leads one area, but everyone contributes across the project.

- **Allison O'Brien**: Monte Carlo Tree Search, including hybrid MCTS/Minimax and heuristic evaluation
- **Emily Tew**: Actor-Critic learning and the neural network (ResNet-style and transformer-based architectures)
- **Rachel Bell**: game environment and evaluation (python-chess integration, self-play loop, Stockfish benchmarking)

## References

1. D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, 362(6419), 2018.
2. L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo planning," *ECML*, 2006.
3. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
