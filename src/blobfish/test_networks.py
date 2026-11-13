"""
tests for the networks folder, makes sure every architecture follows the contract in networks/base.py

checks shapes at every board size, that legal masking works, that a few training steps on real
gardner positions actually lower the loss, and that a saved checkpoint loads back the same network

run from src/blobfish:  python test_networks.py
"""

import os
import random
import tempfile

import lightning.pytorch as pl
import pyffish as sf
import torch

from encoding import PLANES, VARIANTS, encode, policy_size
from move_mapping import legal_mask, move_to_index
from networks import NETWORKS, make_network


# small networks so the tests run fast on a laptop
SMALL = dict(channels=16, blocks=2)


def random_gardner_batch(n, seed=0):
    # play random moves from the gardner start and keep every position we pass through
    # the policy target is one-hot on a random legal move and the value target is random, the point
    # is just to have real boards and real legal masks to push through the network
    rng = random.Random(seed)
    game, size = VARIANTS[5], 5
    start = sf.start_fen(game)
    states, policies, values, masks = [], [], [], []
    while len(states) < n:
        moves = []
        for _ in range(rng.randint(0, 12)):
            legal = sf.legal_moves(game, start, moves)
            if not legal:
                break
            moves.append(rng.choice(legal))
        legal = sf.legal_moves(game, start, moves)
        if not legal:
            continue
        fen = sf.get_fen(game, start, moves)
        black = len(moves) % 2 == 1
        target = torch.zeros(policy_size(size))
        target[move_to_index(rng.choice(legal), size, black)] = 1.0
        states.append(encode(fen))
        policies.append(target)
        values.append(rng.uniform(-1, 1))
        masks.append(torch.from_numpy(legal_mask(legal, size, black)))
    return {
        "state": torch.stack(states),
        "policy": torch.stack(policies),
        "value": torch.tensor(values),
        "mask": torch.stack(masks),
    }


def test_shapes():
    # every registered network at every board size gives (batch, slots) and (batch, 1)
    for name in NETWORKS:
        for size in VARIANTS:
            net = make_network(name, input_shape=(PLANES, size, size),
                               output_shape=(policy_size(size),), **SMALL)
            x = torch.randn(3, PLANES, size, size)
            p, v = net(x)
            assert p.shape == (3, policy_size(size)), f"{name} {size}x{size} policy {tuple(p.shape)}"
            assert v.shape == (3, 1), f"{name} {size}x{size} value {tuple(v.shape)}"
            assert v.abs().max() <= 1, "value should be squashed into -1 to 1"


def test_predict_mask():
    # with a mask illegal moves get exactly 0 and legal ones still add to 1
    batch = random_gardner_batch(8)
    net = make_network("resnet", input_shape=(PLANES, 5, 5), output_shape=(policy_size(5),), **SMALL)
    p, v = net.evaluate(batch["state"], batch["mask"])
    assert v.shape == (8,)
    assert torch.allclose(p.sum(-1), torch.ones(8), atol=1e-5)
    assert (p[batch["mask"] == 0] == 0).all(), "illegal moves got probability"

    # a board with no legal moves shouldnt blow up into nans
    empty = torch.zeros_like(batch["mask"])
    p, _ = net.evaluate(batch["state"], empty)
    assert not torch.isnan(p).any()


def test_training_lowers_loss(policy_loss):
    # overfit one small batch, if the loss doesnt drop something is wired wrong
    torch.manual_seed(0)
    batch = random_gardner_batch(32)
    net = make_network("resnet", input_shape=(PLANES, 5, 5), output_shape=(policy_size(5),),
                       lr=3e-3, policy_loss=policy_loss, **SMALL)
    before = net.compute_loss(batch)[0].item()
    opt = net.configure_optimizers()
    for _ in range(60):
        opt.zero_grad()
        loss = net.compute_loss(batch)[0]
        loss.backward()
        opt.step()
    after = net.compute_loss(batch)[0].item()
    assert after < before, f"{policy_loss}: loss went {before:.3f} -> {after:.3f}"
    return before, after


def test_lightning_fit_and_checkpoint():
    # run the real lightning trainer for one short epoch then save and load a checkpoint
    batch = random_gardner_batch(32)
    rows = [{k: batch[k][i] for k in batch} for i in range(32)]
    loader = torch.utils.data.DataLoader(rows, batch_size=8)
    net = make_network("resnet", input_shape=(PLANES, 5, 5), output_shape=(policy_size(5),), **SMALL)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(net, loader, loader)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.ckpt")
        trainer.save_checkpoint(path)
        loaded = NETWORKS["resnet"].load_from_checkpoint(path)
    assert loaded.hparams.channels == SMALL["channels"] and loaded.hparams.blocks == SMALL["blocks"]
    p1, v1 = net.evaluate(batch["state"])
    p2, v2 = loaded.evaluate(batch["state"])
    assert torch.allclose(p1, p2, atol=1e-6) and torch.allclose(v1, v2, atol=1e-6)


if __name__ == "__main__":
    test_shapes()
    print("shapes ok")
    test_predict_mask()
    print("legal masking ok")
    for mode in ("cross_entropy", "advantage"):
        before, after = test_training_lowers_loss(mode)
        print(f"training ok ({mode}): loss {before:.3f} -> {after:.3f}")
    test_lightning_fit_and_checkpoint()
    print("lightning fit + checkpoint ok")
    print("all network tests passed")
