"""Equivalence of the vectorized executor (hamexp.batched) with the sequential loop (train.train_ff), which T1
ties to the source code. Same seeds, same streams, small data, 4 epochs with the switch after epoch 2."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hamexp import batched, data, ff, train  # noqa: E402

CFG = {"lr": 2e-3, "theta": 1.5, "theta_layers": [0.5 + 0.1 * i for i in range(ff.N_LAYERS)]}


def _setup(n_tr=200, n_val=40):
    x, y = data.load("mnist", "train")
    xn = data.normalize_pixels(x[:n_tr + n_val])
    y = y[:n_tr + n_val]
    sp = SimpleNamespace(train_idx=torch.arange(n_tr), val_idx=torch.arange(n_tr, n_tr + n_val))
    return xn, y, sp


@pytest.mark.parametrize("arm", ff.FF_ARMS)
def test_batched_equals_sequential(arm):
    xn, y, sp = _setup()
    seeds = [69101, 69102, 69103]
    jobs = [{"seed": s, "arm": arm, "cfg": dict(CFG)} for s in seeds]
    recs, net, _ = batched.train_batch(jobs, xn, y, "cpu", epochs=4, switch=2, splits=[sp] * 3)
    for r, j in enumerate(jobs):
        rec, seq = train.train_ff(j, xn[sp.train_idx], y[sp.train_idx], xn[sp.val_idx], y[sp.val_idx], "cpu",
                                  epochs=4, switch=2)
        for l in range(ff.N_LAYERS):
            assert torch.allclose(net.W[l][r].detach(), seq.layers[l].weight.detach(), atol=5e-5, rtol=1e-4), (arm, r, l)  # float32 rounding in Adam near zero grads (≤ 2.5e-5 observed)
            assert torch.allclose(net.b[l][r].detach(), seq.layers[l].bias.detach(), atol=5e-5, rtol=1e-4)
        assert recs[r]["nan"] == rec["nan"]
        assert max(abs(a - b) for a, b in zip(recs[r]["val_curve"], rec["val_curve"])) <= 100.0 / 40 * 1.001  # ≤ 1 example (float32 sum order near ties)
        if arm == "BCM":
            assert all(abs(a - b) < 1e-4 for a, b in zip(net.bcm[r].tolist(), seq.bcm))
        st = net.run_state(r)
        assert len(st["layers"]) == ff.N_LAYERS


def test_batched_nan_isolated():
    xn, y, sp = _setup()
    jobs = [{"seed": 69201, "arm": "STD", "cfg": dict(CFG)},
            {"seed": 69202, "arm": "STD", "cfg": {**CFG, "theta": float("inf")}}]
    recs, net, _ = batched.train_batch(jobs, xn, y, "cpu", epochs=2, switch=1, splits=[sp] * 2)
    assert recs[1]["nan"] is True and recs[1]["nan_epoch"] == 1 and recs[0]["nan"] is False
    rec, seq = train.train_ff(jobs[0], xn[sp.train_idx], y[sp.train_idx], xn[sp.val_idx], y[sp.val_idx], "cpu",
                              epochs=2, switch=1)
    assert torch.allclose(net.W[0][0].detach(), seq.layers[0].weight.detach(), atol=1e-5, rtol=1e-4)


def test_batched_bp_equals_sequential():
    xn, y, sp = _setup()
    jobs = [{"seed": s, "arm": "BP", "cfg": {"lr": 2e-3}} for s in (69301, 69302)]
    recs, net, _ = batched.train_batch_bp(jobs, xn, y, "cpu", epochs=2, splits=[sp] * 2)
    for r, j in enumerate(jobs):
        rec, seq = train.train_bp(j, xn[sp.train_idx], y[sp.train_idx], xn[sp.val_idx], y[sp.val_idx], "cpu", epochs=2)
        st = net.run_state(r)
        ref = seq.state_dict()
        for k in ref:
            assert torch.allclose(st[k], ref[k], atol=5e-5, rtol=1e-4), k
