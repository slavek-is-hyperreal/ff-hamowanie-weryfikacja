"""Training loops for FF arms and the BP arm (PROTOKOL_v6 §3–§5), with E1/E4/E6 logging."""
import math
import time

import torch
import torch.nn as nn

from . import data, ff, streams


def _acc(pred, y):
    return float((pred == y).float().mean().item()) * 100.0


def train_ff(job, xtr, ytr, xval, yval, device, encode=data.encode, orders=None, wrongs=None,
             epochs=ff.EPOCHS, switch=ff.SWITCH, log_val=True):
    """Train one FF run. ``xtr``/``xval`` are prepared pixel tensors on ``device``.

    ``orders``/``wrongs`` (lists per epoch) inject batch order and wrong labels (test T1);
    otherwise they are drawn from streams ``order``/``neg``.
    Returns (record, net). On a non-finite loss/goodness the run stops with nan=True (§3.4).
    """
    seed, arm, cfg = job["seed"], job["arm"], job["cfg"]
    in_dim = encode(xtr[:1], ytr[:1]).shape[1]
    net = ff.FFNet(in_dim, cfg["lr"], cfg.get("theta", 0.0), streams.init_seed(seed),
                   device, theta_layers=cfg.get("theta_layers"))
    n = len(xtr)
    rec = {"nan": False, "nan_epoch": None, "val_curve": [], "bcm_theta": [], "dw_norm": []}
    t0 = time.time()
    for ep in range(epochs):
        formula, mode, lr_f = ff.phase(arm, ep, switch)
        net.set_lr_factor(lr_f)
        order = orders[ep] if orders is not None else torch.randperm(n, generator=streams.order_gen(seed, ep))
        g_neg = None if wrongs is not None else streams.neg_gen(seed, ep)
        before = net.weights_snapshot()
        for bi, s in enumerate(range(0, n, ff.BATCH)):
            idx = order[s:s + ff.BATCH]
            xb, yb = xtr[idx], ytr[idx]
            yw = wrongs[ep][bi].to(yb.device) if wrongs is not None else data.wrong_labels(yb, g_neg)
            if not net.step(encode(xb, yb), encode(xb, yw), formula, mode):
                rec["nan"], rec["nan_epoch"] = True, ep + 1
                break
        if rec["nan"]:
            break
        rec["dw_norm"].append([float((w - b).norm().item()) for w, b in zip(net.weights_snapshot(), before)])
        if mode == "bcm":
            rec["bcm_theta"].append(list(net.bcm))
        if log_val:
            rec["val_curve"].append(_acc(ff.predict(net, xval, encode, arm)["main"], yval))
    rec["train_time_sec"] = time.time() - t0
    rec["epochs_run"] = ep + 1
    return rec, net


def train_bp(job, xtr, ytr, xval, yval, device, epochs=ff.EPOCHS, log_val=True):
    """BP arm (§3.5): 10 x 64 ReLU MLP, CE, Adam, batch 64, normalized pixels (no label)."""
    seed, cfg = job["seed"], job["cfg"]
    net = ff.build_bp(xtr.shape[1], streams.init_seed(seed), device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()
    n = len(xtr)
    rec = {"nan": False, "nan_epoch": None, "val_curve": []}
    t0 = time.time()
    for ep in range(epochs):
        order = torch.randperm(n, generator=streams.order_gen(seed, ep))
        net.train()
        for s in range(0, n, ff.BATCH):
            idx = order[s:s + ff.BATCH]
            loss = lossf(net(xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                rec["nan"], rec["nan_epoch"] = True, ep + 1
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        if rec["nan"]:
            break
        if log_val:
            rec["val_curve"].append(_acc(predict_bp(net, xval), yval))
    rec["train_time_sec"] = time.time() - t0
    rec["epochs_run"] = ep + 1
    return rec, net


@torch.no_grad()
def predict_bp(net, xn):
    net.eval()
    out = net(xn)
    pr = out.argmax(1)
    pr[~torch.isfinite(out).all(1)] = -1
    return pr


def nan_predictions(n, device):
    """§3.4: a run with NaN predicts class 0 on every example (every readout)."""
    z = torch.zeros(n, dtype=torch.long, device=device)
    return {"main": z, "no_l0": z, "cross": z}
