"""Forward-Forward network with simultaneous per-layer training (PROTOKOL_v6 §3–§4).

Ported from the source campaign (``kod/zrodlo/siec_ff.py``, ``zadanie_fashion_mnist.py``,
commit 9d58048): 10 layers x 64, one Adam per layer, all layers updated in every step on the
same positive/negative batch pair, L2-normalized ReLU output passed (detached) to the next
layer, softplus loss around the threshold. Schedules switch formula / threshold / lr between
epoch 12 and 13 without resetting weights or Adam state.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LAYERS, WIDTH, EPOCHS, SWITCH, BATCH = 10, 64, 25, 12, 64
FF_ARMS = ("STD", "CUR", "ANN", "REV", "BCM", "PLW")
BCM_DECAY = 0.99


def normalize(a: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return a / (a.norm(p=2, dim=1, keepdim=True) + eps)


def goodness(z: torch.Tensor, formula: str) -> torch.Tensor:
    p = torch.relu(z)
    if formula == "standard":
        return (p ** 2).sum(dim=1)
    if formula == "hamowanie":
        n = torch.relu(-z)
        return (p ** 2).sum(dim=1) - (n ** 2).sum(dim=1)
    raise ValueError(formula)


def phase(arm: str, epoch: int, switch: int = SWITCH):
    """(formula, threshold_mode, lr_factor) for 0-based ``epoch`` (§4).

    threshold_mode: "theta" (tuned scalar), "zero", "bcm" (tracked per layer), "plw" (fixed per layer).
    """
    late = epoch >= switch
    if arm == "STD":
        return "standard", "theta", 1.0
    if arm == "CUR":
        return ("hamowanie", "zero", 1.0) if late else ("standard", "theta", 1.0)
    if arm == "ANN":
        return "standard", "theta", (0.5 if late else 1.0)
    if arm == "REV":
        return ("standard", "theta", 1.0) if late else ("hamowanie", "zero", 1.0)
    if arm == "BCM":
        return "standard", "bcm", 1.0
    if arm == "PLW":
        return "standard", "plw", 1.0
    raise ValueError(arm)


def final_formula(arm: str) -> str:
    return phase(arm, EPOCHS - 1)[0]


class FFNet:
    def __init__(self, in_dim: int, lr: float, theta: float, init_seed: int, device,
                 theta_layers=None, n_layers: int = N_LAYERS, width: int = WIDTH):
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.layers = []
            d = in_dim
            for _ in range(n_layers):
                self.layers.append(nn.Linear(d, width))
                d = width
        self.layers = [l.to(self.device) for l in self.layers]
        self.opts = [torch.optim.Adam(l.parameters(), lr=lr) for l in self.layers]
        self.lr = lr
        self.theta = theta
        self.bcm = [float(theta)] * n_layers  # BCM thresholds theta_l(t), theta_l(0) = theta
        self.plw = list(theta_layers) if theta_layers is not None else None

    def set_lr_factor(self, f: float):
        for o in self.opts:
            for g in o.param_groups:
                g["lr"] = self.lr * f

    def thresholds(self, mode: str):
        if mode == "theta":
            return [self.theta] * len(self.layers)
        if mode == "zero":
            return [0.0] * len(self.layers)
        if mode == "bcm":
            return list(self.bcm)
        if mode == "plw":
            return list(self.plw)
        raise ValueError(mode)

    def step(self, x_pos, x_neg, formula: str, mode: str) -> bool:
        """One simultaneous update of all layers. Returns False if a loss/goodness is non-finite."""
        a_pos, a_neg = x_pos, x_neg
        ths = self.thresholds(mode)
        for i, (layer, opt) in enumerate(zip(self.layers, self.opts)):
            z_pos, z_neg = layer(a_pos), layer(a_neg)
            g_pos, g_neg = goodness(z_pos, formula), goodness(z_neg, formula)
            loss = F.softplus(ths[i] - g_pos).mean() + F.softplus(g_neg - ths[i]).mean()
            # §3.4: non-finite loss OR goodness (z finite but z^2 overflowing gives zero gradient and
            # would otherwise train on silently; review K1-14)
            if not (torch.isfinite(loss) and torch.isfinite(g_pos).all() and torch.isfinite(g_neg).all()):
                return False
            opt.zero_grad()
            loss.backward()
            opt.step()
            if mode == "bcm":
                mid = 0.5 * float((g_pos.mean() + g_neg.mean()).item())
                if not math.isfinite(mid):
                    return False
                self.bcm[i] = BCM_DECAY * self.bcm[i] + (1.0 - BCM_DECAY) * mid
            a_pos = normalize(torch.relu(z_pos)).detach()
            a_neg = normalize(torch.relu(z_neg)).detach()
        return True

    @torch.no_grad()
    def layer_goodness_terms(self, x):
        """Per layer (sum ReLU(z)^2, sum ReLU(-z)^2), each [n]."""
        out, a = [], x
        for layer in self.layers:
            z = layer(a)
            out.append(((torch.relu(z) ** 2).sum(1), (torch.relu(-z) ** 2).sum(1)))
            a = normalize(torch.relu(z))
        return out

    def state_dict(self):
        return {"layers": [l.state_dict() for l in self.layers], "bcm": list(self.bcm),
                "plw": self.plw, "theta": self.theta, "lr": self.lr}

    def weights_snapshot(self):
        return [l.weight.detach().clone() for l in self.layers]


@torch.no_grad()
def predict(net: FFNet, xn: torch.Tensor, encode, arm: str, chunk: int = 5000):
    """Readouts (§3.3): 'main' (end-of-training formula, all layers), 'no_l0' (main formula
    without layer 0), 'cross' (the other formula, all layers). Rows with non-finite goodness for
    any label get prediction -1 (counted as wrong, §3.4)."""
    main_f = final_formula(arm)
    other_f = "standard" if main_f == "hamowanie" else "hamowanie"
    preds = {"main": [], "no_l0": [], "cross": []}
    for s in range(0, len(xn), chunk):
        xb = xn[s:s + chunk]
        scores = {k: torch.zeros(len(xb), 10, device=xb.device) for k in preds}
        for c in range(10):
            terms = net.layer_goodness_terms(encode(xb, c))
            for li, (p2, n2) in enumerate(terms):
                gm = p2 if main_f == "standard" else p2 - n2
                go = p2 if other_f == "standard" else p2 - n2
                scores["main"][:, c] += gm
                scores["cross"][:, c] += go
                if li > 0:
                    scores["no_l0"][:, c] += gm
        for k, sc in scores.items():
            pr = sc.argmax(1)
            pr[~torch.isfinite(sc).all(1)] = -1
            preds[k].append(pr)
    return {k: torch.cat(v) for k, v in preds.items()}


def build_bp(in_dim: int, init_seed: int, device, n_layers: int = N_LAYERS, width: int = WIDTH):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        mods, d = [], in_dim
        for _ in range(n_layers):
            mods += [nn.Linear(d, width), nn.ReLU()]
            d = width
        mods.append(nn.Linear(d, 10))
        net = nn.Sequential(*mods)
    return net.to(device)
