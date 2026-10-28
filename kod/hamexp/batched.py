"""Vectorized execution of many FF runs of the same arm in one process (ODSTEPSTWA: execution change, not a
method change). Algorithm identical to ``train.train_ff`` / ``ff.FFNet``: per-run layers, per-run Adam
(implemented elementwise exactly as torch.optim.Adam, default betas/eps, no weight decay), simultaneous layer
updates on the same batch pair, softplus loss, detached L2-normalized ReLU outputs, schedules and BCM per §4,
streams per §6 (order / neg / init per seed and epoch), NaN handling per §3.4 (the run stops at the first
non-finite loss or goodness; detection per step via device-side flags, without host synchronization).

Weights of run r, layer l: W[l][r] (out × in), b[l][r]; all R runs share batch size and step count.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data, ff, streams

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def _init_layers(init_seeds, in_dim, device, n_layers=ff.N_LAYERS, width=ff.WIDTH):
    """Same initialization as ff.FFNet: per run fork_rng + manual_seed(init_seed), nn.Linear in order."""
    Ws = [[] for _ in range(n_layers)]
    bs = [[] for _ in range(n_layers)]
    for s in init_seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(s)
            d = in_dim
            for l in range(n_layers):
                lin = nn.Linear(d, width)
                Ws[l].append(lin.weight.detach().clone())
                bs[l].append(lin.bias.detach().clone())
                d = width
    W = [torch.stack(w).to(device).requires_grad_(True) for w in Ws]
    b = [torch.stack(x).to(device).requires_grad_(True) for x in bs]
    return W, b


class BatchedFF:
    def __init__(self, jobs, in_dim, device):
        self.R = len(jobs)
        self.device = torch.device(device)
        arms = {j["arm"] for j in jobs}
        assert len(arms) == 1, "one arm per batch"
        self.arm = arms.pop()
        self.W, self.b = _init_layers([streams.init_seed(j["seed"]) for j in jobs], in_dim, self.device)
        self.params = [p for pair in zip(self.W, self.b) for p in pair]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        f64 = dict(device=self.device, dtype=torch.float64)
        self.lr = torch.tensor([j["cfg"]["lr"] for j in jobs], **f64)
        self.lr_factor = 1.0
        self.theta = torch.tensor([j["cfg"].get("theta", 0.0) for j in jobs], **f64)
        self.bcm = self.theta.unsqueeze(1).repeat(1, ff.N_LAYERS).clone()  # [R, L]
        plw = [j["cfg"].get("theta_layers") for j in jobs]
        self.plw = torch.tensor(plw, **f64) if all(p is not None for p in plw) else None
        self.alive = torch.ones(self.R, dtype=torch.bool, device=self.device)
        self.nan_epoch = torch.zeros(self.R, dtype=torch.long, device=self.device)

    def thresholds(self, mode):
        if mode == "theta":
            return self.theta.unsqueeze(1).expand(-1, ff.N_LAYERS)
        if mode == "zero":
            return torch.zeros(self.R, ff.N_LAYERS, dtype=torch.float64, device=self.device)
        if mode == "bcm":
            return self.bcm
        if mode == "plw":
            return self.plw
        raise ValueError(mode)

    @staticmethod
    def _layer(x, W, b):
        return torch.baddbmm(b.unsqueeze(1), x, W.transpose(1, 2))

    def step(self, x_pos, x_neg, formula, mode, epoch):
        """x_pos/x_neg: [R, B, in]. One simultaneous update of all layers of all live runs."""
        ths = self.thresholds(mode)
        a_pos, a_neg = x_pos, x_neg
        bad = torch.zeros(self.R, dtype=torch.bool, device=self.device)
        losses, mids = [], []
        for l in range(ff.N_LAYERS):
            z_pos, z_neg = self._layer(a_pos, self.W[l], self.b[l]), self._layer(a_neg, self.W[l], self.b[l])
            g_pos, g_neg = ff.goodness(z_pos.reshape(-1, z_pos.shape[-1]), formula).view(self.R, -1), \
                ff.goodness(z_neg.reshape(-1, z_neg.shape[-1]), formula).view(self.R, -1)
            th = ths[:, l].to(torch.float32).unsqueeze(1)  # sequential code uses a Python float in a float32 op
            loss_r = F.softplus(th - g_pos).mean(1) + F.softplus(g_neg - th).mean(1)  # [R]
            fin = torch.isfinite(loss_r) & torch.isfinite(g_pos).all(1) & torch.isfinite(g_neg).all(1)
            bad |= ~fin
            losses.append(torch.where(self.alive & fin, loss_r, torch.zeros_like(loss_r)))
            mids.append((0.5 * (g_pos.mean(1) + g_neg.mean(1))).detach())  # BCM threshold is not differentiated
            a_pos = _norm3(torch.relu(z_pos)).detach()
            a_neg = _norm3(torch.relu(z_neg)).detach()
        # a run with a non-finite value at any layer of this step stops here (sequential code returns before
        # updating later layers; earlier layers of that step were already updated – reproduced below)
        total = torch.stack(losses).sum()
        for p in self.params:
            p.grad = None
        total.backward()
        newbad = bad & self.alive & (self.nan_epoch == 0)
        self.nan_epoch = torch.where(newbad, torch.full_like(self.nan_epoch, epoch + 1), self.nan_epoch)
        self._adam(bad)
        if mode == "bcm":
            mid = torch.stack(mids, 1).to(torch.float64)  # [R, L]
            upd = ff.BCM_DECAY * self.bcm + (1.0 - ff.BCM_DECAY) * mid
            self.bcm = torch.where(self.alive.unsqueeze(1), upd, self.bcm)
        self.alive &= ~bad

    def _adam(self, bad):
        """torch.optim.Adam (single-tensor reference semantics), per run; runs that are already dead or that
        failed in this step are not updated (§3.4: the run stops)."""
        self.t += 1
        bc1, bc2 = 1 - BETA1 ** self.t, 1 - BETA2 ** self.t
        upd_mask = (self.alive & ~bad)
        lr = (self.lr * self.lr_factor).to(torch.float32)
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                g = p.grad
                shape = (self.R,) + (1,) * (p.dim() - 1)
                mk = upd_mask.view(shape)
                m_new = m.lerp(g, 1 - BETA1)  # as torch.optim.Adam: exp_avg.lerp_(grad, 1 - beta1)
                v_new = v.mul(BETA2).addcmul_(g, g, value=1 - BETA2)
                denom = (v_new.sqrt() / math.sqrt(bc2)).add_(EPS)
                step = (lr / bc1).view(shape)
                p_new = p.addcdiv(m_new * step, denom, value=-1.0)
                p.copy_(torch.where(mk, p_new, p))
                m.copy_(torch.where(mk, m_new, m))
                v.copy_(torch.where(mk, v_new, v))

    @torch.no_grad()
    def predict(self, xval, chunk=2500):
        """xval: [R, n, 784] normalized pixels. Readouts main / no_l0 / cross as ff.predict."""
        main_f = ff.final_formula(self.arm)
        other_f = "standard" if main_f == "hamowanie" else "hamowanie"
        n = xval.shape[1]
        out = {k: [] for k in ("main", "no_l0", "cross")}
        for s in range(0, n, chunk):
            xb = xval[:, s:s + chunk]
            sc = {k: torch.zeros(self.R, xb.shape[1], 10, device=self.device) for k in out}
            for c in range(10):
                lab = torch.zeros(self.R, xb.shape[1], 10, device=self.device)
                lab[..., c] = data.LABEL_VALUE
                a = torch.cat([xb, lab], dim=2)
                for l in range(ff.N_LAYERS):
                    z = self._layer(a, self.W[l], self.b[l])
                    p2, n2 = (torch.relu(z) ** 2).sum(2), (torch.relu(-z) ** 2).sum(2)
                    gm = p2 if main_f == "standard" else p2 - n2
                    go = p2 if other_f == "standard" else p2 - n2
                    sc["main"][..., c] += gm
                    sc["cross"][..., c] += go
                    if l > 0:
                        sc["no_l0"][..., c] += gm
                    a = _norm3(torch.relu(z))
            for k, v in sc.items():
                pr = v.argmax(2)
                pr[~torch.isfinite(v).all(2)] = -1
                out[k].append(pr)
        return {k: torch.cat(v, 1) for k, v in out.items()}

    def run_state(self, r):
        """state dict of run r in the format of ff.FFNet.state_dict (for evaluate_test)."""
        layers = [{"weight": self.W[l][r].detach().cpu().clone(), "bias": self.b[l][r].detach().cpu().clone()}
                  for l in range(ff.N_LAYERS)]
        return {"layers": layers, "bcm": self.bcm[r].tolist(), "plw": (self.plw[r].tolist() if self.plw is not None else None),
                "theta": float(self.theta[r]), "lr": float(self.lr[r])}


def _norm3(a, eps=1e-4):
    return a / (a.norm(p=2, dim=-1, keepdim=True) + eps)


def train_batch(jobs, xall, yall, device, epochs=ff.EPOCHS, switch=ff.SWITCH, log_val=True,
                orders=None, wrongs=None, encode=None, splits=None):
    """Train R runs of one arm. xall/yall: normalized pixels [60000, 784] and labels on device.
    Returns per-run records (as train.train_ff) and the BatchedFF object.
    ``orders``/``wrongs``/``encode``/``splits`` are only for the equivalence test."""
    R = len(jobs)
    sp = splits or [data.make_split(j["seed"]) for j in jobs]
    tr = torch.stack([s.train_idx for s in sp]).to(device)  # [R, 20000]
    va = torch.stack([s.val_idx for s in sp]).to(device)
    enc = encode or data.encode
    in_dim = enc(xall[:1], yall[:1]).shape[1]
    net = BatchedFF(jobs, in_dim, device)
    n = tr.shape[1]
    recs = [{"nan": False, "nan_epoch": None, "val_curve": [], "bcm_theta": [], "dw_norm": []} for _ in range(R)]
    xval = xall[va] if log_val else None
    yval = yall[va] if log_val else None
    ar = torch.arange(R, device=device).unsqueeze(1)
    for ep in range(epochs):
        formula, mode, lr_f = ff.phase(net.arm, ep, switch)
        net.lr_factor = lr_f
        if orders is not None:
            order = torch.stack([orders[r][ep] for r in range(R)]).to(device)
        else:
            order = torch.stack([torch.randperm(n, generator=streams.order_gen(j["seed"], ep)) for j in jobs]).to(device)
        if wrongs is None:
            # draws of data.wrong_labels depend only on the batch length, not on labels: precompute shifts per run
            # with the same sequence of generator calls as the sequential code (one call per batch)
            shifts = []
            for j in jobs:
                g = streams.neg_gen(j["seed"], ep)
                shifts.append(torch.cat([torch.randint(1, 10, (min(ff.BATCH, n - s0),), generator=g)
                                         for s0 in range(0, n, ff.BATCH)]))
            shifts = torch.stack(shifts).to(device)  # [R, n]
        before = [w.detach().clone() for w in net.W]
        for bi, s in enumerate(range(0, n, ff.BATCH)):
            idx = tr.gather(1, order[:, s:s + ff.BATCH])  # [R, B] positions in the 60k pool
            xb, yb = xall[idx], yall[idx]
            if wrongs is not None:
                yw = torch.stack([wrongs[r][ep][bi] for r in range(R)]).to(device)
            else:
                yw = (yb + shifts[:, s:s + ff.BATCH]) % 10
            B = xb.shape[1]
            xp = enc(xb.reshape(R * B, -1), yb.reshape(-1)).view(R, B, -1)
            xn_ = enc(xb.reshape(R * B, -1), yw.reshape(-1)).view(R, B, -1)
            net.step(xp, xn_, formula, mode, ep)
        dw = torch.stack([(w.detach() - b0).flatten(1).norm(dim=1) for w, b0 in zip(net.W, before)], 1).cpu()
        alive = net.alive.cpu()
        th = net.bcm.cpu()
        for r in range(R):
            if alive[r]:
                recs[r]["dw_norm"].append(dw[r].tolist())
                if mode == "bcm":
                    recs[r]["bcm_theta"].append(th[r].tolist())
        if log_val:
            pred = net.predict(xval)["main"]
            acc = ((pred == yval).float().mean(1) * 100.0).cpu()
            for r in range(R):
                if alive[r]:
                    recs[r]["val_curve"].append(float(acc[r]))
    ne = net.nan_epoch.cpu()
    for r in range(R):
        if ne[r] > 0:
            recs[r]["nan"], recs[r]["nan_epoch"] = True, int(ne[r])
        recs[r]["epochs_run"] = int(ne[r]) if ne[r] > 0 else epochs
    return recs, net, va


# ---------------------------------------------------------------- BP arm (§3.5), vectorized
class BatchedBP:
    """R independent 10 × 64 ReLU MLPs (+ Linear(64, 10)), cross-entropy, torch-Adam semantics per run."""

    def __init__(self, jobs, in_dim, device):
        self.R = len(jobs)
        self.device = torch.device(device)
        Ws, bs = [], []
        for j in jobs:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(streams.init_seed(j["seed"]))
                mods, d = [], in_dim
                for _ in range(ff.N_LAYERS):
                    mods.append(nn.Linear(d, ff.WIDTH))
                    d = ff.WIDTH
                mods.append(nn.Linear(d, 10))
            Ws.append([m.weight.detach().clone() for m in mods])
            bs.append([m.bias.detach().clone() for m in mods])
        L = len(Ws[0])
        self.W = [torch.stack([w[l] for w in Ws]).to(self.device).requires_grad_(True) for l in range(L)]
        self.b = [torch.stack([b[l] for b in bs]).to(self.device).requires_grad_(True) for l in range(L)]
        self.params = [p for pair in zip(self.W, self.b) for p in pair]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.lr = torch.tensor([j["cfg"]["lr"] for j in jobs], dtype=torch.float32, device=self.device)
        self.alive = torch.ones(self.R, dtype=torch.bool, device=self.device)
        self.nan_epoch = torch.zeros(self.R, dtype=torch.long, device=self.device)

    def forward(self, x):
        a = x
        for l in range(len(self.W)):
            a = torch.baddbmm(self.b[l].unsqueeze(1), a, self.W[l].transpose(1, 2))
            if l < len(self.W) - 1:
                a = torch.relu(a)
        return a

    def step(self, x, y, epoch):
        out = self.forward(x)  # [R, B, 10]
        loss_r = F.cross_entropy(out.reshape(-1, 10), y.reshape(-1), reduction="none").view(self.R, -1).mean(1)
        bad = ~torch.isfinite(loss_r)
        total = torch.where(self.alive & ~bad, loss_r, torch.zeros_like(loss_r)).sum()
        for p in self.params:
            p.grad = None
        total.backward()
        newbad = bad & self.alive & (self.nan_epoch == 0)
        self.nan_epoch = torch.where(newbad, torch.full_like(self.nan_epoch, epoch + 1), self.nan_epoch)
        self.t += 1
        bc1, bc2 = 1 - BETA1 ** self.t, 1 - BETA2 ** self.t
        mk_r = self.alive & ~bad
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                shape = (self.R,) + (1,) * (p.dim() - 1)
                g = p.grad
                m_new = m.lerp(g, 1 - BETA1)
                v_new = v.mul(BETA2).addcmul_(g, g, value=1 - BETA2)
                denom = (v_new.sqrt() / math.sqrt(bc2)).add_(EPS)
                p_new = p.addcdiv(m_new * (self.lr / bc1).view(shape), denom, value=-1.0)
                mk = mk_r.view(shape)
                p.copy_(torch.where(mk, p_new, p))
                m.copy_(torch.where(mk, m_new, m))
                v.copy_(torch.where(mk, v_new, v))
        self.alive &= ~bad

    @torch.no_grad()
    def predict(self, xval):
        out = self.forward(xval)
        pr = out.argmax(2)
        pr[~torch.isfinite(out).all(2)] = -1
        return pr

    def run_state(self, r):
        """state dict in the format of ff.build_bp (nn.Sequential: Linear at even indices)."""
        st = {}
        for l in range(len(self.W)):
            st[f"{2 * l}.weight"] = self.W[l][r].detach().cpu().clone()
            st[f"{2 * l}.bias"] = self.b[l][r].detach().cpu().clone()
        return st


def train_batch_bp(jobs, xall, yall, device, epochs=ff.EPOCHS, log_val=True, splits=None):
    R = len(jobs)
    sp = splits or [data.make_split(j["seed"]) for j in jobs]
    tr = torch.stack([s.train_idx for s in sp]).to(device)
    va = torch.stack([s.val_idx for s in sp]).to(device)
    net = BatchedBP(jobs, xall.shape[1], device)
    n = tr.shape[1]
    recs = [{"nan": False, "nan_epoch": None, "val_curve": []} for _ in range(R)]
    xval, yval = (xall[va], yall[va]) if log_val else (None, None)
    for ep in range(epochs):
        order = torch.stack([torch.randperm(n, generator=streams.order_gen(j["seed"], ep)) for j in jobs]).to(device)
        for s in range(0, n, ff.BATCH):
            idx = tr.gather(1, order[:, s:s + ff.BATCH])
            net.step(xall[idx], yall[idx], ep)
        if log_val:
            acc = ((net.predict(xval) == yval).float().mean(1) * 100.0).cpu()
            alive = net.alive.cpu()
            for r in range(R):
                if alive[r]:
                    recs[r]["val_curve"].append(float(acc[r]))
    ne = net.nan_epoch.cpu()
    for r in range(R):
        if ne[r] > 0:
            recs[r]["nan"], recs[r]["nan_epoch"] = True, int(ne[r])
        recs[r]["epochs_run"] = int(ne[r]) if ne[r] > 0 else epochs
    return recs, net, va
