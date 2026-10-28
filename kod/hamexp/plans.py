"""Run plans and selection rules (PROTOKOL_v6 §5).

Stages: P (pilot), A (tuning: 5 arms on 16 Sobol points + BP on 8 lr), B (top-3 of A on pool B),
A2 (PLW lr, 16 points, pool A, thresholds from BCM of stage B), B2 (PLW top-3 on pool B), C.
Selection scripts read validation means from ``zaslepione/`` and print/write configurations only.

Usage:
  python -m hamexp.plans P  --out ../plany/P.json
  python -m hamexp.plans A  --out ../plany/A.json
  python -m hamexp.plans B  --results ../wyniki --out ../plany/B.json
  python -m hamexp.plans A2 --results ../wyniki --out ../plany/A2.json   (after B: writes BCM part of ETAP_B_WYBOR.json)
  python -m hamexp.plans B2 --results ../wyniki --out ../plany/B2.json
  python -m hamexp.plans SEL --results ../wyniki                          (final ETAP_B_WYBOR.json)
  python -m hamexp.plans C  --results ../wyniki --n N --out ../plany/C.json
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from . import streams
from .ff import FF_ARMS

DATASETS = ("mnist", "fmnist")
TUNED = ("STD", "CUR", "ANN", "REV", "BCM")
POOL_P = list(range(60001, 60011))
POOL_A = list(range(61001, 61006))
POOL_B = list(range(62001, 62011))
LR_RANGE, THETA_RANGE = (1e-4, 1e-2), (0.01, 16.0)
PILOT_CFG = {"lr": 1e-3, "theta": 2.0}
FAIL_ACC = 15.0


def pool_c(n):
    return list(range(63001, 63001 + n))


def sobol_points(n=16):
    eng = torch.quasirandom.SobolEngine(2, scramble=True, seed=streams.seed_from("A-sobol") % (2 ** 31))
    u = eng.draw(n).double().numpy()
    lo = np.log10([LR_RANGE[0], THETA_RANGE[0]])
    hi = np.log10([LR_RANGE[1], THETA_RANGE[1]])
    pts = 10 ** (lo + u * (hi - lo))
    return [{"lr": float(a), "theta": float(b)} for a, b in pts]


def lr_grid(n):
    return [float(v) for v in np.logspace(math.log10(LR_RANGE[0]), math.log10(LR_RANGE[1]), n)]


def shuffled(jobs, stage):
    g = torch.Generator().manual_seed(streams.seed_from(f"{stage}-order") % (2 ** 63))
    perm = torch.randperm(len(jobs), generator=g).tolist()
    return [jobs[i] for i in perm]


def _ff(stage, ds, arm, seed, cfg, point=None):
    j = {"stage": stage, "kind": "ff" if arm != "BP" else "bp", "dataset": ds, "arm": arm, "seed": seed,
         "cfg": cfg}
    if point is not None:
        j["point"] = str(point)
    return j


def plan_p(plw_thresholds=None):
    jobs = []
    for ds in DATASETS:
        for s in POOL_P:
            for arm in ("STD", "CUR", "ANN", "REV", "BCM"):
                jobs.append(_ff("P", ds, arm, s, dict(PILOT_CFG)))
            jobs.append(_ff("P", ds, "BP", s, {"lr": PILOT_CFG["lr"]}))
    return shuffled(jobs, "P")


def plan_p_plw(results):
    """PLW in the pilot gets thresholds from the pilot BCM runs (§5.1), per dataset."""
    jobs = []
    for ds in DATASETS:
        th = mean_bcm_thresholds(results, "P", ds, POOL_P, None)
        for s in POOL_P:
            jobs.append(_ff("P", ds, "PLW", s, {"lr": PILOT_CFG["lr"], "theta_layers": th}))
    return shuffled(jobs, "P-PLW")


def plan_a():
    pts = sobol_points(16)
    jobs = []
    for ds in DATASETS:
        for arm in TUNED:
            for pi, cfg in enumerate(pts):
                for s in POOL_A:
                    jobs.append(_ff("A", ds, arm, s, dict(cfg), point=pi))
        for pi, lr in enumerate(lr_grid(8)):
            for s in POOL_A:
                jobs.append(_ff("A", ds, "BP", s, {"lr": lr}, point=pi))
    return shuffled(jobs, "A")


# ---------------------------------------------------------------- selection helpers
def _val(results, stage, jid):
    with open(os.path.join(results, stage, "zaslepione", jid + ".json")) as f:
        return json.load(f)["val_acc"]


def _run(results, stage, jid):
    with open(os.path.join(results, stage, "runs", jid + ".json")) as f:
        return json.load(f)


def _jid(stage, ds, arm, s, point=None):
    from .runner import job_id
    j = {"stage": stage, "dataset": ds, "arm": arm, "seed": s}
    if point is not None:
        j["point"] = str(point)
    return job_id(j)


def point_scores(results, stage, ds, arm, seeds, points):
    """{point: (mean val acc over seeds, n_failed)}; NaN runs score as class-0 share (already in val_acc)."""
    out = {}
    for p in points:
        accs = [_val(results, stage, _jid(stage, ds, arm, s, p)) for s in seeds]
        fails = sum(1 for a in accs if a <= FAIL_ACC)
        out[p] = (float(np.mean(accs)), fails)
    return out


def top3(scores, lr_of, max_fail):
    """§5.2/§5.3: reject points with more than ``max_fail`` failed seeds; best 3 by mean;
    exact ties -> smaller lr. Returns [] when everything is rejected (§5.6)."""
    ok = [p for p, (m, f) in scores.items() if f <= max_fail]
    ok.sort(key=lambda p: (-scores[p][0], lr_of[p]))
    return ok[:3]


def mean_bcm_thresholds(results, stage, ds, seeds, point):
    ths = []
    for s in seeds:
        r = _run(results, stage, _jid(stage, ds, "BCM", s, point))
        if not r["nan"] and r.get("bcm_theta_final"):
            ths.append(r["bcm_theta_final"])
    if len(ths) < len(seeds) / 2:
        raise RuntimeError(f"BCM {stage}/{ds}: fewer than half of runs without NaN (§4.2/§5.6)")
    return [float(v) for v in np.mean(np.array(ths), axis=0)]


def _a_points(arm):
    if arm == "BP":
        return {i: {"lr": lr} for i, lr in enumerate(lr_grid(8))}
    return dict(enumerate(sobol_points(16)))


def plan_b(results):
    """Top-3 of stage A (strict rule: no failed seed; relaxed §5.6 if all rejected) -> pool B."""
    jobs, chosen = [], {}
    for ds in DATASETS:
        for arm in TUNED + ("BP",):
            pts = _a_points(arm)
            sc = point_scores(results, "A", ds, arm, POOL_A, pts)
            lr_of = {p: c["lr"] for p, c in pts.items()}
            best = top3(sc, lr_of, 0) or top3(sc, lr_of, 1)
            chosen[f"{ds}|{arm}"] = best
            for p in best:
                for s in POOL_B:
                    jobs.append(_ff("B", ds, arm, s, dict(pts[p]), point=p))
    return shuffled(jobs, "B"), chosen


def select_b(results, stage, ds, arm, cand_cfgs):
    """§5.3: best mean on pool B; candidates with > 5 of 10 failed seeds rejected; exact tie -> smaller lr."""
    sc = point_scores(results, stage, ds, arm, POOL_B, list(cand_cfgs))
    lr_of = {p: c["lr"] for p, c in cand_cfgs.items()}
    best = top3(sc, lr_of, 5)
    return best[0] if best else None


def plan_a2(results, sel):
    """After stage B: choose BCM, derive PLW thresholds, PLW lr grid (16) on pool A."""
    jobs = []
    for ds in DATASETS:
        p = sel[f"{ds}|BCM"]["point"]
        if p is None:
            continue
        th = mean_bcm_thresholds(results, "B", ds, POOL_B, p)
        sel[f"{ds}|PLW"] = {"theta_layers": th}
        for pi, lr in enumerate(lr_grid(16)):
            for s in POOL_A:
                jobs.append(_ff("A2", ds, "PLW", s, {"lr": lr, "theta_layers": th}, point=pi))
    return shuffled(jobs, "A2")


def plan_b2(results, sel):
    jobs, chosen = [], {}
    for ds in DATASETS:
        if "theta_layers" not in sel.get(f"{ds}|PLW", {}):
            continue
        th = sel[f"{ds}|PLW"]["theta_layers"]
        pts = {i: {"lr": lr, "theta_layers": th} for i, lr in enumerate(lr_grid(16))}
        sc = point_scores(results, "A2", ds, "PLW", POOL_A, pts)
        best = top3(sc, {p: c["lr"] for p, c in pts.items()}, 0) or top3(sc, {p: c["lr"] for p, c in pts.items()}, 1)
        chosen[ds] = best
        for p in best:
            for s in POOL_B:
                jobs.append(_ff("B2", ds, "PLW", s, dict(pts[p]), point=p))
    return shuffled(jobs, "B2"), chosen


def plan_c(sel, n, without_bp=False):
    """§5.7; ``without_bp`` = the only allowed budget cut (§8.3)."""
    jobs = []
    for ds in DATASETS:
        for arm in FF_ARMS + (() if without_bp else ("BP",)):
            cfg = sel.get(f"{ds}|{arm}", {}).get("cfg")
            if cfg is None:
                continue  # untrainable arm (§5.6)
            for s in pool_c(n):
                jobs.append(_ff("C", ds, arm, s, dict(cfg)))
    return shuffled(jobs, "C")


def _dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--results")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int)
    ap.add_argument("--bez-bp", action="store_true", help="§8.3: drop BP from stage C (only allowed cut)")
    ap.add_argument("--sel", default=None, help="path of ETAP_B_WYBOR.json (default: <results>/../ETAP_B_WYBOR.json)")
    a = ap.parse_args(argv)
    sel_path = a.sel or (os.path.join(a.results, "..", "ETAP_B_WYBOR.json") if a.results else None)
    if a.stage == "P":
        _dump(plan_p(), a.out)
    elif a.stage == "P-PLW":
        _dump(plan_p_plw(a.results), a.out)
    elif a.stage == "A":
        _dump(plan_a(), a.out)
    elif a.stage == "B":
        jobs, chosen = plan_b(a.results)
        _dump(jobs, a.out)
        print("kandydaci do B (numery punktów, posortowane – bez rankingu):", {k: sorted(v) for k, v in chosen.items()})
    elif a.stage == "A2":
        # candidates recomputed from stage A by the §5.2 rule (not from a directory listing; review K1-11)
        _, chosen = plan_b(a.results)
        sel = {}
        for ds in DATASETS:
            for arm in TUNED + ("BP",):
                pts = _a_points(arm)
                cands = {q: pts[q] for q in chosen[f"{ds}|{arm}"]}
                p = select_b(a.results, "B", ds, arm, cands) if cands else None  # missing run files -> hard error
                sel[f"{ds}|{arm}"] = {"point": p, "cfg": (pts[p] if p is not None else None)}
        jobs = plan_a2(a.results, sel)
        _dump(sel, sel_path)
        _dump(jobs, a.out)
        print("ETAP_B_WYBOR.json zapisany (konfiguracje), plan A2:", len(jobs), "biegów")
    elif a.stage == "B2":
        with open(sel_path) as f:
            sel = json.load(f)
        jobs, chosen = plan_b2(a.results, sel)
        _dump(jobs, a.out)
        print("kandydaci PLW do B2 (posortowane – bez rankingu):", {k: sorted(v) for k, v in chosen.items()})
    elif a.stage == "SEL":
        with open(sel_path) as f:
            sel = json.load(f)
        _, chosen = plan_b2(a.results, sel)  # candidates recomputed from A2 by the §5.2 rule
        for ds in DATASETS:
            if "theta_layers" not in sel.get(f"{ds}|PLW", {}):
                continue
            th = sel[f"{ds}|PLW"]["theta_layers"]
            pts = {i: {"lr": lr, "theta_layers": th} for i, lr in enumerate(lr_grid(16))}
            cands = {q: pts[q] for q in chosen.get(ds, [])}
            p = select_b(a.results, "B2", ds, "PLW", cands) if cands else None
            sel[f"{ds}|PLW"].update({"point": p, "cfg": (pts[p] if p is not None else None)})
        _dump(sel, sel_path)
        print("ETAP_B_WYBOR.json zaktualizowany (PLW).")
    elif a.stage == "C":
        with open(sel_path) as f:
            sel = json.load(f)
        _dump(plan_c(sel, a.n, a.bez_bp), a.out)
    else:
        raise SystemExit(f"unknown stage {a.stage}")


if __name__ == "__main__":
    main()
