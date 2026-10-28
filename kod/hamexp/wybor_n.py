"""Choice of N (PROTOKOL_v6 §7.6). Reads validation accuracies from ``zaslepione/`` and prints/writes
ONLY standard deviations of paired differences, rho, failure rates and the resulting N + power table.

  python -m hamexp.wybor_n --results ../wyniki --zrodlo P --out ../wyniki/MINIPILOT.json --params ../wyniki/PARAMETRY.json
  python -m hamexp.wybor_n --results ../wyniki --zrodlo B --sel ../ETAP_B_WYBOR.json --minipilot ../wyniki/MINIPILOT.json \
         --out ../wyniki/SD_ETAP_B.json --params ../wyniki/PARAMETRY.json
"""
import argparse
import json
import os

import numpy as np

from . import analysis as A
from .final import HYP
from .plans import POOL_B, POOL_P, _jid

DATASETS = ("mnist", "fmnist")
N_VAL = 10000
S_UNITS = 50
TARGET = 75  # 1.5·S in counts of 10 000
FAIL_VALUE = (980, 1000)  # class-0 share (counts) MNIST / F-MNIST test (§3.4)
BASE = (9000, 9000)  # fixed accuracy level used only inside the simulation (no observed means are used)


def _val(results, stage, jid):
    with open(os.path.join(results, stage, "zaslepione", jid + ".json")) as f:
        return json.load(f)["val_acc"]


def _ids(stage, ds, arm, seeds, sel):
    if stage == "P":
        return [(stage, _jid(stage, ds, arm, s)) for s in seeds]
    st = "B2" if arm == "PLW" else "B"
    p = sel[f"{ds}|{arm}"]["point"]
    return [(st, _jid(st, ds, arm, s, p)) for s in seeds]


def collect(results, stage, sel=None):
    seeds = POOL_P if stage == "P" else POOL_B
    out = {}
    for h, (B, Ks) in HYP.items():
        out[h] = {"sd": [], "rho": None, "q": {}}
        rhos = []
        for ds in DATASETS:
            acc = {a: np.array([_val(results, st, j) for st, j in _ids(stage, ds, a, seeds, sel)]) for a in [B] + Ks}
            fail = {a: acc[a] <= A.FAIL_ACC for a in acc}
            ok = ~np.any(np.stack([fail[a] for a in acc]), axis=0)
            D = [np.rint((acc[B] - acc[k]) / 100.0 * N_VAL)[ok] for k in Ks]
            out[h]["sd"].append(max(float(np.std(d, ddof=1)) for d in D))
            if len(Ks) == 2 and ok.sum() > 2:
                rhos.append(float(np.corrcoef(D[0], D[1])[0, 1]))
            for a in acc:
                out[h]["q"].setdefault(a, []).append(float(fail[a].mean()))
        if rhos:
            out[h]["rho"] = float(np.mean(rhos))
    return out


def decide(stats_list):
    """σ = max over sources (pilot, B) per hypothesis and dataset; q = max (≥ 1 %); ρ capped to [0, 0.5]."""
    sd, q, rho = {}, {}, 0.0
    for h in HYP:
        sd[h] = tuple(1.29 * max(s[h]["sd"][i] for s in stats_list) for i in range(2))
        B, Ks = HYP[h]
        qq = {a: max(0.01, max(max(s[h]["q"][a]) for s in stats_list)) for a in [B] + Ks}
        q[h] = {"B": qq[B], **{f"K{i}": qq[k] for i, k in enumerate(Ks)}}
        rs = [s[h]["rho"] for s in stats_list if s[h]["rho"] is not None]
        if rs:
            rho = min(max(max(rs), 0.0), 0.5)
    return sd, q, rho


def power_table(n, sd, q, rho, reps, seed):
    tab = {}
    for h, (B, Ks) in HYP.items():
        tab[h] = {}
        for mult, name in ((0, "0"), (1, "S"), (1.5, "1.5S"), (2, "2S")):
            sign = -1 if h == "H4" else 1
            delta = [[sign * mult * S_UNITS] * len(Ks) for _ in range(2)]
            tab[h][name] = A.simulate_joint(n, len(Ks), delta, sd[h], rho if len(Ks) > 1 else 0.0, q[h],
                                            (S_UNITS, S_UNITS), FAIL_VALUE, BASE, reps, seed)
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--zrodlo", choices=["P", "B"], required=True)
    ap.add_argument("--sel")
    ap.add_argument("--minipilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args(argv)
    from .streams import seed_from
    seed = seed_from("choose_n") % (2 ** 32)
    if a.zrodlo == "P":
        st = collect(a.results, "P")
        sources = [st]
    else:
        with open(a.sel) as f:
            sel = json.load(f)
        st = collect(a.results, "B", sel)
        with open(a.minipilot) as f:
            sources = [json.load(f)["stats"], st]
    with open(a.out, "w") as f:
        json.dump({"zrodlo": a.zrodlo, "stats": st}, f, indent=1)
    sd, q, rho = decide(sources)
    m_by_h = {h: len(HYP[h][1]) for h in HYP}
    n, tab = A.choose_n(sd, rho, q, (S_UNITS, S_UNITS), FAIL_VALUE, BASE, (TARGET, TARGET), m_by_h,
                        reps=a.reps, seed=seed)
    params = {}
    if os.path.exists(a.params):
        with open(a.params) as f:
            params = json.load(f)
    key = "N_P" if a.zrodlo == "P" else "N_B"
    params[key] = n
    params[f"{key}_inputs"] = {"sd_units_x1.29": sd, "q": q, "rho": rho}
    params[f"{key}_ladder"] = {str(k): v for k, v in tab.items()}
    params[f"{key}_power_table"] = power_table(n, sd, q, rho, a.reps, seed)
    params["N"] = max(params.get("N_P", 0), params.get("N_B", 0))
    with open(a.params, "w") as f:
        json.dump(params, f, indent=1)
    print(json.dumps({"N": params["N"], key: n, "sd_units_x1.29": sd, "rho": rho, "q": q}, indent=1))


if __name__ == "__main__":
    main()
