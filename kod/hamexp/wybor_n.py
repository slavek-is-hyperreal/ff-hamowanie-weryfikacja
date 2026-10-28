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


def hash_inputs(results, stage, out_path):
    """Timestamp of the inputs of N (R3-09): sha256 of every zaslepione/*.json of the stage(s), to be
    committed together with the output."""
    import glob
    from .runner import sha256
    stages = ["P"] if stage == "P" else ["B", "B2"]
    lines = [f"{sha256(p)}  {os.path.relpath(p, results)}" for st in stages
             for p in sorted(glob.glob(os.path.join(results, st, "zaslepione", "*.json")))]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def run_costs(results, stage, price_usd_h, k):
    import glob
    walls = []
    for st in (["P"] if stage == "P" else ["B", "B2"]):
        for p in glob.glob(os.path.join(results, st, "runs", "*.json")):
            with open(p) as f:
                walls.append(json.load(f)["wall_time_sec"])
    mean_wall = float(np.mean(walls)) if walls else None
    cost = (price_usd_h * mean_wall / 3600.0 / k) if (walls and price_usd_h and k) else None
    return {"n_runs": len(walls), "mean_wall_sec": mean_wall, "k": k, "price_usd_h": price_usd_h,
            "usd_per_run": cost}


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
                out[h]["q"].setdefault(a, []).append([int(fail[a].sum()), int(len(fail[a]))])
        if rhos:
            out[h]["rho"] = float(np.mean(rhos))
    return out


def decide(stats_list):
    """σ = max over sources (pilot, B) per hypothesis and dataset, × 1.29; q per arm and dataset pooled
    over sources (failures / runs), minimum 1 %; ρ = min over sources, capped to [0, 0.5]."""
    sd, q, rho = {}, {}, None
    for h in HYP:
        sd[h] = tuple(1.29 * max(s[h]["sd"][i] for s in stats_list) for i in range(2))
        B, Ks = HYP[h]
        q[h] = []
        for i in range(2):
            pooled = {}
            for a in [B] + Ks:
                f = sum(s[h]["q"][a][i][0] for s in stats_list)
                n = sum(s[h]["q"][a][i][1] for s in stats_list)
                pooled[a] = max(0.01, f / n)
            q[h].append({"B": pooled[B], **{f"K{k}": pooled[c] for k, c in enumerate(Ks)}})
        rs = [s[h]["rho"] for s in stats_list if s[h]["rho"] is not None]
        if rs:
            r = min(max(min(rs), 0.0), 0.5)
            rho = r if rho is None else min(rho, r)
    return sd, q, (0.0 if rho is None else rho)


def power_table(n, sd, q, rho, reps, seed):
    tab = {}
    for h, (B, Ks) in HYP.items():
        tab[h] = {}
        for mult, name in ((0, "0"), (1, "S"), (1.5, "1.5S"), (2, "2S")):
            sign = -1 if h == "H4" else 1
            delta = [[sign * mult * S_UNITS] * len(Ks) for _ in range(2)]
            tab[h][name] = A.simulate_joint(n, len(Ks), delta, sd[h], rho if len(Ks) > 1 else 0.0, q[h],
                                            (S_UNITS, S_UNITS), FAIL_VALUE, BASE, reps, seed)  # joint + per dataset
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
    ap.add_argument("--cena-usd-h", type=float, default=None)
    ap.add_argument("--k", type=int, default=None)
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
    n_hashed = hash_inputs(a.results, a.zrodlo, os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                                             f"HASHE_ZASLEPIONE_{a.zrodlo}.txt"))
    with open(a.out, "w") as f:
        json.dump({"zrodlo": a.zrodlo, "stats": st, "koszty": run_costs(a.results, a.zrodlo, a.cena_usd_h, a.k),
                   "n_plikow_wejsciowych": n_hashed}, f, indent=1)
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
    params["N"] = max(params.get("N_P", 0), params.get("N_B", 0))
    # power table at the FINAL N with the final (latest, most conservative) inputs (review K1-04)
    params["N_power_table"] = {"N": params["N"], "inputs_from": key,
                               "table": power_table(params["N"], sd, q, rho, a.reps, seed)}
    with open(a.params, "w") as f:
        json.dump(params, f, indent=1)
    print(json.dumps({"N": params["N"], key: n, "sd_units_x1.29": sd, "rho": rho, "q": q}, indent=1))


if __name__ == "__main__":
    main()
