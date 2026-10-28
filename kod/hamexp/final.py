"""Frozen analysis script (PROTOKOL_v6 §5.8 step 4, §7). Computes every number and figure of
RAPORT_SZABLON.md.

Usage: python -m hamexp.final --results ../wyniki --params ../wyniki/PARAMETRY.json --out ../wyniki/WYNIKI_KONCOWE.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from . import analysis as A
from .ff import FF_ARMS
from .runner import job_id

HYP = {"H1": ("CUR", ["STD"]), "H2": ("CUR", ["ANN"]), "H3": ("BCM", ["STD", "PLW"]), "H4": ("REV", ["STD"])}
DATASETS = ("mnist", "fmnist")
ALL_ARMS = FF_ARMS + ("BP",)
S_PP = 0.5


def s_units(n_examples, s_pp=S_PP):
    return int(round(s_pp / 100.0 * n_examples))


# ---------------------------------------------------------------- loading
def load_stage_c(results):
    """{(ds, arm): {seed: {"run": rec, "val": val_acc, "test": {set: {readout: bool tensor}}}}}"""
    out = {}
    for rp in sorted(glob.glob(os.path.join(results, "C", "runs", "*.json"))):
        with open(rp) as f:
            r = json.load(f)
        j = r["job"]
        with open(os.path.join(results, "C", "zaslepione", r["id"] + ".json")) as f:
            z = json.load(f)
        tp = os.path.join(results, "test", "C", r["id"] + ".pt")
        test = torch.load(tp, weights_only=False) if os.path.exists(tp) else None
        out.setdefault((j["dataset"], j["arm"]), {})[j["seed"]] = {"run": r, "val": z["val_acc"],
                                                                   "curve": z["val_curve"], "test": test}
    return out


def failed(entry):
    return entry["run"]["nan"] or entry["val"] <= A.FAIL_ACC


def counts(table, ds, arm, seeds, testset, readout="main"):
    return np.array([int(table[(ds, arm)][s]["test"][testset][readout].sum()) for s in seeds], dtype=np.int64)


def correctness(table, ds, arm, seeds, testset, readout="main"):
    return [table[(ds, arm)][s]["test"][testset][readout].numpy() for s in seeds]


# ---------------------------------------------------------------- confirmatory
def hypothesis(table, h, testset_of, readout="main", drop_failed=False, method="wilcoxon", s_pp=S_PP):
    B, Ks = HYP[h]
    per = {}
    for ds in DATASETS:
        if (ds, B) not in table or any((ds, k) not in table for k in Ks):
            per[ds] = {"category": "NIEROZSTRZYGNIETY", "class": "N", "direction": 0, "untrainable": True}
            continue
        seeds = sorted(set(table[(ds, B)]).intersection(*[set(table[(ds, k)]) for k in Ks]))
        if drop_failed:
            seeds = [s for s in seeds if not failed(table[(ds, B)][s]) and not any(failed(table[(ds, k)][s]) for k in Ks)]
        ts = testset_of[ds]
        n_ex = len(table[(ds, B)][seeds[0]]["test"][ts][readout])
        S = s_units(n_ex, s_pp)
        cb = counts(table, ds, B, seeds, ts, readout)
        D = [cb - counts(table, ds, k, seeds, ts, readout) for k in Ks]
        v = A.verdict_dataset(D, S, method=method)
        fb = [failed(table[(ds, B)][s]) for s in seeds]
        st = A.stability(fb, [[failed(table[(ds, k)][s]) for s in seeds] for k in Ks])
        if not drop_failed:
            v = A.apply_stability(v, st)
        v.update({"stability": st, "N": len(seeds), "S_units": S, "n_examples": n_ex,
                  "hl_pp": [100.0 * x / n_ex for x in v["hl"]],
                  "hl_ci_pp": [[100.0 * c / n_ex for c in ci] for ci in v["hl_ci"]],
                  "mean_pp": [100.0 * x / n_ex for x in v["mean"]],
                  "holm_better_comparators": [Ks[i] for i in v["holm_better_comparators"]],
                  "stability_worse_than": [Ks[i] for i in st["holm_worse_than"]]})
        per[ds] = v
    joint = A.combine(per["mnist"], per["fmnist"]) if all("p" in per[d] or per[d].get("untrainable") for d in DATASETS) else None
    stab_joint = (A.combine_stability(per["mnist"]["stability"], per["fmnist"]["stability"])
                  if all("stability" in per[d] for d in DATASETS) else None)
    return {"per_dataset": per, "joint": joint, "stability_joint": stab_joint}


def bootstrap_2d(table, h, R=10000, seed=12345, readout="main", chunk=500):
    """§7.7a: seeds resampled with replacement; within each drawn seed, test examples resampled
    with replacement (same indices for all arms). Statistic: HL of per-seed count differences."""
    B, Ks = HYP[h]
    rng = np.random.default_rng(seed)
    per = {}
    for ds in DATASETS:
        seeds = sorted(set(table[(ds, B)]).intersection(*[set(table[(ds, k)]) for k in Ks]))
        arms = [B] + Ks
        n_ex = len(table[(ds, B)][seeds[0]]["test"][ds][readout])
        S = s_units(n_ex)
        # per seed: pattern counts over arms' correctness (2^(m+1) patterns)
        pats = 2 ** len(arms)
        P = np.zeros((len(seeds), pats))
        for si, s in enumerate(seeds):
            code = np.zeros(n_ex, dtype=np.int64)
            for ai, a in enumerate(arms):
                code |= (table[(ds, a)][s]["test"][ds][readout].numpy().astype(np.int64) << ai)
            P[si] = np.bincount(code, minlength=pats)
        bits = np.array([[(p >> ai) & 1 for ai in range(len(arms))] for p in range(pats)])  # (pats, arms)
        boot = np.zeros((R, len(Ks)))
        for c0 in range(0, R, chunk):
            r = min(chunk, R - c0)
            sidx = rng.integers(0, len(seeds), (r, len(seeds)))
            cnt = rng.multinomial(n_ex, P[sidx] / n_ex)  # (r, N, pats)
            correct = cnt @ bits  # (r, N, arms)
            for k in range(len(Ks)):
                d = correct[..., 0] - correct[..., k + 1]
                i, j = np.triu_indices(d.shape[1])
                w = (d[:, i] + d[:, j]) / 2.0
                boot[c0:c0 + r, k] = np.median(w, axis=1)
        a = A.ALPHA1 / 2
        m = len(Ks)
        pf = lambda cond: cond.mean(axis=0)  # noqa: E731
        sup_S = np.all(pf(boot <= S) <= a)
        sup_0 = np.all(pf(boot <= 0) <= a)
        eq = np.all(np.maximum(pf(boot <= -S), pf(boot >= S)) <= A.ALPHA1)
        inf_S = bool(A.holm_rejected(list(pf(boot >= -S)), a))
        inf_0 = bool(A.holm_rejected(list(pf(boot >= 0)), a))
        cat = "NIEROZSTRZYGNIETY"
        for name, cond in (("GORSZOSC", inf_S), ("PRZEWAGA", sup_S), ("ROWNOWAZNOSC+", eq and (sup_0 or inf_0)),
                           ("GORSZOSC<S", inf_0), ("PRZEWAGA<S", sup_0), ("ROWNOWAZNOSC", eq)):
            if cond:
                cat = name
                break
        per[ds] = {"category": cat, "class": A.CLASS_OF[cat], "direction": 1 if sup_0 else (-1 if inf_0 else 0),
                   "ci_pp": (100.0 * np.quantile(boot, [A.ALPHA1 / 2, 1 - A.ALPHA1 / 2], axis=0).T / n_ex).tolist()}
    return {"per_dataset": per, "joint": A.combine(per["mnist"], per["fmnist"])}


# ---------------------------------------------------------------- descriptive / exploratory
def accuracy_table(table):
    out = {}
    for (ds, arm), runs in sorted(table.items()):
        accs = [100.0 * float(e["test"][ds]["main"].float().mean()) for e in runs.values()]
        out[f"{ds}|{arm}"] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)), "n": len(accs),
                              "n_nan": sum(e["run"]["nan"] for e in runs.values()),
                              "n_failed": sum(failed(e) for e in runs.values())}
    return out


def hl_ci95(d_pp):
    return {"hl": A.hodges_lehmann(d_pp), "ci95": A.hl_ci(d_pp, 0.05)}


def exploratory(table, results):
    ex = {}
    # E1: BCM theta trajectories (mean over seeds), per dataset
    ex["E1_bcm_theta"] = {}
    for ds in DATASETS:
        trs = [np.array(e["run"]["bcm_theta"]) for e in table.get((ds, "BCM"), {}).values() if e["run"]["bcm_theta"]]
        if trs:
            L = min(len(t) for t in trs)
            ex["E1_bcm_theta"][ds] = np.mean([t[:L] for t in trs], axis=0).tolist()
    # E2: BP − best FF (by mean validation of stage C)
    ex["E2_bp_gap"] = {}
    for ds in DATASETS:
        ffm = {a: np.mean([e["val"] for e in table[(ds, a)].values()]) for a in FF_ARMS if (ds, a) in table}
        if not ffm or (ds, "BP") not in table:
            continue
        best = max(ffm, key=ffm.get)
        seeds = sorted(set(table[(ds, best)]) & set(table[(ds, "BP")]))
        n_ex = len(table[(ds, best)][seeds[0]]["test"][ds]["main"])
        d = (counts(table, ds, "BP", seeds, ds) - counts(table, ds, best, seeds, ds)) * 100.0 / n_ex
        ex["E2_bp_gap"][ds] = {"best_ff_by_val": best, **hl_ci95(d)}
    # E3: readouts (mean test accuracy per arm)
    ex["E3_readouts"] = {}
    for (ds, arm), runs in table.items():
        if arm == "BP":
            continue
        ex["E3_readouts"][f"{ds}|{arm}"] = {rd: float(np.mean([100.0 * float(e["test"][ds][rd].float().mean())
                                                               for e in runs.values()])) for rd in ("main", "no_l0", "cross")}
    # E4: weight-change norm per layer per epoch (mean over seeds)
    ex["E4_dw_norm"] = {}
    for (ds, arm), runs in table.items():
        if arm in ("STD", "CUR", "ANN"):
            arr = [np.array(e["run"]["dw_norm"]) for e in runs.values() if len(e["run"]["dw_norm"]) == 25]
            if arr:
                ex["E4_dw_norm"][f"{ds}|{arm}"] = np.mean(arr, axis=0).tolist()
    # E5: tuning landscape of stage A (mean validation per point)
    ex["E5_landscape"] = {}
    for zp in sorted(glob.glob(os.path.join(results, "A", "zaslepione", "*.json"))):
        with open(zp) as f:
            z = json.load(f)
        with open(os.path.join(results, "A", "runs", z["id"] + ".json")) as f:
            j = json.load(f)["job"]
        key = f"{j['dataset']}|{j['arm']}|{j['point']}"
        ex["E5_landscape"].setdefault(key, {"cfg": j["cfg"], "val": []})["val"].append(z["val_acc"])
    for v in ex["E5_landscape"].values():
        v["mean"] = float(np.mean(v["val"]))
    # E6: validation curves
    ex["E6_val_curves"] = {}
    for (ds, arm), runs in table.items():
        cs = [e["curve"] for e in runs.values() if len(e["curve"]) == 25]
        if cs:
            ex["E6_val_curves"][f"{ds}|{arm}"] = np.mean(cs, axis=0).tolist()
    # E7: ANN − STD (descriptive)
    ex["E7_ann_minus_std"] = {}
    for ds in DATASETS:
        if (ds, "ANN") in table and (ds, "STD") in table:
            seeds = sorted(set(table[(ds, "ANN")]) & set(table[(ds, "STD")]))
            n_ex = len(table[(ds, "ANN")][seeds[0]]["test"][ds]["main"])
            d = (counts(table, ds, "ANN", seeds, ds) - counts(table, ds, "STD", seeds, ds)) * 100.0 / n_ex
            ex["E7_ann_minus_std"][ds] = hl_ci95(d)
    return ex


def figures(ex, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    for ds, tr in ex["E1_bcm_theta"].items():
        tr = np.array(tr)
        plt.figure(figsize=(6, 4))
        for l in range(tr.shape[1]):
            plt.plot(range(1, len(tr) + 1), tr[:, l], label=f"warstwa {l}")
        plt.xlabel("epoka"); plt.ylabel("θ_l"); plt.title(f"E1 BCM ({ds})"); plt.legend(fontsize=6)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"E1_bcm_{ds}.png"), dpi=120); plt.close()
    for ds in DATASETS:
        plt.figure(figsize=(6, 4))
        for k, c in ex["E6_val_curves"].items():
            if k.startswith(ds + "|"):
                plt.plot(range(1, 26), c, label=k.split("|")[1])
        plt.axvline(12.5, ls=":", c="gray")
        plt.xlabel("epoka"); plt.ylabel("dokładność walidacyjna [%]"); plt.title(f"E6 ({ds})"); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"E6_krzywe_{ds}.png"), dpi=120); plt.close()
        arms = sorted({k.split("|")[1] for k in ex["E5_landscape"] if k.startswith(ds + "|") and not k.split("|")[1] == "BP"})
        if arms:
            fig, axs = plt.subplots(1, len(arms), figsize=(3 * len(arms), 3), sharey=True)
            for ax, arm in zip(np.atleast_1d(axs), arms):
                pts = [v for k, v in ex["E5_landscape"].items() if k.startswith(f"{ds}|{arm}|")]
                sc = ax.scatter([p["cfg"]["lr"] for p in pts], [p["cfg"]["theta"] for p in pts],
                                c=[p["mean"] for p in pts], cmap="viridis", vmin=10, vmax=100)
                ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(arm); ax.set_xlabel("lr")
            np.atleast_1d(axs)[0].set_ylabel("θ")
            fig.colorbar(sc, ax=axs, label="walidacja [%]")
            fig.savefig(os.path.join(outdir, f"E5_krajobraz_{ds}.png"), dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(argv)
    table = load_stage_c(a.results)
    main_sets = {ds: ds for ds in DATASETS}
    rep = {"accuracy": accuracy_table(table), "confirmatory": {}, "sensitivity": {}}
    for h in HYP:
        rep["confirmatory"][h] = hypothesis(table, h, main_sets)
        rep["sensitivity"][h] = {
            "a_bootstrap_hl": bootstrap_2d(table, h, R=a.boot),
            "b_ttest": hypothesis(table, h, main_sets, method="t"),
            "c_qmnist": hypothesis(table, h, {"mnist": "qmnist", "fmnist": "fmnist"})["per_dataset"]["mnist"],
            "d_no_failed_pairs": hypothesis(table, h, main_sets, drop_failed=True),
            "e_S_0.25": hypothesis(table, h, main_sets, s_pp=0.25)["joint"],
            "e_S_1.0": hypothesis(table, h, main_sets, s_pp=1.0)["joint"],
        }
    rep["exploratory"] = exploratory(table, a.results)
    figures(rep["exploratory"], os.path.join(os.path.dirname(os.path.abspath(a.out)), "rysunki"))
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for h, v in rep["confirmatory"].items():
        print(h, v["joint"]["category"], v["joint"]["per_dataset"], "| stabilność:", v["stability_joint"])


if __name__ == "__main__":
    main()
