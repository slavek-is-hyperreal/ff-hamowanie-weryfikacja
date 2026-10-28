"""Frozen analysis script (PROTOKOL_v6 §5.8 step 5, §7). Computes every number, sentence-template input
and figure of RAPORT_SZABLON.md.

Usage: python -m hamexp.final --results ../wyniki --params ../wyniki/PARAMETRY.json \
           --sel ../ETAP_B_WYBOR.json --out ../wyniki/WYNIKI_KONCOWE.json
Refuses to run on an incomplete stage C (§5.8; review K1-01).
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from scipy import stats

from . import analysis as A
from .ff import FF_ARMS
from .plans import LR_RANGE, THETA_RANGE, pool_c
from .streams import seed_from

HYP = {"H1": ("CUR", ["STD"]), "H2": ("CUR", ["ANN"]), "H3": ("BCM", ["STD", "PLW"]), "H4": ("REV", ["STD"])}
DATASETS = ("mnist", "fmnist")
ALL_ARMS = FF_ARMS + ("BP",)
S_PP = 0.5


def s_units(n_examples, s_pp=S_PP):
    return int(round(s_pp / 100.0 * n_examples))


# ---------------------------------------------------------------- loading and completeness
def load_stage_c(results):
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


def check_complete(table, sel, n, plan):
    """Every planned (dataset, arm) has all N seeds of pool C with a test prediction file; arms with cfg None
    in ETAP_B_WYBOR.json are untrainable (§5.6); BP may be absent from the plan (§8.3 cut; review K2-01)."""
    seeds = set(pool_c(n))
    planned = {(j["dataset"], j["arm"]) for j in plan}
    untrainable = [(ds, a) for ds in DATASETS for a in ALL_ARMS if (sel.get(f"{ds}|{a}") or {}).get("cfg") is None]
    for ds, arm in sorted(planned):
        have = table.get((ds, arm), {})
        missing = seeds - set(have)
        no_test = [s for s, e in have.items() if e["test"] is None]
        if missing or no_test or set(have) - seeds:
            raise SystemExit(f"Etap C niekompletny dla {ds}/{arm}: brak {len(missing)}, bez predykcji "
                             f"{len(no_test)} – analiza zabroniona (§5.8)")
    extra = set(table) - planned
    if extra:
        raise SystemExit(f"biegi spoza planu C: {sorted(extra)}")
    bp_cut = not any(a == "BP" for _, a in planned)
    return untrainable, bp_cut


def failed(entry):
    return bool(entry["run"]["nan"] or entry["val"] <= A.FAIL_ACC)


def counts(table, ds, arm, seeds, testset, readout="main"):
    return np.array([int(table[(ds, arm)][s]["test"][testset][readout].sum()) for s in seeds], dtype=np.int64)


# ---------------------------------------------------------------- confirmatory
def t_ci(d, alpha):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or np.std(d) == 0:
        return [float(d.mean()), float(d.mean())]
    h = stats.t.ppf(1 - alpha / 2, n - 1) * d.std(ddof=1) / math.sqrt(n)
    return [float(d.mean() - h), float(d.mean() + h)]


def hypothesis(table, h, testset_of, n, untrainable, readout="main", drop_failed=False, method="wilcoxon", s_pp=S_PP):
    B, Ks = HYP[h]
    missing = [(ds, a) for ds in DATASETS for a in [B] + Ks if (ds, a) in untrainable]
    if missing:
        note = ("BCM nietrenowalne (" + ", ".join(ds for ds, a in missing if a == "BCM") + ")"
                if any(a == "BCM" for _, a in missing)
                else "ramię nietrenowalne: " + ", ".join(f"{a} ({ds})" for ds, a in missing))
        return {"per_dataset": {}, "joint": {"category": "NIEROZSTRZYGNIETY", "class": "N", "per_dataset": []},
                "stability_joint": None, "untrainable": missing, "note": note}
    per = {}
    seeds_all = pool_c(n)
    for ds in DATASETS:
        seeds = seeds_all
        if drop_failed:
            seeds = [s for s in seeds if not failed(table[(ds, B)][s]) and not any(failed(table[(ds, k)][s]) for k in Ks)]
        if len(seeds) < 2:
            per[ds] = {"category": "NIEROZSTRZYGNIETY", "class": "N", "direction": 0, "N": len(seeds),
                       "note": "za mało par bez awarii"}
            continue
        ts = testset_of[ds]
        n_ex = len(table[(ds, B)][seeds[0]]["test"][ts][readout])
        S = s_units(n_ex, s_pp)
        cb = counts(table, ds, B, seeds, ts, readout)
        D = [cb - counts(table, ds, k, seeds, ts, readout) for k in Ks]
        v = A.verdict_dataset(D, S, method=method)
        v["holm_better_comparators_inf_S"] = [Ks[i] for i in A.holm_rejected(v["p"]["inf_S"], A.ALPHA1 / 2)]
        fb = [failed(table[(ds, B)][s]) for s in seeds]
        st = A.stability(fb, [[failed(table[(ds, k)][s]) for s in seeds] for k in Ks])
        if not drop_failed:
            before = v["class"]
            v = A.apply_stability(v, st)
            v["stab_blocked_R"] = before == "R" and v["class"] == "N"
        to_pp = lambda x: 100.0 * x / n_ex  # noqa: E731
        v.update({"stability": st, "N": len(seeds), "S_units": S, "n_examples": n_ex,
                  "hl_pp": [to_pp(x) for x in v["hl"]],
                  "hl_ci_pp": [[to_pp(c) for c in ci] for ci in v["hl_ci"]],
                  "mean_pp": [to_pp(x) for x in v["mean"]],
                  "t_ci_pp": [[to_pp(c) for c in t_ci(d, A.ALPHA1)] for d in D],
                  "holm_better_comparators": [Ks[i] for i in v["holm_better_comparators"]],
                  "stability_worse_than": [Ks[i] for i in st["holm_worse_than"]],
                  "failures": {"B": int(sum(fb)), **{k: int(sum(failed(table[(ds, k)][s]) for s in seeds)) for k in Ks}}})
        per[ds] = v
    joint = A.combine(per["mnist"], per["fmnist"])
    stab_joint = (A.combine_stability(per["mnist"]["stability"], per["fmnist"]["stability"])
                  if all("stability" in per[d] for d in DATASETS) else None)
    return {"per_dataset": per, "joint": joint, "stability_joint": stab_joint}


def bootstrap_2d(table, h, n, R=10000, readout="main", chunk=500, stab=None):
    """§7.7a: seeds resampled with replacement; within each drawn seed, test examples resampled with
    replacement (same indices for all arms). Statistic: HL of per-seed count differences."""
    B, Ks = HYP[h]
    rng = np.random.default_rng(seed_from("bootstrap", h) % (2 ** 32))
    per = {}
    seeds = pool_c(n)
    for ds in DATASETS:
        arms = [B] + Ks
        n_ex = len(table[(ds, B)][seeds[0]]["test"][ds][readout])
        S = s_units(n_ex)
        pats = 2 ** len(arms)
        P = np.zeros((len(seeds), pats))
        for si, s in enumerate(seeds):
            code = np.zeros(n_ex, dtype=np.int64)
            for ai, a in enumerate(arms):
                code |= (table[(ds, a)][s]["test"][ds][readout].numpy().astype(np.int64) << ai)
            P[si] = np.bincount(code, minlength=pats)
        bits = np.array([[(p >> ai) & 1 for ai in range(len(arms))] for p in range(pats)])
        boot = np.zeros((R, len(Ks)))
        for c0 in range(0, R, chunk):
            r = min(chunk, R - c0)
            sidx = rng.integers(0, len(seeds), (r, len(seeds)))
            cnt = rng.multinomial(n_ex, P[sidx] / n_ex)
            correct = cnt @ bits
            i, j = np.triu_indices(len(seeds))
            for k in range(len(Ks)):
                d = correct[..., 0] - correct[..., k + 1]
                boot[c0:c0 + r, k] = np.median((d[:, i] + d[:, j]) / 2.0, axis=1)
        a = A.ALPHA1 / 2
        pf = lambda cond: cond.mean(axis=0)  # noqa: E731
        p = {"sup_S": pf(boot <= S), "sup_0": pf(boot <= 0), "eq_lo": pf(boot <= -S), "eq_hi": pf(boot >= S),
             "inf_S": pf(boot >= -S), "inf_0": pf(boot >= 0)}
        cat, direction, eq = A.categorize({k: list(v) for k, v in p.items()})
        v = {"category": cat, "class": A.CLASS_OF[cat], "direction": direction}
        if stab and ds in stab:
            v = A.apply_stability(v, stab[ds])  # same R + STAB → N rule as the main analysis (review K2-08)
        per[ds] = {**v, "p": {k: v2.tolist() for k, v2 in p.items()},
                   "ci_pp_1-a": (100.0 * np.quantile(boot, [A.ALPHA1 / 2, 1 - A.ALPHA1 / 2], axis=0).T / n_ex).tolist(),
                   "ci_pp_1-2a": (100.0 * np.quantile(boot, [A.ALPHA1, 1 - A.ALPHA1], axis=0).T / n_ex).tolist()}
    return {"per_dataset": per, "joint": A.combine(per["mnist"], per["fmnist"])}


# ---------------------------------------------------------------- report sentences inputs (§7.4)
NIESPOJNE = {
    "NIESPOJNY-KIERUNEK": "kierunek różnicy przeciwny na obu zbiorach: MNIST: {m}, F-MNIST: {f}",
    "NIESPOJNY-ZGODNY-KIERUNEK": "na jednym zbiorze kategoria zbiorowa, na drugim różnica istotna statystycznie w tym samym kierunku, praktycznie pomijalna (< S, równoważność wykazana): MNIST: {m}, F-MNIST: {f}",
    "NIESPOJNY-CZESCIOWY": "na zbiorze MNIST: {m}; na zbiorze F-MNIST: {f}; kategorie należą do różnych klas, więc werdykt łączny jest nierozstrzygnięty",
}
STAB_ZD = {"STAB-GORSZA": "ulega awarii treningu istotnie częściej niż", "STAB-LEPSZA": "ulega awarii treningu istotnie rzadziej niż każdy komparator:",
           "BRAK": "bez wykazanej różnicy stabilności;"}


def notes(conf, sens, power):
    """All mandatory §7.4 notes as sentences (review K2-03)."""
    out = {}
    for h, c in conf.items():
        B, Ks = HYP[h]
        L = []
        if c.get("note"):
            L.append(c["note"])
            out[h] = L
            continue
        jc = c["joint"]["category"]
        if jc in NIESPOJNE:
            L.append(NIESPOJNE[jc].format(m=c["per_dataset"]["mnist"]["category"], f=c["per_dataset"]["fmnist"]["category"]))
        for ds, v in c["per_dataset"].items():
            if "stability" in v:
                st = v["stability"]
                cnt = "; ".join(f"{B} {x['f_B']} vs {k} {x['f_K']} z {v['N']} ziaren" for k, x in zip(Ks, st["counts"]))
                who = (" " + ", ".join(v["stability_worse_than"])) if st["claim"] == "STAB-GORSZA" else ""
                L.append(f"{ds}: {B} {STAB_ZD[st['claim']]}{who} ({cnt})")
                for k, x in zip([B] + Ks, [st["counts"][0]["f_B"]] + [x["f_K"] for x in st["counts"]]):
                    if x:
                        L.append(f"{ds}: w ramieniu {k} {x} z {v['N']} biegów uległo awarii (NaN lub ≤ 15% walidacji) i zostało ocenione jako najgorsze")
            if v.get("stab_blocked_R"):
                L.append(f"{ds}: równoważność dokładności przy istotnie różnej stabilności")
            if v["category"].endswith("<S"):
                L.append(f"{ds}: różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane")
            if v["category"] == "ROWNOWAZNOSC+":
                L.append(f"{ds}: wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S), kierunek {v['direction']:+d}")
            if v["class"] == "R":
                L.append(f"{ds}: równoważność w granicach ±0.5 p.p. nie wyklucza efektów rzędu 0.4–0.8 p.p. raportowanych w źródle")
            if v["class"] == "G" and v.get("holm_better_comparators"):
                L.append(f"{ds}: komparatory istotnie lepsze (Holm): {v['holm_better_comparators']}")
            tv = sens.get(h, {}).get("b_ttest", {}).get("per_dataset", {}).get(ds)
            if tv and tv.get("class") != v.get("class"):
                L.append(f"{ds}: werdykt testu t ({tv['category']}) różni się klasą od testu głównego ({v['category']})"
                         + (" w obecności awarii" if any(v.get("failures", {}).values()) else ""))
        if h == "H3":
            L.append("θ(0) BCM działa praktycznie tylko w pierwszej epoce; strojenie BCM jest bliższe jednowymiarowemu (§4.1)")
        if c["joint"]["class"] == "N" and power:
            pr = power["table"].get(h, {}).get("0", {}).get("joint", {}).get("R")
            if pr is not None and pr < 0.5:
                L.append(f"badanie nie miało mocy do wykazania równoważności (moc {pr:.2f})")
        q = sens.get(h, {}).get("c_qmnist")
        mn = c["per_dataset"].get("mnist")
        if q and mn and q.get("class") != mn.get("class"):
            L.append(f"na MNIST werdykt zależy od zbioru testowego: oficjalny {mn['category']} (oglądany w źródle), "
                     f"QMNIST-50k {q['category']} (nieoglądany)")
        out[h] = L
    out["H1xH2"] = ["macierz H1 × H2 nie zawiera porównania ANN − STD; wnioski o samym wyżarzaniu LR nie są dopuszczalne"]
    return out


def h1_x_h2(conf):
    c1, c2 = conf["H1"]["joint"]["class"], conf["H2"]["joint"]["class"]
    table = {("P", "P"): "curriculum lepsze zarówno od standardu, jak i od wyżarzania LR",
             ("P", "R"): "przewaga curriculum nad standardem nie została odróżniona od efektu wyżarzania LR (równoważne ±S)",
             ("P", "G"): "curriculum lepsze od standardu, ale wyżarzanie LR lepsze od curriculum",
             ("P", "N"): "przewaga nad standardem; porównanie z wyżarzaniem nierozstrzygnięte",
             ("R", "P"): "curriculum równoważne standardowi, a lepsze od wyżarzania",
             ("R", "R"): "curriculum równoważne standardowi i wyżarzaniu",
             ("R", "G"): "curriculum równoważne standardowi, gorsze od wyżarzania",
             ("G", "P"): "curriculum gorsze od standardu, lepsze od wyżarzania",
             ("G", "R"): "curriculum gorsze od standardu, równoważne wyżarzaniu",
             ("G", "G"): "curriculum gorsze od obu"}
    return {"H1": c1, "H2": c2, "zdanie": table.get((c1, c2), "– (tylko zdania pojedyncze)")}


def boundary(sel):
    """§5.5: chosen configuration within the extreme 10 % of the (log) range of lr or θ."""
    out = []
    for key, v in sel.items():
        cfg = (v or {}).get("cfg") or {}
        for name, rng in (("lr", LR_RANGE), ("theta", THETA_RANGE)):
            if name in cfg:
                lo, hi = math.log10(rng[0]), math.log10(rng[1])
                u = (math.log10(cfg[name]) - lo) / (hi - lo)
                if u <= 0.1 or u >= 0.9:
                    out.append({"arm": key, "param": name, "value": cfg[name], "position": u})
    return out


def versions(table):
    vs = {}
    for runs in table.values():
        for e in runs.values():
            v = e["run"].get("versions", {})
            k = (v.get("gpu"), v.get("driver"), v.get("torch"), v.get("cuda"), v.get("code_commit"), v.get("MKL_CBWR"))
            vs[str(k)] = vs.get(str(k), 0) + 1
    import scipy
    return {"runs_by_environment": vs, "analysis": {"scipy": scipy.__version__, "numpy": np.__version__,
                                                    "torch": torch.__version__}}


# ---------------------------------------------------------------- descriptive / exploratory
def accuracy_table(table):
    out = {}
    for (ds, arm), runs in sorted(table.items()):
        accs = [100.0 * float(e["test"][ds]["main"].float().mean()) for e in runs.values()]
        out[f"{ds}|{arm}"] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)), "n": len(accs),
                              "n_nan": sum(bool(e["run"]["nan"]) for e in runs.values()),
                              "n_failed": sum(failed(e) for e in runs.values())}
    return out


def hl_ci95(d_pp):
    return {"hl": A.hodges_lehmann(d_pp), "ci95": A.hl_ci(d_pp, 0.05), "n": len(d_pp)}


def exploratory(table, results, n):
    """E1–E7. E4/E6 average non-NaN runs only (complete curves); E6 curves of CUR/REV use the end-of-training
    readout formula in every epoch (as logged); 'cross' readout is also computed for BCM/PLW (= hamowanie)."""
    seeds = pool_c(n)
    ex = {"E1_bcm_theta": {}, "E2_bp_gap": {}, "E3_readouts": {}, "E4_dw_norm": {}, "E5_landscape": {},
          "E6_val_curves": {}, "E7_ann_minus_std": {}}
    for ds in DATASETS:
        trs = [np.array(e["run"]["bcm_theta"]) for e in table.get((ds, "BCM"), {}).values() if len(e["run"]["bcm_theta"]) == 25]
        if trs:
            ex["E1_bcm_theta"][ds] = {"mean": np.mean(trs, axis=0).tolist(), "n_runs": len(trs)}
        ffm = {a: np.mean([e["val"] for e in table[(ds, a)].values()]) for a in FF_ARMS if (ds, a) in table}
        if ffm and (ds, "BP") in table and table[(ds, "BP")]:
            best = max(ffm, key=ffm.get)
            n_ex = len(table[(ds, best)][seeds[0]]["test"][ds]["main"])
            d = (counts(table, ds, "BP", seeds, ds) - counts(table, ds, best, seeds, ds)) * 100.0 / n_ex
            ex["E2_bp_gap"][ds] = {"best_ff_by_val": best, **hl_ci95(d)}
        if (ds, "ANN") in table and (ds, "STD") in table:
            n_ex = len(table[(ds, "ANN")][seeds[0]]["test"][ds]["main"])
            d = (counts(table, ds, "ANN", seeds, ds) - counts(table, ds, "STD", seeds, ds)) * 100.0 / n_ex
            ex["E7_ann_minus_std"][ds] = hl_ci95(d)
    for (ds, arm), runs in table.items():
        if arm != "BP":
            ex["E3_readouts"][f"{ds}|{arm}"] = {rd: float(np.mean([100.0 * float(e["test"][ds][rd].float().mean())
                                                                   for e in runs.values()])) for rd in ("main", "no_l0", "cross")}
        if arm in ("STD", "CUR", "ANN"):
            arr = [np.array(e["run"]["dw_norm"]) for e in runs.values() if len(e["run"]["dw_norm"]) == 25]
            if arr:
                ex["E4_dw_norm"][f"{ds}|{arm}"] = {"mean": np.mean(arr, axis=0).tolist(), "n_runs": len(arr)}
        cs = [e["curve"] for e in runs.values() if len(e["curve"]) == 25]
        if cs:
            ex["E6_val_curves"][f"{ds}|{arm}"] = {"mean": np.mean(cs, axis=0).tolist(), "n_runs": len(cs)}
    for st in ("A", "A2"):
        for zp in sorted(glob.glob(os.path.join(results, st, "zaslepione", "*.json"))):
            with open(zp) as f:
                z = json.load(f)
            with open(os.path.join(results, st, "runs", z["id"] + ".json")) as f:
                j = json.load(f)["job"]
            key = f"{st}|{j['dataset']}|{j['arm']}|{j.get('point')}"
            ex["E5_landscape"].setdefault(key, {"cfg": {k: v for k, v in j["cfg"].items() if k != "theta_layers"},
                                                "val": []})["val"].append(z["val_acc"])
    for v in ex["E5_landscape"].values():
        v["mean"] = float(np.mean(v["val"]))
    return ex


def figures(ex, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    for ds, tr in ex["E1_bcm_theta"].items():
        tr = np.array(tr["mean"])
        plt.figure(figsize=(6, 4))
        for l in range(tr.shape[1]):
            plt.plot(range(1, len(tr) + 1), tr[:, l], label=f"warstwa {l}")
        plt.xlabel("epoka"); plt.ylabel("θ_l"); plt.title(f"E1 BCM ({ds})"); plt.legend(fontsize=6)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"E1_bcm_{ds}.png"), dpi=120); plt.close()
    for ds in DATASETS:
        plt.figure(figsize=(6, 4))
        for k, c in ex["E6_val_curves"].items():
            if k.startswith(ds + "|"):
                plt.plot(range(1, 26), c["mean"], label=k.split("|")[1])
        plt.axvline(12.5, ls=":", c="gray")
        plt.xlabel("epoka"); plt.ylabel("dokładność walidacyjna [%]"); plt.title(f"E6 ({ds})"); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"E6_krzywe_{ds}.png"), dpi=120); plt.close()
        arms = sorted({k.split("|")[2] for k in ex["E5_landscape"] if k.startswith(f"A|{ds}|") and k.split("|")[2] != "BP"})
        if arms:
            fig, axs = plt.subplots(1, len(arms), figsize=(3 * len(arms), 3), sharey=True)
            sc = None
            for ax, arm in zip(np.atleast_1d(axs), arms):
                pts = [v for k, v in ex["E5_landscape"].items() if k.startswith(f"A|{ds}|{arm}|")]
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
    ap.add_argument("--sel", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--hashes", required=True, help="HASHE_CHECKPOINTOW.txt from step 1 of §5.8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(argv)
    with open(a.params) as f:
        params = json.load(f)
    with open(a.sel) as f:
        sel = json.load(f)
    n = int(params["N"])
    with open(a.plan) as f:
        plan = json.load(f)
    # records, validation files and predictions must match the hashes fixed before/after the test (review K2-02)
    from .evaluate_test import verify, verify_predictions
    verify(a.results, a.hashes, a.plan)
    verify_predictions(a.results, a.plan)
    table = load_stage_c(a.results)
    untrainable, bp_cut = check_complete(table, sel, n, plan)
    main_sets = {ds: ds for ds in DATASETS}
    rep = {"N": n, "untrainable": untrainable, "bp_cut_8_3": bp_cut, "accuracy": accuracy_table(table), "confirmatory": {}, "sensitivity": {},
           "boundary_5_5": boundary(sel), "versions": versions(table), "power_final_N": params.get("N_power_table")}
    for h in HYP:
        conf = hypothesis(table, h, main_sets, n, untrainable)
        rep["confirmatory"][h] = conf
        if conf.get("untrainable"):
            rep["sensitivity"][h] = {"skipped": "ramię nietrenowalne"}
            continue
        e025 = hypothesis(table, h, main_sets, n, untrainable, s_pp=0.25)
        e10 = hypothesis(table, h, main_sets, n, untrainable, s_pp=1.0)
        rep["sensitivity"][h] = {
            "a_bootstrap_hl": bootstrap_2d(table, h, n, R=a.boot,
                                           stab={d: v["stability"] for d, v in conf["per_dataset"].items() if "stability" in v}),
            "b_ttest": hypothesis(table, h, main_sets, n, untrainable, method="t"),
            "c_qmnist": hypothesis(table, h, {"mnist": "qmnist", "fmnist": "fmnist"}, n, untrainable)["per_dataset"]["mnist"],
            "d_no_failed_pairs": hypothesis(table, h, main_sets, n, untrainable, drop_failed=True),
            "e_S_0.25": {"joint": e025["joint"], "per_dataset": {d: v["category"] for d, v in e025["per_dataset"].items()}},
            "e_S_1.0": {"joint": e10["joint"], "per_dataset": {d: v["category"] for d, v in e10["per_dataset"].items()}},
        }
    rep["notes_7_4"] = notes(rep["confirmatory"], rep["sensitivity"], params.get("N_power_table"))
    rep["H1_x_H2"] = h1_x_h2(rep["confirmatory"])
    rep["exploratory"] = exploratory(table, a.results, n)
    figures(rep["exploratory"], os.path.join(os.path.dirname(os.path.abspath(a.out)), "rysunki"))
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for h, v in rep["confirmatory"].items():
        print(h, v["joint"]["category"], v["joint"].get("per_dataset"), "| stabilność:", v["stability_joint"])


if __name__ == "__main__":
    main()
