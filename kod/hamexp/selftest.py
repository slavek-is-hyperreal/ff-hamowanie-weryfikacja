"""Self-test of the analysis (PROTOKOL_v6 §7.9), scenarios (i)–(xiii).

Usage: python -m hamexp.selftest --out ../wyniki/SELFTEST.json   (exit code 1 on any failure)
"""
import argparse
import json
import math
import sys

import numpy as np

from . import analysis as A
from . import ref_kategorie as R
from .streams import seed_from

S = 50  # 0.5 p.p. on 10 000 examples
FAILV = 980  # class-0 share on MNIST-test (9.80%) in counts


def _d(rng, n, delta, sd=90):
    return np.rint(rng.normal(delta, sd, n)).astype(np.int64)


def check(name, ok, info, results):
    results.append({"scenario": name, "ok": bool(ok), "info": info})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=4000)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(seed_from("autotest") % (2 ** 32))
    res = []

    # (i)–(v): deterministic category scenarios, final implementation vs reference
    scen = {
        "i_wszystkie_P": ([_d(rng, 60, 150)], "P"),
        "ii_R": ([_d(rng, 120, 0, 40)], "R"),
        "iii_G_H3_jeden": ([_d(rng, 60, -150), _d(rng, 60, 200)], "G"),
        "iv_P": ([_d(rng, 60, 120)], "P"),
        "v_G": ([_d(rng, 60, -120)], "G"),
    }
    for name, (D, want) in scen.items():
        v = A.verdict_dataset(D, S)
        ref = R.kategoria([list(d) for d in D], S)
        check(name, v["category"] == ref and v["class"] == want, {"final": v["category"], "ref": ref}, res)

    # (iv)/(v) joint: NIESPÓJNY-KIERUNEK and NIESPÓJNY-ZGODNY-KIERUNEK, (xiii) P + R without "+"
    vP = {"category": "PRZEWAGA<S", "class": "P", "direction": 1}
    vG = {"category": "GORSZOSC<S", "class": "G", "direction": -1}
    vRp = {"category": "ROWNOWAZNOSC+", "class": "R", "direction": 1}
    vR = {"category": "ROWNOWAZNOSC", "class": "R", "direction": 0}
    check("iv_niespojny_kierunek", A.combine(vP, vG)["category"] == "NIESPOJNY-KIERUNEK", {}, res)
    check("v_niespojny_zgodny", A.combine(vP, vRp)["category"] == "NIESPOJNY-ZGODNY-KIERUNEK", {}, res)
    check("xiii_P_R_bez_plus", A.combine(vP, vR)["category"] == "NIESPOJNY-CZESCIOWY", {}, res)

    # (vi) full path (§7.2 incl. stability rule + §7.3), final implementation vs reference, 1 000 data sets
    #      from the sym_nan.py-style generator (normal differences + failures scored as class 0)
    mism = 0
    cover = {}
    for t in range(1000):
        n = [30, 60, 120][t % 3]
        m = 1 + (t % 2)
        per_ds = []
        for _ds in range(2):
            fB = rng.random(n) < rng.choice([0.0, 0.0, 0.03, 0.15])
            D, fKs = [], []
            sd = int(rng.choice([20, 40, 90]))
            for _ in range(m):
                fK = rng.random(n) < rng.choice([0.0, 0.0, 0.03, 0.25])
                d = _d(rng, n, rng.choice([-100, -60, -25, 0, 25, 60, 100]), sd)
                d = np.where(fB & ~fK, FAILV - 9000, np.where(fK & ~fB, 9000 - FAILV, np.where(fB & fK, 0, d)))
                D.append(d.astype(np.int64))
                fKs.append(fK)
            per_ds.append((D, S, fB, fKs))
        vs, sts = [], []
        for D, S_, fB, fKs in per_ds:
            v = A.apply_stability(A.verdict_dataset(D, S_), st := A.stability(fB, fKs))
            vs.append(v)
            sts.append(st)
        mine = (A.combine(vs[0], vs[1])["category"], A.combine_stability(sts[0], sts[1]))
        cover[mine[0]] = cover.get(mine[0], 0) + 1
        for st in sts:
            cover[st["claim"]] = cover.get(st["claim"], 0) + 1
        if mine != R.werdykt(per_ds):
            mism += 1
    check("vi_zgodnosc_pelnej_sciezki", mism == 0, {"niezgodne": mism, "z": 1000, "pokrycie": cover}, res)
    check("vi_pokrycie_kategorii", all(cover.get(k, 0) > 0 for k in (
        "ROWNOWAZNOSC+", "NIESPOJNY-ZGODNY-KIERUNEK", "NIESPOJNY-KIERUNEK", "STAB-LEPSZA", "STAB-GORSZA")), cover, res)

    # vectorized simulation test vs scipy
    X = np.stack([_d(rng, 45, 20) for _ in range(200)])
    X[::7, :3] = 0
    pv = A.wilcoxon_p_vec(X, "greater")
    ps = np.array([A.wilcoxon_p(x, "greater") for x in X])
    check("vec_vs_scipy", np.allclose(pv, ps, atol=1e-12), {"max_abs": float(np.abs(pv - ps).max())}, res)

    # (vii) Holm list in H3
    D = [_d(rng, 60, -200), _d(rng, 60, -200)]
    v = A.verdict_dataset(D, S)
    check("vii_holm_dwa", sorted(v["holm_better_comparators"]) == [0, 1], v["holm_better_comparators"], res)
    D = [_d(rng, 60, -200), _d(rng, 60, 0)]
    v = A.verdict_dataset(D, S)
    check("vii_holm_jeden", v["holm_better_comparators"] == [0], v["holm_better_comparators"], res)

    # (viii) size of rejections p+(0), p-(0) at Δ = 0 (per dataset), N = 30 and 120
    for n in (30, 120):
        X = np.stack([_d(rng, n, 0) for _ in range(a.reps)])
        for alt in ("greater", "less"):
            rate = float((A.wilcoxon_p_vec(X, alt) <= A.ALPHA1 / 2).mean())
            lim = A.ALPHA1 / 2 + 3 * math.sqrt(A.ALPHA1 / 2 * (1 - A.ALPHA1 / 2) / a.reps)
            check(f"viii_rozmiar_{alt}_N{n}", rate <= lim, {"rate": rate, "limit": lim}, res)

    # (viii) H3 variant (m = 2, both comparators Δ = 0): P(class P) per dataset
    for n in (30, 120):
        X1 = np.stack([_d(rng, n, 0) for _ in range(a.reps)])
        X2 = np.stack([_d(rng, n, 0) for _ in range(a.reps)])
        rate = float((A._cat_vec([X1, X2], S, A.ALPHA1) == "P").mean())
        lim = A.ALPHA1 / 2 + 3 * math.sqrt(A.ALPHA1 / 2 * (1 - A.ALPHA1 / 2) / a.reps)
        check(f"viii_H3_klasaP_N{n}", rate <= lim, {"rate": rate, "limit": lim}, res)

    # (ix) H3 variant: first comparator at ±S, second at 0 → P(class R) ≤ α′
    for n in (30, 120):
        for delta in (-S, S):
            X1 = np.stack([_d(rng, n, delta) for _ in range(a.reps)])
            X2 = np.stack([_d(rng, n, 0) for _ in range(a.reps)])
            rate = float((A._cat_vec([X1, X2], S, A.ALPHA1) == "R").mean())
            lim = A.ALPHA1 + 3 * math.sqrt(A.ALPHA1 * (1 - A.ALPHA1) / a.reps)
            check(f"ix_H3_R_{delta}_N{n}", rate <= lim, {"rate": rate, "limit": lim}, res)

    # (ix) class R at the equivalence margin, Δ = −S and +S
    for n in (30, 120):
        for delta in (-S, S):
            X = np.stack([_d(rng, n, delta) for _ in range(a.reps)])
            cls = A._cat_vec([X], S, A.ALPHA1)
            rate = float((cls == "R").mean())
            lim = A.ALPHA1 + 3 * math.sqrt(A.ALPHA1 * (1 - A.ALPHA1) / a.reps)
            check(f"ix_R_na_granicy_{delta}_N{n}", rate <= lim, {"rate": rate, "limit": lim}, res)

    # (x) size of the stability rule at q_B = q_K = 5 %, N = 120
    hits = 0
    for _ in range(a.reps):
        fb, fk = rng.random(120) < 0.05, rng.random(120) < 0.05
        hits += A.stability(fb, [fk])["claim"] == "STAB-GORSZA"
    rate = hits / a.reps
    lim = A.ALPHA1 / 2 + 3 * math.sqrt(A.ALPHA1 / 2 * (1 - A.ALPHA1 / 2) / a.reps)
    check("x_rozmiar_stabilnosci", rate <= lim, {"rate": rate, "limit": lim}, res)

    # (xi) deterministic stability cases
    def f(k, n=60):
        x = np.zeros(n, bool)
        x[:k] = True
        return x
    none = np.zeros(60, bool)
    check("xi_8_0_m1", A.stability(f(8), [none])["claim"] == "STAB-GORSZA", {}, res)
    check("xi_7_0_m1", A.stability(f(7), [none])["claim"] == "BRAK", {}, res)
    check("xi_8_0_H3", A.stability(f(8), [none, none])["claim"] == "BRAK", {}, res)
    check("xi_9_0_H3", A.stability(f(9), [none, none])["claim"] == "STAB-GORSZA", {}, res)
    check("xi_ref_zgodnosc", all(A.stability(f(k), [none] * m)["claim"] == R.stabilnosc(list(f(k)), [list(none)] * m)
                                 for k in range(12) for m in (1, 2)), {}, res)
    vr = A.apply_stability({"category": "ROWNOWAZNOSC", "class": "R", "direction": 0}, {"claim": "STAB-GORSZA"})
    check("xi_R_plus_STAB_daje_N", vr["class"] == "N", vr, res)

    # (xii) pair with failure in both arms (d = 0) and d = ±S exactly
    d = _d(rng, 60, 0)
    d[:5] = 0
    d[5:8] = S
    d[8:11] = -S
    v = A.verdict_dataset([d], S)
    check("xii_zera_i_S", v["category"] == R.kategoria([list(d)], S), {"final": v["category"]}, res)

    ok = all(r["ok"] for r in res)
    with open(a.out, "w") as fo:
        json.dump({"ok": ok, "results": res}, fo, indent=1, default=str)
    for r in res:
        print(("OK  " if r["ok"] else "BŁĄD"), r["scenario"], r["info"] if not r["ok"] else "")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
