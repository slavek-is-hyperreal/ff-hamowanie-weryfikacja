"""Confirmatory statistics (PROTOKOL_v6 §7).

Primary test: Wilcoxon signed-rank on integer differences d − δ (Pratt zeros); exact distribution
for N ≤ 50 without zeros, otherwise normal approximation without continuity correction (scipy
1.14.1). Categories from p-values (§7.2); stability is a separate claim (§7.2); datasets are
combined by §7.3. ``wilcoxon_p_vec`` is a vectorized re-implementation used only in simulations
(``choose_n``, autotests); it is checked against scipy in the tests.
"""
import math
from functools import lru_cache

import numpy as np
from scipy import stats

ALPHA1 = 0.05 / 4
CATEGORIES = ["GORSZOSC", "PRZEWAGA", "ROWNOWAZNOSC+", "GORSZOSC<S", "PRZEWAGA<S", "ROWNOWAZNOSC",
              "NIEROZSTRZYGNIETY"]
CLASS_OF = {"GORSZOSC": "G", "PRZEWAGA": "P", "ROWNOWAZNOSC+": "R", "GORSZOSC<S": "G", "PRZEWAGA<S": "P",
            "ROWNOWAZNOSC": "R", "NIEROZSTRZYGNIETY": "N"}
STRENGTH = {"GORSZOSC": 2, "PRZEWAGA": 2, "GORSZOSC<S": 1, "PRZEWAGA<S": 1, "ROWNOWAZNOSC+": 1,
            "ROWNOWAZNOSC": 1, "NIEROZSTRZYGNIETY": 0}
FAIL_ACC = 15.0


# ---------------------------------------------------------------- primary test (§7.1)
def wilcoxon_p(x, alternative):
    """One-sided p of the Wilcoxon signed-rank test on integer-valued ``x`` (= d − δ)."""
    x = np.asarray(x)
    if np.all(x == 0):
        return 1.0
    method = "exact" if (len(x) <= 50 and not np.any(x == 0)) else "approx"
    kw = {"correction": False} if method == "approx" else {}
    return float(stats.wilcoxon(x, zero_method="pratt", alternative=alternative, method=method, **kw).pvalue)


@lru_cache(maxsize=None)
def _exact_null_sf(n):
    """counts of W+ for n untied ranks; returns survival P(W >= w) for w = 0..n(n+1)/2 (+1 sentinel)."""
    m = n * (n + 1) // 2
    c = np.zeros(m + 1)
    c[0] = 1
    for k in range(1, n + 1):
        c[k:] = c[k:] + c[:-k].copy()
    p = c / c.sum()
    sf = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])
    return sf


def _avg_ranks_abs(x):
    """Average ranks of |x| along the last axis (ties averaged)."""
    return stats.rankdata(np.abs(x), method="average", axis=-1)


def wilcoxon_p_vec(X, alternative):
    """Vectorized ``wilcoxon_p`` over rows of X (reps, N); integer-valued X. Matches scipy 1.14.1
    (checked in tests) for the cases used here."""
    X = np.asarray(X, dtype=float)
    reps, n = X.shape
    ranks = _avg_ranks_abs(X)
    pos = (X > 0)
    nz = (X != 0)
    r_plus = (ranks * pos).sum(1)
    r_minus = (ranks * (X < 0)).sum(1)
    out = np.ones(reps)
    has_zero = (~nz).any(1)
    allzero = ~nz.any(1)
    exact = (n <= 50) & ~has_zero
    if exact.any():
        sf = _exact_null_sf(n)
        m = n * (n + 1) // 2
        # scipy 1.14.1 (_wilcoxon.py): greater -> sf(floor(r_plus)), less -> cdf(ceil(r_plus)) (conservative with ties)
        if alternative == "greater":
            idx = np.floor(r_plus[exact] + 1e-9).astype(int)
        else:
            idx = m - np.ceil(r_plus[exact] - 1e-9).astype(int)  # P(W+ <= c) = P(W+ >= m - c) by symmetry
        idx = idx.clip(0, m + 1)
        out[exact] = sf[idx]
    ap = ~exact & ~allzero
    if ap.any():
        Xa, Ra, NZ = X[ap], ranks[ap], nz[ap]
        # Pratt: zeros ranked then dropped; mean/var from ranks of non-zero entries
        rp = r_plus[ap]
        rm = r_minus[ap]
        tot = (Ra * NZ).sum(1)
        mn = tot / 2.0
        var = ((Ra ** 2) * NZ).sum(1) / 4.0
        z = (rp - mn) / np.sqrt(var)
        out[ap] = stats.norm.sf(z) if alternative == "greater" else stats.norm.cdf(z)
    out[allzero] = 1.0
    return out


# ---------------------------------------------------------------- categories (§7.2)
def _holm_any(pvals, level):
    p = sorted(pvals)
    m = len(p)
    return p[0] <= level / m


def holm_rejected(pvals, level):
    """Holm step-down: indices rejected at family level ``level``."""
    order = np.argsort(pvals)
    m = len(pvals)
    rej = []
    for k, i in enumerate(order):
        if pvals[i] <= level / (m - k):
            rej.append(int(i))
        else:
            break
    return rej


def categorize(p, alpha1=ALPHA1):
    """p: dict of lists (one per comparator) with keys sup_S, sup_0, eq_lo, eq_hi, inf_S, inf_0."""
    a = alpha1 / 2
    sup_S = all(v <= a for v in p["sup_S"])
    sup_0 = all(v <= a for v in p["sup_0"])
    eq = all(lo <= alpha1 and hi <= alpha1 for lo, hi in zip(p["eq_lo"], p["eq_hi"]))
    inf_S = bool(holm_rejected(p["inf_S"], a))
    inf_0 = bool(holm_rejected(p["inf_0"], a))
    for name, cond in (("GORSZOSC", inf_S), ("PRZEWAGA", sup_S), ("ROWNOWAZNOSC+", eq and (sup_0 or inf_0)),
                       ("GORSZOSC<S", inf_0), ("PRZEWAGA<S", sup_0), ("ROWNOWAZNOSC", eq)):
        if cond:
            cat = name
            break
    else:
        cat = "NIEROZSTRZYGNIETY"
    direction = 1 if sup_0 else (-1 if inf_0 else 0)
    return cat, direction, eq


def pvals_wilcoxon(D, S):
    """D: list of integer arrays (one per comparator), S in the same integer units."""
    out = {k: [] for k in ("sup_S", "sup_0", "eq_lo", "eq_hi", "inf_S", "inf_0")}
    for d in D:
        d = np.asarray(d, dtype=np.int64)
        out["sup_S"].append(wilcoxon_p(d - S, "greater"))
        out["sup_0"].append(wilcoxon_p(d, "greater"))
        out["eq_lo"].append(wilcoxon_p(d + S, "greater"))
        out["eq_hi"].append(wilcoxon_p(d - S, "less"))
        out["inf_S"].append(wilcoxon_p(d + S, "less"))
        out["inf_0"].append(wilcoxon_p(d, "less"))
    return out


def pvals_t(D, S):
    """Sensitivity (b): one-sample t on the same integer differences."""
    out = {k: [] for k in ("sup_S", "sup_0", "eq_lo", "eq_hi", "inf_S", "inf_0")}
    for d in D:
        d = np.asarray(d, dtype=float)
        t = lambda x, alt: float(stats.ttest_1samp(x, 0.0, alternative=alt).pvalue) if np.std(x) > 0 else (  # noqa: E731
            0.0 if ((alt == "greater") == (x.mean() > 0)) and x.mean() != 0 else 1.0)
        out["sup_S"].append(t(d - S, "greater"))
        out["sup_0"].append(t(d, "greater"))
        out["eq_lo"].append(t(d + S, "greater"))
        out["eq_hi"].append(t(d - S, "less"))
        out["inf_S"].append(t(d + S, "less"))
        out["inf_0"].append(t(d, "less"))
    return out


def hodges_lehmann(d):
    d = np.asarray(d, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def hl_ci(d, alpha):
    """Descriptive HL interval (1 − alpha) from Walsh averages, normal-approximation ranks."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    i, j = np.triu_indices(n)
    w = np.sort((d[i] + d[j]) / 2.0)
    m = len(w)
    z = stats.norm.ppf(1 - alpha / 2)
    k = int(max(0, math.floor(n * (n + 1) / 4 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24))))
    return [float(w[k]), float(w[m - 1 - k])]


def verdict_dataset(D, S, alpha1=ALPHA1, method="wilcoxon"):
    """Per-dataset accuracy verdict. D: list of integer difference arrays (B − K_j)."""
    p = pvals_wilcoxon(D, S) if method == "wilcoxon" else pvals_t(D, S)
    cat, direction, eq = categorize(p, alpha1)
    worse = holm_rejected(p["inf_0"], alpha1 / 2)
    return {"category": cat, "class": CLASS_OF[cat], "direction": direction, "eq": eq, "p": p,
            "holm_better_comparators": worse,
            "hl": [hodges_lehmann(d) for d in D], "hl_ci": [hl_ci(d, alpha1) for d in D],
            "mean": [float(np.mean(d)) for d in D]}


# ---------------------------------------------------------------- stability (§7.2)
def sign_test_p(k_more, k_less):
    """Exact one-sided sign test: P(X >= k_more), X ~ Bin(k_more + k_less, 1/2)."""
    n = k_more + k_less
    if n == 0:
        return 1.0
    return float(stats.binom.sf(k_more - 1, n, 0.5))


def stability(fail_B, fail_K_list, alpha1=ALPHA1):
    """fail_*: boolean arrays per seed (paired). Returns STAB-GORSZA / STAB-LEPSZA / BRAK and details."""
    fb = np.asarray(fail_B, dtype=bool)
    p_minus, p_plus, counts = [], [], []
    for fk in fail_K_list:
        fk = np.asarray(fk, dtype=bool)
        b_only, k_only = int((fb & ~fk).sum()), int((~fb & fk).sum())
        counts.append({"f_B": int(fb.sum()), "f_K": int(fk.sum()), "tylko_B": b_only, "tylko_K": k_only})
        p_minus.append(sign_test_p(b_only, k_only))
        p_plus.append(sign_test_p(k_only, b_only))
    worse = holm_rejected(p_minus, alpha1 / 2)
    better = all(p <= alpha1 / 2 for p in p_plus)
    claim = "STAB-GORSZA" if worse else ("STAB-LEPSZA" if better else "BRAK")
    return {"claim": claim, "holm_worse_than": worse, "p_minus": p_minus, "p_plus": p_plus, "counts": counts}


def apply_stability(v, stab):
    """Class R is not allowed with a stability difference (§7.2) → N."""
    if v["class"] == "R" and stab["claim"] != "BRAK":
        v = dict(v, category="NIEROZSTRZYGNIETY", stab_blocked=True)
        v["class"] = "N"
    return v


# ---------------------------------------------------------------- combining datasets (§7.3)
def combine(v1, v2):
    c1, c2 = v1["class"], v2["class"]
    k1, k2 = v1["category"], v2["category"]
    if c1 == c2 and c1 != "N":
        weaker = k1 if STRENGTH[k1] <= STRENGTH[k2] else k2
        if c1 == "R":
            weaker = "ROWNOWAZNOSC+" if "ROWNOWAZNOSC+" in (k1, k2) else "ROWNOWAZNOSC"
        return {"category": weaker, "class": c1, "per_dataset": [k1, k2]}
    if c1 == "N" and c2 == "N":
        return {"category": "NIEROZSTRZYGNIETY", "class": "N", "per_dataset": [k1, k2]}
    if {c1, c2} == {"P", "G"}:
        return {"category": "NIESPOJNY-KIERUNEK", "class": "N", "per_dataset": [k1, k2]}
    cat = "NIESPOJNY-CZESCIOWY"
    for a, b in ((v1, v2), (v2, v1)):
        if a["class"] == "P" and b["category"] == "ROWNOWAZNOSC+" and b["direction"] > 0:
            cat = "NIESPOJNY-ZGODNY-KIERUNEK"
        if a["class"] == "G" and b["category"] == "ROWNOWAZNOSC+" and b["direction"] < 0:
            cat = "NIESPOJNY-ZGODNY-KIERUNEK"
    return {"category": cat, "class": "N", "per_dataset": [k1, k2]}


def combine_stability(s1, s2):
    if s1["claim"] == s2["claim"] and s1["claim"] != "BRAK":
        return s1["claim"]
    return "BEZ WYKAZANEJ RÓŻNICY STABILNOŚCI"


# ---------------------------------------------------------------- simulation (§7.6)
def _cat_vec(D_list, S, alpha1):
    """Vectorized categories for simulations. D_list: list (per comparator) of (reps, N) int arrays."""
    a = alpha1 / 2
    m = len(D_list)
    P = {k: [] for k in ("sup_S", "sup_0", "eq_lo", "eq_hi", "inf_S", "inf_0")}
    for D in D_list:
        P["sup_S"].append(wilcoxon_p_vec(D - S, "greater"))
        P["sup_0"].append(wilcoxon_p_vec(D, "greater"))
        P["eq_lo"].append(wilcoxon_p_vec(D + S, "greater"))
        P["eq_hi"].append(wilcoxon_p_vec(D - S, "less"))
        P["inf_S"].append(wilcoxon_p_vec(D + S, "less"))
        P["inf_0"].append(wilcoxon_p_vec(D, "less"))
    A = {k: np.stack(v, 1) for k, v in P.items()}  # (reps, m)

    def holm_any(Pm):
        s = np.sort(Pm, 1)
        return s[:, 0] <= a / m

    sup_S = (A["sup_S"] <= a).all(1)
    sup_0 = (A["sup_0"] <= a).all(1)
    eq = ((A["eq_lo"] <= alpha1) & (A["eq_hi"] <= alpha1)).all(1)
    inf_S, inf_0 = holm_any(A["inf_S"]), holm_any(A["inf_0"])
    cls = np.full(len(sup_S), "N", dtype=object)
    for c, cond in reversed([("G", inf_S), ("P", sup_S), ("R", eq & (sup_0 | inf_0)), ("G", inf_0),
                             ("P", sup_0), ("R", eq)]):
        cls = np.where(cond, c, cls)
    return cls


def _stab_vec(fB, fKs, alpha1):
    """Vectorized stability claim per replicate: 'G' (STAB-GORSZA), 'L' (STAB-LEPSZA) or '-'."""
    a = alpha1 / 2
    m = len(fKs)
    pm, pp = [], []
    for fK in fKs:
        b_only = (fB & ~fK).sum(1)
        k_only = (~fB & fK).sum(1)
        n = b_only + k_only
        pm.append(np.where(n > 0, stats.binom.sf(b_only - 1, np.maximum(n, 1), 0.5), 1.0))
        pp.append(np.where(n > 0, stats.binom.sf(k_only - 1, np.maximum(n, 1), 0.5), 1.0))
    PM = np.sort(np.stack(pm, 1), 1)
    worse = PM[:, 0] <= a / m  # first Holm step decides "at least one rejected"
    better = (np.stack(pp, 1) <= a).all(1)
    return np.where(worse, "G", np.where(better, "L", "-"))


def simulate_joint(n, m, delta_units, sd_units, rho, q, S, fail_value, base, reps, seed, alpha1=ALPHA1):
    """§7.6 simulation of the full procedure (test, Holm/IUT, stability rule R + STAB → N, §7.3).
    q: per dataset dict {'B': .., 'K0': .., ...}. Returns {'joint': {P,R,G,N}, 'per_dataset': [...],
    'stab_worse': [per dataset P(STAB-GORSZA)]}."""
    rng = np.random.default_rng(seed)
    per, stab = [], []
    for ds in range(2):
        sd = sd_units[ds]
        r = rho if m > 1 else 0.5
        sB, sK = sd * math.sqrt(r), sd * math.sqrt(1 - r)
        accB = base[ds] + rng.normal(0, sB, (reps, n))
        failB = rng.random((reps, n)) < q[ds]["B"]
        D_list, fKs = [], []
        for k in range(m):
            accK = base[ds] - delta_units[ds][k] + rng.normal(0, sK, (reps, n))
            failK = rng.random((reps, n)) < q[ds][f"K{k}"]
            b = np.where(failB, fail_value[ds], np.rint(accB))
            kk = np.where(failK, fail_value[ds], np.rint(accK))
            D_list.append((b - kk).astype(np.int64))
            fKs.append(failK)
        cls = _cat_vec(D_list, S[ds], alpha1)
        st = _stab_vec(failB, fKs, alpha1)
        cls = np.where((cls == "R") & (st != "-"), "N", cls)
        per.append(cls)
        stab.append(float((st == "G").mean()))
    c1, c2 = per
    out = np.where((c1 == c2) & (c1 != "N"), c1, "N")
    rate = lambda c: {k: float((c == k).mean()) for k in ("P", "R", "G", "N")}  # noqa: E731
    return {"joint": rate(out), "per_dataset": [rate(c1), rate(c2)], "stab_worse": stab}


def choose_n(sd_units, rho, q, S, fail_value, base, target_delta_units, m_by_h,
             ladder=(30, 45, 60, 90, 120, 150, 200, 250, 300, 400),
             reps=10000, seed=None):
    """§7.6 (ladder D5): smallest N on the ladder meeting power targets for all hypotheses, else 400.
    sd_units: {h: (sd_mnist, sd_fmnist)}; q: {h: [ {'B':..,'K0':..} per dataset ]}; m_by_h: {h: m}.
    Target: joint P for H1–H3 at +target, joint G for H4 at −target; >= 0.8."""
    from .streams import seed_from
    seed = seed_from("choose_n") if seed is None else seed
    table = {}
    for n in ladder:
        ok = True
        table[n] = {}
        for h, m in m_by_h.items():
            sign = -1 if h == "H4" else 1
            delta = [[sign * target_delta_units[ds]] * m for ds in range(2)]
            r = simulate_joint(n, m, delta, sd_units[h], rho if m > 1 else 0.0, q[h], S, fail_value, base,
                               reps, seed % (2 ** 32))
            need = "G" if h == "H4" else "P"
            table[n][h] = r
            ok &= r["joint"][need] >= 0.8
        if ok:
            return n, table
    return ladder[-1], table
