"""Symulacja kontrolna przed R5 (PROTOKOL_v5 §7): test główny = Wilcoxon (Pratt) na d − δ,
kategorie z wartości p (inwersja = przedział Hodgesa–Lehmanna). Bieg z NaN: d ≈ ±77 p.p.
Scenariusze: (a) 1 NaN w ramieniu testowanym, Δ = 0.75; (b) k = 1,3,5 NaN w testowanym, Δ = 0;
(c) NaN symetrycznie (po 1 w każdym ramieniu, różne ziarna), Δ = 0; (d) bez NaN, Δ = 0 – błąd I rodzaju.
Dla porównania test t na tych samych danych. Stałe ziarno."""
import numpy as np
from scipy import stats

A1, S, SIG, BIG = 0.05 / 4, 0.5, 0.9, 77.0


def pv(d, delta, alt, method):
    x = d - delta
    if method == "t":
        return stats.ttest_1samp(x, 0.0, alternative=alt).pvalue
    if np.all(x == 0):
        return 1.0
    return stats.wilcoxon(x, zero_method="pratt", alternative=alt, method="auto").pvalue


def klasa(d, method):
    a = A1 / 2
    inf_S, inf_0 = pv(d, -S, "less", method) <= a, pv(d, 0, "less", method) <= a
    sup_S, sup_0 = pv(d, S, "greater", method) <= a, pv(d, 0, "greater", method) <= a
    eq = pv(d, -S, "greater", method) <= A1 and pv(d, S, "less", method) <= A1
    if inf_S: return "G"
    if sup_S: return "P"
    if eq and (sup_0 or inf_0): return "R"
    if inf_0: return "G"
    if sup_0: return "P"
    if eq: return "R"
    return "N"


def rate(n, delta, k_test=0, k_comp=0, reps=2000, method="w", seed=0):
    rng = np.random.default_rng(seed)
    out = {"P": 0, "R": 0, "G": 0, "N": 0}
    for _ in range(reps):
        d = rng.normal(delta, SIG, n)
        d[:k_test] = -BIG
        d[k_test:k_test + k_comp] = BIG
        out[klasa(d, method)] += 1
    return {c: v / reps for c, v in out.items()}


if __name__ == "__main__":
    for n in (30, 60, 120):
        print(f"== N = {n}")
        for m in ("w", "t"):
            print(f"  [{m}] (a) 1 NaN testowane, Δ=0.75:", rate(n, 0.75, 1, method=m))
            for k in (1, 3, 5):
                print(f"  [{m}] (b) {k} NaN testowane, Δ=0:", rate(n, 0.0, k, method=m))
            print(f"  [{m}] (c) 1+1 NaN symetrycznie, Δ=0:", rate(n, 0.0, 1, 1, method=m))
            print(f"  [{m}] (d) bez NaN, Δ=0:", rate(n, 0.0, method=m, reps=4000))
            print(f"  [{m}] (d2) bez NaN, Δ=0.75:", rate(n, 0.75, method=m))
