"""Reference implementation of the category logic (PROTOKOL_v6 §7.9), written directly from §7.1–7.3
without importing hamexp.analysis. Used only by the self-test."""
from scipy import stats

A1 = 0.05 / 4


def p_one(x, alt):
    x = [int(v) for v in x]
    if all(v == 0 for v in x):
        return 1.0
    if len(x) <= 50 and 0 not in x:
        return float(stats.wilcoxon(x, zero_method="pratt", alternative=alt, method="exact").pvalue)
    return float(stats.wilcoxon(x, zero_method="pratt", alternative=alt, method="approx", correction=False).pvalue)


def holm(ps, level):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    out = []
    for k, i in enumerate(idx):
        if ps[i] <= level / (len(ps) - k):
            out.append(i)
        else:
            break
    return out


def kategoria(D, S):
    """D: list of lists of integer differences (one per comparator)."""
    half = A1 / 2
    sh = lambda d, c: [v - c for v in d]  # noqa: E731
    supS = all(p_one(sh(d, S), "greater") <= half for d in D)
    sup0 = all(p_one(d, "greater") <= half for d in D)
    eq = all(p_one(sh(d, -S), "greater") <= A1 and p_one(sh(d, S), "less") <= A1 for d in D)
    infS = len(holm([p_one(sh(d, -S), "less") for d in D], half)) > 0
    inf0 = len(holm([p_one(d, "less") for d in D], half)) > 0
    if infS:
        return "GORSZOSC"
    if supS:
        return "PRZEWAGA"
    if eq and (sup0 or inf0):
        return "ROWNOWAZNOSC+"
    if inf0:
        return "GORSZOSC<S"
    if sup0:
        return "PRZEWAGA<S"
    if eq:
        return "ROWNOWAZNOSC"
    return "NIEROZSTRZYGNIETY"


def stabilnosc(fB, fKs):
    half = A1 / 2

    def sign_p(a, b):
        n = a + b
        return 1.0 if n == 0 else float(stats.binomtest(a, n, 0.5, alternative="greater").pvalue)

    pm, pp = [], []
    for fK in fKs:
        b_only = sum(1 for x, y in zip(fB, fK) if x and not y)
        k_only = sum(1 for x, y in zip(fB, fK) if y and not x)
        pm.append(sign_p(b_only, k_only))
        pp.append(sign_p(k_only, b_only))
    if holm(pm, half):
        return "STAB-GORSZA"
    if all(p <= half for p in pp):
        return "STAB-LEPSZA"
    return "BRAK"


def klasa(kat):
    return {"GORSZOSC": "G", "GORSZOSC<S": "G", "PRZEWAGA": "P", "PRZEWAGA<S": "P", "ROWNOWAZNOSC+": "R",
            "ROWNOWAZNOSC": "R"}.get(kat, "N")
