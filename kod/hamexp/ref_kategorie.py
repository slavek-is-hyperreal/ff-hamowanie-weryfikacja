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


def laczenie(k1, d1, k2, d2):
    """§7.3 written from the protocol table."""
    c1, c2 = klasa(k1), klasa(k2)
    sila = {"GORSZOSC": 2, "PRZEWAGA": 2}
    if c1 == c2 and c1 != "N":
        if c1 == "R":
            return "ROWNOWAZNOSC+" if "ROWNOWAZNOSC+" in (k1, k2) else "ROWNOWAZNOSC"
        return k1 if sila.get(k1, 1) <= sila.get(k2, 1) else k2
    if c1 == "N" and c2 == "N":
        return "NIEROZSTRZYGNIETY"
    if {c1, c2} == {"P", "G"}:
        return "NIESPOJNY-KIERUNEK"
    if (c1 == "P" and k2 == "ROWNOWAZNOSC+" and d2 > 0) or (c2 == "P" and k1 == "ROWNOWAZNOSC+" and d1 > 0):
        return "NIESPOJNY-ZGODNY-KIERUNEK"
    if (c1 == "G" and k2 == "ROWNOWAZNOSC+" and d2 < 0) or (c2 == "G" and k1 == "ROWNOWAZNOSC+" and d1 < 0):
        return "NIESPOJNY-ZGODNY-KIERUNEK"
    return "NIESPOJNY-CZESCIOWY"


def kierunek(D, S):
    half = A1 / 2
    if all(p_one(d, "greater") <= half for d in D):
        return 1
    if holm([p_one(d, "less") for d in D], half):
        return -1
    return 0


def werdykt(per_ds):
    """per_ds: list of (D, S, fB, fKs) for the two datasets. Full path §7.2 (with stability rule) + §7.3."""
    ks, ds_, st = [], [], []
    for D, S, fB, fKs in per_ds:
        k = kategoria([list(d) for d in D], S)
        s = stabilnosc(list(fB), [list(f) for f in fKs])
        if klasa(k) == "R" and s != "BRAK":
            k = "NIEROZSTRZYGNIETY"
        ks.append(k)
        ds_.append(kierunek([list(d) for d in D], S))
        st.append(s)
    joint = laczenie(ks[0], ds_[0], ks[1], ds_[1])
    stab = st[0] if st[0] == st[1] and st[0] != "BRAK" else "BEZ WYKAZANEJ RÓŻNICY STABILNOŚCI"
    return joint, stab
