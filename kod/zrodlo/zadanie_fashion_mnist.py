"""
Zadanie: klasyfikacja Fashion-MNIST trzema wariantami z planu w
../PRZEKAZANIE.md ("Proponowany ksztalt eksperymentu"):

1. "standard"   -- FF, dobroc Hintona (suma kwadratow po ReLU)
2. "hamowanie"  -- FF, dobroc z odcieta strona (pomysl Slawka)
3. "backprop"   -- zwykla siec ReLU, zwykla propagacja wsteczna,
                    punkt odniesienia (bez sztuczki z etykieta w obrazie
                    -- backprop jej nie potrzebuje)

Pomiar glowny: dokladnosc testowa. Pomiar diagnostyczny, WAZNIEJSZY przy
pierwszym uruchomieniu: separacja dobroci (pozytyw minus negatyw) w
ostatnim kroku treningu kazdej warstwy FF. Jesli separacja jest bliska
zeru albo ujemna, siec nie nauczyla sie odrozniac prawdziwej etykiety od
falszywej -- to jest sygnatura trywialnego rozwiazania opisana w
PRZEKAZANIE.md ("Ryzyko trywialnego rozwiazania"), i wtedy liczba
dokladnosci ponizej nic nie znaczy.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aktywacja_hamujaca import (dopasuj_szerokosc_hamujaca, policz_parametry,
                                 zbuduj_siec_backprop_hamowanie)
from dane import LICZBA_KLAS, WYMIAR_WEJSCIA, losowa_zla_etykieta, podprobka, wczytaj_zbiory, zakoduj_etykiete
from siec_ff import SiecFF
from wspolne import podsumuj_po_ziarnach, ustaw_ziarno, zapisz_wyniki_csv

WYMIAR_WYJSCIA = LICZBA_KLAS

# domyslna szerokosc/glebokosc -- wartosci z pierwszego przebiegu
# (krotka wersja + pelny przebieg 5+10 ziaren, 2026-09-04). Sweep
# szerokosci/glebokosci w kampanii nizej NADPISUJE je przez argumenty
# funkcji, nie zmienia tych stalych -- one sa punktem odniesienia.
SZEROKOSC_DOMYSLNA = 64
LICZBA_WARSTW_UKRYTYCH_DOMYSLNA = 3

# szybkosc uczenia optymalizatora kazdej warstwy FF -- Adam, wartosc
# domyslna z hamowanie/ (dziala tam stabilnie na siedziach podobnej skali)
LR_FF = 1e-3

# Stosunek teta/szerokosc=8/64=0.125 pochodzi z recznego sweepu przy
# szerokosci 64 (scratchpad/kalibracja_teta.py, 2026-09-04) -- teta=64
# (przepis Hintona: teta=szerokosc) dawal separacje ujemna od warstwy 1
# i dokladnosc ~0.30-0.35; wartosci teta w 2-8 dawaly 0.60-0.74 bez
# zapadania sie separacji. Przy INNEJ szerokosci ten stosunek jest
# EKSTRAPOLACJA, nie ponowna kalibracja -- osobny sweep dla kazdej
# szerokosci pochlonalby zbyt duzo budzetu czasu. Flagowane w README.
STOSUNEK_TETA_DOSTROJONY = 0.125


def etykieta_lateralne(beta: float) -> str:
    return f"lateralne_b{beta}"


def etykieta_lateralne_dzielace(gamma: float) -> str:
    return f"lateralne_dzielace_g{gamma}"


def gradient_alfa(ksztalt: str, liczba_warstw: int) -> list:
    """
    Lista alfa (sila formuly hamowanie per warstwa: 0=czysty standard,
    1=czysty hamowanie, patrz dobroc() w siec_ff.py) wedlug wybranego
    ksztaltu zaniku z numerem warstwy -- pomysl Slawka (2026-09-07):
    zamiast skoku "pierwsze K warstw w pelni, reszta wcale"
    (hamowanie_pierwsze_K), plynna funkcja warstwa -> sila.

    t = pozycja warstwy w [0, 1] (0 = pierwsza/najplytsza, 1 = ostatnia/
    najglebsza). Trzy ksztalty do porownania:

    "liniowa": alfa(t) = 1-t -- rownomierny spadek, punkt odniesienia.
    "wykladnicza": alfa(t) = exp(-t/0.3) -- silne TYLKO najblizej wejscia,
      dlugi "ogon" bliski zeru -- test hipotezy "hamowanie pomaga tylko
      tam, gdzie sygnal jest najbardziej surowy/zaszumiony".
    "pierwiastkowa": alfa(t) = sqrt(1-t) -- przeciwienstwo wykladniczej:
      zostaje silne przez wiekszosc warstw, gwaltowny spadek dopiero na
      samym koncu -- test hipotezy "hamowanie szkodzi tylko w ostatnich,
      najglebszych warstwach, wiec wystarczy wygasic je tam".

    Stala 0.3 w formule wykladniczej dobrana tak, by alfa spadlo ponizej
    0.05 (5% sily) juz w okolicach t=0.9 -- czyli "prawie zero" na
    wiekszosci glebokich warstw, nie tylko na samym koncu.
    """
    dzielnik = max(liczba_warstw - 1, 1)
    alfy = []
    for l in range(liczba_warstw):
        t = l / dzielnik
        if ksztalt == "liniowa":
            alfa = 1.0 - t
        elif ksztalt == "wykladnicza":
            alfa = math.exp(-t / 0.3)
        elif ksztalt == "pierwiastkowa":
            alfa = math.sqrt(max(0.0, 1.0 - t))
        else:
            raise ValueError(f"nieznany ksztalt gradientu: {ksztalt}")
        alfy.append(alfa)
    return alfy


def konfiguracje_ff(szerokosc: int, liczba_warstw_ukrytych: int = None,
                     warianty_dozwolone: set = None, teta_dostrojony_override: float = None,
                     teta_per_wariant: dict = None, wspolczynniki_lr: tuple = (),
                     krok_impulsu: tuple = None, punkty_przelaczenia: tuple = (),
                     progi_warstw: tuple = None, dawki_hamowania: tuple = (),
                     bety_lateralne: tuple = (), gammy_lateralne: tuple = (),
                     wagi_vicreg: tuple = (), wagi_vicreg_pelny: tuple = (),
                     harmonogram_progow: list = None):
    """
    Konfiguracje FF do porownania: (etykieta_wyniku, harmonogram(epoki_total)).
    harmonogram to funkcja, ktora dla danego calkowitego budzetu epok zwraca
    liste segmentow (epoki, formula, teta) -- kazdy prosty wariant to jeden
    segment obejmujacy cala pule epok, wariant "curriculum" dzieli pule na
    dwa segmenty z rozna formula W CZASIE (te same warstwy, rozne epoki).

    "hamowanie_pierwsze_K" dzieli SIEC W PRZESTRZENI zamiast w czasie --
    pierwsze K warstw uzywa formuly hamowanie od pierwszego kroku treningu,
    pozostale warstwy formuly standard_dostrojony, PRZEZ CALY trening.
    Pomysl Slawka: skoro pewnosc siebie (hamowanie) pomaga przy plytkich
    sieciach, a szum jest najsurowszy na wejsciu (patrz odwrocenie przy
    glebokosci w README.md), to moze mechanizm powinien byc silny na
    poczatkowych warstwach i gasnac z numerem warstwy, zamiast dzialac
    jednolicie na calej glebokosci. liczba_warstw_ukrytych musi byc podana,
    zeby zbudowac liste per-warstwa wlasciwej dlugosci -- funkcje proste
    (bez podzialu przestrzennego) dzialaja bez tego argumentu (kompatybilnosc
    wsteczna z wywolaniami, ktore nie potrzebuja tej konfiguracji).

    "standard_niedostrojony" uzywa progu WPROST z literatury Hintona
    (teta=szerokosc) -- to jest to, co kazdy, kto implementuje FF wedlug
    przepisu z pracy, dostanie bez dodatkowej pracy.
    "standard_dostrojony" to ta sama formula dobroci, prog przeskalowany
    wedlug STOSUNEK_TETA_DOSTROJONY (patrz komentarz wyzej -- ekstrapolacja
    przy szerokosci innej niz 64).
    "hamowanie" uzywa wlasnego, teoretycznie uzasadnionego progu 0 (naturalny
    punkt neutralny dla roznicy dwoch sum kwadratow) BEZ zadnego strojenia --
    uczciwosc porownania: dostrojony konkurent kontra niestrojony pomysl.
    "curriculum_std_do_ham" -- pomysl Slawka: pierwsza polowa epok formula
    standardowa (dostrojona), zeby zbudowac reprezentacje ogolna, druga
    polowa formula hamowanie NA TYCH SAMYCH WAGACH (bez resetu optymalizatora),
    zeby doglebic kontrast pod koniec treningu. Budzet epok laczny taki sam
    jak dla wariantow prostych -- porownanie jest uczciwe pod wzgledem
    calkowitego czasu treningu.
    "curriculum_1w4_std_3w4_ham" -- ten sam pomysl, dluzsza druga faza (1/4
    epok standard, 3/4 hamowanie zamiast pol na pol). Motywacja z wyniku
    curriculum_std_do_ham przy glebokosci 10 (README.md, sekcja
    "Curriculum"): warstwy glebokie musza jednoczesnie nauczyc sie nowej
    formuly I nadazyc za zmieniajacym sie wyjsciem warstw pod soba --
    dluzsza druga faza daje im na to wiecej czasu.
    """
    if teta_per_wariant is not None:
        # PLAN.md #4: kazda metoda z wlasna teta ze sweepu na pelnej skali.
        return tuple(konf for etykieta, teta in teta_per_wariant.items()
                     for konf in konfiguracje_ff(szerokosc, liczba_warstw_ukrytych,
                                                 {etykieta}, teta, wspolczynniki_lr=wspolczynniki_lr,
                                                 krok_impulsu=krok_impulsu,
                                                 punkty_przelaczenia=punkty_przelaczenia,
                                                 progi_warstw=progi_warstw,
                                                 dawki_hamowania=dawki_hamowania,
                                                 bety_lateralne=bety_lateralne,
                                                 wagi_vicreg=wagi_vicreg,
                                                 wagi_vicreg_pelny=wagi_vicreg_pelny,
                                                 harmonogram_progow=harmonogram_progow))

    teta_dostrojony = (teta_dostrojony_override if teta_dostrojony_override is not None
                       else STOSUNEK_TETA_DOSTROJONY * szerokosc)

    def polowa_na_polowe(epoki_total):
        pierwsza = epoki_total // 2
        druga = epoki_total - pierwsza
        return [(pierwsza, "standard", teta_dostrojony), (druga, "hamowanie", None)]

    def cwiartka_na_reszte(epoki_total):
        pierwsza = epoki_total // 4
        druga = epoki_total - pierwsza
        return [(pierwsza, "standard", teta_dostrojony), (druga, "hamowanie", None)]

    def polowa_na_polowe_odwrocone(epoki_total):
        # Ablacja recenzenta: odwrocona kolejnosc -- najpierw hamowanie (na
        # losowych wagach), potem standard. Test, czy przewaga curriculum
        # bierze sie z KOLEJNOSCI (buduj reprezentacje, potem wyostrz kontrast)
        # czy z samego faktu zmiany formuly w polowie treningu.
        pierwsza = epoki_total // 2
        druga = epoki_total - pierwsza
        return [(pierwsza, "hamowanie", None), (druga, "standard", teta_dostrojony)]

    konfiguracje = [
        ("standard_niedostrojony", lambda e: [(e, "standard", None)]),
        ("standard_dostrojony", lambda e: [(e, "standard", teta_dostrojony)]),
        ("hamowanie", lambda e: [(e, "hamowanie", None)]),
        ("curriculum_std_do_ham", polowa_na_polowe),
        ("curriculum_1w4_std_3w4_ham", cwiartka_na_reszte),
        ("curriculum_ham_do_std", polowa_na_polowe_odwrocone),
    ]

    def kontrola_lr(f):
        # PLAN.md #2: formula standard przez caly trening, LR x f od polowy
        # (ten sam podzial epok co curriculum). Czwarty element segmentu = LR.
        def fn(epoki_total):
            pierwsza = epoki_total // 2
            return [(pierwsza, "standard", teta_dostrojony, LR_FF),
                    (epoki_total - pierwsza, "standard", teta_dostrojony, LR_FF * f)]
        return fn

    for f in wspolczynniki_lr:
        konfiguracje.append((f"kontrola_lr_x{f}", kontrola_lr(f)))

    def kontrola_progu(epoki_total):
        # PLAN.md #2d: formula standard przez caly trening, prog teta -> 0 od
        # polowy (curriculum w fazie hamowania tez ma teta=0). Oddziela "kara za
        # ReLU(-z)" od "zmiana progu".
        pierwsza = epoki_total // 2
        return [(pierwsza, "standard", teta_dostrojony), (epoki_total - pierwsza, "standard", 0.0)]

    konfiguracje.append(("kontrola_progu", kontrola_progu))

    # PLAN.md #10: impuls hamowania -- 3 epoki hamowania na starcie albo w epokach
    # 13-15, poza tym standard. Test okna krytycznego.
    konfiguracje.append(("impuls_wczesny", lambda e: [(3, "hamowanie", None),
                                                       (e - 3, "standard", teta_dostrojony)]))
    konfiguracje.append(("impuls_pozny", lambda e: [(12, "standard", teta_dostrojony),
                                                     (3, "hamowanie", None),
                                                     (e - 15, "standard", teta_dostrojony)]))

    # PLAN.md E3 (efekt dawki): k epok hamowania na starcie, potem standard do konca.
    for k in dawki_hamowania:
        konfiguracje.append((f"hamowanie_start_{k}",
                             lambda e, k=k: [(k, "hamowanie", None), (e - k, "standard", teta_dostrojony)]))
    # PLAN.md #12: krotsze curriculum 6+6 epok, niezaleznie od epoki_ff.
    konfiguracje.append(("curriculum_6_6", lambda e: [(6, "standard", teta_dostrojony), (6, "hamowanie", None)]))

    if krok_impulsu is not None:
        # PLAN.md #10, kontrola kroku: standard z LR x f tylko w epokach impulsu.
        f_wcz, f_poz = krok_impulsu
        konfiguracje.append(("krok_wczesny", lambda e: [(3, "standard", teta_dostrojony, LR_FF * f_wcz),
                                                         (e - 3, "standard", teta_dostrojony, LR_FF)]))
        konfiguracje.append(("krok_pozny", lambda e: [(12, "standard", teta_dostrojony, LR_FF),
                                                       (3, "standard", teta_dostrojony, LR_FF * f_poz),
                                                       (e - 15, "standard", teta_dostrojony, LR_FF)]))

    def curriculum_teta8(epoki_total):
        # PLAN.md #2e: hamowanie od polowy, ale z teta z fazy standardowej
        # (izoluje kare od zmiany progu).
        pierwsza = epoki_total // 2
        return [(pierwsza, "standard", teta_dostrojony), (epoki_total - pierwsza, "hamowanie", teta_dostrojony)]

    def curriculum_lr_polowa(epoki_total):
        # PLAN.md #2f: curriculum z kompensacja kroku (LR x0.5 od przelaczenia).
        pierwsza = epoki_total // 2
        return [(pierwsza, "standard", teta_dostrojony, LR_FF),
                (epoki_total - pierwsza, "hamowanie", None, LR_FF * 0.5)]

    konfiguracje.append(("curriculum_teta8", curriculum_teta8))
    konfiguracje.append(("curriculum_lr_x0.5", curriculum_lr_polowa))

    def rampa_alfa(epoki_total):
        # PLAN.md #9: polowa epok standard, potem alfa rosnie liniowo co epoke
        # od 1/n do 1, teta = (1 - alfa) * teta_dostrojony (regula z gradient_hamowania).
        pierwsza = epoki_total // 2
        n = epoki_total - pierwsza
        return [(pierwsza, "standard", teta_dostrojony)] + [
            (1, k / n, (1.0 - k / n) * teta_dostrojony) for k in range(1, n + 1)]

    konfiguracje.append(("rampa_alfa", rampa_alfa))
    # PLAN.md #11: prog przesuwny BCM, start od teta_dostrojony, bez strojenia.
    konfiguracje.append(("prog_bcm", lambda e: [(e, "standard_bcm", teta_dostrojony)]))
    # PLAN.md #16: faza standardowa z progiem BCM, druga polowa hamowanie teta=0.
    konfiguracje.append(("curriculum_bcm", lambda e: [(e // 2, "standard_bcm", teta_dostrojony),
                                                       (e - e // 2, "hamowanie", None)]))
    # PLAN.md #18: faza hamowania z przesuwnym progiem BCM (start od 0).
    konfiguracje.append(("curriculum_ham_bcm", lambda e: [(e // 2, "standard", teta_dostrojony),
                                                           (e - e // 2, "hamowanie_bcm", None)]))
    if progi_warstw is not None:
        # PLAN.md #17: standard ze STALYMI progami per warstwa (koncowe srednie BCM z E8).
        konfiguracje.append(("prog_staly_warstwy", lambda e: [(e, "standard", list(progi_warstw))]))

    if harmonogram_progow is not None:
        # PLAN.md #19 krok 2: sztywny replay sredniej trajektorii BCM
        def replay_progow(epoki_total):
            assert len(harmonogram_progow) >= epoki_total, (
                f"harmonogram_progow ma {len(harmonogram_progow)} epok, a zadano {epoki_total}")
            return [(1, "standard", list(harmonogram_progow[ep])) for ep in range(epoki_total)]
        konfiguracje.append(("prog_harmonogram", replay_progow))

    def curriculum_przelacz(e_przel):
        # PLAN.md #14: curriculum z przelaczeniem po epoce e_przel.
        return lambda e: [(e_przel, "standard", teta_dostrojony), (e - e_przel, "hamowanie", None)]

    def curriculum_ham_do_std_vicreg(w):
        # PLAN.md #8 (E4): odwrocone curriculum z kara wariancji na warstwie 0
        # w fazie hamowania. Waga jedzie jako atrybut harmonogramu, bo konfiguracje
        # sa parami (etykieta, harmonogram_fn) -- patrz waga_vicreg().
        def fn(epoki_total):
            pierwsza = epoki_total // 2
            return [(pierwsza, "hamowanie_vicreg", None), (epoki_total - pierwsza, "standard", teta_dostrojony)]
        fn.vicreg_waga = float(w)
        return fn

    for w in wagi_vicreg:
        konfiguracje.append((f"curriculum_ham_do_std_vicreg_w{w}", curriculum_ham_do_std_vicreg(w)))

    # PLAN.md E4 (decyzja 2026-09-29): pelny VICReg = wariancja + kowariancja na warstwie 0.
    for w in wagi_vicreg_pelny:
        fn = curriculum_ham_do_std_vicreg(w)
        fn.vicreg_kowariancja = True
        konfiguracje.append((f"curriculum_ham_do_std_vicregpelny_w{w}", fn))

    # PLAN.md #20: faza 1 z progiem BCM, faza 2 hamowanie teta=0 z LR x0.5.
    konfiguracje.append(("curriculum_bcm_lr_x0.5", lambda e: [(e // 2, "standard_bcm", teta_dostrojony, LR_FF),
                                                              (e - e // 2, "hamowanie", None, LR_FF * 0.5)]))

    for e_przel in punkty_przelaczenia:
        konfiguracje.append((f"curriculum_przelacz_e{e_przel}", curriculum_przelacz(e_przel)))

    # PLAN.md #13/#21: hamowanie lateralne (subtraktywne beta oraz dzielace gamma);
    # parametry nie mieszcza sie w segmencie harmonogramu,
    # ustawia je uruchom_zadanie_fashion_mnist.
    for beta in bety_lateralne:
        konfiguracje.append((etykieta_lateralne(beta), lambda e: [(e, "lateralne", teta_dostrojony)]))
    for gamma in gammy_lateralne:
        konfiguracje.append((etykieta_lateralne_dzielace(gamma), lambda e: [(e, "lateralne_dzielace", teta_dostrojony)]))

    if liczba_warstw_ukrytych is not None:
        def hamowanie_pierwsze_k(k):
            warianty = ["hamowanie"] * k + ["standard"] * (liczba_warstw_ukrytych - k)
            tety = [None] * k + [teta_dostrojony] * (liczba_warstw_ukrytych - k)
            return lambda e: [(e, warianty, tety)]

        # k=0 to standard_dostrojony (juz na liscie), k=liczba_warstw_ukrytych
        # to czysty hamowanie (juz na liscie) -- testujemy tylko wartosci
        # POSREDNIE, zeby zobaczyc trend "ile pierwszych warstw wystarczy".
        # k=4 dolozone 2026-09-23 -- gestszy sweep wokol znalezionego
        # maksimum K=3 przy glebokosci 10 (README.md, "Podzial przestrzenny"),
        # zeby sprawdzic czy to maksimum jest realne czy przypadek 5 ziaren.
        for k in sorted(set(k for k in (1, 2, 3, 4, 5) if 0 < k < liczba_warstw_ukrytych)):
            konfiguracje.append((f"hamowanie_pierwsze_{k}", hamowanie_pierwsze_k(k)))

        def k_curriculum(k):
            # PLAN.md #3 -- polaczenie obu wymiarow: K warstw W PRZESTRZENI
            # (jak hamowanie_pierwsze_k) + TYLKO DRUGA POLOWA treningu W CZASIE
            # (jak curriculum_std_do_ham). Pierwsza polowa epok: WSZYSTKIE
            # warstwy standard (buduj reprezentacje). Druga polowa: tylko
            # pierwsze K warstw przelacza na hamowanie, reszta zostaje standard.
            def fn(epoki_total):
                pierwsza = epoki_total // 2
                druga = epoki_total - pierwsza
                warianty_druga = ["hamowanie"] * k + ["standard"] * (liczba_warstw_ukrytych - k)
                tety_druga = [None] * k + [teta_dostrojony] * (liczba_warstw_ukrytych - k)
                return [(pierwsza, "standard", teta_dostrojony), (druga, warianty_druga, tety_druga)]
            return fn

        for k in (1, 2):
            if k < liczba_warstw_ukrytych:
                konfiguracje.append((f"k{k}_curriculum", k_curriculum(k)))

        def gradient_hamowania(ksztalt: str):
            alfy = gradient_alfa(ksztalt, liczba_warstw_ukrytych)
            # teta interpoluje LINIOWO wzgledem alfa niezaleznie od ksztaltu
            # zaniku alfa -- izoluje "ksztalt zaniku" jako jedyna testowana
            # zmienna, zamiast dokladac druga, nieliniowa decyzje projektowa
            # naraz. alfa=1 (czysty hamowanie) -> teta=0; alfa=0 (czysty
            # standard) -> teta=teta_dostrojony.
            tety = [(1.0 - a) * teta_dostrojony for a in alfy]
            return lambda e: [(e, alfy, tety)]

        # Trzy ksztalty do porownania, patrz docstring gradient_alfa():
        # "liniowa" -- rownomierny spadek, punkt odniesienia
        # "wykladnicza" -- silne TYLKO najblizej wejscia, dlugi ogon bliski zeru
        # "pierwiastkowa" -- zostaje silne przez wiekszosc warstw, spada
        #   dopiero na samym koncu (przeciwienstwo wykladniczej)
        for ksztalt in ("liniowa", "wykladnicza", "pierwiastkowa"):
            konfiguracje.append((f"gradient_{ksztalt}", gradient_hamowania(ksztalt)))

    # PLAN.md #15: jak curriculum_std_do_ham, ale kara w fazie hamowania
    # przetasowana miedzy przykladami partii. Na koncu listy, bo generator
    # partii jest wspolny dla konfiguracji -- nie zmienia kolejnosci pozostalym.
    konfiguracje.append(("curriculum_tasowane", lambda e: [(e // 2, "standard", teta_dostrojony),
                                                            (e - e // 2, "hamowanie_tasowane", None)]))

    if warianty_dozwolone is not None:
        konfiguracje = [(etykieta, fn) for etykieta, fn in konfiguracje
                         if etykieta in warianty_dozwolone]

    return tuple(konfiguracje)


def waga_vicreg(harmonogram_fn) -> float:
    """Waga kary wariancji dla konfiguracji (0.0 dla wszystkich poza *_vicreg_w*)."""
    return getattr(harmonogram_fn, "vicreg_waga", 0.0)


def kowariancja_vicreg(harmonogram_fn) -> bool:
    """Czy konfiguracja uzywa pelnego VICReg (czlon kowariancji) -- *_vicregpelny_w*."""
    return getattr(harmonogram_fn, "vicreg_kowariancja", False)


def dokladnosc_ff(siec: SiecFF, X: torch.Tensor, y: torch.Tensor, liczba_klas: int) -> float:
    predykcje = siec.przewiduj(X, liczba_klas, zakoduj_etykiete)
    return float((predykcje == y).float().mean().item())


def trenuj_ff(siec: SiecFF, X_train: torch.Tensor, y_train: torch.Tensor,
              epoki: int, rozmiar_partii: int, generator: np.random.Generator):
    """
    Petla treningowa FF: kazda partia dostaje pozytywne przejscie (obraz
    + prawdziwa etykieta) i negatywne (ten sam obraz + losowa zla
    etykieta), oba licza sie w jednym kroku przez wszystkie warstwy.
    Zwraca straty z OSTATNIEGO kroku, tylko jako wglad w zbieznosc --
    separacja dobroci do raportowania liczy sie OSOBNO, po treningu, na
    calym zbiorze testowym (patrz separacja_diagnostyczna w siec_ff.py),
    bo separacja z pojedynczej mini-partii jest zbyt zaszumiona.
    """
    n = X_train.shape[0]
    straty_ostatnie = None
    for _ in range(epoki):
        kolejnosc = generator.permutation(n)
        for start in range(0, n, rozmiar_partii):
            idx = kolejnosc[start:start + rozmiar_partii]
            idx_t = torch.from_numpy(idx).to(X_train.device)
            xb, yb = X_train[idx_t], y_train[idx_t]

            x_pos = zakoduj_etykiete(xb, yb, LICZBA_KLAS)
            y_zle = losowa_zla_etykieta(yb, LICZBA_KLAS, generator)
            x_neg = zakoduj_etykiete(xb, y_zle, LICZBA_KLAS)

            straty_ostatnie, _ = siec.krok_treningowy(x_pos, x_neg)

    return straty_ostatnie


def trenuj_ff_ze_sledzeniem(siec: SiecFF, X_train: torch.Tensor, y_train: torch.Tensor,
                             X_test: torch.Tensor, y_test: torch.Tensor,
                             epoki: int, rozmiar_partii: int, generator: np.random.Generator):
    """
    Jak trenuj_ff, ale mierzy dokladnosc na zbiorze testowym PO KAZDEJ
    epoce -- pytanie Slawka (2026-09-07): czy silniejszy sygnal wczesnie
    (hamowanie_pierwsze_K) przyspiesza uczenie, nie tylko podnosi wynik
    koncowy. Uzywana TYLKO dla konfiguracji o STALEJ formule przez caly
    trening (nie curriculum -- tam formula sama sie zmienia w czasie, co
    zaklocaloby pomiar "jak szybko dochodzi do danego poziomu").

    Zwraca liste (epoka, dokladnosc); epoka=0 to dokladnosc PRZED
    treningiem (losowa siec, punkt odniesienia -- powinna byc bliska
    1/LICZBA_KLAS).
    """
    n = X_train.shape[0]
    krzywa = [(0, dokladnosc_ff(siec, X_test, y_test, LICZBA_KLAS))]
    for epoka in range(1, epoki + 1):
        kolejnosc = generator.permutation(n)
        for start in range(0, n, rozmiar_partii):
            idx = kolejnosc[start:start + rozmiar_partii]
            idx_t = torch.from_numpy(idx).to(X_train.device)
            xb, yb = X_train[idx_t], y_train[idx_t]

            x_pos = zakoduj_etykiete(xb, yb, LICZBA_KLAS)
            y_zle = losowa_zla_etykieta(yb, LICZBA_KLAS, generator)
            x_neg = zakoduj_etykiete(xb, y_zle, LICZBA_KLAS)

            siec.krok_treningowy(x_pos, x_neg)
        krzywa.append((epoka, dokladnosc_ff(siec, X_test, y_test, LICZBA_KLAS)))
    return krzywa


def trenuj_ff_z_harmonogramem(siec: SiecFF, X_train: torch.Tensor, y_train: torch.Tensor,
                               harmonogram, rozmiar_partii: int, generator: np.random.Generator,
                               po_epoce=None):
    """
    Trening w segmentach, kazdy z WLASNA formula dobroci -- pomysl
    Slawka: "cykle standardowego uczenia, a potem mojego". Przelacza
    formule miedzy segmentami przez siec.przelacz_formule(), ktore NIE
    resetuje wag ani stanu optymalizatorow Adam -- to jest istota testu:
    czy zmiana CELU w trakcie treningu na tych samych wagach dziala
    lepiej niz jeden cel od poczatku do konca.

    harmonogram: lista (epoki, formula, teta) -- kazdy element to jeden
    segment, wykonywany w podanej kolejnosci. teta=None uzywa wartosci
    domyslnej dla danej formuly (patrz prog_teta w siec_ff.py).
    """
    straty_ostatnie = None
    for segment in harmonogram:
        epoki, formula, teta = segment[:3]
        if len(segment) == 4:
            for optymalizator in siec.optymalizatory:
                for grupa in optymalizator.param_groups:
                    grupa["lr"] = segment[3]
        siec.przelacz_formule(formula, teta_override=teta)
        if po_epoce is None:
            straty_ostatnie = trenuj_ff(siec, X_train, y_train, epoki, rozmiar_partii, generator)
        else:
            # epoka po epoce: ta sama sekwencja losowan generatora co w jednym wywolaniu
            for _ in range(epoki):
                straty_ostatnie = trenuj_ff(siec, X_train, y_train, 1, rozmiar_partii, generator)
                po_epoce(siec)
    return straty_ostatnie


def zbuduj_siec_backprop(wymiar_wejscia, wymiar_wyjscia, liczba_warstw_ukrytych, szerokosc,
                          device=None):
    warstwy = []
    wymiar_biezacy = wymiar_wejscia
    for _ in range(liczba_warstw_ukrytych):
        warstwy += [nn.Linear(wymiar_biezacy, szerokosc), nn.ReLU()]
        wymiar_biezacy = szerokosc
    warstwy.append(nn.Linear(wymiar_biezacy, wymiar_wyjscia))
    model = nn.Sequential(*warstwy)
    return model.to(device) if device is not None else model


def trenuj_backprop(model, X_train, y_train, epoki, rozmiar_partii, lr=1e-3):
    zbior = TensorDataset(X_train, y_train)
    loader = DataLoader(zbior, batch_size=rozmiar_partii, shuffle=True)
    optymalizator = torch.optim.Adam(model.parameters(), lr=lr)
    strata_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epoki):
        for xb, yb in loader:
            optymalizator.zero_grad()
            strata = strata_fn(model(xb), yb)
            strata.backward()
            optymalizator.step()
    return model


def uruchom_zadanie_fashion_mnist(katalog_wyjsciowy: Path, katalog_danych: Path,
                                   ziarna=(0,), n_train=3000, n_test=1000,
                                   epoki_ff=8, epoki_backprop=8, rozmiar_partii=64,
                                   szybki_test=False,
                                   szerokosc=SZEROKOSC_DOMYSLNA,
                                   liczba_warstw_ukrytych=LICZBA_WARSTW_UKRYTYCH_DOMYSLNA,
                                   device=None, nazwa_zbioru="fashion_mnist",
                                   warianty_dozwolone=None, pomin_backprop_hamowanie=False,
                                   teta_dostrojony_override=None, lr_backprop=1e-3,
                                   teta_per_wariant=None, wspolczynniki_lr=(), n_walidacja_krzywe=0,
                                   krok_impulsu=None, punkty_przelaczenia=(), czysty_test=False,
                                   progi_warstw=None, loguj_progi=False, dawki_hamowania=(),
                                   bety_lateralne=(), gammy_lateralne=(),
                                   wagi_vicreg=(), wagi_vicreg_pelny=(),
                                   harmonogram_progow=None):
    if szybki_test:
        ziarna = tuple(ziarna[:1])
        n_train, n_test, epoki_ff, epoki_backprop = 200, 100, 1, 1

    katalog_wyjsciowy.mkdir(parents=True, exist_ok=True)

    # device=None -> autodetekcja: GPU jesli dostepne (Kaggle), inaczej CPU
    # (maszyna deweloperska Slawka, i5-3450, bez CUDA -- kod ma dzialac
    # identycznie tam, tylko wolniej). Dane wczytujemy i podprobkowujemy
    # na CPU (tansze, numpy), na docelowe urzadzenie przenosimy JEDEN raz,
    # PO podprobkowaniu -- przenoszenie calych 60000 obrazkow byloby
    # marnotrawstwem, gdy trenujemy na podzbiorze.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"FF na {nazwa_zbioru} -- urzadzenie={device} szerokosc={szerokosc} "
          f"warstwy_ukryte={liczba_warstw_ukrytych} n_train={n_train} n_test={n_test} "
          f"epoki_ff={epoki_ff} epoki_backprop={epoki_backprop}")

    X_all_train, y_all_train, X_all_test, y_all_test = wczytaj_zbiory(katalog_danych, nazwa_zbioru)
    konfiguracje = konfiguracje_ff(szerokosc, liczba_warstw_ukrytych, warianty_dozwolone,
                                    teta_dostrojony_override, teta_per_wariant,
                                    wspolczynniki_lr=wspolczynniki_lr, krok_impulsu=krok_impulsu,
                                    punkty_przelaczenia=tuple(punkty_przelaczenia),
                                    progi_warstw=tuple(progi_warstw) if progi_warstw else None,
                                    dawki_hamowania=tuple(dawki_hamowania),
                                    bety_lateralne=tuple(bety_lateralne),
                                    gammy_lateralne=tuple(gammy_lateralne),
                                    wagi_vicreg=tuple(wagi_vicreg),
                                    wagi_vicreg_pelny=tuple(wagi_vicreg_pelny),
                                    harmonogram_progow=harmonogram_progow)
    beta_dla_etykiety = {etykieta_lateralne(b): b for b in bety_lateralne}
    gamma_dla_etykiety = {etykieta_lateralne_dzielace(g): g for g in gammy_lateralne}

    if czysty_test:
        # PLAN.md, swieze potwierdzenie FMNIST: 10000 przykladow testowych wydzielonych
        # na stale ze zbioru TRENINGOWEGO (ziarno podzialu 2026), rozlacznych z
        # walidacjami obu dawnych sweepow; trening losowany z reszty. Oryginalny
        # zbior testowy nieuzywany.
        wykluczone = set()
        for z_sw in (100, 101):
            perm = np.random.default_rng(z_sw).permutation(X_all_train.shape[0])
            wykluczone |= set(perm[3000:4000].tolist()) | set(perm[20000:23000].tolist())
        kandydaci = np.array(sorted(set(range(X_all_train.shape[0])) - wykluczone))
        idx_test = np.random.default_rng(2026).choice(kandydaci, size=10000, replace=False)
        maska = np.ones(X_all_train.shape[0], dtype=bool)
        maska[idx_test] = False
        idx_test_t, idx_pula_t = torch.from_numpy(np.sort(idx_test)), torch.from_numpy(np.nonzero(maska)[0])
        X_all_test, y_all_test = X_all_train[idx_test_t], y_all_train[idx_test_t]
        X_all_train, y_all_train = X_all_train[idx_pula_t], y_all_train[idx_pula_t]
        n_test = 10000
        print(f"czysty test: {len(idx_test)} z treningu, wykluczone walidacje sweepow: {len(wykluczone)}, "
              f"pula treningowa: {len(idx_pula_t)}")

    # Dopasowanie szerokosci backprop_hamowanie do liczby parametrow
    # zwyklego backpropu -- macierz g dodaje szerokosc^2 parametrow na
    # warstwe, wiec przy tej samej szerokosci siec hamujaca mialaby wiecej
    # parametrow. Liczy sie raz, nie zalezy od ziarna (patrz
    # dopasuj_szerokosci w ../hamowanie/aktywacje.py po ten sam wzorzec).
    model_bp_wzorcowy = zbuduj_siec_backprop(WYMIAR_WEJSCIA, WYMIAR_WYJSCIA,
                                              liczba_warstw_ukrytych, szerokosc)
    n_parametry_bp = policz_parametry(model_bp_wzorcowy)
    szerokosc_bp_ham, n_parametry_bp_ham, blad_dopasowania = dopasuj_szerokosc_hamujaca(
        WYMIAR_WEJSCIA, WYMIAR_WYJSCIA, liczba_warstw_ukrytych, n_parametry_bp)
    print(f"Backprop -- dopasowanie parametrow: plain szerokosc={szerokosc} "
          f"parametry={n_parametry_bp}; hamujaca szerokosc={szerokosc_bp_ham} "
          f"parametry={n_parametry_bp_ham} blad={blad_dopasowania:.2%}")

    wyniki_wiersze = []
    for ziarno in ziarna:
        generator = np.random.default_rng(ziarno)
        X_val = None
        if n_walidacja_krzywe > 0:
            # PLAN.md #10: walidacja do krzywych wydzielona z puli treningowej
            # PRZED losowaniem treningu; test tylko na wynik koncowy.
            perm = generator.permutation(X_all_train.shape[0])
            idx_val, idx_reszta = torch.from_numpy(perm[:n_walidacja_krzywe]), torch.from_numpy(perm[n_walidacja_krzywe:])
            X_val, y_val = X_all_train[idx_val].to(device), y_all_train[idx_val].to(device)
            X_train, y_train = podprobka(X_all_train[idx_reszta], y_all_train[idx_reszta], n_train, generator)
        else:
            X_train, y_train = podprobka(X_all_train, y_all_train, n_train, generator)
        X_test, y_test = podprobka(X_all_test, y_all_test, n_test, generator)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        for etykieta, harmonogram_fn in konfiguracje:
            ustaw_ziarno(ziarno)
            harmonogram = harmonogram_fn(epoki_ff)
            pierwsza_formula, pierwsza_teta = harmonogram[0][1], harmonogram[0][2]
            siec = SiecFF(WYMIAR_WEJSCIA, liczba_warstw_ukrytych, szerokosc, pierwsza_formula,
                          lr=LR_FF, teta_override=pierwsza_teta, device=device)
            siec.beta_lateralne = beta_dla_etykiety.get(etykieta)
            siec.gamma_lateralne = gamma_dla_etykiety.get(etykieta)
            siec.vicreg_waga = waga_vicreg(harmonogram_fn)
            siec.vicreg_kowariancja = kowariancja_vicreg(harmonogram_fn)
            krzywa, progi_epok = [], []

            def po_epoce(s, k=krzywa, pe=progi_epok):
                if X_val is not None:
                    k.append(dokladnosc_ff(s, X_val, y_val, LICZBA_KLAS))
                if loguj_progi:
                    pe.append(list(s.tety))  # PLAN.md #19: trajektoria progow per warstwa
            straty = trenuj_ff_z_harmonogramem(
                siec, X_train, y_train, harmonogram, rozmiar_partii, generator,
                po_epoce=po_epoce if (X_val is not None or loguj_progi) else None)
            for e, progi in enumerate(progi_epok, start=1):
                for i, t in enumerate(progi):
                    wyniki_wiersze.append({"ziarno": ziarno, "wariant": f"ff_{etykieta}",
                                           "pomiar": "teta_warstwy", "warstwa": i,
                                           "wartosc": t, "dodatkowe": json.dumps({"epoka": e})})
            if X_val is not None:
                for e, d in enumerate(krzywa, start=1):
                    wyniki_wiersze.append({"ziarno": ziarno, "wariant": f"ff_{etykieta}",
                                           "pomiar": "dokladnosc_walidacja", "warstwa": -1,
                                           "wartosc": d, "dodatkowe": json.dumps({"epoka": e})})
            dokladnosc = dokladnosc_ff(siec, X_test, y_test, LICZBA_KLAS)

            # separacja koncowa -- na CALYM zbiorze testowym, nie na
            # ostatniej mini-partii treningowej (patrz docstring trenuj_ff)
            y_zle_test = losowa_zla_etykieta(y_test, LICZBA_KLAS, generator)
            x_pos_test = zakoduj_etykiete(X_test, y_test, LICZBA_KLAS)
            x_neg_test = zakoduj_etykiete(X_test, y_zle_test, LICZBA_KLAS)
            separacje = siec.separacja_diagnostyczna(x_pos_test, x_neg_test)
            if etykieta in beta_dla_etykiety or etykieta in gamma_dla_etykiety:
                # PLAN.md #13/#21: badanie odsetka martwych neuronow w warstwie 0.
                # Martwy = z'<=0 na wszystkich przykladach testu.
                war = "lateralne" if etykieta in beta_dla_etykiety else "lateralne_dzielace"
                with torch.no_grad():
                    for typ, x in (("pozytywna", x_pos_test), ("negatywna", x_neg_test)):
                        z = siec.warstwy[0](x)
                        z_prim = siec.z_efektywne(z, war)
                        wyniki_wiersze.append({
                            "ziarno": ziarno, "wariant": f"ff_{etykieta}", "pomiar": "martwe_warstwa0",
                            "warstwa": 0, "wartosc": float((z_prim <= 0).all(dim=0).float().mean().item()),
                            "dodatkowe": json.dumps({"probka": typ, "beta": siec.beta_lateralne,
                                                      "gamma": siec.gamma_lateralne,
                                                      "martwe_przed_odjeciem": float(
                                                          (z <= 0).all(dim=0).float().mean().item())})})

            print(f"  ziarno={ziarno} wariant=ff_{etykieta} ({siec.opis_formuly()}): "
                  f"dokladnosc={dokladnosc:.3f} "
                  f"separacja_dobroci_per_warstwa={['%.2f' % s for s in separacje]}")

            wyniki_wiersze.append({
                "ziarno": ziarno, "wariant": f"ff_{etykieta}", "pomiar": "dokladnosc_test",
                "warstwa": -1, "wartosc": dokladnosc, "dodatkowe": "",
            })
            for i, sep in enumerate(separacje):
                wyniki_wiersze.append({
                    "ziarno": ziarno, "wariant": f"ff_{etykieta}", "pomiar": "separacja_dobroci",
                    "warstwa": i, "wartosc": sep,
                    "dodatkowe": json.dumps({"strata_ostatnia_warstwa": straty[-1],
                                              "wariant_warstwy": siec.warianty[i], "teta": siec.tety[i]}),
                })

        # punkt odniesienia -- zwykla siec, zwykla propagacja wsteczna
        ustaw_ziarno(ziarno)
        model_bp = zbuduj_siec_backprop(WYMIAR_WEJSCIA, WYMIAR_WYJSCIA,
                                         liczba_warstw_ukrytych, szerokosc, device=device)
        trenuj_backprop(model_bp, X_train, y_train, epoki_backprop, rozmiar_partii, lr=lr_backprop)
        model_bp.eval()
        with torch.no_grad():
            dokladnosc_bp = float((model_bp(X_test).argmax(dim=1) == y_test).float().mean().item())
        print(f"  ziarno={ziarno} wariant=backprop: dokladnosc={dokladnosc_bp:.3f}")
        wyniki_wiersze.append({
            "ziarno": ziarno, "wariant": "backprop", "pomiar": "dokladnosc_test",
            "warstwa": -1, "wartosc": dokladnosc_bp, "dodatkowe": "",
        })

        # backprop z aktywacja hamujaca -- "zwykla metoda, ale z
        # uwzglednieniem bledu wynikajacego z ReLU" (pomysl Slawka):
        # ta sama propagacja wsteczna, ale ReLU(x) zastapione przez
        # ReLU(x) - g(ReLU(-x)) w kazdej warstwie ukrytej. Szerokosc
        # dopasowana do liczby parametrow zwyklego backpropu (patrz wyzej).
        # Pomijalna w kampanii potwierdzenia n=30 -- watek juz rozstrzygniety
        # przy n=5 (README archiwum), nie jest sercem tezy K=3.
        if not pomin_backprop_hamowanie:
            ustaw_ziarno(ziarno)
            model_bp_ham = zbuduj_siec_backprop_hamowanie(WYMIAR_WEJSCIA, WYMIAR_WYJSCIA,
                                                           liczba_warstw_ukrytych, szerokosc_bp_ham,
                                                           device=device)
            trenuj_backprop(model_bp_ham, X_train, y_train, epoki_backprop, rozmiar_partii)
            model_bp_ham.eval()
            with torch.no_grad():
                dokladnosc_bp_ham = float(
                    (model_bp_ham(X_test).argmax(dim=1) == y_test).float().mean().item())
            print(f"  ziarno={ziarno} wariant=backprop_hamowanie: dokladnosc={dokladnosc_bp_ham:.3f}")
            wyniki_wiersze.append({
                "ziarno": ziarno, "wariant": "backprop_hamowanie", "pomiar": "dokladnosc_test",
                "warstwa": -1, "wartosc": dokladnosc_bp_ham, "dodatkowe": "",
            })

    zapisz_wyniki_csv(wyniki_wiersze, katalog_wyjsciowy / "fashion_mnist_ff_wyniki.csv")

    podsumowanie_dokladnosci = podsumuj_po_ziarnach(
        [w for w in wyniki_wiersze if w["pomiar"] == "dokladnosc_test"],
        klucze_grupujace=["wariant"])
    with open(katalog_wyjsciowy / "fashion_mnist_ff_podsumowanie_dokladnosc.json", "w",
              encoding="utf-8") as f:
        json.dump(podsumowanie_dokladnosci, f, indent=2, ensure_ascii=False)

    podsumowanie_separacji = podsumuj_po_ziarnach(
        [w for w in wyniki_wiersze if w["pomiar"] == "separacja_dobroci"],
        klucze_grupujace=["wariant", "warstwa"])
    with open(katalog_wyjsciowy / "fashion_mnist_ff_podsumowanie_separacji.json", "w",
              encoding="utf-8") as f:
        json.dump(podsumowanie_separacji, f, indent=2, ensure_ascii=False)

    print("\nDokladnosc testowa, srednia +/- odchylenie po ziarnach:")
    for wpis in sorted(podsumowanie_dokladnosci, key=lambda w: w["wariant"]):
        print(f"  {wpis['wariant']}: {wpis['srednia']:.3f} +/- {wpis['odchylenie']:.3f} (n={wpis['n']})")

    print("\nSeparacja dobroci (pozytyw-negatyw) w ostatnim kroku, per warstwa -- "
          "PIERWSZA RZECZ DO SPRAWDZENIA, blisko zera albo ujemna = trywialne rozwiazanie:")
    for wpis in sorted(podsumowanie_separacji, key=lambda w: (w["wariant"], w["warstwa"])):
        print(f"  wariant={wpis['wariant']} warstwa={wpis['warstwa']}: "
              f"{wpis['srednia']:.2f} +/- {wpis['odchylenie']:.2f} (n={wpis['n']})")

    return wyniki_wiersze
