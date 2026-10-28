"""
Funkcje wspolne: ziarna losowosci, zapis CSV, podsumowania wynikow po
ziarnach. Kopia odpowiednika z ../hamowanie/wspolne.py -- kazdy
eksperyment w tym katalogu jest niezalezny, wiec nie importuje kodu
z innych eksperymentow.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def ustaw_ziarno(ziarno: int) -> None:
    random.seed(ziarno)
    np.random.seed(ziarno)
    torch.manual_seed(ziarno)


def zapisz_wyniki_csv(wiersze, sciezka: Path) -> None:
    if not wiersze:
        return
    pola = list(wiersze[0].keys())
    with open(sciezka, "w", newline="", encoding="utf-8") as f:
        pisarz = csv.DictWriter(f, fieldnames=pola)
        pisarz.writeheader()
        pisarz.writerows(wiersze)


def podsumuj_po_ziarnach(wiersze, klucze_grupujace, klucz_wartosci="wartosc"):
    grupy = defaultdict(list)
    for w in wiersze:
        klucz = tuple(w[k] for k in klucze_grupujace)
        grupy[klucz].append(w[klucz_wartosci])

    podsumowanie = []
    for klucz, wartosci in grupy.items():
        tablica = np.array(wartosci, dtype=float)
        wpis = dict(zip(klucze_grupujace, klucz))
        wpis["srednia"] = float(tablica.mean())
        wpis["odchylenie"] = float(tablica.std(ddof=0)) if len(tablica) > 1 else 0.0
        wpis["n"] = len(tablica)
        podsumowanie.append(wpis)
    return podsumowanie
