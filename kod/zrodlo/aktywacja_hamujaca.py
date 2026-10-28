"""
Aktywacja hamujaca do wariantu propagacji wstecznej, ktory "uwzglednia
blad wynikajacy z ReLU" -- pomysl Slawka polaczyc dwa watki: material
odciety przez ReLU (w `siec_ff.py` uzyty jako sygnal Forward-Forward)
uzyty tutaj jako TOR HAMUJACY w sieci uczonej ZWYKLA propagacja wsteczna,
zamiast zwyklego ReLU.

Kod skopiowany z ../hamowanie/aktywacje.py (WarstwaHamujaca) -- ten sam
mechanizm, ta sama parametryzacja, to samo uzasadnienie. Eksperymenty w
tym katalogu sa niezalezne od `hamowanie/` (patrz wspolne.py), wiec kod
jest duplikowany, nie importowany.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inicjalizuj_wagi_g(v: torch.Tensor, szerokosc: int) -> None:
    """
    Startowa skala wag toru hamujacego dobrana tak, by na poczatku
    treningu hamowanie mialo podobny rzad wielkosci co typowa
    inicjalizacja Kaiming dla ReLU (odchylenie std = sqrt(2/fan_in)).
    Patrz ../hamowanie/aktywacje.py po pelne uzasadnienie -- identyczna
    logika.
    """
    cel = math.sqrt(2.0 / szerokosc)
    v_start = math.log(cel)
    with torch.no_grad():
        v.normal_(mean=v_start, std=0.5)


class WarstwaHamujaca(nn.Module):
    """
    y = ReLU(x) - g(ReLU(-x)). g to PELNA warstwa liniowa bez wyrazu
    wolnego, wagi WYLACZNIE nieujemne przez parametryzacje w=softplus(v)
    (gradient plynie normalnie, nic nie trzeba rzutowac po kroku
    optymalizatora -- rzutowanie po fakcie psuje momentum Adama).
    """

    def __init__(self, szerokosc: int):
        super().__init__()
        self.szerokosc = szerokosc
        self.v = nn.Parameter(torch.empty(szerokosc, szerokosc))
        _inicjalizuj_wagi_g(self.v, szerokosc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strona_dodatnia = torch.relu(x)
        strona_ujemna = torch.relu(-x)
        g_wagi = F.softplus(self.v)
        return strona_dodatnia - strona_ujemna @ g_wagi.t()


def zbuduj_siec_backprop_hamowanie(wymiar_wejscia: int, wymiar_wyjscia: int,
                                    liczba_warstw_ukrytych: int, szerokosc: int,
                                    device=None) -> nn.Sequential:
    warstwy = []
    wymiar_biezacy = wymiar_wejscia
    for _ in range(liczba_warstw_ukrytych):
        warstwy.append(nn.Linear(wymiar_biezacy, szerokosc))
        warstwy.append(WarstwaHamujaca(szerokosc))
        wymiar_biezacy = szerokosc
    warstwy.append(nn.Linear(wymiar_biezacy, wymiar_wyjscia))
    model = nn.Sequential(*warstwy)
    return model.to(device) if device is not None else model


def policz_parametry(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dopasuj_szerokosc_hamujaca(wymiar_wejscia: int, wymiar_wyjscia: int,
                                liczba_warstw_ukrytych: int,
                                docelowa_liczba_parametrow: int,
                                zakres_szerokosci=(2, 400)):
    """
    Macierz g dodaje szerokosc^2 parametrow na warstwe -- siec hamujaca
    potrzebuje wezszych warstw niz plain ReLU backprop tej samej
    docelowej liczby parametrow. Przeszukanie liniowe, tak jak w
    ../hamowanie/aktywacje.py -- liczy sie raz na caly eksperyment, nie
    zalezy od ziarna losowego.
    """
    najlepsza_szerokosc, najlepszy_blad, najlepsza_liczba = None, math.inf, None
    for szerokosc in range(zakres_szerokosci[0], zakres_szerokosci[1] + 1):
        model = zbuduj_siec_backprop_hamowanie(wymiar_wejscia, wymiar_wyjscia,
                                                liczba_warstw_ukrytych, szerokosc)
        n = policz_parametry(model)
        blad = abs(n - docelowa_liczba_parametrow) / docelowa_liczba_parametrow
        if blad < najlepszy_blad:
            najlepszy_blad, najlepsza_szerokosc, najlepsza_liczba = blad, szerokosc, n
    return najlepsza_szerokosc, najlepsza_liczba, najlepszy_blad
