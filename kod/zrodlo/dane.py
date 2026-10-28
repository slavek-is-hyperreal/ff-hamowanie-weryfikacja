"""
Wczytywanie Fashion-MNIST i kodowanie etykiety w obrazie -- metoda
Hintona (2022) dla wersji nadzorowanej Forward-Forward: dziesiec
pierwszych pikseli obrazu (splaszczonego) zastapione kodowaniem
jeden-z-N etykiety, wartosc rowna maksimum znormalizowanych pikseli (1.0).
Siec musi wtedy nauczyc sie uzywac tego sygnalu, zeby dobroc rosla dla
prawdziwej etykiety i malala dla falszywej -- to jest zrodlo "negatywnych
danych" u Hintona: ten sam obraz, zla etykieta, nie osobno generowany
falszywy obraz.
"""

from pathlib import Path

import numpy as np
import torch

WYMIAR_WEJSCIA = 28 * 28
LICZBA_KLAS = 10


def wczytaj_zbiory(katalog_danych: Path, nazwa_zbioru: str = "fashion_mnist"):
    """
    nazwa_zbioru: "fashion_mnist" (domyslny, uzywany we wszystkich
    dotychczasowych testach) albo "mnist" (zwykle cyfry -- test
    generalizacji na inne dane, patrz README.md "Nastepne kroki").
    Oba maja identyczny ksztalt (28x28, 10 klas), wiec reszta kodu
    (WYMIAR_WEJSCIA, LICZBA_KLAS, kodowanie etykiety) dziala bez zmian.
    """
    from torchvision import datasets

    klasa_zbioru = {"fashion_mnist": datasets.FashionMNIST, "mnist": datasets.MNIST}[nazwa_zbioru]

    zbior_treningowy = klasa_zbioru(root=str(katalog_danych), train=True, download=True)
    zbior_testowy = klasa_zbioru(root=str(katalog_danych), train=False, download=True)

    X_train = zbior_treningowy.data.reshape(-1, WYMIAR_WEJSCIA).float() / 255.0
    y_train = zbior_treningowy.targets.long()
    X_test = zbior_testowy.data.reshape(-1, WYMIAR_WEJSCIA).float() / 255.0
    y_test = zbior_testowy.targets.long()
    return X_train, y_train, X_test, y_test


def podprobka(X_all: torch.Tensor, y_all: torch.Tensor, n: int, generator: np.random.Generator):
    n = min(n, len(X_all))
    idx = generator.choice(len(X_all), size=n, replace=False)
    idx_t = torch.from_numpy(idx)
    return X_all[idx_t], y_all[idx_t]


def zakoduj_etykiete(x: torch.Tensor, y: torch.Tensor, liczba_klas: int = LICZBA_KLAS,
                      wartosc: float = 1.0) -> torch.Tensor:
    """
    Zwraca kopie x z pierwszymi liczba_klas pikselami zastapionymi
    kodowaniem jeden-z-N etykiety y. Kopia, nie modyfikacja w miejscu --
    wywolujacy czesto potrzebuje tego samego x z rozna etykieta (pozytyw
    i negatyw z jednego obrazu).
    """
    x_mod = x.clone()
    x_mod[:, :liczba_klas] = 0.0
    x_mod[torch.arange(x.shape[0], device=x.device), y] = wartosc
    return x_mod


def losowa_zla_etykieta(y_prawdziwe: torch.Tensor, liczba_klas: int,
                         generator: np.random.Generator) -> torch.Tensor:
    """
    Losuje etykiete rozna od prawdziwej dla kazdego przykladu (przesuniecie
    o 1..liczba_klas-1 modulo liczba_klas -- gwarantuje roznosc bez
    odrzucania probek, prostsze i szybsze niz losowanie z odrzuceniem).
    """
    przesuniecie = generator.integers(1, liczba_klas, size=len(y_prawdziwe))
    # .cpu() przed .numpy() -- na GPU y_prawdziwe siedzi na karcie, .numpy()
    # dziala tylko na tensorach CPU. Wynik wraca na urzadzenie y_prawdziwe,
    # zeby dalsze porownania/indeksowanie nie mieszaly urzadzen.
    zle = (y_prawdziwe.cpu().numpy() + przesuniecie) % liczba_klas
    return torch.from_numpy(zle).long().to(y_prawdziwe.device)
