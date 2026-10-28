"""
Siec Forward-Forward (Hinton, 2022) z dwoma wariantami dobroci warstwy.

Rdzen FF: kazda warstwa uczy sie LOKALNIE, wlasnym optymalizatorem, na
podstawie "dobroci" (goodness) wlasnego wyjscia -- bez propagacji
wstecznej miedzy warstwami. Kazdy krok dostaje dwa przejscia tego samego
obrazu: pozytywne (z prawdziwa etykieta wklejona w piksele) i negatywne
(z etykieta losowo zla). Warstwa ma podniesc dobroc dla pozytywnego i
obnizyc dla negatywnego -- to jest jedyny sygnal uczacy, gradient nie
plynie do poprzednich warstw.

Dwa warianty dobroci -- to jest wlasciwy przedmiot tego eksperymentu:

- "standard" (Hinton): dobroc = suma kwadratow aktywacji po ReLU,
  prog = szerokosc warstwy (przyblizenie u Hintona: prog rzedu liczby
  neuronow).
- "hamowanie" (pomysl Slawka): dobroc = suma kwadratow strony DODATNIEJ
  minus suma kwadratow strony UJEMNEJ (ta, ktora zwykle ReLU wyrzuca).
  Prog = 0, bo to juz jest roznica, nie surowa suma. Hipoteza: material
  odciety przez ReLU niesie informacje o tym, jak bardzo dany przyklad
  NIE pasuje -- wlaczenie go do dobroci wprost, zamiast wyrzucania,
  powinno dawac ostrzejszy sygnal separacji niz standard.

Metoda generowania negatywow (zla etykieta w tym samym obrazie) jest
IDENTYCZNA dla obu wariantow -- to jest kontrolowana zmienna. Jedyna
roznica to wzor na dobroc. Bez tego porownanie nic by nie znaczylo.

Normalizacja dlugosci wektora aktywacji miedzy warstwami (normalizuj())
jest zabezpieczeniem Hintona przed trywialnym rozwiazaniem: bez niej
warstwa moglaby "oszukiwac" przekazujac dalej sama dlugosc wektora
(ktora latwo zrobic duza w obu przejsciach naraz), zamiast uczyc sie
czegos o roznicy miedzy pozytywem a negatywem. Uzyta dla obu wariantow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

WARIANTY_FF = ("standard", "hamowanie", "standard_bcm", "hamowanie_bcm", "lateralne", "lateralne_dzielace",
               "hamowanie_tasowane", "hamowanie_vicreg")
# Formuly liczace dobroc, prog i predykcje dokladnie jak "hamowanie". hamowanie_tasowane tasuje
# kare tylko w treningu (dobroc_tasowana); hamowanie_vicreg wlacza kare wariancji na warstwie 0.
WARIANTY_HAMOWANIA = ("hamowanie", "hamowanie_bcm", "hamowanie_vicreg", "hamowanie_tasowane")


def normalizuj(a: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    norma = a.norm(p=2, dim=1, keepdim=True)
    return a / (norma + eps)


def odejmij_lateralne(z: torch.Tensor, beta: float) -> torch.Tensor:
    """PLAN.md #13: z'_i = z_i - beta * srednia_{j != i} ReLU(z_j), per przyklad,
    po neuronach tej samej warstwy."""
    p = torch.relu(z)
    srednia_sasiadow = (p.sum(dim=1, keepdim=True) - p) / (z.shape[1] - 1)
    return z - beta * srednia_sasiadow


def dzielace_lateralne(z: torch.Tensor, gamma: float) -> torch.Tensor:
    """PLAN.md #21: Biologiczna normalizacja dzielaca (Carandini & Heeger 2012):
    z'_i = ReLU(z_i) / (1 + gamma * srednia_{j != i} ReLU(z_j)), per przyklad,
    po neuronach tej samej warstwy."""
    p = torch.relu(z)
    srednia_sasiadow = (p.sum(dim=1, keepdim=True) - p) / (z.shape[1] - 1)
    return p / (1.0 + gamma * srednia_sasiadow)


def waliduj_wariant(w) -> None:
    """String z WARIANTY_FF albo liczba (alfa mieszania w [0, 1], patrz
    dobroc()) -- poza tym blad jawny zamiast cichego zlego zachowania."""
    if isinstance(w, (int, float)):
        assert 0.0 <= float(w) <= 1.0, f"alfa mieszania musi byc w [0,1], dostal {w}"
        return
    assert w in WARIANTY_FF, f"nieznany wariant: {w}"


def dobroc(z: torch.Tensor, wariant) -> torch.Tensor:
    """
    wariant to albo string ("standard"/"hamowanie" -- dwie skrajnosci,
    dotychczasowe uzycie) albo liczba w [0, 1] -- pomysl Slawka: standard
    i hamowanie to nie dwa odrebne wzory, tylko dwa konce JEDNEGO
    pokretla. Suma(ReLU(z)^2) - alfa*suma(ReLU(-z)^2): alfa=0 daje
    dokladnie standard, alfa=1 daje dokladnie hamowanie, wartosci
    posrednie mieszaja formuly w sposob ciagly. Pozwala to na dowolna
    funkcje zaniku alfa(warstwa) zamiast skokowego "pierwsze K warstw
    w pelni, reszta wcale" (patrz konfiguracje_ff w zadanie_fashion_mnist.py).
    """
    p = torch.relu(z)
    # "lateralne" i "lateralne_dzielace": z musi byc JUZ po modyfikacji sasiadow (SiecFF.z_efektywne)
    if wariant in ("standard", "standard_bcm", "lateralne", "lateralne_dzielace"):
        return (p ** 2).sum(dim=1)
    elif wariant in WARIANTY_HAMOWANIA:
        n = torch.relu(-z)
        return (p ** 2).sum(dim=1) - (n ** 2).sum(dim=1)
    elif isinstance(wariant, (int, float)):
        alfa = float(wariant)
        n = torch.relu(-z)
        return (p ** 2).sum(dim=1) - alfa * (n ** 2).sum(dim=1)
    raise ValueError(f"nieznany wariant FF: {wariant}")


def dobroc_tasowana(z: torch.Tensor) -> torch.Tensor:
    """
    PLAN.md #15, kontrola z przetasowaniem kary: suma(ReLU(z)^2) biezacego
    przykladu minus suma(ReLU(-z)^2) INNEGO przykladu tej samej partii.
    Kara ma ten sam rozklad i skale co w "hamowanie", ale nie niesie
    informacji o biezacym przykladzie.
    """
    p = torch.relu(z)
    n = torch.relu(-z)
    # zwykly randperm z RNG torcha (deterministyczny po manual_seed); punkt
    # staly (przyklad dostaje wlasna kare) z p-stwem ok. 1/n -- akceptowalne
    perm = torch.randperm(z.shape[0], device=z.device)
    return (p ** 2).sum(dim=1) - (n ** 2).sum(dim=1)[perm]


def prog_teta(wariant, szerokosc: int) -> float:
    # Hinton: prog rzedu liczby neuronow warstwy (kazda jednostka wnosi
    # ok. 1 do sumy kwadratow przy typowej aktywacji). Dla wariantu
    # "hamowanie" dobroc jest juz roznica dwoch takich sum, wiec naturalny
    # punkt neutralny to zero, nie szerokosc.
    if wariant == "standard":
        return float(szerokosc)
    elif wariant in WARIANTY_HAMOWANIA:
        return 0.0
    elif wariant in ("lateralne", "lateralne_dzielace"):
        raise ValueError(f"wariant {wariant} wymaga jawnego teta_override (PLAN.md #13/#21: teta=8)")
    # wariant mieszany (alfa liczbowe, 0..1) NIE MA jednego "naturalnego"
    # progu -- domyslny prog standardu (=szerokosc) i hamowanie (=0) to
    # dwie rozne skrajnosci, a interpolacja miedzy nimi zalezy od tego,
    # jak dobroc skaluje sie z alfa (nieliniowo -- alfa mnozy tylko
    # skladnik ujemny). Cichy blad tutaj (np. brak teta_override) dalby
    # kazdej warstwie posredniej prog=0, poprawny tylko dla alfa=1 -- wiec
    # ZAMIAST zgadywac, wymuszamy jawny teta_override dla takich warstw.
    raise ValueError(
        f"prog_teta wymaga jawnego teta_override dla wariantu mieszanego (alfa={wariant}) -- "
        "nie ma bezpiecznej wartosci domyslnej, patrz komentarz w kodzie")


def kara_wariancji(a: torch.Tensor) -> torch.Tensor:
    """Czlon wariancji VICReg (Bardes, Ponce, LeCun 2022): srednia po jednostkach
    z relu(1 - std), std liczone po partii."""
    std = torch.sqrt(a.var(dim=0) + 1e-4)
    return torch.relu(1.0 - std).mean()


def kara_kowariancji(a: torch.Tensor) -> torch.Tensor:
    """Czlon kowariancji VICReg: suma kwadratow pozadiagonalnych elementow kowariancji
    po partii, podzielona przez liczbe jednostek (PLAN.md, E4 pelny VICReg)."""
    n, d = a.shape
    ac = a - a.mean(dim=0, keepdim=True)
    kow = (ac.T @ ac) / max(n - 1, 1)
    pozadiagonalne = kow - torch.diag(torch.diag(kow))
    return (pozadiagonalne ** 2).sum() / d


class WarstwaFF(nn.Module):
    def __init__(self, wymiar_wejscia: int, szerokosc: int):
        super().__init__()
        self.liniowa = nn.Linear(wymiar_wejscia, szerokosc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.liniowa(x)  # z, PRZED aktywacja -- dobroc() liczy relu sama


class SiecFF:
    """
    Lista warstw FF, kazda z wlasnym optymalizatorem Adam. Trening
    przechodzi warstwa po warstwie w JEDNYM kroku (ta sama para partii
    pozytyw/negatyw), odlaczajac wyjscie od grafu miedzy warstwami --
    to jest to, co odroznia FF od propagacji wstecznej: warstwa 2 nigdy
    nie dostaje gradientu z warstwy 3.
    """

    def __init__(self, wymiar_wejscia: int, liczba_warstw_ukrytych: int,
                 szerokosc: int, wariant, lr: float = 1e-3, teta_override=None,
                 device=None):
        """
        wariant/teta_override moga byc POJEDYNCZA wartoscia (ta sama
        formula dla kazdej warstwy -- dotychczasowe uzycie) albo LISTA
        dlugosci liczba_warstw_ukrytych (rozna formula per warstwa --
        pomysl Slawka "hamowanie powinno zanikac z numerem warstwy",
        patrz konfiguracja hamowanie_zanikajace w zadanie_fashion_mnist.py).
        Pojedyncza wartosc jest wewnetrznie rozszerzana na liste tej samej
        wartosci powtorzonej -- reszta kodu zawsze operuje na listach.

        device -- gdzie trzymac wagi warstw (CPU domyslnie, "cuda" na
        Kaggle z GPU). Wejscia (x_pos/x_neg) MUSZA byc na tym samym
        urzadzeniu -- ta klasa tego nie sprawdza, PyTorch zglosi blad
        niezgodnosci sam, jesli sie pomylimy.
        """
        self.liczba_warstw_ukrytych = liczba_warstw_ukrytych
        self.szerokosc = szerokosc
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.warianty = self._rozszerz_na_liste(wariant, liczba_warstw_ukrytych)
        for w in self.warianty:
            waliduj_wariant(w)
        tety_zadane = self._rozszerz_na_liste(teta_override, liczba_warstw_ukrytych)
        # teta_override -- do kalibracji progu ff_standard przed dlugim
        # przebiegiem (patrz PRZEKAZANIE.md); None per warstwa uzywa
        # wartosci domyslnej z prog_teta() dla formuly TEJ warstwy
        self.tety = [t if t is not None else prog_teta(w, szerokosc)
                     for w, t in zip(self.warianty, tety_zadane)]
        # PLAN.md #13: sila hamowania lateralnego; ustawiana po konstrukcji,
        # wymagana tylko przez warstwy z wariantem "lateralne"
        self.beta_lateralne = None
        # PLAN.md #21: sila biologicznej normalizacji dzielacej; ustawiana po konstrukcji,
        # wymagana tylko przez warstwy z wariantem "lateralne_dzielace"
        self.gamma_lateralne = None
        # PLAN.md #8: waga kary wariancji; dziala tylko na warstwie 0 i tylko
        # gdy jej formula to "hamowanie_vicreg" -- przy 0.0 kod liczy dokladnie jak dawniej
        self.vicreg_waga = 0.0
        self.vicreg_kowariancja = False  # E4 pelny VICReg: dodatkowo czlon kowariancji
        self.warstwy = []
        self.optymalizatory = []
        wymiar_biezacy = wymiar_wejscia
        for _ in range(liczba_warstw_ukrytych):
            warstwa = WarstwaFF(wymiar_biezacy, szerokosc).to(self.device)
            self.warstwy.append(warstwa)
            self.optymalizatory.append(torch.optim.Adam(warstwa.parameters(), lr=lr))
            wymiar_biezacy = szerokosc

    @staticmethod
    def _rozszerz_na_liste(wartosc, dlugosc: int) -> list:
        if isinstance(wartosc, (list, tuple)):
            assert len(wartosc) == dlugosc, (
                f"lista musi miec dlugosc {dlugosc} (liczba warstw ukrytych), "
                f"dostala {len(wartosc)}")
            return list(wartosc)
        return [wartosc] * dlugosc

    def parametry_ogolem(self) -> int:
        return sum(p.numel() for w in self.warstwy for p in w.parameters())

    @property
    def wariant(self):
        """Zwraca wariant, jesli WSZYSTKIE warstwy maja ten sam -- inaczej AssertionError.
        Uzyj self.warianty (lista) dla sieci z formula per-warstwa."""
        assert len(set(self.warianty)) == 1, (
            "siec ma rozne formuly per warstwa, uzyj self.warianty (lista)")
        return self.warianty[0]

    @property
    def teta(self):
        """Analogicznie do wariant -- zwraca wspolny prog, jesli wszystkie warstwy go dziela."""
        assert len(set(self.tety)) == 1, (
            "siec ma rozne progi per warstwa, uzyj self.tety (lista)")
        return self.tety[0]

    def opis_formuly(self) -> str:
        """Krotki, czytelny opis formuly/progu do logow -- dziala rowniez
        gdy formula rozni sie miedzy warstwami (gdzie self.wariant/self.teta
        rzucilyby AssertionError)."""
        if len(set(self.warianty)) == 1:
            return f"{self.warianty[0]}(teta={self.tety[0]:.1f})"
        return "|".join(f"{w}:{t:.1f}" for w, t in zip(self.warianty, self.tety))

    def przelacz_formule(self, wariant, teta_override=None):
        """
        Zmienia formule dobroci (i prog) BEZ resetu wag ani stanu
        optymalizatorow Adam -- pomysl Slawka: ucz najpierw standardowa
        dobrocia (buduje reprezentacje ogolna), potem przelacz na
        hamowanie na TYCH SAMYCH wagach (doglebia kontrast pod koniec
        treningu). Zachowanie momentum Adama miedzy formulami jest
        celowe -- to jest test, czy przelaczenie CELU w trakcie treningu
        dziala lepiej niz trzymanie jednego celu od poczatku do konca,
        nie test nowej inicjalizacji.

        Przyjmuje pojedyncza wartosc (broadcast na wszystkie warstwy --
        dotychczasowe uzycie w curriculum) albo liste per warstwa.
        """
        nowe_warianty = self._rozszerz_na_liste(wariant, self.liczba_warstw_ukrytych)
        for w in nowe_warianty:
            waliduj_wariant(w)
        nowe_tety_zadane = self._rozszerz_na_liste(teta_override, self.liczba_warstw_ukrytych)
        self.warianty = nowe_warianty
        self.tety = [t if t is not None else prog_teta(w, self.szerokosc)
                     for w, t in zip(self.warianty, nowe_tety_zadane)]

    def z_efektywne(self, z: torch.Tensor, wariant) -> torch.Tensor:
        """z widziane przez dobroc i nastepna warstwe: dla "lateralne" po
        odjeciu sasiadow, dla "lateralne_dzielace" po normalizacji dzielacej,
        dla pozostalych wariantow bez zmian (ten sam obiekt)."""
        if wariant == "lateralne":
            assert self.beta_lateralne is not None, "wariant lateralne wymaga ustawienia siec.beta_lateralne"
            return odejmij_lateralne(z, self.beta_lateralne)
        elif wariant == "lateralne_dzielace":
            assert self.gamma_lateralne is not None, "wariant lateralne_dzielace wymaga ustawienia siec.gamma_lateralne"
            return dzielace_lateralne(z, self.gamma_lateralne)
        return z

    def krok_treningowy(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        """
        Jeden krok lokalnego uczenia wszystkich warstw. Zwraca (straty,
        separacje) -- po jednej wartosci na warstwe. separacje =
        srednia(dobroc_pozytyw - dobroc_negatyw) w tym kroku -- to jest
        diagnostyka ryzyka trywialnego rozwiazania z PRZEKAZANIE.md: jesli
        separacja nie rosnie w czasie treningu i zostaje bliska zeru,
        siec nie uczy sie odrozniac pozytywu od negatywu, tylko np.
        jednostajnie skaluje wszystko w gore.
        """
        straty = []
        separacje = []
        wejscie_pos, wejscie_neg = x_pos, x_neg

        for i, (warstwa, optymalizator, wariant_w, teta_w) in enumerate(zip(
                self.warstwy, self.optymalizatory, self.warianty, self.tety)):
            z_pos = self.z_efektywne(warstwa(wejscie_pos), wariant_w)
            z_neg = self.z_efektywne(warstwa(wejscie_neg), wariant_w)
            if wariant_w == "hamowanie_tasowane":
                # tasowanie osobno w pozytywach i osobno w negatywach -- kara
                # pozytywu pochodzi z innego pozytywu, negatywu z innego negatywu
                g_pos = dobroc_tasowana(z_pos)
                g_neg = dobroc_tasowana(z_neg)
            else:
                g_pos = dobroc(z_pos, wariant_w)
                g_neg = dobroc(z_neg, wariant_w)

            # Hinton: logistyczna strata wokol progu, osobno dla kazdego
            # kierunku -- pozytyw ma byc POWYZEJ progu, negatyw PONIZEJ.
            strata = (F.softplus(teta_w - g_pos).mean()
                      + F.softplus(g_neg - teta_w).mean())
            if i == 0 and wariant_w == "hamowanie_vicreg" and self.vicreg_waga > 0:
                # na ReLU(z), bo to jest reprezentacja przekazywana dalej (i mierzona przez PR sondy)
                strata = strata + self.vicreg_waga * (kara_wariancji(torch.relu(z_pos))
                                                      + kara_wariancji(torch.relu(z_neg)))
                if self.vicreg_kowariancja:
                    strata = strata + self.vicreg_waga * (kara_kowariancji(torch.relu(z_pos))
                                                          + kara_kowariancji(torch.relu(z_neg)))

            optymalizator.zero_grad()
            strata.backward()
            optymalizator.step()

            straty.append(float(strata.item()))
            separacje.append(float((g_pos - g_neg).mean().item()))

            if wariant_w in ("standard_bcm", "hamowanie_bcm"):
                # PLAN.md #11: prog przesuwny w stylu BCM -- srednia wykladnicza
                # (0.99 na krok) polowy sumy srednich dobroci pozytywu i negatywu.
                srodek = 0.5 * float((g_pos.mean() + g_neg.mean()).item())
                self.tety[i] = 0.99 * self.tety[i] + 0.01 * srodek

            # odlaczenie od grafu -- kolejna warstwa uczy sie lokalnie,
            # nie dostaje gradientu z warstw pozniejszych. Dla "lateralne" dalej idzie
            # reprezentacja PO hamowaniu (z_pos/z_neg to juz z').
            wejscie_pos = normalizuj(torch.relu(z_pos)).detach()
            wejscie_neg = normalizuj(torch.relu(z_neg)).detach()

        return straty, separacje

    @torch.no_grad()
    def suma_dobroci(self, x: torch.Tensor) -> torch.Tensor:
        """Suma dobroci po wszystkich warstwach dla jednego wejscia -- uzywane przy predykcji."""
        wejscie = x
        suma = torch.zeros(x.shape[0], device=x.device)
        for warstwa, wariant_w in zip(self.warstwy, self.warianty):
            z = self.z_efektywne(warstwa(wejscie), wariant_w)
            suma = suma + dobroc(z, wariant_w)
            wejscie = normalizuj(torch.relu(z))
        return suma

    @torch.no_grad()
    def separacja_diagnostyczna(self, x_pos: torch.Tensor, x_neg: torch.Tensor):
        """
        Separacja dobroci (pozytyw minus negatyw) per warstwa, liczona na
        CALYM podanym zbiorze naraz, bez treningu. W odroznieniu od
        separacji zwracanej przez krok_treningowy() (jedna mini-partia,
        ostatni krok -- zaszumiona), to jest srednia po wszystkich
        przykladach na wejsciu. Uzyj tego do raportowania koncowego stanu
        sieci i do kalibracji progu -- nie do sledzenia przebiegu treningu
        krok po kroku (na to sluzy krok_treningowy()).
        """
        wejscie_pos, wejscie_neg = x_pos, x_neg
        separacje = []
        for warstwa, wariant_w in zip(self.warstwy, self.warianty):
            z_pos = self.z_efektywne(warstwa(wejscie_pos), wariant_w)
            z_neg = self.z_efektywne(warstwa(wejscie_neg), wariant_w)
            g_pos = dobroc(z_pos, wariant_w)
            g_neg = dobroc(z_neg, wariant_w)
            separacje.append(float((g_pos - g_neg).mean().item()))
            wejscie_pos = normalizuj(torch.relu(z_pos))
            wejscie_neg = normalizuj(torch.relu(z_neg))
        return separacje

    @torch.no_grad()
    def przewiduj(self, x: torch.Tensor, liczba_klas: int, zakoduj_etykiete_fn) -> torch.Tensor:
        """
        FF nie ma jednej "glowy klasyfikujacej" -- klasyfikacja to proba
        kazdej z liczba_klas etykiet wklejonych w obraz i wybor tej, ktora
        daje najwyzsza sumaryczna dobroc po wszystkich warstwach. Kosztuje
        liczba_klas razy wiecej przejsc w przod niz zwykla siec, ale bez
        propagacji wstecznej to i tak jest tanie.
        """
        dobroci_na_klase = []
        for klasa in range(liczba_klas):
            etykiety = torch.full((x.shape[0],), klasa, dtype=torch.long, device=x.device)
            x_z_etykieta = zakoduj_etykiete_fn(x, etykiety, liczba_klas)
            dobroci_na_klase.append(self.suma_dobroci(x_z_etykieta))
        macierz = torch.stack(dobroci_na_klase, dim=1)
        return macierz.argmax(dim=1)
