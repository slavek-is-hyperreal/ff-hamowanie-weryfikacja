# Protokół v1: curriculum hamowania i próg BCM w Forward-Forward – sprawdzenie konfirmacyjne

Autor: Sławomir Majcher. Szkic do pętli recenzji (R1…). Decyzje: `DECYZJE_AUTORA.md` (D1–D3). Audyt kampanii źródłowej: `00_AUDYT.md`. Infrastruktura i reguły statystyczne wzorowane na `../eksperyment_ff_od_zera/PROTOKOL_v5.md` (dalej „FF-v5”); każde odwołanie do FF-v5 dotyczy wyłącznie mechaniki, a wszystkie liczby tego badania są ustalone tutaj.

## 1. Cel i zakres

Sprawdzić konfirmacyjnie cztery twierdzenia z kampanii „hamowanie” (wpis `blog/forward_forward_hamowanie.md`) w **poprawionym układzie wejścia** (D2), z równym strojeniem, jednorazowym użyciem zbioru testowego i kontrolą błędu.

Zakres wszystkich wniosków: MLP 10 × 64, Forward-Forward z jednoczesnym treningiem warstw (§3), 20 000 przykładów treningowych, 25 epok, MNIST i Fashion-MNIST, oficjalny zbiór testowy.

## 2. Hipotezy konfirmacyjne

Δ = Acc(ramię testowane) − Acc(komparator), parowane po ziarnie, w p.p.

| H | Ramię testowane | Komparator(y) | Przewidywanie autora (kierunek z blogu) |
|---|---|---|---|
| **H1** | `CUR` | `STD` | przewaga |
| **H2** | `CUR` | `ANN` | przewaga |
| **H3** | `BCM` | `STD`, `PLW` | przewaga |
| **H4** | `REV` | `STD` | gorszość |

**Przewidywanie zapisane przed badaniem (asystent, H3):** po normalizacji wejścia (D2) asymetria skali między warstwą 0 a resztą znika, więc oczekiwana przewaga BCM nad stałym progiem jest mała lub zerowa. Wynik R/N w H3 będzie zgodny z tym przewidywaniem, a nie porażką badania.

Interpretacja każdej hipotezy wyłącznie przez kategorie §7.3 i łączenie §7.4.

## 3. Model i trening (wspólne dla wszystkich ramion FF)

Przeniesione z kodu źródłowego (`siec_ff.py`, `zadanie_fashion_mnist.py`, commit `9d58048`), z wyjątkiem wejścia (§3.2).

### 3.1. Sieć i krok uczenia
- 10 warstw liniowych po 64 jednostki; wyjście warstwy: `z`; dalej `normalizuj(ReLU(z))` (L2 per przykład, eps = 1e-4), odłączone od grafu.
- **Każda warstwa ma własny Adam** (β = (0.9, 0.999), eps = 1e-8, bez weight decay). **W każdym kroku wszystkie warstwy uczą się jednocześnie** na tej samej parze partii pozytyw/negatyw (warstwa po warstwie w jednym kroku, jak w źródle). To **nie** jest uczenie zachłanne warstwa po warstwie z FF-v5.
- Strata warstwy: `softplus(θ − g_pos).mean() + softplus(g_neg − θ).mean()`.
- Dobroć: `standard` g = Σ ReLU(z)²; `hamowanie` g = Σ ReLU(z)² − Σ ReLU(−z)².
- Negatyw: ten sam obraz z losową **złą** etykietą (przesunięcie o 1…9 mod 10), ze strumienia `neg`.
- Partia 64, **25 epok, bez wczesnego zatrzymania** (punkty przełączenia w `CUR`, `ANN`, `REV` są zdefiniowane w epokach i nie mogą zależeć od zatrzymania).
- Przełączenie formuły/LR/progu **nie resetuje** wag ani stanu Adama (jak w źródle).

### 3.2. Wejście (zmiana względem źródła, D2) – ustalone przed pilotem
- Piksele: x/255, następnie **normalizacja L2 wektora 784 pikseli do normy 1**.
- Etykieta: one-hot o wartości **1.0 doklejona** jako 10 dodatkowych wymiarów (wejście 794). Brak dalszej normalizacji całego wektora 794 (norma = √2 dla każdego przykładu; obraz i etykieta mają jednakowy udział normy).
- Uzasadnienie: warstwa 0 dostaje wektor o normie stałej i porównywalnej z wejściem warstw ukrytych (norma 1), etykieta nie niszczy pikseli. Wybór skali etykiety (1.0) jest ustalony teraz, nie strojony.

### 3.3. Predykcja
- Dla każdej z 10 etykiet: suma dobroci **wszystkich 10 warstw** (jak w źródle; formuła dobroci każdej warstwy taka, jaka obowiązuje na końcu treningu danego ramienia); argmax.
- Analiza eksploracyjna E3: suma bez warstwy 0 (jak u Hintona).

## 4. Ramiona

Wszystkie ramiona FF mają **te same dwa strojone hiperparametry**: `lr` (wspólne dla wszystkich warstw i faz) i `θ` (próg skalarny; w `BCM` – próg początkowy). Pozostałe parametry są ustalone jak w źródle.

| Ramię | Epoki 1–12 | Epoki 13–25 | Strojone |
|---|---|---|---|
| `STD` | standard, θ | standard, θ | lr, θ |
| `CUR` | standard, θ | hamowanie, θ = 0 | lr, θ |
| `ANN` | standard, θ, lr | standard, θ, **lr × 0.5** | lr, θ |
| `REV` | hamowanie, θ = 0 | standard, θ | lr, θ |
| `BCM` | standard, θ_l(t) = 0.99·θ_l(t−1) + 0.01·½(ḡ_pos + ḡ_neg), θ_l(0) = θ, aktualizacja co krok, osobno dla każdej warstwy | jak obok | lr, θ (początkowe) |
| `PLW` | standard, **stały próg per warstwa** θ_l = średnia końcowych θ_l ramienia `BCM` | jak obok | brak (lr = lr wybrane dla `BCM`) |
| `BP` (eksploracyjne) | backprop, ta sama architektura 10 × 64, ReLU, głowa liniowa 10, CE, Adam | – | lr |

Uwagi:
- Punkt przełączenia (epoka 12/13), θ = 0 w fazie hamowania i współczynnik 0.5 w `ANN` są przeniesione ze źródła i **nie są strojone** (nie zwiększają wymiaru przestrzeni strojenia żadnego ramienia).
- `PLW` (definicja H3): θ_l wyznaczane w Etapie B jako średnia po ziarnach strojenia z θ_l na końcu treningu `BCM` przy wybranej konfiguracji `BCM`. Odpowiada pytaniu „czy dynamiczne śledzenie progu jest potrzebne, czy wystarczy stały próg per warstwa o tej samej wartości” (#17/#19 źródła). `PLW` nie ma własnego strojenia; dostaje progi z przebiegu `BCM`, więc różnica `BCM` − `PLW` izoluje samą dynamikę progu. Drugi komparator H3, `STD`, ma pełne strojenie.

## 5. Etapy

### 5.1. Mini-pilot (P)
- Cel: σ_d (SD różnic parowanych) dla każdej hipotezy i zbioru, ρ dla H3, przepustowość i koszt.
- 6 ramion FF × 2 zbiory × 10 ziaren (pula P), konfiguracja domyślna: lr = 1e-3 (jak w źródle), θ = 2 (Hinton; θ = 8 ze źródła było dobrane pod nienormalizowane wejście i po D2 nie ma uzasadnienia). `PLW` w pilocie dostaje progi z pilotowego `BCM`.
- Ewaluacja wyłącznie na walidacji. Wyniki pilota nie wchodzą do analizy konfirmacyjnej.

### 5.2. Etap A – strojenie
- Przestrzeń: lr ∈ [1e-4, 1e-2] (log), θ ∈ [0.25, 16] (log). 16 punktów Sobola (scramble, ziarno stałe), **te same punkty dla każdego ramienia i zbioru**.
- **5 ziaren na punkt** (pula A), lekcja z blogu (2 ziarna wybrały źle).
- Kryterium: średnia dokładność walidacyjna (10 000 z puli treningowej, rozłączne z treningiem); wybór: najlepsza średnia po **uśrednieniu z 2 najbliższymi sąsiadami w przestrzeni log** (reguła z blogu, `PLAN.md` #4). Konfiguracje z NaN lub dokładnością ≤ 15% u któregokolwiek ziarna odrzucane.
- `BP`: 8 punktów lr, 5 ziaren.

### 5.3. Etap B – wyznaczenie `PLW` i kontrola wybranych konfiguracji
- `BCM` w wybranej konfiguracji na 10 ziarnach (pula B): zapis θ_l na końcu treningu; θ_l(`PLW`) = średnia po ziarnach.
- Wszystkie ramiona w wybranych konfiguracjach na tych samych 10 ziarnach: kontrola stabilności (walidacja).

### 5.4. Etap C – konfirmacja
- Wszystkie ramiona (6 FF + `BP`) × 2 zbiory × N ziaren (pula C), konfiguracje zamrożone po Etapie B. Kolejność biegów permutowana ziarnem stałym.
- Walidacyjne metryki do `zaslepione/`.

### 5.5. Ewaluacja końcowa (zaślepiona)
Jak FF-v5 §5.6, z poprawką operacyjną: hashe checkpointów liczone **na instancji**, plik hashy commitowany i wypychany na GitHub, **ewaluacja testowa na tej samej instancji GPU**, pobranie predykcji, dopiero potem zniszczenie instancji.

## 6. Dane, ziarna, determinizm
- Pula treningowa per ziarno: 20 000 losowane z 50 000 (podział 50k/10k ziarnem `split`); walidacja 10 000 stała per ziarno; test: **pełne 10 000 oficjalne**, QMNIST-50k dla MNIST (wrażliwość).
- Strumienie losowe i ziarna jak FF-v5 §4.3 (SHA-256 krotek; `split`, `order`, `neg`, `init`).
- Pule ziaren (rozłączne, nowe względem źródła): P 60001–60010, A 61001–61005, B 62001–62010, C 63001–63000+N.
- Determinizm GPU: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`; test 9.1.9 FF-v5 przed każdym etapem na nowej instancji.

## 7. Analiza

### 7.1–7.4
Test elementarny, kategorie i łączenie zbiorów – **jak FF-v5 §7.2–7.4**, z α′ = **0.05/4 = 0.0125**. Przy jednym komparatorze IUT i test sumy redukują się do pojedynczego testu (H1, H2, H4); H3 ma dwa komparatory (IUT po `STD`, `PLW`; gorszość: Holm na 2 wartościach p).

### 7.5. SESOI (S) – ustalone teraz
**S = 0.5 p.p. dla obu zbiorów i wszystkich hipotez.** Uzasadnienie: w siatce progów źródła różnice między sąsiednimi punktami θ przy dobrze dobranym standardzie wynosiły ok. 0.5 p.p. (blog, sekcja „Sweep na pełnej skali”); efekt mniejszy nie przetrwałby zmiany strojenia, więc nie ma znaczenia praktycznego dla wyboru metody.

### 7.6. Wybór N – reguła ustalona teraz
- Symulacja mocy jak FF-v5 (rozkład t z σ_d z pilota × 1.29 na niepewność σ, ρ z pilota ograniczone do [0, 0.5]).
- Wymagana moc ≥ 0.8 dla przewagi przy prawdziwym Δ = 1.5·S w H1, H2, H3 na każdym zbiorze z osobna.
- **Drabina budżetowa:** N ∈ {30, 45, 60, 90, 120}; wybierane jest najmniejsze N spełniające wymóg mocy; jeśli żadne nie spełnia w limicie 10 USD (D3), wybierane jest największe N mieszczące się w budżecie, a raport podaje osiągniętą moc. N tylko w górę.

### 7.7. Analizy wrażliwości
(a) bootstrap 2D (przykłady × ziarna) jak FF-v5; (b) Wilcoxon; (c) QMNIST-50k dla MNIST.

### 7.8. Eksploracje (bez korekty, oznaczane)
- E1: trajektorie θ_l w `BCM` (per warstwa, per epoka).
- E2: luka `BP` − najlepsze ramię FF.
- E3: predykcja bez warstwy 0.
- E4: `CUR` − `ANN` w drugiej połowie treningu: krok wag (norma zmiany wag na epokę), jak pomiar kroku w źródle.

### 7.9. Skrypt analizy
Jeden zamrożony skrypt (`final.py`) liczy **każdą liczbę i każdy rysunek wymagany przez szablon raportu**, łącznie z listą komparatorów istotnych wg Holma, eksploracjami E1–E4 i analizami wrażliwości. Przed zamrożeniem: autotest na danych syntetycznych ze znanym wynikiem (jak FF-v5 §9.2).

## 8. Wykonanie i koszt
- Vast.ai; typ GPU wybierany po benchmarku przepustowości (sieć 10 × 64 jest ograniczona narzutem uruchamiania kerneli, nie FLOP-ami); test 9.1.9 przy zmianie typu urządzenia.
- Limit 10 USD łącznie (D3).

## 9. Prerejestracja
- Folder publikowany jako osobne publiczne repozytorium GitHub (subtree split tylko `eksperyment_hamowanie_od_zera/`) – **za zgodą autora przed utworzeniem repozytorium**.
- Tag zamrożenia przed Etapem A; każda zmiana po tagu → `ODSTEPSTWA.md`.

## 10. Ograniczenia znane z góry
1. Brak ramienia 1:1 (D2) – wynik negatywny nie jest przypisywalny naprawie wejścia.
2. Reżim słabej pojemności (10 × 64, 20 000 przykładów, 25 epok).
3. `PLW` nie jest strojone niezależnie – H3 pyta o „dynamiczny vs stały próg o tej samej wartości” oraz „BCM vs strojony próg skalarny”, a nie o optymalny próg per warstwa.
4. Punkty przełączenia i θ = 0 w fazie hamowania nie są strojone (ustalone ze źródła).
