# Protokół v2: curriculum hamowania i próg BCM w Forward-Forward – sprawdzenie konfirmacyjne

Autor: Sławomir Majcher. Wersja po recenzji R1 (`recenzje/R1.md`, odpowiedź `odpowiedzi/O1.md`). Decyzje: `DECYZJE_AUTORA.md` (D1–D3). Audyt źródła: `00_AUDYT.md`. Protokół jest **samowystarczalny**: wszystkie liczby, reguły i zdania raportu są tutaj; odwołania do `../eksperyment_ff_od_zera/PROTOKOL_v5.md` („FF-v5”) wskazują wyłącznie pochodzenie mechaniki.

**Źródło metody:** `/my_data/VectorLegis/eksperymenty/forward_forward/potwierdzenie_30_ziaren/` (commit `9d58048`): `siec_ff.py`, `zadanie_fashion_mnist.py`, `dane.py`; opis: `blog/forward_forward_hamowanie.md`.

## 1. Cel, zakres, status wiedzy przed badaniem

Cel: sprawdzić konfirmacyjnie cztery twierdzenia kampanii „hamowanie” w poprawionym układzie wejścia (D2), z równym budżetem strojenia, jednorazową ewaluacją testową i kontrolą błędu.

Zakres wszystkich wniosków: MLP 10 × 64 z jednoczesnym treningiem warstw FF (§3), 20 000 przykładów treningowych, 25 epok, MNIST i Fashion-MNIST.

**Wiedza autora i asystenta przed badaniem (R1-01).** Oficjalne zbiory testowe MNIST i Fashion-MNIST były w kampanii źródłowej oglądane wielokrotnie (30 ziaren × podpróbka 3000; `PLAN.md` źródła, l. 553–555); „czysty test” F-MNIST z puli treningowej też. Autor zna kierunki i przybliżone wielkości efektów źródłowych (blog). Konsekwencje:
- Ten protokół nie korzysta z żadnych wyników testowych źródła przy wyborze konfiguracji (strojenie wyłącznie na walidacji, §5).
- Hipotezy i ich kierunki pochodzą ze źródła i są zapisane jawnie (§2) wraz z prognozami liczbowymi w `PROGNOZY.md` (commit przed pilotem, §9).
- **QMNIST-50k** (50 000 cyfr spoza oficjalnego testu, nieużywane w źródle) jest dla MNIST raportowane na pierwszej stronie raportu obok wyniku głównego (§7.7c). Dla F-MNIST nie istnieje niezależny zbiór testowy; to ograniczenie (§12.1).

## 2. Hipotezy konfirmacyjne

Δ = Acc(ramię testowane) − Acc(komparator) [p.p.], parowane po ziarnie, osobno dla każdego zbioru.

| H | Ramię testowane | Komparator(y) | Kierunek ze źródła | Wielkość w źródle (MNIST / F-MNIST) |
|---|---|---|---|---|
| **H1** | `CUR` | `STD` | przewaga | +1.4 / +0.4…+0.8 |
| **H2** | `CUR` | `ANN` | przewaga | +1.44 / −0.17 (blog: na F-MNIST brak różnicy) |
| **H3** | `BCM` | `STD`, `PLW` (per zbiór) | przewaga | +1.49 / +0.68 vs stały θ; vs stały próg per warstwa: brak rozstrzygnięcia w źródle |
| **H4** | `REV` | `STD` | gorszość | −2.0 / −0.4 |

**Przewidywania asystenta przed badaniem (per zbiór, R1-19, R1-20):**
- H2: na F-MNIST oczekiwana RÓWNOWAŻNOŚĆ albo NIEROZSTRZYGNIĘTY, więc łączny werdykt H2 najpewniej w klasie N (NIESPÓJNY-*), zgodnie ze źródłem.
- H3: po normalizacji wejścia (D2) asymetria skali warstwy 0 maleje, więc oczekiwana przewaga `BCM` jest mała. Przewidywanie „brak praktycznej przewagi” potwierdza tylko kategoria klasy R; klasa N oznacza brak rozstrzygnięcia.

**Definicja H3 (R1-21).** Audyt formułował H3 jako „BCM > próg dostrojony per warstwa”. Ten protokół zawęża to do: BCM > (a) **skalarny próg dostrojony** (`STD`) i (b) **stały próg per warstwa o wartościach z przebiegu BCM**, z osobno dostrojonym lr (`PLW`). Nie testuje progu per warstwa strojonego niezależnie (10 wymiarów – równy budżet niemożliwy).

## 3. Model i trening

### 3.1. Sieć (jak w źródle)
- 10 warstw `Linear(·, 64)`; wejście warstwy l+1 = `normalizuj(ReLU(z_l)).detach()`, `normalizuj(a) = a / (‖a‖₂ + 1e-4)` per przykład.
- **Każda warstwa ma własny Adam** (β = (0.9, 0.999), eps = 1e-8, bez weight decay). W każdym kroku wszystkie 10 warstw uczy się po kolei na tej samej parze partii (jednoczesny trening, jak `SiecFF.krok_treningowy` źródła).
- Strata warstwy: `softplus(θ_l − g_pos).mean() + softplus(g_neg − θ_l).mean()`.
- Dobroć: `standard` g = Σ ReLU(z)²; `hamowanie` g = Σ ReLU(z)² − Σ ReLU(−z)².
- Inicjalizacja: domyślna `nn.Linear` (Kaiming-uniform), ziarno ze strumienia `init`.
- Partia 64; kolejność przykładów: permutacja co epokę; **ostatnia, niepełna partia (20 000 mod 64 = 32) jest używana** (jak w źródle). 25 epok, **bez wczesnego zatrzymania**.
- Przełączenie fazy nie resetuje wag ani stanu Adama.

### 3.2. Wejście (D2) – ustalone przed pilotem
- Piksele: x/255, następnie L2 do normy 1 (784 wymiary).
- Etykieta: one-hot o wartości **λ = 0.1** doklejona (wejście 794). Brak normalizacji całego wektora; norma wejścia = √1.01.
- Uzasadnienie λ (R1-10): w źródle stosunek normy etykiety (1.0) do normy obrazu x/255 (ok. 9–12, zależnie od zbioru; do zmierzenia w T8) wynosił ok. 0.1; λ = 0.1 zachowuje ten stosunek po normalizacji obrazu. Wartość nie jest strojona.
- Pozostała asymetria: warstwa 0 dostaje wektor o normie ≈ 1 o innej strukturze niż warstwy ukryte (gęste piksele vs rzadkie ReLU); progi BCM mogą się więc nadal różnić między warstwami – to jest przedmiot E1, nie błąd.
- Negatyw: ta sama część pikselowa, zła etykieta (przesunięcie o 1…9 mod 10).

### 3.3. Predykcja
- **Odczyt główny** (jak w źródle): dla każdej z 10 etykiet suma dobroci wszystkich 10 warstw, formułą dobroci obowiązującą w danym ramieniu na końcu treningu (`CUR`: hamowanie; pozostałe: standard); argmax.
- Odczyty eksploracyjne (E3, R1-11), liczone w tym samym poleceniu testowym: (a) suma bez warstwy 0; (b) odczyt krzyżowy – formuła standard dla `CUR` i formuła hamowanie dla `STD`, `ANN`, `REV`.

### 3.4. NaN i rozjechanie treningu (R1-03, R1-25)
- Bieg, w którym strata którejkolwiek warstwy lub dobroć stanie się nieskończona/NaN, zostaje zatrzymany i oznaczony `nan = true`; checkpoint = stan z końca ostatniej pełnej epoki bez NaN (zapisywany co epokę w pamięci).
- **Analiza główna:** bieg z NaN ma dokładność testową liczona z tego checkpointu (jeśli NaN w epoce 1: 10%, poziom losowy). Uzasadnienie: niestabilność jest kosztem metody i nie może być usuwana z analizy.
- Wrażliwość: (b) test Wilcoxona (odporny na wartości odstające) i (d) analiza bez par z NaN.
- Kategoria raportowana z dopiskiem, gdy t i Wilcoxon się różnią w obecności NaN (jak FF-v5).

### 3.5. Ramię `BP` (eksploracyjne, R1-12)
- Ta sama architektura 10 × 64, ReLU, głowa `Linear(64, 10)`, entropia krzyżowa, Adam, partia 64, 25 epok, bez wczesnego zatrzymania.
- Wejście: 784 piksele po L2 (bez etykiety).
- Strojenie: 8 punktów lr log-równomiernie w [1e-4, 1e-2], 5 ziaren puli A.

## 4. Ramiona FF

Ramiona różnią się wyłącznie harmonogramem (formuła, próg, lr) w czasie.

| Ramię | Epoki 1–12 | Epoki 13–25 | Strojone | Gdzie działa strojone θ |
|---|---|---|---|---|
| `STD` | standard, θ | standard, θ | lr, θ | cały trening |
| `CUR` | standard, θ | hamowanie, θ = 0 | lr, θ | epoki 1–12 |
| `ANN` | standard, θ, lr | standard, θ, lr × 0.5 | lr, θ | cały trening |
| `REV` | hamowanie, θ = 0 | standard, θ | lr, θ | epoki 13–25 |
| `BCM` | standard, θ_l(t) BCM | standard, θ_l(t) BCM | lr, θ(0) | tylko start (§4.1) |
| `PLW` | standard, θ_l stałe z `BCM` | jak obok | lr (1D) | – |

Zgodność ze źródłem (sprawdzona w R1): `CUR` = `polowa_na_polowe`, `ANN` = `kontrola_lr(0.5)`, `REV` = `polowa_na_polowe_odwrocone`, `BCM` = `prog_bcm`, `PLW` = #17 `prog_staly_warstwy`. Punkt przełączenia (12/13), θ = 0 w fazie hamowania i współczynnik 0.5 są ustalone ze źródła i nie są strojone.

### 4.1. `BCM`
θ_l(t) = 0.99·θ_l(t−1) + 0.01·½(mean g_pos + mean g_neg), aktualizacja po każdym kroku optymalizatora warstwy l, θ_l(0) = θ (wspólne dla warstw). **Uwaga (R1-06):** pamięć EMA to ok. 100 kroków (≈ 1/3 epoki); po epoce wpływ θ(0) spada do ok. 4% (0.99³¹³). Strojenie θ(0) działa więc praktycznie tylko przez pierwszą epokę, a `BCM` stroi w istocie samo lr. To asymetria wbudowana w metodę źródłową; raport podaje ją przy H3, a krajobraz strojenia (E5) jest obowiązkowy.

### 4.2. `PLW` (R1-08)
- θ_l = średnia po ziarnach puli B wartości θ_l na końcu treningu `BCM` w wybranej konfiguracji `BCM`, **osobno dla każdego zbioru** (ziarna z NaN pomijane; jeśli NaN ma > 5 z 10 ziaren – §5.6).
- lr `PLW` strojone osobno w Etapie B2 (§5.4): 8 punktów log-równomiernych w [1e-4, 1e-2], 5 ziaren puli A.

## 5. Etapy i reguły wyboru

### 5.1. Mini-pilot (P)
- Ramiona `STD`, `CUR`, `ANN`, `REV`, `BCM`, `PLW` (z progami pilotowego `BCM`) × 2 zbiory × 10 ziaren puli P, konfiguracja domyślna: lr = 1e-3 (źródło), θ = 2 (wartość umowna, często używana w reimplementacjach FF; Hinton nie podaje liczby – R1-18). `BP`: lr = 1e-3.
- Wyjście: σ_d dla każdego kontrastu H1–H4 i zbioru, ρ dla H3, przepustowość, czas i koszt biegu, odsetek NaN. Tylko walidacja.
- `PLW` w pilocie dzieli ziarna z `BCM` (progi z tych samych biegów), co może zaniżać σ_d i zawyżać ρ; dlatego σ do reguły N = max(pilot, Etap B) (§7.6).

### 5.2. Etap A – strojenie
- Przestrzeń wspólna dla `STD`, `CUR`, `ANN`, `REV`, `BCM`: lr ∈ [1e-4, 1e-2], θ ∈ [0.01, 16] (oba w skali log). Dolna granica θ obniżona względem v1, bo po D2 dobroć startowa wynosi ok. 0.04 (warstwa 0) – 0.3 (warstwy ukryte) (R1-07).
- 16 punktów Sobola (scramble, ziarno stałe `seed_from("A-sobol")`), identycznych dla każdego ramienia i zbioru. 5 ziaren na punkt (pula A).
- Kryterium: średnia dokładność walidacyjna z 5 ziaren. Punkt odrzucany, jeśli którykolwiek bieg ma NaN albo dokładność walidacyjną ≤ 15%.
- **Do Etapu B przechodzą 3 najlepsze nieodrzucone punkty** danego ramienia i zbioru (R1-07; zastępuje uśrednianie z sąsiadami).

### 5.3. Etap B – wybór konfiguracji
- Dla każdego ramienia i zbioru 3 kandydaci × 10 ziaren (pula B). Wybór: najwyższa średnia walidacyjna; remis (< 0.05 p.p.) – mniejszy lr.
- Z biegów `BCM` w wybranej konfiguracji: θ_l końcowe → `PLW` (§4.2).
- Wyjście: `ETAP_B_WYBOR.json` (konfiguracje, średnie, θ_l `PLW`), σ_d z Etapu B.

### 5.4. Etap B2 – strojenie lr `PLW` i `BP`
- `PLW`: 8 lr × 5 ziaren (pula A), progi z §4.2; wybór najlepszej średniej walidacyjnej.
- `BP`: jak §3.5 (może iść równolegle z Etapem A).

### 5.5. Reguła brzegowa (R1-07)
Jeśli wybrana konfiguracja któregokolwiek ramienia leży w skrajnych 10% zakresu (log) lr lub θ: raport to podaje jako ograniczenie; **brak rozszerzenia przestrzeni** (budżet). Krajobraz E5 pokazuje, czy optimum mogło leżeć poza zakresem.

### 5.6. Sytuacje awaryjne
- Wszystkie 16 punktów ramienia odrzucone w Etapie A: kryterium odrzucenia łagodzone do „mniej niż 2 z 5 ziaren z NaN/≤ 15%”; jeśli nadal brak – ramię uznane za nietrenowalne w tym układzie, hipotezy z jego udziałem otrzymują kategorię NIEROZSTRZYGNIĘTY z dopiskiem „ramię nietrenowalne”.
- `BCM` w Etapie B z NaN w > 5 z 10 ziaren dla wybranego kandydata: wybierany kolejny kandydat.
- Awaria instancji: biegi są atomowe (zapis po zakończeniu); runner wznawia brakujące. Zmiana typu GPU lub k w trakcie etapu jest zabroniona (R1-23); nowa instancja tego samego typu wymaga testu determinizmu (§8.2) przed startem.

### 5.7. Etap C – konfirmacja
- 6 ramion FF + `BP` × 2 zbiory × N ziaren (pula C), konfiguracje zamrożone w `ETAP_B_WYBOR.json`. Kolejność biegów permutowana ziarnem `seed_from("C-order")` – **we wszystkich etapach** (R1-23).
- Metryki walidacyjne → `zaslepione/` (nieotwierane do kroku 2 §5.8).

### 5.8. Ewaluacja końcowa (R1-16)
1. Po zakończeniu Etapu C: hashe SHA-256 wszystkich checkpointów liczone **na instancji**, plik pobrany, commit i push na GitHub.
2. **Dopiero potem** dane testowe (MNIST-test, F-MNIST-test, QMNIST) są kopiowane na instancję (wcześniej ich tam nie ma; runner treningowy nie ma ścieżki do testu).
3. Jedno polecenie na instancji: weryfikacja hashy, odczyt główny i odczyty eksploracyjne E3, QMNIST; zapis predykcji per przykład.
4. Pobranie predykcji, weryfikacja hashy plików predykcji, commit.
5. Zniszczenie instancji.
- Awaria instancji między krokami 1 a 4: checkpointy są synchronizowane lokalnie na bieżąco (rsync + weryfikacja hashy); ewaluacja może zostać powtórzona na nowej instancji tego samego typu z tymi samymi hashami.
- Powtórki biegów przed krokiem 3 wyłącznie po błędzie kodu (definicja i procedura jak FF-v5 §5.6 pkt 4: test, diff i lista biegów commitowane przed powtórką, biegi pierwotne zachowane i raportowane).

## 6. Dane, ziarna, strumienie losowe (R1-09, R1-22)

- Podział per ziarno (strumień `split`): 60 000 treningowych → 10 000 walidacyjnych (stałe dla danego ziarna) + 50 000; z tych 50 000 losowane 20 000 treningowych. Walidacja i trening są rozłączne (test jednostkowy T4).
- Test: pełne oficjalne 10 000; QMNIST-50k (`test50k`) dla modeli MNIST.
- Strumienie (generator CPU, ziarno = pierwsze 63 bity SHA-256 reprezentacji krotki):
  - `split`: (s, 1)
  - `order`: (s, 2, epoka) – permutacja 20 000 w epoce
  - `neg`: (s, 3, epoka) – złe etykiety w epoce
  - `init`: (s, 4)
  - **Bez indeksu warstwy** (warstwy uczą się jednocześnie na tej samej partii). `BP` używa `order` (s, 2, epoka), `init` (s, 4).
- Strumienie są identyczne dla wszystkich ramion przy danym ziarnie (parowanie).
- Pule ziaren (rozłączne, nowe względem źródła): P 60001–60010, A 61001–61005, B 62001–62010, C 63001–(63000 + N).
- Determinizm: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`.

## 7. Analiza

### 7.1. Test elementarny
- d_i = Acc_B(s_i) − Acc_K(s_i) [p.p.], i = 1…N. Przedział t-Studenta dla średniej d.
- α = 0.05, Bonferroni na 4 hipotezy: **α′ = 0.0125**.
- Kontrola błędu dla każdego typu twierdzenia: klasa P (przewaga) na poziomie α′/2 (jednostronnie), klasa G (gorszość) na α′/2, klasa R (równoważność, TOST) na α′. W całym badaniu ≤ 0.05 dla każdego typu twierdzenia osobno.

### 7.2. Kategorie na jednym zbiorze
Oznaczenia: CI_K – dwustronny przedział (1 − α′) dla Δ_K; CI2_K – przedział (1 − 2α′) (TOST); S – §7.5.

| Warunek | Definicja |
|---|---|
| sup(S) | dla każdego K: dolna granica CI_K > S (IUT) |
| sup(0) | dla każdego K: dolna granica CI_K > 0 (IUT) |
| eq | dla każdego K: CI2_K ⊂ (−S, S) (IUT, TOST) |
| inf(S) | jednostronne p_K dla H₀: Δ_K ≥ −S; Holm na m komparatorach na poziomie α′/2; odrzucona ≥ 1 |
| inf(0) | jak inf(S), H₀: Δ_K ≥ 0 |

Dla H1, H2, H4 m = 1 (Holm = pojedynczy test na α′/2); dla H3 m = 2.

| # | Kategoria | Warunek | Klasa |
|---|---|---|---|
| 1 | GORSZOŚĆ | inf(S) | G |
| 2 | PRZEWAGA | sup(S) | P |
| 3 | RÓWNOWAŻNOŚĆ+ | eq ∧ (sup(0) ∨ inf(0)) | R |
| 4 | GORSZOŚĆ<S | inf(0) | G |
| 5 | PRZEWAGA<S | sup(0) | P |
| 6 | RÓWNOWAŻNOŚĆ | eq | R |
| 7 | NIEROZSTRZYGNIĘTY | – | N |

### 7.3. Łączenie zbiorów
| Sytuacja | Werdykt łączny |
|---|---|
| Ta sama klasa na obu zbiorach | ta klasa; kategoria łączna = słabsza z dwóch (P → PRZEWAGA tylko przy PRZEWADZE na obu; analogicznie G; R → RÓWNOWAŻNOŚĆ+, jeśli wystąpi na choć jednym) |
| P na jednym, G na drugim | NIESPÓJNY-KIERUNEK (N) |
| P na jednym i R+ o dodatnim kierunku na drugim (albo G i R+ o ujemnym) | NIESPÓJNY-ZGODNY-KIERUNEK (N) |
| Pozostałe | NIESPÓJNY-CZĘŚCIOWY (N) |

### 7.4. Zdania raportu (jedyne dopuszczalne interpretacje konfirmacyjne)

| H | P | R | G | N |
|---|---|---|---|---|
| H1 | Curriculum (standard → hamowanie) daje wyższą dokładność niż standard. | Curriculum równoważne standardowi (±S). | Curriculum daje niższą dokładność niż standard. | Porównanie curriculum ze standardem nierozstrzygnięte. |
| H2 | Curriculum daje wyższą dokładność niż standard z wyżarzaniem LR (× 0.5 od połowy). | Curriculum równoważne standardowi z wyżarzaniem LR (±S). | Curriculum daje niższą dokładność niż standard z wyżarzaniem LR. | Porównanie curriculum z wyżarzaniem LR nierozstrzygnięte. |
| H3 | Próg BCM daje wyższą dokładność niż strojony próg skalarny i niż stały próg per warstwa o wartościach z BCM. | BCM równoważny obu stałym progom (±S). | Co najmniej jeden stały próg daje wyższą dokładność niż BCM. | Porównanie BCM ze stałymi progami nierozstrzygnięte. |
| H4 | Odwrócone curriculum daje wyższą dokładność niż standard. | Odwrócone curriculum równoważne standardowi (±S). | Odwrócone curriculum daje niższą dokładność niż standard. | Porównanie odwróconego curriculum ze standardem nierozstrzygnięte. |

Obowiązkowe dopiski:
- Kategorie „<S”: „różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane”.
- RÓWNOWAŻNOŚĆ+: „wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S)”, z kierunkiem na każdym zbiorze i komparatorami, które go wyznaczają.
- NIESPÓJNY-ZGODNY-KIERUNEK / NIESPÓJNY-KIERUNEK: zdania jak FF-v5 §7.4.
- H3 w klasie G: lista komparatorów istotnie lepszych od `BCM` **wg procedury Holma** (nie wg przedziałów; lekcja R6-01), liczona przez `final.py`.
- H3: zawsze dopisek o §4.1 (θ(0) BCM działa tylko w pierwszej epoce).
- NaN: „werdykt testu t (…) i testu Wilcoxona (…) różnią się w obecności biegów z NaN (…)”, gdy dotyczy.

**Odczyt łączny H1 × H2 (opisowy, bez nowego testu):** jeśli H1 = P i H2 ∈ {R, N}, raport stwierdza: „przewaga curriculum nad standardem nie została odróżniona od efektu wyżarzania LR”. Jeśli H1 = P i H2 = P: „curriculum daje wyższą dokładność zarówno niż standard, jak i niż wyżarzanie LR”.

Słowa zakazane: „dowodzi”, „przełom”, „bezprecedensowy”, „idealny”, „rozwiązuje”, „deklasuje”, „ratuje”.

### 7.5. SESOI (R1-02)
**S = 0.5 p.p. dla obu zbiorów i wszystkich hipotez.**
- Uzasadnienie (uczciwe): S to próg istotności praktycznej przyjęty arbitralnie przez autora jako „różnica, dla której warto zmienić metodę treningu małego MLP FF”. Nie wynika z danych.
- Autor zna efekty źródłowe (§2): na F-MNIST mieszczą się w 0.4–0.8 p.p., więc kategoria PRZEWAGA (> S) na F-MNIST jest mało prawdopodobna nawet przy prawdziwym efekcie; wynik w klasie P będzie zwykle PRZEWAGĄ<S. To zapisane z góry.
- Wrażliwość na S: kategorie przeliczane dla S ∈ {0.25, 1.0} (§7.7e), raportowane w tabeli wrażliwości.

### 7.6. Wybór N (R1-05)
- Wejście skryptu `choose_n`: **wyłącznie SD różnic** (σ_d) i ρ, nigdy średnie. σ_d = max(pilot, Etap B) dla każdego kontrastu i zbioru, × 1.29 (niepewność σ, jak FF-v5); ρ (H3) = min(ρ̂, 0.5), ≥ 0.
- Cel mocy (symulacja, 20 000 powtórzeń, rozkład t):
  - H1, H2, H3: prawdopodobieństwo **łącznej klasy P** (na obu zbiorach P) ≥ 0.8 przy prawdziwym Δ = 1.0 p.p. (= 2S) na obu zbiorach;
  - H4: prawdopodobieństwo łącznej klasy G ≥ 0.8 przy prawdziwym Δ = −1.0 p.p. na obu zbiorach.
  - Moc równoważności (klasa R przy Δ = 0) raportowana, bez wymogu.
- **Drabina:** N ∈ {30, 45, 60, 90, 120}; wybierane najmniejsze N spełniające cel dla wszystkich czterech hipotez; jeśli żadne – N = 120 (o ile mieści się w budżecie §8.3), z raportowaną mocą osiągniętą. N tylko w górę.

### 7.7. Analizy wrażliwości
(a) bootstrap 2D (przykłady testowe × ziarna, 10 000 losowań) dla kategorii; (b) Wilcoxon; (c) QMNIST-50k (MNIST) – na pierwszej stronie raportu; (d) bez par z NaN; (e) S ∈ {0.25, 1.0}.

### 7.8. Eksploracje (bez korekty; oznaczane)
- E1: trajektorie θ_l `BCM` (warstwa × epoka; zapis co epokę w rekordzie biegu).
- E2: luka `BP` − najlepsze ramię FF; „najlepsze” wybierane wg średniej walidacyjnej Etapu C (`zaslepione/` po kroku 2 §5.8).
- E3: odczyty (a) bez warstwy 0, (b) krzyżowy (§3.3).
- E4: norma zmiany wag na epokę per warstwa (`CUR`, `ANN`, `STD`; zapis w rekordzie biegu) – czy przełączenie zmienia krok.
- E5: krajobraz strojenia Etapu A (mapy lr × θ per ramię i zbiór) – **obowiązkowy**.
- E6: dokładność walidacyjna co epokę (krzywe) dla wszystkich ramion – zapis w rekordzie biegu.

### 7.9. Skrypt analizy (`final.py`)
Liczy każdą liczbę, tabelę i rysunek z `RAPORT_SZABLON.md` (w tym: listę Holma dla H3, odsetki NaN, E1–E6, wrażliwości a–e, rysunki). Autotest przed zamrożeniem na danych syntetycznych ze znanymi werdyktami: scenariusze (i) wszystkie P, (ii) wszystkie R, (iii) G w H3 z jednym komparatorem, (iv) NIESPÓJNY-KIERUNEK, (v) NIESPÓJNY-ZGODNY-KIERUNEK, (vi) NaN z rozbieżnym t/Wilcoxon; porównanie z implementacją referencyjną (jak FF-v5 §9.2).

## 8. Wykonanie

### 8.1. Kod (R1-13)
Runner, zapis atomowy, synchronizacja z weryfikacją hashy, monitor, ewaluacja i analiza – przeniesione z `../eksperyment_ff_od_zera/kod/ffexp` i dostosowane; model i pętla treningu – przeniesione ze źródła (§3). Specyfikacja implementacyjna (tabela plik → sekcja protokołu) w `kod/README.md` przed zamrożeniem.

### 8.2. Testy kodu (przed pilotem; wszystkie muszą przejść)
- T1: zgodność pętli treningu ze źródłem – w źródłowym układzie wejścia (bez D2), przy **wstrzykniętych identycznych** kolejnościach partii, złych etykietach i wagach początkowych (oba kody dostają te same tablice), wagi po 2 epokach i straty są zgodne (`allclose`, tolerancja 1e-6) dla `STD`, `CUR`, `BCM`, `ANN`, `REV` (CPU). Strumienie losowe różnią się z założenia (§6), więc porównanie bit w bit bez wstrzyknięcia nie jest możliwe.
- T2: harmonogramy – w epoce 13 `CUR` liczy formułę hamowanie z θ = 0, `ANN` ma lr × 0.5, `REV` przełącza na standard; stan Adama zachowany.
- T3: BCM – aktualizacja θ_l zgodna ze wzorem §4.1 na ręcznie policzonym przykładzie.
- T4: rozłączność walidacji, treningu i testu; runner treningowy nie ma ścieżki do testu.
- T5: identyczność strumieni między ramionami przy tym samym ziarnie.
- T6: determinizm – ten sam bieg dwa razy na GPU daje identyczne predykcje walidacyjne (FF-v5 9.1.6, R1-17).
- T7: obsługa NaN (§3.4) na sztucznie wymuszonym NaN.
- T8: wejście – norma pikseli = 1, etykieta λ = 0.1 na pozycjach 784–793, piksele nienaruszone.

### 8.3. Budżet i sprzęt (R1-15)
- Dostępne: saldo + doładowanie, łącznie ≤ 9.8 USD.
- Szacunek: P 140 biegów, A ok. 1 040, B ok. 360, B2 ok. 160, C (7 × 2 × N) = 840 (N = 60) – 1 680 (N = 120). Koszt biegu mierzony w pilocie.
- Pod-limity: P + A + B + B2 ≤ 4 USD; C ≤ 5 USD; rezerwa 0.8 USD na ewaluację.
- Przekroczenie pod-limitu → wstrzymanie i decyzja autora; cięcia w kolejności: (1) `BP` w Etapie C tylko na 30 ziarnach, (2) N w dół drabiny (przed zamrożeniem N).
- Sprzęt: wybór po benchmarku przepustowości (sieć 10 × 64 ograniczona narzutem kerneli); k procesów na GPU ustalone w benchmarku i stałe w etapie.

## 9. Kolejność i zamrożenie (R1-14)
1. Commit: protokół (wersja zamrażana), kod, testy T1–T8 przechodzące, `PROGNOZY.md` (prognozy liczbowe autora i asystenta dla H1–H4 per zbiór), `RAPORT_SZABLON.md`.
2. Benchmark i pilot (P).
3. `MINIPILOT.json` (tylko SD, ρ, koszty), `PARAMETRY.json` z N z reguły §7.6 – commit.
4. Tag `hamowanie-prereg` i push na publiczne repozytorium GitHub (za zgodą autora, D-repo). Bez zgody na publiczne repozytorium: tag lokalny + jawne zastrzeżenie w raporcie „brak zewnętrznego znacznika czasu” (§12).
5. Etapy A, B, B2, C, ewaluacja (§5.8), `final.py`, raport.
Każda zmiana po tagu → `ODSTEPSTWA.md` (data, treść, powód, przed/po ewaluacji testowej).

Pilot pokazuje kierunki efektów (na walidacji) przed zamrożeniem; dlatego z pilota do protokołu trafiają wyłącznie SD, ρ i koszty (skrypt nie wypisuje średnich różnic).

## 10. Raport
`RAPORT_SZABLON.md` (przed pilotem): strona pierwsza z werdyktami H1–H4, QMNIST, informacją o oglądanym teście (§1) i liczbą odstępstw; tabele dokładności; tabele Δ z CI per hipoteza; wrażliwości; eksploracje E1–E6 z oznaczeniem; ograniczenia §12.

## 11. Zmiany względem źródła (R1-24)
| Element | Źródło | Tu | Powód |
|---|---|---|---|
| Wejście | x/255, etykieta nadpisuje piksele 0–9 | L2 pikseli + etykieta λ = 0.1 doklejona | D2 |
| lr FF | 1e-3 stałe | strojone (Etap A) | równe strojenie (A6) |
| θ | 8 wspólne / ze sweepu 2 ziaren | strojone osobno per ramię | A6 |
| θ(0) BCM | 8 | strojone | j.w. |
| `PLW` | progi z trajektorii BCM (#17) | średnie końcowe θ_l z Etapu B + lr strojone | §4.2 |
| Test | podpróbka 3000 per ziarno, wielokrotnie | pełne 10 000, raz | A3 |
| Losowość | jeden generator numpy | strumienie per cel (§6) | parowanie, reprodukowalność |
| Ziarna | 0–149 | 60001+ (nowe pule) | A9 |
| NaN | brak reguły | §3.4 | R1-03 |

## 12. Ograniczenia znane z góry
1. Oficjalne zbiory testowe były oglądane w źródle; dla F-MNIST brak niezależnego zbioru (§1).
2. Brak ramienia 1:1 (D2): wynik negatywny nie jest przypisywalny naprawie wejścia.
3. Reżim słabej pojemności (10 × 64, 20 000 przykładów, 25 epok).
4. H3 nie obejmuje progu per warstwa strojonego niezależnie (§2); θ(0) BCM działa tylko w pierwszej epoce (§4.1).
5. Punkty przełączenia, θ = 0 w fazie hamowania i współczynnik 0.5 ustalone ze źródła (nie strojone).
6. S = 0.5 p.p. arbitralne (§7.5); PRZEWAGA na F-MNIST mało prawdopodobna przy efektach rzędu źródła.
7. Reguła brzegowa bez rozszerzenia przestrzeni (§5.5).
8. Bez zgody na publiczne repozytorium – brak zewnętrznego znacznika czasu (§9).
