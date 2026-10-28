# Protokół v6: curriculum hamowania i próg BCM w Forward-Forward – sprawdzenie konfirmacyjne

> **Wersja do zamrożenia** (tag `hamowanie-prereg` po pilocie, §9).

Autor: Sławomir Majcher. Wersja po recenzjach R1–R5 (`recenzje/`, odpowiedzi `odpowiedzi/O1.md`–`O5.md`). Decyzje: `DECYZJE_AUTORA.md` (D1–D4). Audyt źródła: `00_AUDYT.md`. Protokół jest **samowystarczalny**: wszystkie liczby, reguły i zdania raportu są tutaj; odwołania do `../eksperyment_ff_od_zera/PROTOKOL_v5.md` („FF-v5”) wskazują wyłącznie pochodzenie mechaniki.

**Źródło metody:** `/my_data/VectorLegis/eksperymenty/forward_forward/potwierdzenie_30_ziaren/` (commit `9d58048`): `siec_ff.py`, `zadanie_fashion_mnist.py`, `dane.py`; opis: `blog/forward_forward_hamowanie.md`.

## 1. Cel, zakres, status wiedzy przed badaniem

Cel: sprawdzić konfirmacyjnie cztery twierdzenia kampanii „hamowanie” w poprawionym układzie wejścia (D2), z równym budżetem strojenia, jednorazową ewaluacją testową i kontrolą błędu.

Zakres wszystkich wniosków: MLP 10 × 64 z jednoczesnym treningiem warstw FF (§3), 20 000 przykładów treningowych, 25 epok, MNIST i Fashion-MNIST.

**Wiedza autora i asystenta przed badaniem (R1-01).** Oficjalne zbiory testowe MNIST i Fashion-MNIST były w kampanii źródłowej oglądane wielokrotnie (30 ziaren × podpróbka 3000; `PLAN.md` źródła, l. 553–555); „czysty test” F-MNIST z puli treningowej też. Autor zna kierunki i przybliżone wielkości efektów źródłowych (blog). Konsekwencje:
- Ten protokół nie korzysta z żadnych wyników testowych źródła przy wyborze konfiguracji (strojenie wyłącznie na walidacji, §5).
- Hipotezy i ich kierunki pochodzą ze źródła i są zapisane jawnie (§2) wraz z prognozami liczbowymi w `PROGNOZY.md` (commit przed pilotem, §9).
- **QMNIST-50k** (50 000 cyfr spoza oficjalnego testu, nieużywane w źródle) jest dla MNIST raportowane na pierwszej stronie raportu obok wyniku głównego, z własną kategorią (§7.7c). Dla F-MNIST nie istnieje niezależny zbiór testowy; to ograniczenie (§12.1).
- **Estymand:** pseudomediana (Hodges–Lehmann) parowanych różnic dokładności na oficjalnych zbiorach testowych przy zamrożonych konfiguracjach, z biegami z awarią ocenionymi jak w §3.4 (ocena najgorszą rangą). Pseudomediana jest prawie niewrażliwa na rzadkie awarie (5% awarii przesuwa ją o ok. −0.09 p.p.), dlatego **stabilność jest osobnym, współraportowanym twierdzeniem** (§7.2). Średnia różnica i test t są analizą wrażliwości. Wnioski konfirmacyjne są warunkowe względem tych zbiorów, oglądanych (zagregowanie, nie per przykład) w kampanii źródłowej; nie były one użyte do żadnego wyboru w tym badaniu.

## 2. Hipotezy konfirmacyjne

Δ = Acc(ramię testowane) − Acc(komparator) [p.p.], parowane po ziarnie, osobno dla każdego zbioru.

| H | Ramię testowane | Komparator(y) | Kierunek ze źródła | Wielkość w źródle (MNIST / F-MNIST) |
|---|---|---|---|---|
| **H1** | `CUR` | `STD` | przewaga | +1.4 / +0.4…+0.8 |
| **H2** | `CUR` | `ANN` | przewaga | +1.44 / −0.17 (blog: na F-MNIST brak różnicy) |
| **H3** | `BCM` | `STD`, `PLW` (per zbiór) | przewaga | vs stały θ: +1.49 / +0.68; vs stały próg per warstwa (#17 źródła): +0.72 [+0.35; +1.10] / +0.48 [+0.05; +0.91]; nierozstrzygnięty w źródle był tylko #19 (harmonogram progu) |
| **H4** | `REV` | `STD` | gorszość | −2.0 / −0.4 |

**Przewidywania asystenta przed badaniem (per zbiór, R1-19, R1-20):**
- H2: na F-MNIST oczekiwana RÓWNOWAŻNOŚĆ albo NIEROZSTRZYGNIĘTY, więc łączny werdykt H2 najpewniej w klasie N (NIESPÓJNY-*), zgodnie ze źródłem.
- H3: po normalizacji wejścia (D2) asymetria skali warstwy 0 maleje, więc oczekiwana przewaga `BCM` jest mała. Przewidywanie „brak praktycznej przewagi” potwierdza tylko kategoria klasy R; klasa N oznacza brak rozstrzygnięcia.

**Sprawdzalność przewidywań równoważności (R3-10).** Przy N z reguły §7.6 moc wykazania **łącznej** klasy R (oba zbiory) przy Δ = 0 wynosi ok. 0.07–0.27 dla σ_d ≈ 0.7–1.0 (σ planistyczne = 1.29·σ_d; obliczenia R3 dla testu t, test rangowy nieco niżej); przy σ_d ≈ 0.4 ok. 0.98. **Per zbiór** (np. F-MNIST w H2) moc R jest wyższa: przy σ_d ≈ 0.9 ok. 0.5 (N = 30) – 0.9 (N = 60) (R5). Wiążące liczby: tabela mocy per zbiór i łącznie w `PARAMETRY.json` (§7.6). Przewidywania „brak różnicy” dla H2 (F-MNIST) i H3 są więc praktycznie nie do potwierdzenia; najbardziej prawdopodobny wynik przy ich prawdziwości to klasa N.

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
- Piksele: x/255, następnie L2 do normy 1 (784 wymiary): x / max(‖x‖₂, 1e-12).
- Etykieta: one-hot o wartości **λ = 0.1** doklejona (wejście 794). Brak normalizacji całego wektora; norma wejścia = √1.01.
- Uzasadnienie λ (R1-10): w źródle stosunek normy etykiety (1.0) do średniej normy obrazu x/255 (MNIST 9.21, F-MNIST 12.15; zmierzone w R2 na zbiorach treningowych) wynosił 0.109 / 0.082; λ = 0.1 zachowuje ten stosunek. W źródle stosunek zmieniał się per przykład, tu jest stały (§11). Wartość nie jest strojona.
- Pozostała asymetria: normalizacja wyrównuje normę wejścia, nie skalę preaktywacji ani tempa uczenia. Warstwa 0 dostaje gęsty wektor (norma L1 średnio 10.8 MNIST / 17.7 F-MNIST) wobec ≤ 8 dla wejścia warstw ukrytych; dobroć startowa warstwy 0 ok. 0.027 vs ok. 0.3 w warstwach ukrytych. Progi BCM mogą się więc nadal różnić między warstwami – to przedmiot E1 (§12.9).
- Negatyw: ta sama część pikselowa, zła etykieta (przesunięcie o 1…9 mod 10).

### 3.3. Predykcja
- **Odczyt główny** (jak w źródle): dla każdej z 10 etykiet suma dobroci wszystkich 10 warstw, formułą dobroci obowiązującą w danym ramieniu na końcu treningu (`CUR`: hamowanie; pozostałe: standard); argmax.
- Odczyty eksploracyjne (E3, R1-11), liczone w tym samym poleceniu testowym: (a) suma bez warstwy 0; (b) odczyt krzyżowy – formuła standard dla `CUR` i formuła hamowanie dla `STD`, `ANN`, `REV`.

### 3.4. NaN i rozjechanie treningu (R1-03, R1-25)
- Bieg, w którym strata którejkolwiek warstwy lub dobroć stanie się nieskończona/NaN, zostaje zatrzymany i oznaczony `nan = true`. **Nie ma cofania do wcześniejszej epoki** (to byłoby wczesne zatrzymanie tylko dla biegów rozbieżnych).
- **Dokładność biegu z NaN w każdym etapie (P, A, B, B2, C) i każdym odczycie** (walidacja, test, QMNIST, E3) = **udział klasy 0** w danym zbiorze, bo predykcja biegu z NaN na każdym przykładzie to klasa 0 (MNIST-test 9.80%, F-MNIST-test 10.00%, QMNIST-50k 9.94%, walidacja ok. 9.9% ± 0.3). Uzasadnienie: niestabilność jest kosztem metody i nie może być usuwana z analizy ani maskowana; w teście rangowym (§7.1) taki bieg to jedna najgorsza ranga, a nie weto.
- NaN/nieskończoność dobroci przy ewaluacji (bieg bez NaN w treningu): przykłady z niezdefiniowaną dobrocią którejkolwiek etykiety liczone jako błędne.
- Wrażliwość: (b) test t (średnia różnic) na tych samych danych i (d) test główny bez par z awarią (§7.2). Wcześniejsze wersje (v3: reguła „t ≠ Wilcoxon → N”; v4: test t z NaN jako głównym) tworzyły weto – jedna wartość odstająca rzędu −87 p.p. uniemożliwiała jakiekolwiek rozstrzygnięcie testem opartym na średniej (R3-01, R4-01). Test rangowy jako **jedyny** test główny, niezależnie od obecności NaN, usuwa ten problem bez warunkowego wyboru testu.
- Ta sama zasada obejmuje biegi bez NaN, których trening zapadł się do poziomu losowego – test rangowy traktuje je identycznie. Awarie ramienia ocenia osobne twierdzenie o stabilności (§7.2).
- **Symulacja kontrolna** (`kod/sym_nan.py`, wynik `kod/sym_nan_wynik.txt`, commit przed R5): test t przy jednym NaN daje klasę N z prawdopodobieństwem 1.00 dla N = 30, 60, 120 (weto); test rangowy przy jednym NaN w ramieniu testowanym i Δ = 0.75 daje klasę P z prawdopodobieństwem 0.84 (N = 30) i 1.00 (N = 60, 120); rozmiar testu sup(0) (odrzucenia niezależnie od kategorii) wynosi ok. 0.006 dla N = 30–120 (R5, Tab. 1; nominalnie α′/2 = 0.00625); odsetek klasy P jest niższy, bo przy dużym N odrzucenie trafia do RÓWNOWAŻNOŚCI+. **Awaria biegu** = `nan = true` albo końcowa dokładność **walidacyjna** ≤ 15% – cecha biegu ustalana przed otwarciem testu, ta sama we wszystkich odczytach i analizach. Liczba awarii per ramię i zbiór jest raportowana na pierwszej stronie.
- **Biegi z NaN nie mają użytecznego checkpointu.** Rekord biegu zawiera znacznik `nan`; w pliku hashy checkpointów biegi z NaN mają wpis `NAN <sha256 rekordu biegu> <id>`.

### 3.5. Ramię `BP` (eksploracyjne, R1-12)
- Ta sama architektura 10 × 64, ReLU, głowa `Linear(64, 10)`, entropia krzyżowa, Adam, partia 64, 25 epok, bez wczesnego zatrzymania.
- Wejście: 784 piksele po L2 (bez etykiety).
- Strojenie: 8 punktów lr log-równomiernie w [1e-4, 1e-2], 5 ziaren puli A; 3 najlepsze → 10 ziaren puli B; wybór jak §5.3.

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
θ_l(t) = 0.99·θ_l(t−1) + 0.01·½(mean g_pos + mean g_neg), aktualizacja po każdym kroku optymalizatora warstwy l, z g_pos, g_neg policzonych w tym kroku **przed** aktualizacją wag (jak źródło), θ_l(0) = θ (wspólne dla warstw). **Uwaga (R1-06):** pamięć EMA to ok. 100 kroków (≈ 1/3 epoki); po epoce wpływ θ(0) spada do ok. 4% (0.99³¹³). Strojenie θ(0) działa więc praktycznie tylko przez pierwszą epokę, a strojenie `BCM` jest w praktyce bliższe jednowymiarowemu (lr) niż dwuwymiarowemu. To asymetria wbudowana w metodę źródłową; raport podaje ją przy H3, a krajobraz strojenia (E5) jest obowiązkowy.

### 4.2. `PLW` (R1-08)
- θ_l = średnia po ziarnach puli B wartości θ_l na końcu treningu `BCM` w wybranej konfiguracji `BCM`, **osobno dla każdego zbioru** (ziarna z NaN pomijane; jeśli NaN ma > 5 z 10 ziaren – §5.3).
- lr `PLW` strojone z **tym samym budżetem co pozostałe ramiona** (§5.4): 16 punktów lr log-równomiernych w [1e-4, 1e-2] × 5 ziaren puli A, 3 najlepsze × 10 ziaren puli B (110 biegów na zbiór, jak 16 × 5 + 3 × 10 dla ramion z §5.2). Dzięki temu `PLW` ma biegi na puli B sparowane z `BCM` (σ i ρ kontrastu `BCM` − `PLW` z Etapu B). Zastrzeżenie: progi `PLW` są średnią z tych samych 10 biegów `BCM` puli B (waga każdego ziarna 1/10), więc σ_d z Etapu B może być lekko zaniżone – podobnie jak w pilocie; czynnik 1.29 częściowo to kompensuje.

## 5. Etapy i reguły wyboru

### 5.1. Mini-pilot (P)
- Ramiona `STD`, `CUR`, `ANN`, `REV`, `BCM`, `PLW` (z progami pilotowego `BCM`) × 2 zbiory × 10 ziaren puli P, konfiguracja domyślna: lr = 1e-3 (źródło), θ = 2 (wartość umowna, często używana w reimplementacjach FF; Hinton nie podaje liczby – R1-18). `BP`: lr = 1e-3.
- Wyjście: σ_d dla każdego kontrastu H1–H4 i zbioru, ρ dla H3, przepustowość, czas i koszt biegu, odsetek NaN. Tylko walidacja; metryki walidacyjne pilota zapisywane do `zaslepione/P/` (nieprzeglądane poza skryptem `choose_n`).
- `PLW` w pilocie dzieli ziarna z `BCM` (progi z tych samych biegów), co może zaniżać σ_d i zawyżać ρ; ponadto `STD`, `CUR` i `ANN` są w pilocie identyczne bit w bit przez epoki 1–12 (ta sama konfiguracja), co zaniża σ_d dla H1 i H2. Dlatego N wyznaczane jest dwukrotnie: z pilota przed tagiem i z Etapu B przed Etapem C, N = max (§7.6).

### 5.2. Etap A – strojenie
- Przestrzeń wspólna dla `STD`, `CUR`, `ANN`, `REV`, `BCM`: lr ∈ [1e-4, 1e-2], θ ∈ [0.01, 16] (oba w skali log). Dolna granica θ obniżona względem v1, bo po D2 dobroć startowa wynosi ok. 0.027 (warstwa 0) – 0.3 (warstwy ukryte) (R1-07).
- 16 punktów Sobola (scramble, ziarno stałe `seed_from("A-sobol")`), identycznych dla każdego ramienia i zbioru. 5 ziaren na punkt (pula A).
- Kryterium: średnia dokładność walidacyjna z 5 ziaren. Punkt odrzucany, jeśli którykolwiek bieg ma NaN albo dokładność walidacyjną ≤ 15%.
- **Do Etapu B przechodzą 3 najlepsze nieodrzucone punkty** danego ramienia i zbioru (R1-07; zastępuje uśrednianie z sąsiadami). Przy 1–2 nieodrzuconych punktach przechodzą wszystkie nieodrzucone.
- Kolejność biegów w etapie permutowana ziarnem `seed_from("<etap>-order")` (osobnym dla każdego etapu).

### 5.3. Etap B – wybór konfiguracji
- Dla każdego ramienia i zbioru 3 kandydaci × 10 ziaren (pula B). Wybór: najwyższa średnia walidacyjna (biegi z NaN oceniane jak w §3.4); dokładna równość średnich – mniejszy lr. Kandydat z awarią (NaN albo walidacja ≤ 15%, §3.4) w > 5 z 10 ziaren jest odrzucany (dowolne ramię); jeśli odrzuceni wszyscy – §5.6.
- Z biegów `BCM` w wybranej konfiguracji: θ_l końcowe → `PLW` (§4.2).
- Wyjście: `ETAP_B_WYBOR.json` – **wyłącznie** wybrane konfiguracje i θ_l `PLW` (bez średnich); średnie walidacyjne kandydatów zapisywane do `zaslepione/B/` (otwierane po kroku 3 §5.8, do raportu). Skrypt wyboru czyta średnie, ale wypisuje tylko konfiguracje.

### 5.4. Etap B2 – `PLW` (po wyborze `BCM`) i `BP`
- `PLW`: progi z §4.2 (z biegów `BCM` Etapu B); A-PLW: 16 lr × 5 ziaren (pula A); B-PLW: 3 najlepsze × 10 ziaren (pula B); wybór jak §5.3.
- `BP`: jak §3.5 (A-BP może iść równolegle z Etapem A).
- Wyjście: uzupełnione `ETAP_B_WYBOR.json`, `SD_ETAP_B.json` (jedyny plik z σ_d i ρ z Etapu B; tylko SD różnic i ρ, §7.6), N_B – commit i push **przed** Etapem C.

### 5.5. Reguła brzegowa (R1-07)
Jeśli wybrana konfiguracja któregokolwiek ramienia leży w skrajnych 10% zakresu (log) lr lub θ: raport to podaje jako ograniczenie; **brak rozszerzenia przestrzeni** (budżet). Krajobraz E5 pokazuje, czy optimum mogło leżeć poza zakresem.

### 5.6. Sytuacje awaryjne
- Wszystkie 16 punktów ramienia odrzucone w Etapie A: kryterium odrzucenia łagodzone do „mniej niż 2 z 5 ziaren z NaN/≤ 15%”; jeśli nadal brak – ramię uznane za nietrenowalne w tym układzie, hipotezy z jego udziałem otrzymują kategorię NIEROZSTRZYGNIĘTY z dopiskiem „ramię nietrenowalne”.
- Wszyscy kandydaci ramienia odrzuceni w Etapie B (§5.3): ramię nietrenowalne (j.w.). Jeśli nietrenowalne jest `BCM`, `PLW` nie ma progów i nie jest uruchamiane; H3 = NIEROZSTRZYGNIĘTY z dopiskiem „`BCM` nietrenowalne”.
- Reguły odrzucenia (§5.2) i przejścia top-3 (§5.3) stosują się identycznie do A-PLW, B-PLW, A-BP i B-BP.
- Awaria instancji: biegi są atomowe (zapis po zakończeniu); runner wznawia brakujące. **W obrębie etapu** zmiana typu GPU jest zabroniona; nowa instancja tego samego typu wymaga T6 przed startem. **Między etapami** zmiana typu GPU jest dozwolona po T6 na nowym typie, z wpisem w `ODSTEPSTWA.md`. Zmniejszenie k przy braku pamięci (OOM) jest dozwolone w każdej chwili z wpisem w `ODSTEPSTWA.md` (k nie wpływa na wynik biegu; T6).

### 5.7. Etap C – konfirmacja
- 6 ramion FF + `BP` × 2 zbiory × N ziaren (pula C), konfiguracje zamrożone w `ETAP_B_WYBOR.json`. Kolejność biegów permutowana ziarnem `seed_from("C-order")` (każdy etap ma własne ziarno kolejności, §5.2).
- Etap C może być podzielony na kilka instancji **tego samego typu GPU** (kolejne części permutowanego planu), każda po T6; wpis w `ODSTEPSTWA.md`.
- Metryki walidacyjne → `zaslepione/` (nieotwierane do kroku 2 §5.8).

### 5.8. Ewaluacja końcowa (R1-16)
1. Po zakończeniu Etapu C: hashe SHA-256 wszystkich checkpointów liczone **na instancji**, plik pobrany, commit i push na GitHub.
2. **Dopiero potem** dane testowe (MNIST-test, F-MNIST-test, QMNIST) są kopiowane na instancję (wcześniej ich tam nie ma; runner treningowy nie ma ścieżki do testu).
3. Jedno polecenie na instancji: weryfikacja hashy, odczyt główny i odczyty eksploracyjne E3, QMNIST; zapis predykcji per przykład.
4. Pobranie predykcji, weryfikacja hashy plików predykcji, commit.
5. Zniszczenie instancji.
- Awaria instancji między krokami 1 a 4: checkpointy są synchronizowane lokalnie na bieżąco (rsync + weryfikacja hashy); ewaluacja może zostać powtórzona na nowej instancji tego samego typu z tymi samymi hashami – z wpisem w `ODSTEPSTWA.md`.
- Etap C na kilku instancjach: hashe checkpointów liczone na każdej instancji przed jej zniszczeniem; ewaluacja na jednej instancji po zebraniu wszystkich checkpointów (zweryfikowanych hashami).
- **Powtórki przed krokiem 3** wyłącznie po błędzie kodu: błąd kodu = niezgodność implementacji z §3–§6 wykazana testem, który nie przechodzi na kodzie pierwotnym i przechodzi na poprawionym; test, diff, lista biegów (wyznaczona automatycznie z diffu; w razie wątpliwości cały etap) i powód commitowane przed powtórką; biegi pierwotne zachowane, oceniane w kroku 3 i raportowane jako wrażliwość.
- **Powtórki po kroku 3** – tylko jako analiza wrażliwości; wynik główny pozostaje pierwotny.
- **Zakaz ewaluacji testowej częściowego Etapu C** (jak FF-v5 §5.6 pkt 6). Jeśli Etap C nie zostanie ukończony (budżet, awaria), raport ma status „nieukończone”, a zbiór testowy nie jest otwierany. Brak wyjątków.

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
- Dane: pliki źródłowe z oficjalnych źródeł, hashe w `dane/SHA256SUMS` (weryfikowane przy każdym wczytaniu).
- Determinizm: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, float32, TF32 wyłączone, `torch.set_num_threads(1)` na proces; wersje: obraz `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (torch 2.5.1, Python 3.11) – zapisywane w każdym rekordzie biegu.

## 7. Analiza

### 7.1. Test elementarny
- d_i = Acc_B(s_i) − Acc_K(s_i) [p.p.], i = 1…N.
- α = 0.05, Bonferroni na 4 hipotezy: **α′ = 0.0125**.
- **Test elementarny (jedyny test główny): test Wilcoxona rangowany ze znakiem na d − δ** dla δ ∈ {−S, 0, +S}, jednostronnie; zera metodą Pratta; `scipy.stats.wilcoxon(…, zero_method="pratt", method="auto")` (dokładny rozkład dla N ≤ 50 bez wiązań i zer, w pozostałych przypadkach przybliżenie normalne z poprawką na wiązania). Kategorie z wartości p (inwersja testu = przedział Hodgesa–Lehmanna; HL raportowany jako estymata punktowa z przedziałem (1 − α′)). **Brak testu normalności i brak warunkowego wyboru testu.** Test t – wrażliwość (§7.7b).
- Asymptotyczna efektywność względna testu rangowego wobec testu t przy normalnych różnicach: 0.955 (ok. 5% więcej ziaren dla tej samej mocy). Test zakłada symetrię d wokół pseudomediany; przy ≥ 10% awarii w jednym ramieniu i N = 30 rozmiar ok. 1.4 × nominał (R5; §12).
- **Wartości d w liczbach całkowitych:** d = (liczba poprawnych odpowiedzi B) − (liczba poprawnych K) na danym zbiorze; S w tych jednostkach = 50 (10 000 przykładów) albo 250 (QMNIST-50k); d − δ liczone dokładnie. Raport przelicza na p.p.
- Metoda jawnie: `scipy.stats.wilcoxon(d − δ, zero_method="pratt", alternative=…, method="exact")` przy N ≤ 50 i braku zer w d − δ, w pozostałych przypadkach `method="approx", correction=False`; scipy 1.14.1 (przypięte w `kod/README.md` i w rekordzie analizy). Decyzje wyłącznie z wartości p; przedział HL jest opisowy.
- Kontrola błędu dla każdego typu twierdzenia: klasa P (przewaga) na poziomie α′/2 (jednostronnie), klasa G (gorszość) na α′/2, klasa R (równoważność, TOST) na α′; **stabilność – osobny typ twierdzenia, α′/2 na kierunek** (§7.2). W całym badaniu ≤ 0.05 dla każdego typu twierdzenia osobno.

### 7.2. Kategorie na jednym zbiorze
Oznaczenia: p⁺(δ) – jednostronne p testu Wilcoxona dla H₀: pseudomediana(d − δ) ≤ 0; p⁻(δ) – dla H₀: pseudomediana(d − δ) ≥ 0; S – §7.5. (Decyzje z wartości p; przedziały HL wyłącznie opisowe.)

| Warunek | Definicja |
|---|---|
| sup(S) | dla każdego K: p⁺_K(S) ≤ α′/2 (IUT) |
| sup(0) | dla każdego K: p⁺_K(0) ≤ α′/2 (IUT) |
| eq | dla każdego K: p⁺_K(−S) ≤ α′ i p⁻_K(S) ≤ α′ (IUT, TOST) |
| inf(S) | p⁻_K(−S); Holm na m komparatorach na poziomie α′/2; odrzucona ≥ 1 |
| inf(0) | jak inf(S) z p⁻_K(0) |

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

**Stabilność (per zbiór, osobne twierdzenie).**
- Awaria biegu = `nan = true` albo dokładność walidacyjna ≤ 15% (cecha biegu, ta sama dla wszystkich odczytów i analiz §7.7).
- Dla każdego K: dokładny jednostronny test znaków na ziarnach niezgodnych (awaria tylko w jednym ramieniu) w obu kierunkach, p_st⁻_K (ramię testowane ma więcej awarii) i p_st⁺_K.
- STAB-GORSZA: Holm po K na α′/2 dla p_st⁻ (≥ 1 odrzucona).
- STAB-LEPSZA: dla każdego K p_st⁺_K ≤ α′/2 (IUT).
- Wynik stabilności nie zmienia kategorii dokładności, z jednym wyjątkiem: klasa R jest niedopuszczalna przy STAB-GORSZA lub STAB-LEPSZA i wtedy przechodzi w N z dopiskiem „równoważność dokładności przy istotnie różnej stabilności”.
- Łącznie (§7.3): STAB-GORSZA/LEPSZA, gdy wystąpi na obu zbiorach; w przeciwnym razie „bez wykazanej różnicy stabilności”.
- Pierwsza strona raportu: werdykt dokładności i werdykt stabilności per hipoteza.
- Czułość (zapisana z góry, R5-02): test znaków wykrywa dopiero ≥ 8 (m = 1) / ≥ 9 (H3, pierwszy krok Holma) awarii niezgodnych przy braku awarii w drugim ramieniu; moc przy q = 10% w ramieniu testowanym i 1% w komparatorze ok. 0.17 (N = 60) / 0.69 (N = 120). Rzadkie, nieistotne awarie (np. 5/120) zostawiają klasę dokładności bez zmian, z obowiązkowym dopiskiem o liczbie awarii (§7.4, §12).

### 7.3. Łączenie zbiorów
| Sytuacja | Werdykt łączny |
|---|---|
| Ta sama klasa na obu zbiorach | ta klasa; kategoria łączna = słabsza z dwóch (P → PRZEWAGA tylko przy PRZEWADZE na obu; analogicznie G; R → RÓWNOWAŻNOŚĆ+, jeśli wystąpi na choć jednym) |
| P na jednym, G na drugim | NIESPÓJNY-KIERUNEK (N) |
| P na jednym i R+ o dodatnim kierunku na drugim (albo G i R+ o ujemnym) | NIESPÓJNY-ZGODNY-KIERUNEK (N) |
| Pozostałe | NIESPÓJNY-CZĘŚCIOWY (N) |

### 7.4. Zdania raportu (jedyne dopuszczalne interpretacje konfirmacyjne)

„Wyższa/niższa dokładność” w zdaniach oznacza dodatnią/ujemną pseudomedianę parowanych różnic (§1), nie różnicę średnich.

| H | P | R | G | N |
|---|---|---|---|---|
| H1 | Curriculum (standard → hamowanie) daje wyższą dokładność niż standard. | Curriculum równoważne standardowi (±S). | Curriculum daje niższą dokładność niż standard. | Porównanie curriculum ze standardem nierozstrzygnięte. |
| H2 | Curriculum daje wyższą dokładność niż standard z wyżarzaniem LR (× 0.5 od połowy). | Curriculum równoważne standardowi z wyżarzaniem LR (±S). | Curriculum daje niższą dokładność niż standard z wyżarzaniem LR. | Porównanie curriculum z wyżarzaniem LR nierozstrzygnięte. |
| H3 | Próg BCM daje wyższą dokładność niż strojony próg skalarny i niż stały próg per warstwa o wartościach z BCM (przy równym budżecie biegów strojenia; θ(0) BCM działa tylko w pierwszej epoce). | BCM równoważny obu stałym progom (±S). | Co najmniej jeden stały próg daje wyższą dokładność niż BCM. | Porównanie BCM ze stałymi progami nierozstrzygnięte. |
| H4 | Odwrócone curriculum daje wyższą dokładność niż standard. | Odwrócone curriculum równoważne standardowi (±S). | Odwrócone curriculum daje niższą dokładność niż standard. | Porównanie odwróconego curriculum ze standardem nierozstrzygnięte. |

Obowiązkowe dopiski:
- Kategorie „<S”: „różnica istotna statystycznie, mniejsza niż przyjęty próg S; jej praktyczne znaczenie nie zostało wykazane”.
- RÓWNOWAŻNOŚĆ+: „wykryto różnicę istotną statystycznie, ale praktycznie pomijalną (< S)”, z kierunkiem na każdym zbiorze i komparatorami, które go wyznaczają.
- NIESPÓJNY-KIERUNEK: „kierunek różnicy przeciwny na obu zbiorach: MNIST: {kategoria}, F-MNIST: {kategoria}”.
- NIESPÓJNY-ZGODNY-KIERUNEK: „na zbiorze {X}: {kategoria}; na zbiorze {Y}: różnica istotna statystycznie w tym samym kierunku, praktycznie pomijalna (< S, równoważność wykazana)”.
- NIESPÓJNY-CZĘŚCIOWY: „na zbiorze {X}: {kategoria}; na zbiorze {Y}: {kategoria}; kategorie należą do różnych klas, więc werdykt łączny jest nierozstrzygnięty”.
- Klasa R (RÓWNOWAŻNOŚĆ, RÓWNOWAŻNOŚĆ+): „równoważność w granicach ±S = ±0.5 p.p. nie wyklucza efektów rzędu 0.4–0.8 p.p. raportowanych w źródle”.
- Moc równoważności < 0.5 (z §7.6, przy N z Etapu C): przy klasie N dopisek „badanie nie miało mocy do wykazania równoważności (moc {…})”.
- STAB-GORSZA: „{ramię testowane} ulega awarii treningu (NaN lub dokładność walidacyjna ≤ 15%) istotnie częściej niż {komparator(y) z listy Holma}: {f_B} vs {f_K} z {N} ziaren”.
- STAB-LEPSZA: „{ramię testowane} ulega awarii treningu istotnie rzadziej niż każdy komparator: {f_B} vs {f_K} z {N} ziaren”.
- Brak różnicy stabilności: „bez wykazanej różnicy stabilności ({f_B} vs {f_K} awarii z {N} ziaren)” – **przy każdej hipotezie i każdym zbiorze**.
- Ramię nietrenowalne (§5.6): „{ramię} nietrenowalne w tym układzie (wszystkie konfiguracje odrzucone)”.
- QMNIST (§7.7c): kategoria na QMNIST obok kategorii MNIST; przy różnej klasie: „na MNIST werdykt zależy od zbioru testowego: oficjalny {X} (oglądany w źródle), QMNIST-50k {Y} (nieoglądany)”.
- H3 w klasie G: lista komparatorów istotnie lepszych od `BCM` **wg procedury Holma, osobno dla każdego zbioru** (nie wg przedziałów; lekcja R6-01), liczona przez `final.py`.
- H3: zawsze dopisek o §4.1 (θ(0) BCM działa tylko w pierwszej epoce).
- Awarie: przy każdej klasie, gdy któreś z ramion ma awarie: „w ramieniu {X} {k} z {N} biegów uległo awarii (NaN lub ≤ 15% walidacji) i zostało ocenione jako najgorsze”; gdy klasa testu t (wrażliwość) różni się od głównej – dopisek z obiema kategoriami.
- „Macierz H1 × H2 nie zawiera porównania `ANN` − `STD`; wnioski o samym wyżarzaniu LR nie są dopuszczalne” (opisowo: E7).
- Klasa P lub G z R bez „+” na drugim zbiorze: zdanie NIESPÓJNY-CZĘŚCIOWY.

**Odczyt łączny H1 × H2 (opisowy, bez nowego testu; klasy łączne):**

| H1 \ H2 | P | R | G | N |
|---|---|---|---|---|
| P | curriculum lepsze zarówno od standardu, jak i od wyżarzania LR | przewaga curriculum nad standardem nie została odróżniona od efektu wyżarzania LR (równoważne ±S; przy RÓWNOWAŻNOŚCI+ dopisek z kierunkiem) | curriculum lepsze od standardu, ale wyżarzanie LR lepsze od curriculum | przewaga nad standardem; porównanie z wyżarzaniem nierozstrzygnięte |
| R | curriculum równoważne standardowi, a lepsze od wyżarzania | curriculum równoważne standardowi i wyżarzaniu | curriculum równoważne standardowi, gorsze od wyżarzania | – (tylko zdania pojedyncze) |
| G | curriculum gorsze od standardu, lepsze od wyżarzania | curriculum gorsze od standardu, równoważne wyżarzaniu | curriculum gorsze od obu | – |
| N | – | – | – | – |

Komórki „–”: raport podaje wyłącznie zdania pojedyncze H1 i H2.

**Pierwsza strona raportu (R6-08):** werdykty H1–H4 (łączne i per zbiór), QMNIST, zestawienie z `PROGNOZY.md` (które prognozy autora i asystenta się nie sprawdziły), informacja o oglądanym teście (§1), liczba odstępstw.

Słowa zakazane: „dowodzi”, „przełom”, „bezprecedensowy”, „idealny”, „rozwiązuje”, „deklasuje”, „ratuje”.

### 7.5. SESOI (R1-02)
**S = 0.5 p.p. dla obu zbiorów i wszystkich hipotez.**
- Uzasadnienie (uczciwe): S to próg istotności praktycznej przyjęty arbitralnie przez autora jako „różnica, dla której warto zmienić metodę treningu małego MLP FF”. Nie wynika z danych.
- Autor zna efekty źródłowe (§2): na F-MNIST mieszczą się w 0.4–0.8 p.p., więc kategoria PRZEWAGA (> S) na F-MNIST jest mało prawdopodobna nawet przy prawdziwym efekcie; wynik w klasie P będzie zwykle PRZEWAGĄ<S. To zapisane z góry.
- Wrażliwość na S: kategorie przeliczane dla S ∈ {0.25, 1.0} (§7.7e), raportowane w tabeli wrażliwości.

### 7.6. Wybór N (R1-05, R2-01, R2-03)
- Wejście skryptu `choose_n`: **wyłącznie SD różnic** (σ_d) i ρ, nigdy średnie; skrypt nie wypisuje średnich (zasada skryptu; surowe rekordy zawierają walidację i asystent ich nie przegląda przed Etapem C poza SD – R2-11). σ_d liczone **z par bez awarii**. Model symulacji: dla ziarna s i ramienia a dokładność = μ_a + u_s + e_{s,a} (wspólny efekt ziarna u_s i szum ramienia dobrane tak, by odtworzyć σ_d i ρ); niezależnie każdy bieg ulega awarii z prawdopodobieństwem q_a (per ramię i zbiór; dla N_P z pilota, dla N_B z wybranej konfiguracji Etapu B łącznie z pilotem; minimum 1%), a bieg z awarią ma dokładność = udział klasy 0 (para z awarią obu ramion daje d = 0); dokładności zaokrąglane do liczby poprawnych odpowiedzi (siatka jak w §7.1); stosowana jest pełna procedura §7.2–§7.3 (test główny, Holm/IUT, łączenie zbiorów; stabilność raportowana obok); 10 000 powtórzeń; obliczenia lokalnie (CPU), nie na Vast; σ_d × 1.29 (niepewność σ, jak FF-v5); ρ (H3) = min(ρ̂, 0.5), ≥ 0. Symulacja ma **stałe ziarno** `seed_from("choose_n")`.
- Cel mocy (symulacja jak wyżej, różnice normalne o SD σ_d per zbiór):
  - H1, H2, H3: P(łączna klasa P) ≥ 0.8 przy prawdziwym **Δ = 1.5·S = 0.75 p.p.** na obu zbiorach;
  - H4: P(łączna klasa G) ≥ 0.8 przy Δ = −0.75 p.p. na obu zbiorach;
  - moc równoważności (klasa R przy Δ = 0) raportowana, bez wymogu;
  - `PARAMETRY.json` zawiera tabelę mocy (klasy P, R, G) przy Δ ∈ {0, S, 1.5·S, 2·S} dla wybranego N; moc R w raporcie liczona z σ planistycznego, nie obserwowanego.
- **Drabina:** N ∈ {30, 45, 60, 90, 120}; najmniejsze N spełniające cel dla wszystkich hipotez; jeśli żadne – N = 120 z raportowaną mocą.
- **Dwa wyznaczenia:** N_P z pilota (przed tagiem, w `PARAMETRY.json`); N_B z Etapu B i B2 (σ = max(pilot, B) per kontrast i zbiór; dla `BCM` − `PLW` z B-PLW). **N = max(N_P, N_B)**, commit i push przed Etapem C. N tylko w górę; N nie podlega cięciom budżetowym (§8.3).

### 7.7. Analizy wrażliwości
(a) bootstrap 2D: 10 000 losowań; w każdym ziarna losowane ze zwracaniem, a w obrębie każdego wylosowanego ziarna przykłady testowe losowane ze zwracaniem (te same indeksy dla obu ramion – parowanie); statystyka: estymator HL (ta sama co w teście głównym); kategorie z przedziałów percentylowych (1 − α′) i (1 − 2α′) oraz jednostronnych p bootstrapu (Holm w H3); (b) test t (średnia różnic): kategorie §7.2 z przedziałów t (1 − α′), (1 − 2α′) i jednostronnych p (Holm w H3); (c) QMNIST-50k (MNIST) – kategoria §7.2 (test główny) liczona na QMNIST dla modeli MNIST, na pierwszej stronie raportu (§7.4); (d) bez par z awarią (§7.2; reguła stabilności nie ma tu zastosowania); (e) S ∈ {0.25, 1.0}.

### 7.8. Eksploracje (bez korekty; oznaczane)
- E1: trajektorie θ_l `BCM` (warstwa × epoka; zapis co epokę w rekordzie biegu).
- E2: luka `BP` − najlepsze ramię FF; „najlepsze” wybierane wg średniej walidacyjnej Etapu C (`zaslepione/` po kroku 2 §5.8).
- E3: odczyty (a) bez warstwy 0, (b) krzyżowy (§3.3).
- E4: norma zmiany wag na epokę per warstwa (`CUR`, `ANN`, `STD`; zapis w rekordzie biegu) – czy przełączenie zmienia krok.
- E5: krajobraz strojenia Etapu A (mapy lr × θ per ramię i zbiór) – **obowiązkowy**.
- E6: dokładność walidacyjna co epokę (krzywe) dla wszystkich ramion – zapis w rekordzie biegu.
- E7: `ANN` − `STD` (opisowo, 95% CI HL).

### 7.9. Skrypt analizy (`final.py`)
Liczy każdą liczbę, tabelę i rysunek z `RAPORT_SZABLON.md` (w tym: listę Holma dla H3, odsetki NaN, E1–E6, wrażliwości a–e, rysunki). Autotest przed zamrożeniem na danych syntetycznych ze znanymi werdyktami: scenariusze (i) wszystkie P, (ii) wszystkie R, (iii) G w H3 z jednym komparatorem, (iv) NIESPÓJNY-KIERUNEK, (v) NIESPÓJNY-ZGODNY-KIERUNEK, (vi) `final.py` i implementacja referencyjna dają **identyczne klasy** na zbiorach danych wygenerowanych generatorem z `kod/sym_nan.py` (te same ziarna), (vii) H3 z m = 2: lista Holma (jeden i dwa komparatory istotne), (viii) rozmiar: przy Δ = 0 (oba komparatory w H3) odsetek odrzuceń p⁺(0) ≤ α′/2 oraz p⁻(0) ≤ α′/2 (niezależnie od kategorii) ≤ α′/2 + 3·SE, dla N = 30 i 120, na siatce całkowitej; (ix) przy Δ = −S i Δ = +S (w H3: pierwszy komparator ±S, drugi 0) odsetek klasy R ≤ α′ + 3·SE; (x) rozmiar reguły stabilności przy q_B = q_K = 5%, N = 120: P(STAB-GORSZA) ≤ α′/2 + 3·SE; (xi) deterministycznie: 8 : 0 → STAB-GORSZA (m = 1); w H3 8 : 0 → brak, 9 : 0 → STAB-GORSZA; klasa R + STAB-* → N; (xii) para z awarią obu ramion (d = 0) i d = ±S dokładnie; (xiii) P + R bez „+” → NIESPÓJNY-CZĘŚCIOWY. Symulacje: 4 000 powtórzeń, ziarno stałe `seed_from("autotest")`, SE = √(α(1 − α)/4000). **Implementacja referencyjna:** niezależnie napisana druga implementacja logiki kategorii (osobny moduł, bezpośrednio z definicji §7.2–7.3, `scipy.stats`), bez współdzielenia kodu z `final.py`; obie muszą dać identyczne kategorie na wszystkich scenariuszach i na 1 000 losowych zestawów danych.

## 8. Wykonanie

### 8.1. Kod (R1-13)
Runner, zapis atomowy, synchronizacja z weryfikacją hashy, monitor, ewaluacja i analiza – przeniesione z `../eksperyment_ff_od_zera/kod/ffexp` i dostosowane; model i pętla treningu – przeniesione ze źródła (§3). Specyfikacja implementacyjna (tabela plik → sekcja protokołu) w `kod/README.md` przed zamrożeniem.

### 8.2. Testy kodu (przed pilotem; wszystkie muszą przejść)
- T1: zgodność pętli treningu ze źródłem – w źródłowym układzie wejścia (bez D2), przy **wstrzykniętych identycznych** kolejnościach partii, złych etykietach i wagach początkowych (oba kody dostają te same tablice), wagi i straty po 4 epokach z **przełączeniem po epoce 2** (skrócony harmonogram testowy) są zgodne (`torch.allclose`, atol = 1e-6, rtol = 1e-5) dla `STD`, `CUR`, `ANN`, `REV`, `BCM` i `PLW` (CPU). Źródło: zamrożona kopia plików źródłowych w `kod/zrodlo/` z hashami w `kod/zrodlo/SHA256SUMS`. Strumienie losowe różnią się z założenia (§6), więc porównanie bit w bit bez wstrzyknięcia nie jest możliwe.
- T2: harmonogramy – w epoce 13 `CUR` liczy formułę hamowanie z θ = 0, `ANN` ma lr × 0.5, `REV` przełącza na standard; stan Adama zachowany.
- T3: BCM – aktualizacja θ_l zgodna ze wzorem §4.1 na ręcznie policzonym przykładzie.
- T4: rozłączność walidacji, treningu i testu; runner treningowy nie ma ścieżki do testu.
- T5: identyczność strumieni między ramionami przy tym samym ziarnie.
- T6: determinizm – ten sam bieg dwa razy na GPU daje identyczne predykcje walidacyjne (FF-v5 9.1.6, R1-17).
- T7: obsługa NaN (§3.4) na sztucznie wymuszonym NaN, także w ewaluacji (§5.8 krok 3): predykcja klasa 0, wpis `NAN <sha256 rekordu> <id>` weryfikowany.
- T8: wejście – norma pikseli = 1 (±1e-6), etykieta λ = 0.1 na pozycjach 784–793, piksele nienaruszone; runner w trybie etapów P/A/B/C odrzuca źródłowy tryb wejścia (używany tylko w T1).
- T6 obejmuje też niezależność od k: ten sam bieg uruchomiony samodzielnie i jako jeden z k równoległych daje identyczne predykcje.
- Monitor postępu pokazuje wyłącznie liczbę ukończonych biegów, błędów i NaN – bez metryk.

### 8.3. Budżet i sprzęt (R1-15, R2-07)
- Dostępne: łącznie ≤ 9.8 USD (twardy limit D3; obecne saldo + jednorazowe doładowanie do tej kwoty).
- Liczba biegów (oba zbiory):
  - P: 7 ramion × 10 ziaren × 2 = 140;
  - A: 5 ramion × 16 × 5 × 2 = 800; A-BP: 8 × 5 × 2 = 80;
  - B: 5 ramion × 3 × 10 × 2 = 300; B-BP: 3 × 10 × 2 = 60;
  - B2 (`PLW`): 16 × 5 × 2 + 3 × 10 × 2 = 220;
  - razem przed C: 1 600;
  - C: 7 ramion × N × 2 = 420 (N = 30) … 1 680 (N = 120).
- Koszt biegu (z logowaniem E1/E4/E6) mierzony w pilocie; pod-limity: P + A + B + B2 ≤ 4 USD; C ≤ 5 USD; rezerwa 0.8 USD na ewaluację, powtórki po błędzie kodu i awarie.
- **Prognoza przed Etapem C:** koszt C = (średni koszt biegu w Etapie B; przy zmianie typu GPU między B a C – koszt zmierzony w benchmarku/T6 na nowym typie) × 14·N. Jeśli prognoza > (pozostałe środki − 0.8 USD): `BP` usuwane z Etapu C (jedyne dopuszczalne cięcie; prognoza 12·N). Jeśli nadal za mało: wstrzymanie; zwiększenie limitu wymaga zmiany D3 przez autora i wpisu w `ODSTEPSTWA.md` przed Etapem C. N nie jest cięte.
- Przekroczenie pod-limitu w trakcie etapu → wstrzymanie i decyzja autora (j.w.).
- Sprzęt: wybór po benchmarku przepustowości; k procesów na GPU ustalone w benchmarku.

## 9. Kolejność i zamrożenie (R1-14)
1. Commit: protokół (wersja zamrażana), kod, testy T1–T8 przechodzące, `SHA256SUMS` (protokół, `PROGNOZY.md`, kod, `RAPORT_SZABLON.md`), `PROGNOZY.md` (prognozy liczbowe autora i asystenta dla H1–H4 per zbiór), `RAPORT_SZABLON.md`.
2. Benchmark i pilot (P).
3. `MINIPILOT.json` (jedyny plik z σ_d i ρ pilota; tylko SD, ρ, koszty; metryki pilota w `zaslepione/P/`), `PARAMETRY.json` z N z reguły §7.6 – commit.
4. Tag `hamowanie-prereg` i push na publiczne repozytorium GitHub (zgoda autora: D4). Bez zgody na publiczne repozytorium: tag lokalny + jawne zastrzeżenie w raporcie „brak zewnętrznego znacznika czasu” (§12).
5. Etapy A, B, B2; `SD_ETAP_B.json` i N = max(N_P, N_B) – commit i push; Etap C; ewaluacja (§5.8); `final.py`; raport.
Każda zmiana po tagu → `ODSTEPSTWA.md` (data, treść, powód, przed/po ewaluacji testowej).

Pilot pokazuje kierunki efektów (na walidacji) przed zamrożeniem; dlatego z pilota do protokołu trafiają wyłącznie SD, ρ i koszty (skrypt nie wypisuje średnich różnic).

## 10. Raport
`RAPORT_SZABLON.md` (przed pilotem): strona pierwsza z werdyktami H1–H4, QMNIST, informacją o oglądanym teście (§1) i liczbą odstępstw; tabele dokładności; tabele HL z przedziałem (1 − α′) oraz średnich różnic z CI t (wrażliwość) per hipoteza; werdykt stabilności per hipoteza; wrażliwości; eksploracje E1–E6 z oznaczeniem; ograniczenia §12.

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
| Stosunek etykieta/obraz | zmienny per przykład (norma obrazu zmienna) | stały 0.1 | D2, §3.2 |
| NaN | brak reguły | §3.4 | R1-03 |

## 12. Ograniczenia znane z góry
1. Oficjalne zbiory testowe były oglądane w źródle; dla F-MNIST brak niezależnego zbioru (§1).
2. Brak ramienia 1:1 (D2): wynik negatywny nie jest przypisywalny naprawie wejścia.
3. Reżim słabej pojemności (10 × 64, 20 000 przykładów, 25 epok).
4. H3 nie obejmuje progu per warstwa strojonego niezależnie (§2); θ(0) BCM działa tylko w pierwszej epoce (§4.1).
5. Punkty przełączenia, θ = 0 w fazie hamowania i współczynnik 0.5 ustalone ze źródła (nie strojone).
6. S = 0.5 p.p. arbitralne (§7.5); PRZEWAGA na F-MNIST mało prawdopodobna przy efektach rzędu źródła.
7. Reguła brzegowa bez rozszerzenia przestrzeni (§5.5).
8. Bez zgody na publiczne repozytorium (D4) – brak zewnętrznego znacznika czasu dla protokołu i dla N_B (§9).
9. Normalizacja wejścia wyrównuje normę, nie skalę preaktywacji ani tempo uczenia warstwy 0 (§3.2).
10. Strojenie `BCM` jest w praktyce bliższe jednowymiarowemu (§4.1), przy równym budżecie biegów; asymetria działa **na korzyść `BCM`** (ten sam budżet na mniej efektywnych wymiarów). Budżet biegów wszystkich ramion (w tym `PLW`) jest równy.
11. Test rangowy zakłada symetrię d wokół pseudomediany; przy ≥ 10% awarii w jednym ramieniu i N = 30 rozmiar ok. 1.4 × nominał. Reguła stabilności ma niską moc przy rzadkich awariach (§7.2); metoda ulegająca awarii w kilku procentach biegów może otrzymać klasę R (z dopiskiem o awariach).
