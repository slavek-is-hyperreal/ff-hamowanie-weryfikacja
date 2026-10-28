# K1 – recenzja kodu `kod/hamexp/` względem PROTOKOL_v6 (§3–§9)

Data: 2026-10-28. Recenzent: asystent (rola: czepialski recenzent statystyki i PyTorch). Zakres: `ff.py`, `train.py`, `runner.py`, `plans.py`, `wybor_n.py`, `analysis.py`, `final.py`, `evaluate_test.py`, `selftest.py`, `ref_kategorie.py`, `gpu_check.py`, `data.py`, `streams.py`, `monitor.py`, `sync_verify.py`, `kod/tests/`, `kod/vast/*.sh`, `RAPORT_SZABLON.md`; porównanie z `kod/zrodlo/` (tylko odczyt). Kodu nie edytowano. Pliki robocze: `/tmp/claude-1000/k1/` (nic nie zapisano w `wyniki/`, żadnego treningu na ziarnach z pul P/A/B/C).

## 0. Co uruchomiono i co sprawdzono pozytywnie

- `pytest tests/`: 13 przeszło, 1 pominięty (T6 – brak GPU). `python -m hamexp.selftest`: 28/28 OK.
- Wektorowy Wilcoxon `wilcoxon_p_vec` względem `scipy.stats.wilcoxon` 1.14.1 (moja próba: N ∈ {30, 45, 60, 120}, SD ∈ {3, 20, 90}, zera w co piątym wierszu, obie alternatywy, 300 wierszy na przypadek): maks. |różnica| = **0**. Ścieżki exact (z wiązaniami, floor/ceil jak scipy) i approx (Pratt, poprawka na wiązania, bez poprawki ciągłości) są zgodne ze scipy.
- **Zgodne z protokołem i źródłem (sprawdzone czytaniem kodu i T1):**
  - sieć 10 × 64 z Adamem per warstwa i jednoczesnym treningiem na tej samej parze partii; `normalize` z eps 1e-4; detach; softplus wokół progu (`ff.py:93–113` ↔ `siec_ff.py:292–350`);
  - BCM aktualizowane po `opt.step()` z g policzonym przed krokiem; strata używa θ_l sprzed aktualizacji, θ_l(0) = θ (§4.1);
  - przełączenie 12/13: `phase()` z `epoch >= 12` (0-based) daje epoki 13–25. To zgadza się ze źródłem (`epoki_total // 2` = 12), a stan Adama nie jest resetowany;
  - harmonogramy `CUR`, `ANN`, `REV`, `BCM`, `PLW` (§4); ostatnia niepełna partia jest używana;
  - wejście L2 + λ = 0.1 doklejone (§3.2, T8); podział 10 000 / 20 000 z 50 000 (§6); strumienie (s, 1), (s, 2, ep), (s, 3, ep), (s, 4), bez indeksu warstwy, wspólne dla ramion; pule ziaren;
  - permutacja kolejności per etap (`shuffled(jobs, "<etap>")`);
  - predykcja: argmax sumy dobroci, odczyty `main`, `no_l0`, `cross`; dobroć nieskończona przy ewaluacji daje −1, czyli błąd (§3.3–§3.4);
  - biegi z NaN: brak cofania, predykcja klasy 0 w walidacji (runner) i w teście/QMNIST/E3 (`evaluate_test`), wpis `NAN <sha256 rekordu> <id>` (§3.4);
  - BP (§3.5); top-3, przypadek 1–2 punktów, złagodzenie „< 2 z 5” tylko gdy odrzucono wszystkie 16 punktów, kryterium „> 5 z 10” w B, remis → mniejszy lr (§5.2–§5.6); progi `PLW` per zbiór z puli B wybranego `BCM` z pominięciem NaN; równy budżet 16 × 5 + 3 × 10 dla `PLW`;
  - kategorie §7.2 z kolejnością 1–7, Holm dla inf, IUT dla sup/eq; stabilność (Holm dla STAB-GORSZA, IUT dla STAB-LEPSZA, R + STAB → N); łączenie §7.3; S w jednostkach 50 / 250 (25 / 100 we wrażliwości e);
  - `wybor_n`: σ z par bez awarii, × 1.29, ρ ∈ [0, 0.5], q ≥ 1%, d = 0 przy awarii obu ramion, siatka całkowita, cel 1.5·S = 75, drabina, N = max, stałe ziarno `seed_from("choose_n")`. **Skrypt nie wypisuje ani nie zapisuje średnich** (tylko SD, ρ, odsetki awarii, koszty);
  - bootstrap 2D (§7.7a): losowanie ziaren ze zwracaniem, a w ziarnie multinomial po wzorcach poprawności ramion. Jest to dokładnie równoważne losowaniu przykładów z tymi samymi indeksami dla wszystkich ramion. Statystyka HL, Holm w H3 – poprawnie;
  - monitor i synchronizacja wypisują wyłącznie liczności; dane testowe są wykluczone z `wyslij.sh`; `wyslij_test.sh` sprawdza, że plik hashy jest na GitHubie.

Rdzeń treningowy jest więc wierny protokołowi i źródłu. Uwagi poniżej dotyczą głównie warstwy analizy i kontroli procesu – tam, gdzie poprzednie badanie (R6) miało błędy.

## 1. Uwagi

### Krytyczne

**K1-01 (krytyczna) – nic nie wymusza zakazu ewaluacji i analizy niepełnego Etapu C (§5.8: „Brak wyjątków”).**
- Gdzie:
  - `evaluate_test.py:129` – `--plan` ma `default=None`, a bez planu `verify()` w ogóle nie porównuje zbioru biegów z planem (l. 59–64);
  - `evaluate_test.py:27–39` – `write_hashes` hashuje, co jest w `C/runs/`. Pliki `.error` i brakujące biegi są pomijane bez ostrzeżenia;
  - `verify()` nie sprawdza też, czy plik hashy nie zawiera wpisów nadmiarowych (biegu usuniętego po zhashowaniu);
  - `final.py:65` – analiza bierze przecięcie ziaren obecnych na dysku;
  - `final.py:254` – `--params` jest wczytywane z argumentów, ale nigdzie nieużywane, więc N z `PARAMETRY.json` nie jest porównywane;
  - żaden skrypt w `vast/` ani `kod/README.md` nie podaje polecenia kroku 3 z `--plan`.
- Dowód: syntetyczny katalog z `tests/test_final_e2e.py`, z którego usunąłem 3 ziarna `CUR` (MNIST). `final.py` kończy się sukcesem i wydaje werdykt H1 na N = 7 (H3 na N = 10), bez ostrzeżenia (`/tmp/claude-1000/k1/probe2.py`, wynik `(b) H1 mnist N used: 7`).
- Poprawka:
  - `--plan` wymagane w `evaluate_test` (także przy `--write-hashes`), z porównaniem zbioru biegów z planem w obie strony i odmową przy pliku `.error`;
  - `verify()` porównuje zbiór wpisów pliku hashy z zbiorem biegów (równość);
  - `final.py` czyta `PARAMETRY.json` i sprawdza (twardy błąd) komplet: dla każdego zbioru, ramienia z `ETAP_B_WYBOR.json` z niepustym `cfg` i każdego z N ziaren C – rekord, `zaslepione/*.json` i plik testowy (`test is None` dziś daje `TypeError` dopiero w `counts`).

### Istotne

**K1-02 (istotna) – ramię nietrenowalne (§5.6): `final.py` się wywraca, a werdykt i dopiski są niezgodne z §5.6.**
- Gdzie: `final.py:91–138` (`bootstrap_2d` bez sprawdzenia obecności ramienia), `final.py:62–64, 85–87`.
- Dowód: usunięte biegi `PLW` dla F-MNIST (symulacja „`PLW` nietrenowalne na F-MNIST”).
  - `final.main` kończy się wyjątkiem `KeyError ('fmnist', 'PLW')` w `bootstrap_2d`.
  - `hypothesis("H3")` zwraca joint `NIESPOJNY-CZESCIOWY` (MNIST ROWNOWAZNOSC, F-MNIST untrainable) i `stability_joint = None`.
  - §5.6 wymaga: „hipotezy z jego udziałem otrzymują kategorię NIEROZSTRZYGNIĘTY z dopiskiem »ramię nietrenowalne«”, a dla `BCM`: „H3 = NIEROZSTRZYGNIĘTY z dopiskiem »`BCM` nietrenowalne«”.
  - Słownik `untrainable` nie zapisuje, które ramię jest nietrenowalne, więc nie da się wygenerować właściwego dopisku ani odróżnić „`BCM` nietrenowalne” (wtedy `PLW` nie istnieje) od „`PLW` nietrenowalne”.
- Poprawka:
  - `hypothesis`: gdy brakuje ramienia na którymkolwiek zbiorze, joint ma kategorię `NIEROZSTRZYGNIETY`, a `untrainable` zawiera listę ramion i zbiorów; dopisek według §5.6;
  - `bootstrap_2d` i `exploratory` pomijają takie hipotezy;
  - dodać scenariusz do `test_final_e2e`.
- Do decyzji autora: protokół nie mówi, co gdy ramię jest nietrenowalne tylko na jednym zbiorze. Dosłownie §5.6 daje NIEROZSTRZYGNIĘTY, a nie NIESPÓJNY-CZĘŚCIOWY. Warto dopisać jedno zdanie przed zamrożeniem.

**K1-03 (istotna) – `wybor_n` nie tworzy tabeli mocy per zbiór.**
- Gdzie: `analysis.py:282–304` (`simulate_joint` zwraca tylko klasy łączne), `wybor_n.py:100–109`.
- Dowód: §2: „Wiążące liczby: tabela mocy per zbiór i łącznie w `PARAMETRY.json`”; szablon §1: „Moc planistyczna (łącznie / per zbiór)”. W kodzie nie ma per zbiór. Moc R per zbiór to dokładnie ta liczba, na której §2 opiera ocenę sprawdzalności przewidywania H2 (F-MNIST).
- Poprawka: `simulate_joint` zwraca też `{ds: {P, R, G, N}}`; `power_table` zapisuje oba poziomy.

**K1-04 (istotna) – tabela mocy nie jest liczona przy końcowym N ani przy końcowym σ, a `final.py` jej nie czyta.**
- Gdzie: `wybor_n.py:148–153`. `N_P_power_table` liczone jest przy N_P z σ pilota, a `N_B_power_table` przy N_B z σ = max(P, B). Gdy N_P > N_B, N = N_P, ale tabela przy N_P istnieje tylko dla σ pilota.
- Skutek: dopisek §7.4 „moc równoważności < 0.5 (… przy N z Etapu C)” nie ma jednoznacznego źródła, a `final.py` i tak nie czyta `PARAMETRY.json` (K1-01).
- Poprawka:
  - po wyznaczeniu N = max policzyć `power_table(N, σ_końcowe, q_końcowe, ρ_końcowe)` jako `N_power_table` (per zbiór i łącznie);
  - `final.py` czyta z niej moc R i generuje dopisek przy klasie N.

**K1-05 (istotna) – q (awaryjność) jest agregowane niezgodnie z §7.6, a agregacja przez maksimum zawyża moc.**
- Gdzie: `wybor_n.py:92`: `max(0.01, max(max(s[h]["q"][a]) for s in stats_list))` – maksimum po zbiorach i po źródłach (pilot, B). Protokół: „q_a per ramię i zbiór; … dla N_B z wybranej konfiguracji Etapu B łącznie z pilotem” (czyli osobno per zbiór, z połączonych liczności).
- Dowód, że maksimum nie jest zachowawcze (`/tmp/claude-1000/k1/probe3.py`; N = 30, Δ = 75, σ_d = 110, 2000 powtórzeń):
  - moc P rośnie z awaryjnością **komparatora**: q_K = 0.01 → 0.611, 0.05 → 0.702, 0.10 → 0.795;
  - w H4 moc G rośnie z awaryjnością **ramienia testowanego**: q_B = 0.01 → 0.613, 0.10 → 0.793.
  - Zawyżenie q_K (H1–H3) lub q_B (H4) zawyża więc moc i może dać za małe N.
- Poprawka:
  - `simulate_joint` z q per zbiór;
  - q_N_B = (awarie pilota + awarie B w wybranej konfiguracji) / (10 + 10) per ramię i zbiór, minimum 1%;
  - jeśli autor chce zachowawczości, właściwym kierunkiem jest maksimum q ramienia „po złej stronie” i minimum po drugiej, ale to wymaga zmiany protokołu.

**K1-06 (istotna) – §7.1 jest wewnętrznie sprzeczny co do metody exact; kod realizuje tylko jedną z dwóch wersji.**
- Gdzie: PROTOKOL_v6 l. 168: „dokładny rozkład dla N ≤ 50 **bez wiązań i zer**”; l. 171: `method="exact"` „przy N ≤ 50 i **braku zer**”.
- Kod (`analysis.py:31`, `ref_kategorie.py:160`) i scipy 1.14.1 `auto` wybierają exact przy braku zer, także przy wiązaniach. Scipy zaokrągla wtedy W⁺ zachowawczo (floor/ceil, gh-19872).
- Dowód, że to nie jest kosmetyka: dla d całkowitych przy N = 30 i SD ≈ 1.16 p.p. (116 jednostek) **82%** zestawów ma wiązania |d| (`probe1.py`). Pierwsze zdanie opisuje więc zachowanie niezgodne z kodem w większości realnych przypadków dla N ∈ {30, 45}.
- Poprawka: przed zamrożeniem poprawić l. 168 na „dokładny rozkład (scipy; przy wiązaniach statystyka zaokrąglana zachowawczo) dla N ≤ 50 bez zer, w pozostałych przypadkach przybliżenie normalne z poprawką na wiązania”. Kod zostaje.

**K1-07 (istotna) – `final.py` nie liczy części pól wymaganych przez §7.4, §7.7, §7.9, §10 i `RAPORT_SZABLON.md` (lekcja R6).** §7.9: „Liczy każdą liczbę, tabelę i rysunek z `RAPORT_SZABLON.md`”. Brakuje:
- (a) **CI t (1 − α′)** dla średnich różnic: §10 wymaga „tabele … średnich różnic z CI t (wrażliwość)”. `pvals_t` daje tylko p, a `verdict_dataset` tylko średnią.
- (b) Bootstrap: zapisany jest tylko przedział (1 − α′) (`final.py:137`). Przedział (1 − 2α′) (z którego §7.7a każe wyprowadzać kategorie) nie jest zapisywany, a p bootstrapu nie są zapisywane wcale. Reguła R + STAB → N nie jest stosowana w (a), a w (b) jest – niespójność do rozstrzygnięcia.
- (c) **Reguła brzegowa §5.5** („Konfiguracje na brzegu zakresu” w szablonie): nie jest liczona. Uwaga: siatka lr `PLW`/`BP` (`np.logspace` z końcami) zawiera punkty dokładnie na brzegach, więc wybór brzegowy jest realny.
- (d) **Dopiski §7.4** nie są generowane, choć zawierają liczby:
  - „w ramieniu {X} {k} z {N} biegów uległo awarii…”;
  - STAB-* z listą Holma;
  - „równoważność dokładności przy istotnie różnej stabilności” (jest tylko flaga `stab_blocked`, a pierwotna kategoria R+ / R jest nadpisywana – `analysis.py:219` – i ginie);
  - dopisek o różnej klasie testu t;
  - rozbieżność QMNIST („na MNIST werdykt zależy od zbioru testowego…”);
  - moc R < 0.5;
  - odczyt H1 × H2.
  Szablon (l. 21) każe je wpisać z JSON, czyli ręcznie, a tu R6 znalazł błędy.
- (e) Lista ramion nietrenowalnych, wersje/GPU/k z rekordów biegów (szablon §1), wersja scipy w rekordzie analizy (§7.1: „przypięte … w rekordzie analizy”).
- (f) Wrażliwości (e) S ∈ {0.25, 1.0} zapisane są tylko jako `joint` – bez kategorii per zbiór i bez stabilności. Do uzupełnienia dla przejrzystości.
- Poprawka: dodać te pola do `WYNIKI_KONCOWE.json` (z `PARAMETRY.json`, `ETAP_B_WYBOR.json`, rekordów biegów) i generować dopiski jako listę zdań. Do `test_final_e2e` dodać asercje dla każdego pola szablonu (dziś test sprawdza obecność kluczy, nie kompletność).

**K1-08 (istotna) – autotest i implementacja referencyjna nie spełniają §7.9.**
- Gdzie: `selftest.py`, `ref_kategorie.py`.
- (vi): protokół – „na zbiorach danych wygenerowanych generatorem z `kod/sym_nan.py` (te same ziarna)” oraz „na 1 000 losowych zestawów danych”. Kod: własny generator `_d` z ziarnem `autotest`, **300** zestawów (`selftest.py:60`).
- (viii): protokół – „(oba komparatory w H3)”. Kod: tylko m = 1 (`selftest.py:91–96`).
- (ix): protokół – „w H3: pierwszy komparator ±S, drugi 0”. Kod: tylko m = 1 (`selftest.py:99–105`).
- `ref_kategorie.py` nie zawiera §7.3 (łączenia) ani reguły R + STAB → N. Scenariusze (iv), (v), (xiii) testują więc `analysis.combine` na ręcznych słownikach, a nie zgodność z niezależną implementacją. Protokół: referencja „bezpośrednio z definicji §7.2–7.3”.
- Referencja porównywana jest z `analysis.verdict_dataset`, a nie z `final.hypothesis`. Nie jest więc sprawdzone liczenie d z plików predykcji, przecięcie ziaren, stabilność ani łączenie w ścieżce `final.py`.
- (i) „wszystkie P” to jeden zbiór, jeden komparator. Etykiety scenariuszy `iv_P`, `v_G` sugerują pokrycie (iv)/(v), którego nie ma.
- Poprawka: uzupełnić referencję o §7.3 i stabilność, porównywać pełną ścieżkę (H1–H4, oba zbiory, joint + stabilność) na 1 000 zestawów z generatora `sym_nan.py`, dodać H3 do (viii)/(ix).

**K1-09 (istotna) – T7 nie testuje tego, czego wymaga §8.2.**
- Gdzie: `tests/test_protokol.py:138–145`.
- Protokół: „T7: obsługa NaN … także w ewaluacji (§5.8 krok 3): predykcja klasa 0, wpis `NAN <sha256 rekordu> <id>` weryfikowany”. Test sprawdza tylko flagę `nan` po θ = inf i funkcję `nan_predictions`.
- Nie są testowane: `runner.run_job` (brak checkpointu, `val_acc` = udział klasy 0, `ckpt_sha256 = None`), `write_hashes` / `verify` z wpisem NAN (w tym wykrycie podmiany rekordu) ani `evaluate()` dla biegu z NaN (klasa 0 we wszystkich odczytach i na QMNIST).
- §8.2 wymaga przejścia T1–T8 **przed pilotem**.
- Poprawka: test na ziarnie 69001+ i małych danych: runner → write_hashes → verify → evaluate z podmienionym `data.DATA_DIR` lub mockiem zbiorów testowych.

**K1-10 (istotna) – brak hashy dla plików, od których zależy definicja awarii i wynik testu.**
- Gdzie: `evaluate_test.py:27–39`, brak skryptu kroku 4 §5.8.
- Awaria biegu (`final.failed`) = `nan` lub `val_acc ≤ 15` z `C/zaslepione/<id>.json`. Te pliki nie są objęte plikiem hashy kroku 1. Status awarii, który decyduje o stabilności, analizie (d) i E2, nie ma więc znacznika czasu sprzed otwarcia testu (§3.4: „cecha biegu ustalana przed otwarciem testu”).
- Krok 4 §5.8 („weryfikacja hashy plików predykcji”) nie ma implementacji – `evaluate()` nie tworzy pliku hashy `test/C/*.pt`.
- Poprawka:
  - w `write_hashes` dodać sha256 `C/zaslepione/*.json` i `C/runs/*.json` dla wszystkich biegów, a `verify` je sprawdza;
  - `evaluate()` na końcu zapisuje `HASHE_PREDYKCJI.txt`, który porównuje się lokalnie po pobraniu.

**K1-11 (istotna) – kandydaci B/B2 są odtwarzani z listingu katalogu, a nie z reguły wyboru Etapu A.**
- Gdzie: `plans.py:253`, `plans.py:275`.
- Kandydat, którego żaden bieg B nie został zapisany (np. wszystkie biegi skończyły się `.error`), znika z wyboru bez komunikatu. Wybór odbywa się wtedy spośród 2 kandydatów zamiast 3, co narusza §5.3.
- Poprawka: w `A2` i `SEL` przeliczyć kandydatów regułą `plan_b` / `plan_b2` (deterministycznie z wyników A / A2) i wymagać kompletu 10 biegów per kandydat. Brak pliku ma dać twardy błąd (`point_scores` już go rzuca, jeśli kandydat jest znany).

### Drobne

**K1-12 (drobna) – symulacja mocy nie stosuje reguły R + STAB → N i nie raportuje stabilności „obok”.**
- Gdzie: `analysis.py:253–304`.
- §7.6: „stosowana jest pełna procedura §7.2–§7.3 (…; stabilność raportowana obok)”. Moc R przy q_B ≠ q_K jest zawyżona, a moc STAB-* nie jest raportowana.
- Poprawka: w symulacji policzyć test znaków z wektorów awarii i zastosować regułę; dodać kolumnę mocy STAB-GORSZA.

**K1-13 (drobna) – σ i ρ w H3 są zwijane do jednej liczby.**
- Gdzie: `wybor_n.py:76` (σ H3 = max po dwóch kontrastach, zamiast per kontrast), `wybor_n.py:82` (ρ = średnia po zbiorach), `wybor_n.py:96` (min po źródłach).
- Kierunki są zachowawcze, ale niezapisane w protokole („σ = max(pilot, B) per kontrast i zbiór”). Udokumentować w `kod/README.md` albo zaimplementować σ per kontrast (dwa SD szumu komparatorów).
- `ok.sum() < 2` daje NaN σ, a ramię nietrenowalne daje `KeyError` w `_ids`: brak obsługi.

**K1-14 (istotna; podniesiona z drobnej po próbie) – wykrywanie NaN sprawdza tylko stratę, nie dobroć; nieskończona dobroć przy skończonym z nie jest wykrywana wcale.**
- Gdzie: `ff.py:100–102`.
- §3.4: „strata … lub dobroć stanie się nieskończona/NaN”.
- Mechanizm: przy skończonym z, ale z² przepełnionym do +∞ (float32), g_pos = +∞. Strata `softplus(θ − ∞) + softplus(g_neg − θ)` jest wtedy skończona, a gradient softplus w −∞ wynosi dokładnie 0. W efekcie backward daje 2·p·0 = 0 i do wag nie trafia NaN.
- Dowód (`/tmp/claude-1000/k1/probe4.py`; mała sieć na ziarnie 69001, wagi warstwy 0 × 1e20):
  - `z finite: True`, `g_pos finite: False`;
  - `step ok: True`, wagi wszystkich warstw skończone, drugi krok też `True`.
- Skutek: bieg trenuje dalej z `nan = false` mimo nieskończonej dobroci. Przy ewaluacji przykłady z niezdefiniowaną dobrocią dostają −1, więc bieg jest oceniany częściowo jako błędny zamiast „klasa 0 wszędzie” (§3.4). Może też nie zostać zaliczony do awarii, jeśli walidacja > 15%.
- Poprawka: przed `backward` sprawdzać `torch.isfinite(g_pos).all() and torch.isfinite(g_neg).all()` (oraz stratę). Dodać przypadek do T7.

**K1-15 (drobna) – lista Holma w H3 jest zawsze liczona z p⁻(0).**
- Gdzie: `analysis.py:184`.
- Przy kategorii GORSZOŚĆ (inf(S)) zdanie „co najmniej jeden stały próg daje wyższą dokładność” z listą z p⁻(0) może wymienić komparator istotnie lepszy, ale nie o więcej niż S.
- Zapisywać obie listy (`holm_inf_S`, `holm_inf_0`) i w raporcie używać tej, która odpowiada kategorii.

**K1-16 (drobna) – E6 i E3: semantyka odczytów.**
- Krzywa walidacyjna `CUR` w epokach 1–12 jest liczona formułą hamowanie (końcową), choć sieć trenuje wtedy formułą standard; dla `REV` odwrotnie (`train.py:49`, `ff.final_formula`). To dopuszczalne, ale musi być opisane przy rysunku E6 – inaczej „skok przy przełączeniu” będzie źle czytany.
- Odczyt `cross` jest liczony i raportowany także dla `BCM` i `PLW` (`final.py:178–182`), choć §3.3 definiuje go tylko dla `CUR`, `STD`, `ANN`, `REV`.
- E4/E6 pomijają biegi z NaN (`final.py:187, 204`) – opisać.

**K1-17 (drobna) – `plans B` wypisuje kandydatów w kolejności rankingu średniej walidacyjnej.**
- Gdzie: `plans.py:247` – `top3` sortuje po średniej.
- Ujawnia to ranking punktów A (§5.3: „wypisuje tylko konfiguracje”). Wypisywać posortowane po numerze punktu.

**K1-18 (drobna) – ziarna nieopisane w protokole.**
- Sobol: `seed_from("A-sobol") % 2**31` (`plans.py:42`).
- Bootstrap: `seed=12345` (`final.py:91`), zamiast np. `seed_from("bootstrap")`.
- Zapisać w protokole lub README przed zamrożeniem.

**K1-19 (drobna) – drobiazgi wykonawcze.**
- (a) Średnie kandydatów B nie są zapisywane do `zaslepione/B/` (§5.3) – są tylko pośrednio w rekordach.
- (b) E5 nie obejmuje A2 (`PLW`) ani A-BP (krzywe 1D).
- (c) `plan_c` nie ma opcji usunięcia `BP` (jedyne dopuszczalne cięcie, §8.3).
- (d) Nazwa obrazu Dockera nie trafia do rekordu biegu (§6: „zapisywane w każdym rekordzie”).
- (e) `vast/uruchom.sh` zapisuje `HEAD` bez sprawdzenia czystości drzewa (`git diff --quiet`), a `wyslij.sh` wysyła drzewo robocze.
- (f) `apply_stability` zostawia śmieciowy klucz `class_` (`analysis.py:219`).
- (g) `hypothesis(drop_failed=True)` przy braku par bez awarii daje `IndexError` (`final.py:69`).
- (h) `hl_ci` używa przybliżenia normalnego bez poprawki na wiązania i z niesprawdzoną konwencją indeksu (w[k] vs w[k−1]) – opisowe, ale warto porównać z dokładnym przedziałem w teście.

**K1-20 (drobna) – słabe testy T1/T2/T3/T5/T8 (przechodzą, ale nie sprawdzają wszystkiego, co mówi §8.2).**
- T1 porównuje wagi, ale nie straty („wagi i straty … są zgodne”); θ BCM z tolerancją 1e-4 zamiast atol/rtol z protokołu.
- T2 nie sprawdza zachowania stanu Adama przez przełączenie.
- T3 używa x_pos = x_neg i tylko warstwy 0, więc nie wykryje zamiany g_pos/g_neg ani błędu w warstwach > 0.
- T5 porównuje generator sam ze sobą zamiast kolejności i złych etykiet w `train_ff` dla dwóch ramion.
- T8 nie testuje „runner … odrzuca źródłowy tryb wejścia”. Runner w ogóle nie ma takiego trybu, więc warunek jest spełniony konstrukcyjnie – wystarczy test, że `run_job` używa `data.encode`.

## 2. Werdykt

- **Rdzeń treningowy** (`ff.py`, `train.py`, `data.py`, `streams.py`, `runner.py`, plany P/A) jest zgodny z §3–§6 i ze źródłem (T1 dla 6 ramion, przełączenie 12/13, BCM). Wyjątek: wykrywanie nieskończonej dobroci (K1-14).
- **Dlaczego jedna bramka dla pilota i zamrożenia:** według §9 krok 1 (commit protokołu, kodu, testów T1–T8 i `SHA256SUMS`) poprzedza benchmark i pilot (krok 2), a tag (krok 4) tylko znakuje to, co już zacommitowano. Commit `80f8e61` jest właśnie tym krokiem 1. Każda zmiana `final.py`, `evaluate_test.py`, `selftest.py`, `ref_kategorie.py`, `wybor_n.py` lub tekstu §7.1 po pilocie oznaczałaby, że hashe sprzed pilota nie obejmują kodu, który przeanalizuje dane. Przed tym chroni ostatni akapit §9. Dla kodu „gotowy do pilota” i „gotowy do zamrożenia” to więc ta sama bramka.
- **Pilot: NIE, dopóki krok 1 §9 nie zostanie powtórzony**, to znaczy:
  - poprawione K1-01 – K1-11 i K1-14 (w tym tekst §7.1 z K1-06);
  - ponownie uruchomione `pytest` i `selftest`;
  - nowy commit z przeliczonymi `SHA256SUMS` (protokół, `PROGNOZY.md`, kod, `RAPORT_SZABLON.md`);
  - dopiero potem benchmark i pilot.
- To jest powtórzenie kroku 1, a nie wpis w `ODSTEPSTWA.md`, bo tagu jeszcze nie ma.
- Uwagi drobne (K1-12, K1-13, K1-15 – K1-20) nie blokują. Zalecane jest poprawienie K1-12 i K1-13 razem z K1-03 – K1-05, bo dotyczą tej samej symulacji.
