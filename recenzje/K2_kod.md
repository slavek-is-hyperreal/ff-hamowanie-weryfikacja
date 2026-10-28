# K2 – recenzja weryfikacyjna kodu (po odpowiedzi na K1)

Data: 2026-10-28. Recenzent: asystent (rola: czepialski recenzent). Zakres: `git diff 80f8e61 43d4b50 -- eksperyment_hamowanie_od_zera/`, `K1_kod.md`, `K1_odpowiedz.md`, PROTOKOL_v6 §5–§9.
- Kodu nie edytowano. Nic nie zapisano w `wyniki/`. Nie było treningu na ziarnach z pul P/A/B/C (jedyny trening to T7 w pytest na ziarnach 69001–69002).
- Pliki robocze (próby): `/tmp/claude-1000/k2/` – `probe_final.py`, `probe_eval.py`, `probe_sim.py`. Wynik selftestu: `/tmp/claude-1000/k2_st.json`.

## 0. Co uruchomiono

- `pytest tests/`: **16 przeszło, 1 pominięty** (T6 – brak GPU). Doszły trzy nowe testy: `test_refuses_incomplete_stage_c`, `test_untrainable_arm`, `test_T7_pipeline_with_nan`.
- `python -m hamexp.selftest`: **34/34 OK**. Wynik jest identyczny z zacommitowanym `wyniki/SELFTEST.json` (porównanie JSON: `True`), więc zacommitowany SELFTEST pochodzi z zacommitowanego kodu.
- `SHA256SUMS`: `sha256sum -c` daje wszystkie 42 pliki OK. Każdy plik z `kod/hamexp`, `kod/tests`, `kod/vast`, `sym_nan.py`, `README.md`, `RAPORT_SZABLON.md`, `PROGNOZY.md` i `PROTOKOL_v6.md` jest pokryty.
- Próba K1-14 (`/tmp/claude-1000/k1/probe4.py`, ziarno 69001, wagi × 1e20): `z finite: True`, `g_pos finite: False`, `step ok: False`, czyli nieskończona dobroć jest teraz wykrywana.
- `_stab_vec` (wektorowa stabilność w symulacji N) porównana z `analysis.stability` na 6 000 wierszach (m = 1 i 2, q_B = 12%, q_K = 2%): **0 niezgodności**.

## 1. Status uwag K1

R – poprawione i działa; Cz – częściowo; N – nie poprawione.

| Uwaga | Status | Dowód / komentarz |
|---|---|---|
| K1-01 niepełny Etap C | **R** | `final.py` odmawia (`SystemExit`) przy braku 3 ziaren `CUR`, przy ziarnie spoza puli C i przy braku pliku predykcji. `evaluate_test` odmawia przy `.error`, przy niezgodności z planem (w obie strony) i bez `--plan` (argparse, kod 2). **Uwaga:** wymaganie kompletu oparte na `ETAP_B_WYBOR.json` koliduje z §8.3 (K2-01). |
| K1-02 ramię nietrenowalne | **R** (z uwagą) | Próba „`BCM` nietrenowalne na F-MNIST”: H3 = NIEROZSTRZYGNIETY, dopisek „BCM nietrenowalne”, E1 tylko dla MNIST, bez wyjątku. Próba „`STD` nietrenowalne na F-MNIST”: H1, H3, H4 = NIEROZSTRZYGNIETY, H2 policzone, wrażliwości pominięte. Uwagi co do dopisków: K2-03. |
| K1-03 moc per zbiór | **R** | `simulate_joint` zwraca `joint`, `per_dataset` i `stab_worse` (sprawdzone w `probe_sim.py`). |
| K1-04 tabela mocy przy końcowym N | **R** | `N_power_table` liczona przy `N = max(N_P, N_B)` z ostatnich wejść; `final.py` czyta ją do dopisku mocy. |
| K1-05 agregacja q | **R** | q per ramię i zbiór z połączonych liczności (awarie / biegi), minimum 1%. Zgodne z nowym tekstem §7.6. |
| K1-06 opis exact w §7.1 | **R** | L. 168 opisuje teraz zachowanie scipy (floor/ceil przy wiązaniach) i jest spójna z l. 171. |
| K1-07 pola raportu | **Cz** | Dodano: CI t, oba przedziały bootstrapu i p, regułę brzegową, wersje, (e) per zbiór, H1 × H2. Dopiski §7.4 są niekompletne lub mają zły format (K2-03). R + STAB → N nadal nie jest stosowane w bootstrapie (K2-08). |
| K1-08 autotest / referencja | **Cz** | Jest 1 000 zestawów i pełna ścieżka §7.2 + stabilność + §7.3, a referencja ma `laczenie` i `werdykt`. Braki: generator tylko naśladuje `sym_nan.py` (protokół: „generatorem z `kod/sym_nan.py` (te same ziarna)”); porównanie nadal z `analysis.*`, nie z `final.hypothesis`; pokrycie kategorii słabe (K2-06). |
| K1-09 T7 w ewaluacji | **R** (z uwagą) | Test przechodzi pełną ścieżkę runner → write_hashes → verify → evaluate → verify_predictions. Moje próby: `nan_predictions` daje 0 we wszystkich odczytach, a podmiana rekordu NaN jest wykrywana (`hash mismatch: C/runs/…CUR….json`). Test sam tych dwóch rzeczy nie sprawdza (K2-07). |
| K1-10 hashe rekordów / zaślepionych / predykcji | **Cz** | Linie `FILE` są w pliku hashy, a podmiana `zaslepione` jest wykrywana. Ale: (1) `verify` przechodzi po usunięciu linii `FILE`; (2) `final.py` nie weryfikuje żadnych hashy przed analizą (K2-02). |
| K1-11 kandydaci B/B2 z reguły | **R** | `A2` i `SEL` przeliczają kandydatów przez `plan_b` / `plan_b2`. Brak pliku daje `FileNotFoundError` w `point_scores`. Klucze są zgodne (`ds\|arm` dla B, `ds` dla B2). |
| K1-12 R + STAB → N w symulacji | **R** | `analysis.py:282–323`; `_stab_vec` zgodne z `stability` (0/6000). Moc STAB-GORSZA tylko per zbiór, bez wersji łącznej (wystarczy). |
| K1-13 σ/ρ w H3, NaN σ, ramię nietrenowalne | **N** | Opisane tylko w docstringu `decide`. Brak obsługi NaN σ i ramienia nietrenowalnego (K2-04, K2-05). |
| K1-14 nieskończona dobroć | **R** | `ff.py:101–104`; próba probe4 daje `step ok: False`. |
| K1-15 lista Holma z inf(S) | **Cz** | Pole `holm_better_comparators_inf_S` istnieje, ale `notes()` (`final.py:194–195`) dla klasy G zawsze używa listy z inf(0), niezależnie od kategorii. Dopisek jest też generowany dla H4, choć §7.4 przewiduje go tylko dla H3. |
| K1-16 semantyka E3/E6 | **R** | §7.8 uzupełniony; docstring `exploratory`. |
| K1-17 ranking w wydruku | **R** | `plans.py:353, 378` sortują po numerze punktu. |
| K1-18 ziarna | **R** | §6 (l. 161): Sobol, bootstrap `seed_from("bootstrap", H)`, `choose_n`, `autotest`; `final.py:137` używa `seed_from`. |
| K1-19 drobiazgi | **Cz** | (b) R: E5 obejmuje A2. (c) R: `--bez-bp`, ale K2-01. (f) R: `class_` usunięte. (g) R: < 2 pary dają N z dopiskiem. (a) N: średnie kandydatów B nie są zapisywane do `zaslepione/B/`. (d) N: nazwy obrazu nie ma w rekordzie, choć §6 l. 162 mówi „zapisywane w każdym rekordzie biegu”. (e) N: `vast/uruchom.sh:6` bierze `HEAD` bez `git diff --quiet`. (h) N: brak odpowiedzi. |
| K1-20 słabe T1/T2/T3/T5/T8 | **N** | Odpowiedź mówi „słabości zapisane jako znane”, ale `kod/README.md` nie zawiera takiego wpisu. §8.2 T1 wprost wymaga zgodności strat, a test porównuje tylko wagi (K2-10). |

## 2. Nowe uwagi K2

### Istotne (blokujące pilot / zamrożenie)

**K2-01 (istotna) – jedyne dopuszczalne cięcie budżetu (§8.3, `plan_c --bez-bp`) uniemożliwia analizę.**
- Gdzie: `final.py:49–65`. `check_complete` iteruje po `ALL_ARMS = FF_ARMS + ("BP",)`, a oczekiwany komplet bierze z `ETAP_B_WYBOR.json`, w którym `BP` zawsze ma `cfg` (wpisuje go etap `A2`). Planu C nie czyta.
- Dowód (`probe_final.py`, „cięcie BP (§8.3)”): po usunięciu wszystkich biegów `BP` z syntetycznego C wynik to `SystemExit: Etap C niekompletny dla mnist/BP: brak 10 … analiza zabroniona`. Obejście przez wpisanie `BP.cfg = None` zmienia zamrożony `ETAP_B_WYBOR.json` i fałszywie oznacza `BP` jako „nietrenowalne”.
- Skutek: §8.3 przewiduje to cięcie jako ścieżkę zgodną z protokołem, a po tagu wymagałoby ono zmiany `final.py`. Przed tym właśnie chroni jedna bramka z K1.
- Poprawka: `final.py --plan ../plany/C.json`. Zbiór oczekiwanych (zbiór, ramię, ziarno) brać z planu i sprawdzić, że N planu = `PARAMETRY.N` i że ramiona planu = ramiona z `cfg` w `ETAP_B_WYBOR.json` minus ewentualnie `BP`. Zapisać `"bp_cut": true` z dopiskiem i pominąć E2. Dodać scenariusz do `test_final_e2e`.

**K2-02 (istotna) – analiza nie sprawdza, że status awarii i predykcje są tymi zhashowanymi przed otwarciem testu; `verify` akceptuje plik hashy bez linii `FILE`.**
- Gdzie:
  - `evaluate_test.py:76–85`: `listed` porównuje z biegami tylko wpisy checkpointów i `NAN`. Wpisy `FILE` są sprawdzane tylko wtedy, gdy są obecne.
  - `final.py:34–46`: wczytuje `C/runs`, `C/zaslepione` i `test/C` z dysku lokalnego bez żadnej weryfikacji.
- Dowód (`probe_eval.py`):
  - podmiana `zaslepione/<STD>.json` (val 90 → 10, czyli bieg staje się „awarią”) daje odmowę `hash mismatch` – dobrze;
  - ta sama podmiana po usunięciu linii `FILE` z pliku hashy daje **„verify PRZESZŁO”**.
- Najważniejsze: `final.py` działa lokalnie, **po** otwarciu testu, i nie porównuje niczego z plikiem hashy z kroku 1 ani z `HASHE_PREDYKCJI.txt`. Status awarii decyduje o stabilności, o regule R + STAB → N i o analizie (d). Jego zmiana po otwarciu testu nie zostałaby wykryta. To jest sedno K1-10.
- Poprawka:
  - `verify`: zbiór kluczy `FILE` = dokładnie {`C/runs/<id>.json`, `C/zaslepione/<id>.json` dla każdego id z planu}.
  - `final.py --hashes HASHE_C.txt` wywołuje tę weryfikację (bez wymogu checkpointów lokalnie – wystarczą linie `FILE` i `NAN`) oraz `verify_predictions`, a przy niezgodności odmawia.
  - Test: podmiana `zaslepione` po kroku 1 daje odmowę `final.py`.

**K2-03 (istotna) – dopiski §7.4 generowane przez `notes()` są niepełne, a część ma format niezgodny z §7.4.**
- Gdzie: `final.py:175–206`. Szablon (`RAPORT_SZABLON.md`, nagłówek) każe przepisywać wartości z JSON „bez zmian”, więc braki trafią do raportu albo zostaną dopisane ręcznie – a ręczne dopisywanie to właśnie lekcja R6.
- Dowód (`probe_final.py`, przebieg „pełny” i „BCM nietrenowalne”):
  1. **Brak dopisku o różnej klasie testu t.** H4: test główny GORSZOSC, test t NIEROZSTRZYGNIETY, a `notes_7_4.H4` nie zawiera zdania z obiema kategoriami (§7.4 „Awarie… gdy klasa testu t różni się od głównej – dopisek z obiema kategoriami”).
  2. **Brak zdań NIESPÓJNY-\*.** H3 ma joint NIESPOJNY-CZESCIOWY, ale zdania „na zbiorze {X}: …; na zbiorze {Y}: …; kategorie należą do różnych klas…” nie ma. To samo dotyczy KIERUNEK i ZGODNY-KIERUNEK.
  3. **Brak stałych dopisków:** H3 §4.1 („θ(0) BCM działa tylko w pierwszej epoce”) oraz „Macierz H1 × H2 nie zawiera porównania ANN − STD…”.
  4. **Format awarii i stabilności:** wypisywany jest słownik `awarie {'B': 1, 'STD': 0}`, a nie „w ramieniu REV 1 z 10 biegów uległo awarii … i zostało ocenione jako najgorsze”. Zdania STAB-GORSZA, STAB-LEPSZA i „bez wykazanej różnicy stabilności ({f_B} vs {f_K} awarii z {N} ziaren)” nie zawierają liczności ani N, choć liczności są w `stability.counts`.
  5. **RÓWNOWAŻNOŚĆ+:** brak „komparatorów, które wyznaczają kierunek” (§7.4).
  6. **Hipoteza z ramieniem nietrenowalnym** dostaje dopisek mocy („badanie nie miało mocy… (moc 0.20)”), choć klasa N wynika z §5.6, a nie z mocy. Dopisek „BCM nietrenowalne” nie wskazuje zbioru, choć nowy tekst §5.6 wymaga „ramię i zbiór”. Brzmienie różni się też od §7.4 („{ramię} nietrenowalne w tym układzie (wszystkie konfiguracje odrzucone)”).
  7. K1-15: dla kategorii GORSZOŚĆ lista Holma powinna (albo przynajmniej może) pochodzić z `holm_better_comparators_inf_S`. Zdanie G dla H3 ograniczyć do H3.
- Poprawka: generować zdania §7.4 dosłownie z szablonów tekstowych (lista zdań per hipoteza i per zbiór plus zdania łączne), z licznościami. W `test_final_e2e` dodać asercje na obecność każdego wymaganego zdania w sytuacjach, które je wywołują (fixture już daje H3 NIESPÓJNY-CZĘŚCIOWY, H4 z różną klasą t i awarię w `REV`).

**K2-04 (istotna) – `wybor_n` (N_B) wywraca się przy ramieniu nietrenowalnym (§5.6), więc nie da się wyznaczyć N przed Etapem C.**
- Gdzie: `wybor_n.py:31–36` i `:72`.
- Dowód (`_ids` wywołane bezpośrednio):
  - `sel = {}` daje `KeyError 'fmnist|BCM'` / `'fmnist|PLW'`;
  - `fmnist|BCM.point = None` daje id bez punktu (`B__fmnist__BCM__s62001`), a w `_val` będzie `FileNotFoundError`.
- §5.6 i §7.6 przewidują tę ścieżkę: N_B ma być policzone, a hipoteza z ramieniem nietrenowalnym jest N z definicji. Naprawa po tagu oznaczałaby zmianę kodu wyboru N.
- Poprawka: `collect`/`decide`/`choose_n` pomijają hipotezy z ramieniem bez `cfg` na którymkolwiek zbiorze (bez celu mocy, z wpisem w `PARAMETRY.json`). Dodać test.

### Drobne

**K2-05 (drobna) – NaN σ jest cicho gubione albo propagowane, zależnie od kolejności źródeł.**
- Gdzie: `wybor_n.py:76` (`ok.sum() < 2` daje `np.std` = NaN) i `:91` (`max(...)` po źródłach).
- Dowód:
  - `max(nan, 100.) = nan`, ale `max(100., nan) = 100.`. NaN z Etapu B (drugie źródło) zostaje więc cicho zastąpiony σ pilota.
  - NaN z pilota trafia do `simulate_joint`, które nie zgłasza błędu: `RuntimeWarning: invalid value in cast`, per zbiór G = 1.0, joint N = 1.0, co prowadzi do N = 120 bez komunikatu (`probe_sim.py`).
- Poprawka: twardy błąd przy < 3 parach bez awarii (albo jawna reguła w §7.6), `np.isfinite` przed symulacją.

**K2-06 (drobna) – autotest (vi) słabo pokrywa kategorie, a porównanie nie obejmuje ścieżki `final.py`.**
- Dowód: replika generatora (vi) na 1 000 zestawach daje łącznie RÓWNOWAŻNOŚĆ+ **0**, RÓWNOWAŻNOŚĆ 4, NIESPÓJNY-ZGODNY-KIERUNEK 2, PRZEWAGA 9; stabilność STAB-LEPSZA **0**, STAB-GORSZA 44; R + STAB → N 17 zdarzeń per zbiór. Zgodność „na 1 000 zestawach” prawie nie testuje gałęzi R+, ZGODNY i STAB-LEPSZA.
- Generator to nie `sym_nan.py` (wymóg §7.9 „te same ziarna”), a porównanie idzie z `analysis.*`, nie z `final.hypothesis` (liczenie d z plików predykcji, ziarna, `failed()` z `val_acc`).
- Poprawka: warstwować przesunięcia, tak by każda kategoria łączna i każde twierdzenie stabilności miały ≥ 30 przypadków. Wywoływać `final.hypothesis` na syntetycznej tabeli (jak w `test_final_e2e`). Albo poprawić tekst §7.9 na „generator w stylu `sym_nan.py`”.
- Dodatkowo (viii) H3 sprawdza tylko P(klasa P), a nie rozmiar p⁻(0) z Holmem przy Δ = 0. Protokół mówi „odsetek odrzuceń p⁺(0) … oraz p⁻(0) … (oba komparatory w H3)”.

**K2-07 (drobna) – `test_T7_pipeline_with_nan` nie sprawdza tego, co mówi §8.2 T7.**
- Gdzie: `tests/test_protokol.py` (nowy test, koniec pliku). Sprawdzane są tylko klucze `{"mnist", "qmnist"}` × `{main, no_l0, cross}`, a nie to, że predykcje są klasą 0 (`res == (y == 0)`). Podmiana rekordu NaN nie jest testowana – jest tylko usunięcie rekordu.
- Moje próby pokazują, że zachowanie jest poprawne, ale test by tego nie wykrył.
- Poprawka: w `fake_load` zwracać stałe `y`, sprawdzać `res_nan[nm][k] == (y == 0)`, dodać przypadek „zmieniony rekord NaN, a potem `verify`, czyli odmowa”.

**K2-08 (drobna) – niespójność reguły R + STAB → N w wrażliwościach.**
- Gdzie: `final.py:133–171`. Reguła jest stosowana w (b) t, (c) QMNIST i (e), a w (a) bootstrapie nie.
- Poprawka: zastosować w (a) albo zapisać w §7.7, że (a) jej nie stosuje (jak (d)).

**K2-09 (drobna) – `verify_predictions` nie sprawdza kompletności ani zgodności z planem.**
- Gdzie: `evaluate_test.py:150–157`. Weryfikuje tylko pliki wymienione w `HASHE_PREDYKCJI.txt`, który powstaje na instancji razem z predykcjami (kontrola transferu, nie znacznik czasu). Brak predykcji wyłapie co prawda `final.check_complete`.
- Poprawka: zestaw plików = plan, w ramach K2-02.

**K2-10 (drobna) – K1-20 i K1-19(d) zostają w sprzeczności z literą protokołu w chwili zamrożenia.**
- §8.2 T1: „wagi **i straty**”. Test porównuje tylko wagi.
- §6 l. 162: wersja obrazu „zapisywana w każdym rekordzie biegu”. Rekord nie ma nazwy obrazu (jest w README).
- Odpowiedź K1 zapowiada wpis „znane słabości”, którego w `kod/README.md` nie ma.
- Poprawka: albo dodać asercję strat w T1 i `HAMEXP_IMAGE` w `_versions` (oraz `uruchom.sh` z `git diff --quiet`), albo zmienić tekst §8.2/§6 przed zamrożeniem. Zamrożenie nie powinno się zaczynać od udokumentowanej niezgodności.

## 3. Zgodność z §5–§9 (nowe elementy)

- §5.2/§5.3/§5.6 (`plans A2/SEL`): zgodne. Ramię bez kandydatów dostaje `cfg = None`, a `BCM` bez punktu daje brak `PLW` (`plan_a2`, `plan_b2`). Pozostaje stara kwestia: `mean_bcm_thresholds` rzuca `RuntimeError`, gdy wybrany `BCM` ma < 50% biegów bez NaN. Taki punkt przy kryterium „> 5 z 10” mógł zostać wybrany, więc ścieżka nie jest opisana w §5.6 (do dopisania jednym zdaniem).
- §5.8: kroki 1, 3 i 4 mają implementację. Krok „analiza po kroku 4” nie weryfikuje hashy (K2-02).
- §7.6: q, R + STAB → N, tabela przy końcowym N – zgodne. Ramię nietrenowalne i NaN σ – K2-04, K2-05.
- §7.7: (a) zgodne co do przedziałów i p, ale bez reguły stabilności (K2-08). (b) kategorie z p są równoważne przedziałom t, (1 − 2α′) CI t nie jest zapisywany (opisowo, akceptowalne). (c), (d), (e) zgodne.
- §8.3: K2-01.
- §9 krok 1: hashe i SELFTEST są spójne z commitem `43d4b50`. Każda poprawka powyżej wymaga ponownego kroku 1 (testy, selftest, `SHA256SUMS`, commit).

## 4. Werdykt

**Kod gotowy do pilota i zamrożenia po wymienionych poprawkach; K3 nie jest potrzebna, jeśli poprawki ograniczą się do zakresu poniżej i zostaną pokryte testami.**

- Logika decyzyjna jest poprawna i zweryfikowana: kategorie, Holm/IUT, stabilność, łączenie, symulacja N z q per zbiór i R + STAB → N, wykrywanie NaN/inf. Żadna z nowych uwag jej nie dotyczy.
- Blokujące (przed pilotem, ta sama zasada jednej bramki co w K1):
  - **K2-01** (cięcie `BP` §8.3 blokuje `final.py`);
  - **K2-02** (`final.py` bez weryfikacji hashy; `verify` akceptuje brak linii `FILE`);
  - **K2-03** (dopiski §7.4);
  - **K2-04** (`wybor_n` przy ramieniu nietrenowalnym).
  Każda z nich to ścieżka przewidziana w protokole, która po tagu wymagałaby zmiany kodu analizy albo wyboru N.
- Zalecane w tym samym commicie: K2-05, K2-07, K2-10 (albo zmiana tekstu §8.2/§6). Pozostałe (K2-06, K2-08, K2-09) mogą trafić do protokołu jako doprecyzowania.
- Po poprawkach: powtórzyć §9 krok 1 (pytest, selftest, `SHA256SUMS`, commit). Jeśli autor poprawi także generator (vi), czyli zmieni logikę autotestu, krótka K3 tylko dla `selftest.py` / `final.notes` jest wskazana, ale nie wymagana.
