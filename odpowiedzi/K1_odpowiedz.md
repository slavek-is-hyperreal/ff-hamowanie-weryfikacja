# Odpowiedź na recenzję kodu K1 (przed zamrożeniem; powtórzenie §9 krok 1)

| Uwaga | Zmiana |
|---|---|
| K1-01 | `evaluate_test`: `--plan` wymagane (także `--write-hashes`), komplet biegów = plan (w obie strony), odmowa przy `.error`, plik hashy = dokładnie zbiór biegów; `final.py` czyta `PARAMETRY.json` i `ETAP_B_WYBOR.json` i odmawia przy niekompletnym C (test `test_refuses_incomplete_stage_c`) |
| K1-02 | `final.hypothesis`: ramię nietrenowalne (choćby na jednym zbiorze) → NIEROZSTRZYGNIĘTY z dopiskiem (BCM / ramię + zbiór), wrażliwości pomijane; §5.6 doprecyzowany; test `test_untrainable_arm` |
| K1-03, K1-04, K1-12 | `simulate_joint` zwraca klasy łącznie i per zbiór oraz P(STAB-GORSZA), stosuje R + STAB → N; tabela mocy przy końcowym N i końcowych wejściach (`N_power_table`), czytana przez `final.py` (dopisek mocy R); §7.6 |
| K1-05 | q per ramię i zbiór z połączonych liczności (awarie / biegi), min 1%; §7.6 |
| K1-06 | §7.1 – opis metody scipy zgodny z kodem (exact przy wiązaniach, bez zer) |
| K1-07 | `final.py`: CI t, przedziały bootstrapu (1 − α′) i (1 − 2α′) + p, reguła brzegowa §5.5, dopiski §7.4 (`notes_7_4`), odczyt H1 × H2, wersje (GPU, sterownik, torch, scipy, MKL_CBWR), (e) per zbiór; test e2e sprawdza pola |
| K1-08 | selftest: (vi) pełna ścieżka (§7.2 + stabilność + §7.3) vs rozszerzona referencja na 1 000 zestawach z generatora jak `sym_nan.py`; (viii)/(ix) wariant H3; 34/34 |
| K1-09 | test `test_T7_pipeline_with_nan`: runner → write_hashes (NAN) → verify → evaluate (zamockowany test) → verify_predictions; odmowa przy brakującym rekordzie |
| K1-10 | plik hashy obejmuje `C/runs/*.json` i `C/zaslepione/*.json`; `HASHE_PREDYKCJI.txt` + `--verify-predictions` (krok 4); §5.8 |
| K1-11 | `plans A2`/`SEL`: kandydaci przeliczani regułą §5.2 z wyników A / A2; brak pliku → twardy błąd |
| K1-14 | `FFNet.step`: NaN także przy nieskończonej dobroci (z² → inf) |
| K1-15 | lista Holma także z inf(S) (`holm_better_comparators_inf_S`) |
| K1-16 | §7.8 E4/E6 opisane (formuła końcowa, biegi bez NaN, liczba biegów) |
| K1-17 | `plans` wypisuje kandydatów posortowanych po numerze punktu (bez rankingu) |
| K1-18 | §6 – ziarna Sobola, bootstrapu, symulacji i autotestu |
| K1-19 | `plan_c --bez-bp` (§8.3); `class_` usunięte; (d) przy < 2 parach → N z dopiskiem; E5 obejmuje A2. Nazwa obrazu: w `kod/README.md` (rekord biegu zawiera wersje torch/CUDA/GPU/sterownik) |
| K1-13, K1-20 | przyjęte częściowo: kierunki agregacji σ/ρ opisane w docstringu `wybor_n.decide`; testy T1–T8 bez zmian poza T7 (pozostałe słabości zapisane jako znane) |
