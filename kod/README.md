# Kod badania (PROTOKOL_v6)

| Plik | Sekcja protokołu |
|---|---|
| `hamexp/streams.py` | §6 strumienie losowe |
| `hamexp/data.py` | §3.2 wejście (L2 + etykieta λ = 0.1), §6 podział, blokada testu |
| `hamexp/ff.py` | §3.1 sieć FF (jednoczesny trening warstw, Adam per warstwa), §4 harmonogramy ramion, §3.3 odczyty, §3.5 sieć BP |
| `hamexp/train.py` | §3–§5 pętle treningu, §3.4 NaN, logowanie E1/E4/E6 |
| `hamexp/runner.py` | §5, §8 wykonanie, zapis atomowy, metryki walidacyjne tylko w `zaslepione/` |
| `hamexp/plans.py` | §5.1–5.7 plany i reguły wyboru; katalogi etapów: `P`, `A` (5 ramion + BP), `B` (top-3 z A, także BP), `A2`/`B2` (= Etap B2 protokołu dla `PLW`: pula A / pula B), `C` |
| `hamexp/wybor_n.py` | §7.6 wybór N (tylko SD, ρ, odsetki awarii) |
| `hamexp/analysis.py` | §7.1–7.3 test Wilcoxona (Pratt), kategorie, Holm, stabilność, łączenie, symulacja |
| `hamexp/final.py` | §7.4–7.9 zamrożony skrypt analizy (wszystkie pola `RAPORT_SZABLON.md`, rysunki) |
| `hamexp/selftest.py`, `hamexp/ref_kategorie.py` | §7.9 autotest (i)–(xiii), niezależna implementacja referencyjna |
| `hamexp/evaluate_test.py` | §5.8 hashe checkpointów i jednorazowa ewaluacja testowa |
| `hamexp/gpu_check.py` | §8.2 T6 (powtarzalność, niezależność od k), benchmark przepustowości |
| `hamexp/monitor.py`, `hamexp/sync_verify.py`, `vast/*.sh` | §8 monitor (bez metryk), synchronizacja, wysyłka (dane testowe dopiero `wyslij_test.sh` po pushu hashy) |
| `tests/test_protokol.py` | §8.2 testy T1–T8 |
| `tests/test_final_e2e.py` | §7.9 test całego `final.py` na danych syntetycznych |
| `zrodlo/` | zamrożona kopia kodu źródłowego (commit 9d58048) z `SHA256SUMS` – tylko dla T1 |
| `sym_nan.py` | symulacja kontrolna obsługi NaN (§3.4) |

Środowisko: obraz `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (torch 2.5.1, Python 3.11) na Vast; lokalnie torch 2.5.1+cpu, scipy 1.14.1 (przypięte, §7.1), numpy 2.1.x.

**Lokalnie (CPU bez AVX2, i5-3450):** uruchamiać z `MKL_CBWR=COMPATIBLE` – bez tego Intel MKL w torch 2.5.1 losowo wybiera jądra AVX-512 (`mkl_vml_kernel_sSqrt_Z0…`, `…sTanh_Z0…`) i proces kończy się SIGILL (zob. `../eksperyment_ff_od_zera/ODSTEPSTWA.md`).

Testy: `cd kod && MKL_CBWR=COMPATIBLE python -m pytest -q tests/`

## Znane słabości (zapisane przed zamrożeniem; recenzje K1/K2)

- T1 porównuje wagi i progi BCM (nie straty), T2 nie sprawdza stanu Adama wprost, T3 sprawdza aktualizację BCM tylko na warstwie 0, T5 jest prosty (K1-20).
- Agregacja σ (maksimum po źródłach i po dwóch kontrastach H3) i ρ (minimum) w `wybor_n.decide` jest zachowawcza, ale nie wynika z jednej reguły ogólnej (K1-13).
- Autotest (vi) używa generatora w stylu `sym_nan.py` (różnice normalne + awarie jako klasa 0), nie samego `sym_nan.py`; porównuje pełną ścieżkę `analysis` (bez warstwy wczytywania `final.hypothesis`, którą pokrywa `tests/test_final_e2e.py`).
- Rekord biegu zawiera nazwę obrazu (`HAMEXP_IMAGE`, ustawiane przez `vast/uruchom.sh`).
