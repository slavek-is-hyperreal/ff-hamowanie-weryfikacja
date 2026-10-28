# Kod badania (PROTOKOL_v6)

| Plik | Sekcja protokołu |
|---|---|
| `hamexp/streams.py` | §6 strumienie losowe |
| `hamexp/data.py` | §3.2 wejście (L2 + etykieta λ = 0.1), §6 podział, blokada testu |
| `hamexp/ff.py` | §3.1 sieć FF (jednoczesny trening warstw, Adam per warstwa), §4 harmonogramy ramion, §3.3 odczyty, §3.5 sieć BP |
| `hamexp/train.py` | §3–§5 pętle treningu, §3.4 NaN, logowanie E1/E4/E6 |
| `tests/test_protokol.py` | §8.2 testy T1–T8 |
| `zrodlo/` | zamrożona kopia kodu źródłowego (commit 9d58048) z `SHA256SUMS` – tylko dla T1 |
| `sym_nan.py` | symulacja kontrolna obsługi NaN (§3.4) |

Środowisko: obraz `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (torch 2.5.1, Python 3.11) na Vast; lokalnie torch 2.5.1+cpu, scipy 1.14.1 (przypięte, §7.1), numpy 2.1.x.

**Lokalnie (CPU bez AVX2, i5-3450):** uruchamiać z `MKL_CBWR=COMPATIBLE` – bez tego Intel MKL w torch 2.5.1 losowo wybiera jądra AVX-512 (`mkl_vml_kernel_sSqrt_Z0…`, `…sTanh_Z0…`) i proces kończy się SIGILL (zob. `../eksperyment_ff_od_zera/ODSTEPSTWA.md`).

Testy: `cd kod && MKL_CBWR=COMPATIBLE python -m pytest -q tests/`
