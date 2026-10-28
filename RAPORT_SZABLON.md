# Raport: curriculum hamowania i próg BCM w Forward-Forward – sprawdzenie konfirmacyjne

> Szablon napisany przed pilotem (PROTOKOL_v6 §9, §10). Pola `{…}` wypełnia `kod/hamexp/final.py` (`wyniki/WYNIKI_KONCOWE.json`) – wartości przepisywane bez zmian, ze wskazaniem klucza JSON. Zdania interpretacji konfirmacyjnej – wyłącznie z §7.4 protokołu. Wszystko inne – w części „Wyniki eksploracyjne”.
>
> Słowa zakazane: „dowodzi”, „przełom”, „bezprecedensowy”, „idealny”, „rozwiązuje”, „deklasuje”, „ratuje”.

**Status:** prerejestrowane z zewnętrznym znacznikiem czasu: https://github.com/slavek-is-hyperreal/ff-hamowanie-weryfikacja (tag `hamowanie-prereg`, {data pusha}); {ukończone | nieukończone (§5.8)}.
**Zakres wniosków:** MLP 10 × 64, Forward-Forward z jednoczesnym treningiem warstw, 20 000 przykładów, 25 epok, MNIST i Fashion-MNIST, oficjalne zbiory testowe (oglądane w kampanii źródłowej – §1 protokołu).
**Estymand:** pseudomediana (Hodges–Lehmann) parowanych różnic dokładności; „wyższa/niższa dokładność” = dodatnia/ujemna pseudomediana.

## Pierwsza strona

| H | Pytanie | MNIST | F-MNIST | Łącznie | QMNIST-50k (MNIST) | Stabilność (łącznie) |
|---|---|---|---|---|---|---|
| H1 | curriculum vs standard | {confirmatory.H1.per_dataset.mnist.category} | {…fmnist.category} | {confirmatory.H1.joint.category} | {sensitivity.H1.c_qmnist.category} | {confirmatory.H1.stability_joint} |
| H2 | curriculum vs wyżarzanie LR | | | | | |
| H3 | BCM vs stały próg (skalarny, per warstwa) | | | | | |
| H4 | odwrócone curriculum vs standard | | | | | |

- Zdania (§7.4 protokołu): H1 {…}; H2 {…}; H3 {…}; H4 {…}; odczyt H1 × H2 {…}.
- Obowiązkowe dopiski (§7.4): {lista}.
- Awarie treningu (NaN lub walidacja ≤ 15%) per ramię i zbiór: {accuracy.*.n_failed, n_nan}.
- Zestawienie z `PROGNOZY.md`: prognozy, które się **nie** sprawdziły: {lista}.
- Zbiór testowy oglądany w źródle (§1 protokołu); dla MNIST kategoria na nieoglądanym QMNIST-50k: {…}; rozbieżność: {brak | zdanie §7.4}.
- Odstępstwa od protokołu: {liczba} pozycji w `ODSTEPSTWA.md`.

## 1. Wykonanie

| Element | Wartość |
|---|---|
| N (Etap C), N_P, N_B | {PARAMETRY.json: N, N_P, N_B} |
| Moc planistyczna (łącznie / per zbiór) przy Δ ∈ {0, S, 1.5S, 2S} | {PARAMETRY.json: N_*_power_table} |
| Wybrane konfiguracje (lr, θ) per ramię i zbiór; θ_l `PLW` | {ETAP_B_WYBOR.json} |
| Konfiguracje na brzegu zakresu (§5.5) | {lista} |
| GPU, k, obraz, wersje (z rekordów biegów) | {…} |
| Koszt łączny | {USD} z 9.8 USD |
| Ramiona nietrenowalne (§5.6) | {brak | lista} |

## 2. Dokładność testowa (średnia ± SD po N ziarnach; liczba awarii)

| Ramię | MNIST | F-MNIST | Awarie M/F (NaN) |
|---|---|---|---|
| `STD`, `CUR`, `ANN`, `REV`, `BCM`, `PLW`, `BP` | {accuracy.ds|arm.mean ± sd} | | {n_failed (n_nan)} |

## 3. Hipotezy konfirmacyjne

Dla każdej hipotezy i zbioru: HL [p.p.] z opisowym przedziałem (1 − α′), średnia różnic (opisowo), kategoria (§7.2), lista Holma (H3), stabilność (liczby awarii f_B, f_K, p testu znaków).

| H | Komparator | HL MNIST [CI] | HL F-MNIST [CI] | Kat. MNIST | Kat. F-MNIST | Stab. MNIST | Stab. F-MNIST |
|---|---|---|---|---|---|---|---|
| H1 | `STD` | {hl_pp, hl_ci_pp} | | | | {stability.claim, counts} | |
| H2 | `ANN` | | | | | | |
| H3 | `STD` | | | | | | |
| H3 | `PLW` | | | | | | |
| H4 | `STD` | | | | | | |

Łącznie: {joint per H}; stabilność łącznie: {stability_joint per H}.

## 4. Analizy wrażliwości (§7.7)

| H | Główny | (a) bootstrap HL | (b) test t | (c) QMNIST | (d) bez par z awarią | (e) S = 0.25 | (e) S = 1.0 |
|---|---|---|---|---|---|---|---|
| H1–H4 | {joint} | {a_bootstrap_hl.joint} | {b_ttest.joint} | {c_qmnist.category} | {d_no_failed_pairs.joint} | {e_S_0.25} | {e_S_1.0} |

Twierdzenia o rozkładzie danych dopuszczalne tylko tam, gdzie (a) daje tę samą klasę łączną co analiza główna.

---

## 5. Wyniki eksploracyjne – bez wniosków konfirmacyjnych (bez korekty; mechanizmy jako hipotezy)

- **E1** – trajektorie progów BCM per warstwa: rysunek `wyniki/rysunki/E1_bcm_{ds}.png`; {opis}.
- **E2** – `BP` − najlepsze ramię FF (wybór wg walidacji Etapu C): {E2_bp_gap: HL [95% CI]}.
- **E3** – odczyty: bez warstwy 0, krzyżowy: {E3_readouts}.
- **E4** – norma zmiany wag na epokę (`STD`, `CUR`, `ANN`), czy przełączenie zmienia krok: {E4_dw_norm}.
- **E5** – krajobraz strojenia Etapu A (obowiązkowy): `wyniki/rysunki/E5_krajobraz_{ds}.png`.
- **E6** – krzywe walidacyjne: `wyniki/rysunki/E6_krzywe_{ds}.png`.
- **E7** – `ANN` − `STD`: {E7_ann_minus_std: HL [95% CI]} („wnioski o samym wyżarzaniu nie są dopuszczalne konfirmacyjnie”).

## 6. Ograniczenia

PROTOKOL_v6 §12 (1–11) oraz `ODSTEPSTWA.md`.
