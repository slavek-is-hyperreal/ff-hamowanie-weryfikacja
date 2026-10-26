# Audyt kampanii „hamowanie / curriculum / BCM” w Forward-Forward

Stan na 2026-10-26. Przedmiot audytu: `/my_data/VectorLegis/eksperymenty/` – wpis `blog/forward_forward_hamowanie.md`, kampania `forward_forward/potwierdzenie_30_ziaren/` (`PLAN.md`, `DZIENNIK.md`, kod), oraz piloty `hamowanie/` (Mandelbrot, filtry lustrzane) i `forward_forward_bramka/`. Audyt czyta kod i plan, nie uruchamia ponownie biegów.

Cel: wskazać, co trzeba poprawić, żeby główne twierdzenia blogu dało się sprawdzić z taką samą precyzją jak w `eksperyment_ff_od_zera` (prerejestracja z zewnętrznym znacznikiem czasu, test oglądany raz, kontrola błędu, równe strojenie, SESOI i moc).

## Co jest zrobione dobrze

- **Samokorekta jest jawna.** Blog i `PLAN.md` zostawiają obalone wyniki (K=3, synergia, okno krytyczne) i dopisują korekty interpretacji (np. test przetasowania permutuje przykłady, a nie neurony).
- **Hipotezy zapisywane przed biegami** w `PLAN.md`, z kryteriami.
- **Kontrole alternatywnych wyjaśnień:** odwrócone curriculum, wyżarzanie LR, większy krok, sam próg, pomiar rzeczywistego kroku Adama.
- **Testy determinizmu** (powtórka pierwszego ziarna) i sparowanie ziaren.
- **„Czysty test”** wydzielony z puli treningowej w późnej fazie kampanii.

## Problemy (od najpoważniejszych)

| # | Problem | Dowód | Skutek |
|---|---|---|---|
| A1 | **Brak zewnętrznego znacznika czasu.** „Prerejestracja” to commity w lokalnym repozytorium bez zdalnego (`git remote -v` puste). | `.git` w `eksperymenty/` | Nie da się wykazać komuś z zewnątrz, że kryterium powstało przed wynikiem. |
| A2 | **Wielokrotne testowanie bez korekty.** Ok. 24 numerowanych hipotez/kontroli (#1–#21, E1–E9, B1, Z1, Z2), werdykty z pojedynczych testów t przy p < 0.05. Korekta Holma pojawia się tylko raz, post hoc (`PLAN.md`, wiersz ~1780). | `PLAN.md` | Przy ~24 testach oczekuje się ~1 fałszywie pozytywnego werdyktu. Wyniki z t ≈ 2.0–2.6 (np. curriculum na F-MNIST t = 2.20, #12 MNIST t = 2.57, BCM F-MNIST t = 2.13) nie przeszłyby korekty. |
| A3 | **Wielokrotne użycie zbioru testowego do decyzji sekwencyjnych.** Każdy bieg ocenia na podpróbce oficjalnego testu, a kolejne hipotezy i warianty dobierano po zobaczeniu wyników testowych (przeciek θ = 8 – przyznany; wybór kolejnych kontroli, kierunek kontroli „większy krok” – przyznany). | `zadanie_fashion_mnist.py` (`X_test` = podpróbka testu per ziarno); blog | Ogród rozwidlających się ścieżek: nawet poprawnie policzony pojedynczy test jest warunkowy na wcześniejszych spojrzeniach na test. |
| A4 | **Etykieta nadpisuje 10 pierwszych pikseli.** `zakoduj_etykiete` zastępuje piksele 0–9. Na F-MNIST część obrazów ma tam niezerowe piksele (w naszym mini-pilocie: 38.9% obrazów uszkodzonych). | `dane.py` | Systematyczne obciążenie wyników F-MNIST; różnice między wariantami mogą zależeć od tego, jak wariant radzi sobie z uszkodzonym wejściem. W `ff_od_zera` rozwiązane decyzją D4 (etykieta doklejana, wejście 794). |
| A5 | **Asymetria normalizacji wejścia.** Warstwa 0 dostaje surowe x/255 (norma ~10), kolejne – wektory o normie 1. Wspólne θ dla wszystkich warstw jest przez to źle dopasowane. Blog to zauważa, ale wszystkie testy (curriculum, BCM) są w tym układzie. | `siec_ff.py`, blog | Główny „zysk” BCM i część zysku curriculum mogą być kompensacją tej asymetrii, a nie własnością metody. |
| A6 | **Nierówne strojenie.** Standard: θ z siatki 5 wartości przy 2 ziarnach (sam blog pokazuje, że wybór był błędny); LR FF stałe 1e-3 dla wszystkich wariantów; BCM, curriculum, kontrole bez osobnego strojenia; backprop strojony tylko LR. | `PLAN.md` #4, `LR_FF = 1e-3` | Porównania mogą mierzyć dopasowanie domyślnych hiperparametrów, a nie metody (pułapka K1 z naszego audytu). |
| A7 | **Brak SESOI, testów równoważności i analizy mocy** (poza jednym marginesem non-inferiority −0.5 pp w #19, przy przyznanej niskiej mocy). N = 15 (MNIST) / 30 (F-MNIST) bez uzasadnienia mocą. Zdania typu „nie wygrywa i nie przegrywa” opierają się na braku istotności. | `PLAN.md` | Brak istotności interpretowany opisowo jako remis; nie wiadomo, jakie efekty dało się wykryć. |
| A8 | **Metaanaliza z pul wybieranych sekwencyjnie** (curriculum F-MNIST, 4 pule, 120 ziaren, model stałych efektów), w tym pul z różnymi θ i warunkami. | blog, `metaanaliza_fmnist.json` | Pule nie są niezależne od decyzji badawczych; łączny p < 10⁻⁶ jest zawyżony. |
| A9 | **Ziarna współdzielone między wieloma testami.** Te same ziarna 0–29 służą wielu hipotezom. | `PLAN.md`, wyniki | Werdykty nie są niezależnymi replikacjami; błędy losowe są skorelowane między werdyktami. |
| A10 | **Reżim słabej pojemności** (10×64, 20 000 przykładów). Przyznany w blogu. | blog | Ograniczona uogólnialność; to ograniczenie, a nie błąd. |
| A11 | **Piloty `hamowanie/` i `bramka/` mają n = 2** (np. bramka 0.431 vs standard 0.597 przy n = 2). | `forward_forward_bramka/wyniki/` | Tylko eksploracja; żadnych wniosków. |
| A12 | **Predykcja sumuje dobroć wszystkich warstw, łącznie z warstwą 0**, która ma inną skalę (A5). | `SiecFF.suma_dobroci` | Warstwa 0 może dominować w decyzji; wpływa na porównania wariantów zmieniających skalę warstwy 0 (BCM). |

## Które twierdzenia blogu warto sprawdzić konfirmacyjnie

Kandydaci (wszystkie w reżimie 10×64; do wyboru 2–3, nie więcej):

1. **Curriculum (standard → hamowanie) > standard** – najczęściej powtarzany wynik blogu (MNIST +1.4 pp, F-MNIST +0.4–0.8 pp).
2. **Curriculum > kontrola wyżarzania (LR × 0.5 od połowy)** – czy curriculum to coś więcej niż zmiana harmonogramu (blog: tak na MNIST, nie na F-MNIST).
3. **Próg BCM > stały próg dostrojony per warstwa** – czy BCM robi coś ponad automatyczne dobranie progu per warstwa (blog sam podejrzewa, że głównie kompensuje A5).
4. **Odwrócone curriculum < standard** – „najczystszy wynik kampanii” wg blogu.

## Co trzeba zmienić w nowym protokole

- Publiczny znacznik czasu (GitHub, tag) przed Etapem A, jak w `ff_od_zera`.
- Zbiór testowy oglądany raz, jednym poleceniem, po zahashowaniu checkpointów.
- Etykieta doklejana (D4) i normalizacja wejścia warstwy 0 (albo oba warianty jako czynnik – do decyzji).
- Równy budżet strojenia (Sobol, te same punkty i ziarna dla każdego ramienia), w tym LR i θ (lub θ per warstwa).
- SESOI, symulacja mocy z mini-pilota, kontrola błędu (Bonferroni na hipotezy, IUT/Holm w hipotezach), TOST.
- Nowe, rozłączne pule ziaren; analiza wrażliwości na „czystym teście” albo QMNIST.
- Reużycie infrastruktury `ff_od_zera/kod/ffexp` (strumienie losowe, runner, zaślepienie, ewaluacja, analiza).
