# Prognozy przed pilotem (PROTOKOL_v6 §9 pkt 1)

Zapisane 2026-10-28, przed pilotem i przed jakimkolwiek wynikiem tego badania. Jedyna wcześniejsza informacja z tego układu: jedna krzywa walidacyjna z testu dymnego (`DZIENNIK.md`). Podstawa prognoz: blog źródłowy, wyniki `eksperyment_ff_od_zera`, literatura (Polar-FFA, Akkus i in.).

Format: kategoria per zbiór → łącznie; „HL” – oczekiwana pseudomediana różnicy [p.p.].

## Asystent (Claude)

| H | MNIST | F-MNIST | Łącznie | Pewność | Uzasadnienie |
|---|---|---|---|---|---|
| H1 `CUR` vs `STD` | PRZEWAGA<S (HL ≈ +0.4) | NIEROZSTRZYGNIĘTY / RÓWNOWAŻNOŚĆ (HL ≈ +0.1) | NIESPÓJNY-CZĘŚCIOWY | 40% | źródło: MNIST +1.4, F-MNIST +0.4–0.8, ale część zysku źródła to niedostrojony standard i asymetria wejścia; równe strojenie zmniejszy efekt |
| H2 `CUR` vs `ANN` | NIEROZSTRZYGNIĘTY (HL ≈ +0.2) | RÓWNOWAŻNOŚĆ (HL ≈ 0) | klasa N | 45% | źródło: na F-MNIST remis z wyżarzaniem; na MNIST zysk mógł zależeć od nienormalizowanego wejścia |
| H3 `BCM` vs `STD`, `PLW` | NIEROZSTRZYGNIĘTY / RÓWNOWAŻNOŚĆ | NIEROZSTRZYGNIĘTY / RÓWNOWAŻNOŚĆ | klasa N lub R | 55% | po normalizacji wejścia asymetria warstwy 0 maleje (choć test dymny pokazał, że progi nadal się różnią); z `PLW` (te same progi) różnica powinna zniknąć |
| H4 `REV` vs `STD` | GORSZOŚĆ<S lub GORSZOŚĆ (HL ≈ −1.0) | GORSZOŚĆ<S (HL ≈ −0.4) | klasa G | 60% | „najczystszy wynik” źródła; mechanizm (zapaść reprezentacji warstwy 0 przy hamowaniu na losowych wagach) nie zależy od wejścia |
| Stabilność | brak różnic stabilności we wszystkich H | | | 70% | NaN rzadkie przy wybranych konfiguracjach |
| Ogólnie | brak PRZEWAGI (pełnej, > S) łącznie w H1–H3 | | | 80% | |

## Autor (Sławek)

Brak własnej prognozy autora. Autor (2026-10-28): „wpisz swoją prognozę” – polecił zarejestrować wyłącznie prognozę asystenta (powyżej).
