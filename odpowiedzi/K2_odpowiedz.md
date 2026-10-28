# Odpowiedź na recenzję kodu K2

| Uwaga | Zmiana |
|---|---|
| K2-01 | `final.py --plan`: lista ramion z planu C; brak `BP` (cięcie §8.3) nie blokuje analizy (`bp_cut_8_3`); test `test_bp_cut_and_tampering` |
| K2-02 | `verify` wymaga wpisów FILE dla dokładnie wszystkich rekordów i plików `zaslepione` C; `final.py --hashes` uruchamia `verify` i `verify_predictions` przed analizą; test podmiany pliku `zaslepione` i rekordu NaN |
| K2-03 | `notes`: zdania NIESPÓJNY-*, rozbieżność klasy testu t, dopisek §4.1 przy H3, zdanie o braku `ANN − STD`, awarie i stabilność jako zdania z licznościami i N, brak dopisku mocy przy ramieniu nietrenowalnym, „BCM nietrenowalne (zbiór)” |
| K2-04 | `wybor_n.collect/decide`: hipotezy z ramieniem nietrenowalnym pomijane (bez celu mocy) |
| K2-05 | NaN/inf w σ → twardy błąd z komunikatem |
| K2-06 | generator (vi) poszerzony; nowy scenariusz pokrycia kategorii (R+, NIESPÓJNY-ZGODNY/KIERUNEK, STAB-LEPSZA/GORSZA) – 35/35 |
| K2-07 | T7: predykcja klasy 0 dla biegu z NaN sprawdzana wprost; podmiana rekordu NaN odrzucana |
| K2-08 | bootstrap stosuje R + STAB → N (stabilność z analizy głównej) |
| K2-09 | `verify_predictions` porównuje zbiór plików z planem; brak pliku = odmowa |
| K2-10 | §8.2 T1 – tekst zgodny z testem (wagi i progi BCM); `HAMEXP_IMAGE` w rekordzie biegu; znane słabości w `kod/README.md` |
