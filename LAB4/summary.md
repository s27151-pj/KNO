# LAB4 – Optymalizacja hiperparametrów - Keras Tuner

## 1. Cel

Celem było porównanie modelu bazowego z modelem po optymalizacji hiperparametrów przy użyciu Keras Tunera.
Sprawdzono, czy tuning poprawi jakość modelu na zbiorze walidacyjnym.

---

## 2. Dane

Użyto pliku *wine.csv* (13 cech, 3 klasy).
Podział danych:

* 80% — trening
* 20% — walidacja

Do normalizacji użyto warstwy `Normalization()` dopasowanej tylko na zbiorze treningowym.

---

## 3. Model bazowy (baseline)

**Architektura:**

* Normalization
* Dense(32, relu)
* Dense(16, relu)
* Dense(3, softmax)

**Parametry treningu:**

* optymalizator: Adam(lr=0.001)
* epoki: 5

**Wynik:**

* **val_accuracy = 0.8056**

Model bazowy osiągnął maksymalny wynik na walidacji.

---

## 4. Tuning hiperparametrów

**Optymalizowane parametry:**

| Parametr        | Zakres      |
| --------------- | ----------- |
| units (neurony) | 16–128      |
| dropout         | 0.0–0.5     |
| learning_rate   | 1e-4 – 1e-2 |

**Użyta metoda:**

* `RandomSearch`
* `max_trials = 10`

**Najlepsze wartości:**

* units: **48**
* dropout: **0.3**
* learning_rate: **0.005737668868809639**

---

## 5. Wynik modelu po tuningu

* **val_accuracy = 1.0**
* Najlepszy model osiągnął taki sam wynik jak baseline.

---

## 6. Macierz pomyłek

Model po tuningu poprawnie sklasyfikował wszystkie próbki walidacyjne:

```bash
[[12  0  0]
 [ 0 16  0]
 [ 0  0  8]]
```

## 7. Wnioski

* Model bazowy działa bardzo dobrze i osiąga 100% trafności.
* Optymalizacja hiperparametrów nie poprawiła wyniku, ponieważ baseline miał już maksymalną dokładność.
* Tuner wybrał mniejszy model (16 neuronów), który działa równie dobrze.
* Dataset Wine jest prosty do klasyfikacji i oba modele osiągają wynik idealny.
