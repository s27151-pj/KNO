import argparse
import json
import numpy as np
import tensorflow as tf

# Nazwy cech z pliku wine.data
FEATURE_NAMES = [
    "Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
    "Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins",
    "Color_intensity","Hue","OD280/OD315_of_diluted_wines","Proline"
]

def load_scaler(npz_path):
    # [Z9] Odczyt parametrów standaryzacji zapisanych podczas treningu
    data = np.load(npz_path, allow_pickle=True)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    return mean, std

def standardize(x, mean, std):
    # [Z9] Standaryzacja wejścia użytkownika
    return (x - mean) / std

def main():
    # [Z9] Parser CLI — wymagane 13 argumentów cech + ścieżki do modelu i skalera
    parser = argparse.ArgumentParser(description="Predict wine class (1, 2, or 3)")
    parser.add_argument("--model", required=True, help="Path to trained model (.keras)")
    parser.add_argument("--scaler", required=True, help="Path to scaler_stats.npz")
    
    # dodaj parametry dla każdej cechy
    for f in FEATURE_NAMES:
        parser.add_argument(f"--{f}", type=float, required=True)
    
    args = parser.parse_args()

    # [Z9] Zebranie cech w kolejności FEATURE_NAMES
    # Wczytaj parametry wina do listy
    x_input = np.array([[getattr(args, f) for f in FEATURE_NAMES]], dtype=np.float32)

    # [Z9] Standaryzacja + predykcja softmax
    mean, std = load_scaler(args.scaler)
    model = tf.keras.models.load_model(args.model)

    # Przekształć dane i wykonaj predykcję
    x_std = standardize(x_input, mean, std)
    probs = model.predict(x_std, verbose=0)[0]
    predicted_class = int(np.argmax(probs) + 1)  # klasy 1-3

    # [Z9] Wynik numeryczny + prawdopodobieństwa
    print(json.dumps({
        "Predicted_class": predicted_class,
        "Probabilities": [round(float(p), 4) for p in probs]
    }, indent=2))

if __name__ == "__main__":
    main()
