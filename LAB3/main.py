import argparse, json, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, initializers

from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
    "Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins",
    "Color_intensity","Hue","OD280/OD315_of_diluted_wines","Proline"
]

def load_csv(path):
    # [Z1] Wczytanie surowego pliku CSV
    df = pd.read_csv(path, header=None)
    df.columns = ["Class"] + FEATURE_NAMES
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["Class"].values.astype(np.int32)  # 1..3
    return X, y


def standardize_fit(X_train):
    # [Z1] Standaryzacja
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0
    return mean, std

def standardize_apply(X, mean, std):
    # [Z1] Zastosowanie standaryzacji do zbiorów (train/test)
    return (X - mean) / std

def one_hot(y, num_classes=3):
    # [Z2] One-hot: klasy 1..3 → wektor 3D (softmax na wyjściu)
    # y in {1,2,3} -> {0,1,2}
    return tf.keras.utils.to_categorical(y - 1, num_classes=num_classes)

def build_model_A(input_dim, lr):
    # [Z3][Z4] Model A (Sequential): Dense-ReLU + He, wyjście softmax, loss=categorical_crossentropy
    # ReLU + He normal
    model = models.Sequential(name="WineNet_A_ReLU_He")
    model.add(layers.Input(shape=(input_dim,), name="input"))
    model.add(layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="dense_64"))
    model.add(layers.Dropout(0.2, name="dropout_20"))
    model.add(layers.Dense(32, activation="relu", kernel_initializer="he_normal", name="dense_32"))
    model.add(layers.Dense(3, activation="softmax", kernel_initializer="glorot_uniform", name="output"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy", # [Z3] funkcja celu dla one-hot
        metrics=["accuracy"]
    )
    return model

def build_model_B(input_dim, lr):
    # [Z3][Z4] Model B (Sequential): SELU + LeCunNormal + AlphaDropout (self-normalizing)
    # SELU + LeCun normal (+ AlphaDropout)
    model = models.Sequential(name="WineNet_B_SELU_LeCun")
    model.add(layers.Input(shape=(input_dim,), name="input"))
    model.add(layers.Dense(128, activation="selu", kernel_initializer="lecun_normal", name="dense_128"))
    model.add(layers.AlphaDropout(0.1, name="alpha_dropout_10"))
    model.add(layers.Dense(64, activation="selu", kernel_initializer="lecun_normal", name="dense_64"))
    model.add(layers.Dense(3, activation="softmax", kernel_initializer="lecun_normal", name="output"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy", # [Z3] funkcja celu dla one-hot
        metrics=["accuracy"]
    )
    return model

def main():
    # [Z5] Parametry eksperymentu (epochs, batch_size, lr) + ścieżki logów (TensorBoard)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Ścieżka do wine.csv")
    ap.add_argument("--logdir", type=str, default="runs", help="Katalog z logami TensorBoard")
    ap.add_argument("--outdir", type=str, default="artifacts", help="Katalog na modele/wykresy")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    
    # [Z1] Stratyfikowany podział train/test (po wcześniejszym potasowaniu wewnętrznym)
    # Wczytanie + tasowanie + podział (stratyfikacja)
    X, y = load_csv(args.csv)
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # [Z1] Standaryzacja — fit na TRAIN, apply na TRAIN/TEST
    # Standaryzacja (zapamiętujemy mean/std)
    mean, std = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mean, std)
    X_test  = standardize_apply(X_test,  mean, std)

    # [Z2] One-hot etykiet (3 klasy)
    # One-hot
    y_train = one_hot(y_train_int, num_classes=3)
    y_test  = one_hot(y_test_int,  num_classes=3)

    # [Z5] Callbacki: EarlyStopping + ModelCheckpoint + TensorBoard (krzywe)
    # Callbacki (EarlyStopping, ModelCheckpoint, TensorBoard)
    def cbs(run_name):
        return [
            callbacks.EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(args.outdir, f"{run_name}.keras"),
                monitor="val_accuracy", save_best_only=True, mode="max"
            ),
            callbacks.TensorBoard(log_dir=os.path.join(args.logdir, run_name))
        ]

    # [Z4] Dwie różne architektury Sequential
    # MODELE
    modelA = build_model_A(X_train.shape[1], args.lr)
    modelB = build_model_B(X_train.shape[1], args.lr)

    # [Z5] Trening (zapis historii metryk do wykresów)
    # Trening
    historyA = modelA.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=cbs("modelA")
    )
    historyB = modelB.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=cbs("modelB")
    )

    # [Z6] Ewaluacja na zbiorze testowym
    test_loss_A, test_acc_A = modelA.evaluate(X_test, y_test, verbose=0)
    test_loss_B, test_acc_B = modelB.evaluate(X_test, y_test, verbose=0)

    # [Z6] Zapis parametrów standaryzacji (do predykcji CLI)
    np.savez(os.path.join(args.outdir, "scaler_stats.npz"), mean=mean, std=std, features=np.array(FEATURE_NAMES, dtype=object))

    # [Z7] Rysowanie krzywych uczenia (Matplotlib) — loss/accuracy dla train/val
    import matplotlib.pyplot as plt

    def plot_history(hist, title, out_png):
        hist = hist.history
        epochs = range(1, len(hist["loss"])+1)
        # Loss
        plt.figure()
        plt.plot(epochs, hist["loss"], label="train_loss")
        plt.plot(epochs, hist["val_loss"], label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title} - Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, out_png.replace(".png", "_loss.png")), dpi=150)
        plt.close()
        # Accuracy
        plt.figure()
        plt.plot(epochs, hist["accuracy"], label="train_acc")
        plt.plot(epochs, hist["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title} - Accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, out_png.replace(".png", "_acc.png")), dpi=150)
        plt.close()

    plot_history(historyA, "Model A (ReLU/He)", "modelA.png")
    plot_history(historyB, "Model B (SELU/LeCun)", "modelB.png")

    # [Z8] Raport: który model lepszy + parametry eksperymentu
    best_name = "modelA" if test_acc_A >= test_acc_B else "modelB"
    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "test_size": args.test_size,
        "test_acc_A": float(test_acc_A),
        "test_acc_B": float(test_acc_B),
        "better_model": best_name,
        "artifacts_dir": args.outdir
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
