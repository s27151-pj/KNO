# train_engine.py
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from models import build_model


def load_fashion_mnist():
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalizacja wartości pikseli 0–255 → 0–1
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Dodanie wymiaru kanału (TensorFlow oczekuje kształtu 28×28×1)
    x_train_full = np.expand_dims(x_train_full, -1)
    x_test = np.expand_dims(x_test, -1)

    X_train, X_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    return X_train, X_val, y_train, y_val, x_test, y_test


def train_model(model_type, use_augmentation, epochs, batch_size, model_path, metrics_path):
    X_train, X_val, y_train, y_val, X_test, y_test = load_fashion_mnist()

    model = build_model(model_type=model_type, hp=None, use_augmentation=use_augmentation)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    metrics = {
        "model_type": model_type,
        "use_augmentation": use_augmentation,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "history": history.history
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
