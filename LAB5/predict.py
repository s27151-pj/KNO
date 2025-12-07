# predict.py
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def preprocess_image(path):
    """
    Wczytuje obrazek i przetwarza go tak, aby pasował do Fashion-MNIST.
    1. Grayscale
    2. Resize 28x28
    3. Negatyw
    4. Normalizacja [0,1]
    """
    img = Image.open(path).convert("L")  # grayscale

    # Zmiana rozmiaru
    img = img.resize((28, 28))

    img = np.array(img).astype("float32")

    # NEGATYW: Fashion-MNIST = białe tło, ciemny obiekt
    img = 255 - img

    # Normalizacja
    img = img / 255.0

    # Dodaj wymiar kanału
    img = np.expand_dims(img, axis=-1)  # (28, 28, 1)

    # Dodaj batch dimension
    img = np.expand_dims(img, axis=0)   # (1, 28, 28, 1)

    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Predict class of clothing image")
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("--image", required=True, help="Path to image file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Wczytaj model
    model = load_model(args.model)

    # Przetwarzamy obraz
    img = preprocess_image(args.image)

    # Predykcja
    probs = model.predict(img)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]

    print(f"Predicted class: {CLASS_NAMES[pred_class]}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
