# generate_test_images.py
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf

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


def main():
    # Wczytaj Fashion-MNIST
    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

    # Upewnij się, że folder istnieje
    output_dir = "test_samples"
    os.makedirs(output_dir, exist_ok=True)

    # Wybierz 5 losowych indeksów
    indices = random.sample(range(len(x_train)), 5)

    for i, idx in enumerate(indices, start=1):
        img = x_train[idx]  # (28, 28)
        label = CLASS_NAMES[y_train[idx]]

        # PIL wymaga typu uint8
        img_pil = Image.fromarray(img.astype("uint8"), mode="L")

        # Zapisz do pliku
        filename = os.path.join(output_dir, f"test{i}.jpg")
        img_pil.save(filename)

        print(f"Saved {filename}  ->  {label}")


if __name__ == "__main__":
    main()
