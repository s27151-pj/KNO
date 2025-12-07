# models.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def get_augmentation_layer():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def build_dense_model(hp=None, use_augmentation=False):
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs

    if use_augmentation:
        x = get_augmentation_layer()(x)

    x = layers.Flatten()(x)

    if hp:
        units1 = hp.Int("dense_units1", 64, 256, step=64)
        units2 = hp.Int("dense_units2", 32, 128, step=32)
        dropout_rate = hp.Float("dense_dropout", 0.0, 0.5, step=0.1)
    else:
        units1, units2, dropout_rate = 128, 64, 0.3

    x = layers.Dense(units1, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(units2, activation="relu")(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="fashion_dense")

    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log") if hp else 1e-3

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(hp=None, use_augmentation=False):
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs

    if use_augmentation:
        x = get_augmentation_layer()(x)

    if hp:
        f1 = hp.Int("conv_filters1", 32, 64, step=16)
        f2 = hp.Int("conv_filters2", 64, 128, step=16)
        dense_units = hp.Int("cnn_dense_units", 64, 256, step=64)
        dropout_rate = hp.Float("cnn_dropout", 0.0, 0.5, step=0.1)
    else:
        f1, f2, dense_units, dropout_rate = 32, 64, 128, 0.3

    x = layers.Conv2D(f1, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(f2, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="fashion_cnn")

    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log") if hp else 1e-3

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(model_type="dense", hp=None, use_augmentation=False):
    if model_type == "dense":
        return build_dense_model(hp=hp, use_augmentation=use_augmentation)
    elif model_type == "cnn":
        return build_cnn_model(hp=hp, use_augmentation=use_augmentation)
    else:
        raise ValueError("Unknown model_type: choose dense or cnn")
