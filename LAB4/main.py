import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras_tuner as kt

def load_data():
    data = pd.read_csv("../LAB3/wine.csv")

    X = data.drop("Class", axis=1).values.astype(np.float32)
    y = data["Class"].values.astype(np.int32) - 1  # 0–2

    y = tf.keras.utils.to_categorical(y, num_classes=3)

    return X, y

X, y = load_data()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Warstwa normalizacji
norm_layer = layers.Normalization()
norm_layer.adapt(X_train)

def create_baseline_model():
    model = models.Sequential([
        layers.Input(shape=(13,)),
        norm_layer,
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def build_model_hp(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(13,)))
    model.add(norm_layer)

    units = hp.Int("units", min_value=16, max_value=128, step=16)
    dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")

    model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

baseline_model = create_baseline_model()
history = baseline_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    verbose=0
)

baseline_val_acc = history.history["val_accuracy"][-1]
print("Baseline Validation Accuracy:", baseline_val_acc)

baseline_model.save("baseline_model.keras")

tuner = kt.RandomSearch(
    build_model_hp,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory="tuner",
    project_name="wine_hp"
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]

y_pred = np.argmax(best_model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion matrix:")
print(cm)

best_model.save("best_model_hp.keras")

print("\nBest Hyperparameters:", best_hp.values)
