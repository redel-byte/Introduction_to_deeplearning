"""A baseline CNN with one optional, easy-to-compare change."""

import tensorflow as tf
from tensorflow.keras import layers


def build_cnn(extra_convolution=False, dropout=0.5):
    inputs = layers.Input(shape=(224, 224, 1), name="mri")
    x = inputs
    for block, filters in enumerate([32, 64, 128], start=1):
        # More filters let deeper layers combine simple edges into richer patterns.
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv_{block}")(x)
        if extra_convolution:
            # Turn this on in a separate experiment, keeping other settings fixed.
            x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"extra_conv_{block}")(x)
        x = layers.MaxPooling2D(name=f"pool_{block}")(x)
    # Averaging avoids a very large Dense layer and keeps the baseline small.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    # Keep raw class scores separate so Grad-CAM can inspect them before softmax.
    logits = layers.Dense(4, name="class_scores")(x)
    outputs = layers.Activation("softmax", name="probabilities")(logits)
    return tf.keras.Model(inputs, outputs)


def compile_model(model, learning_rate=0.0003):
    # Labels are one-hot, so categorical (not sparse categorical) is the matching loss.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
