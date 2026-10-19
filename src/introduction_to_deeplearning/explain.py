"""A small Grad-CAM implementation following the official Keras example.

Reference: https://keras.io/examples/vision/grad_cam/
"""

import cv2
import numpy as np
import tensorflow as tf


def make_gradcam(image, model, class_index=None):
    # image is a single uint8 image, shaped (224, 224, 1).
    batch = image[np.newaxis].astype(np.float32) / 255.0
    # Find the deepest convolution, including the optional extra convolution.
    last_conv = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    gradient_model = tf.keras.Model(model.input, [last_conv.output, model.get_layer("class_scores").output])
    with tf.GradientTape() as tape:
        features, logits = gradient_model(batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        selected_score = logits[:, class_index]
    # Gradients measure how a small feature change affects the selected class score.
    gradients = tape.gradient(selected_score, features)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(features[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0).numpy()
    # A zero map is possible; avoid division by zero in that case.
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return cv2.resize(heatmap, (image.shape[1], image.shape[0]))
