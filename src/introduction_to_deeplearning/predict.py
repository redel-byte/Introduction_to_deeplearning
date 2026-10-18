"""Use the same image conversion for training, file prediction and uploads."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import read_image_bytes


def load_predictor(run_dir):
    run_dir = Path(run_dir)
    model = tf.keras.models.load_model(run_dir / "best.keras")
    config = json.loads((run_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
    return model, config["class_names"]


def predict_bytes(contents, model, class_names):
    image = read_image_bytes(contents)
    # Add the batch dimension, then normalize exactly once, just like make_dataset.
    batch = image[np.newaxis].astype(np.float32) / 255.0
    scores = model(batch, training=False).numpy()[0]
    return {"class": class_names[int(scores.argmax())],
            "scores": dict(zip(class_names, scores.astype(float).tolist())),
            "image": image}


def predict_image(path, run_dir):
    """Example: predict_image('scan.jpg', 'models/my_run')."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")
    model, class_names = load_predictor(run_dir)
    return predict_bytes(path.read_bytes(), model, class_names)
