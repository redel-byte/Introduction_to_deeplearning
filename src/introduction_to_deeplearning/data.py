"""Small, reusable functions shared by the notebooks and the application."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED, VALID_EXTENSIONS


def read_image_bytes(contents):
    """Decode uploaded bytes exactly as we decode a training image."""
    if not contents:
        raise ValueError("The image file is empty.")
    try:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise ValueError("This file cannot be read as an image.") from error
    if image is None:
        raise ValueError("This file cannot be read as an image.")
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image[..., np.newaxis]  # Add the single grayscale channel.


def load_images(data_dir):
    """Read images, reject unreadable files, and remove exact duplicates."""
    images, labels, records, rejected = [], [], [], []
    seen = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        folder = Path(data_dir) / class_name
        if not folder.is_dir():
            raise ValueError(f"Missing class folder: {folder}")
        # Sorting makes file order repeatable before the seeded split.
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VALID_EXTENSIONS:
                rejected.append({"path": str(path), "reason": "unsupported extension"})
                continue
            try:
                image = read_image_bytes(path.read_bytes())
            except (OSError, ValueError, cv2.error) as error:
                rejected.append({"path": str(path), "reason": str(error)})
                continue
            # Hash the actual model input, so identical resized images stay out of different splits.
            signature = hashlib.sha256(image.tobytes()).hexdigest()
            if signature in seen:
                if seen[signature] != class_name:
                    raise ValueError(f"Same image has conflicting labels: {path}. Inspect it first.")
                rejected.append({"path": str(path), "reason": "exact duplicate"})
                continue
            seen[signature] = class_name
            images.append(image)
            labels.append(class_index)
            records.append({"path": str(path), "label": class_name, "pixel_hash": signature})
    if not images:
        raise ValueError("No readable images found. Check the dataset path.")
    return np.asarray(images, dtype=np.uint8), np.asarray(labels), pd.DataFrame(records), pd.DataFrame(rejected, columns=["path", "reason"])


def save_splits(images, labels, records, output_dir):
    """Split indices first, keeping each image, label and source path together."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = np.arange(len(images))
    train_indices, remaining = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    val_indices, test_indices = train_test_split(
        remaining, test_size=0.50, stratify=labels[remaining], random_state=RANDOM_SEED
    )
    for name, selection in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        np.save(output_dir / f"X_{name}.npy", images[selection])
        np.save(output_dir / f"y_{name}_int.npy", labels[selection])
        # One-hot label: class 2 becomes [0, 0, 1, 0]. This matches categorical_crossentropy.
        one_hot = np.eye(len(CLASS_NAMES), dtype=np.float32)[labels[selection]]
        np.save(output_dir / f"y_{name}.npy", one_hot)
        records.iloc[selection].to_csv(output_dir / f"{name}_manifest.csv", index=False)
        print(f"{name}: {len(selection)} images")
    config = {
        "class_names": list(CLASS_NAMES), "image_size": list(IMAGE_SIZE),
        "channels": 1, "stored_image_dtype": "uint8", "seed": RANDOM_SEED,
        "batch_size": 32, "normalization": "divide by 255 exactly once in make_dataset",
        "label_format": "one_hot", "split_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "split_unit": "unique resized image; patient independence is not established",
        "duplicate_check": "SHA256 of resized grayscale pixels; near-duplicates not checked",
    }
    (output_dir / "preprocessing_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_split(processed_dir, name):
    """Check stored arrays before feeding them to the model."""
    images = np.load(Path(processed_dir) / f"X_{name}.npy")
    labels = np.load(Path(processed_dir) / f"y_{name}.npy")
    if images.dtype != np.uint8 or images.shape[1:] != (224, 224, 1):
        raise ValueError("Expected uint8 images with shape (N, 224, 224, 1).")
    if labels.shape != (len(images), 4):
        raise ValueError("Expected four one-hot labels for each image.")
    if not np.all((labels == 0) | (labels == 1)) or not np.all(labels.sum(axis=1) == 1):
        raise ValueError("Labels must be one-hot: exactly one 1 in each row.")
    return images, labels


def make_dataset(images, labels, training=False, augment=True, batch_size=32, seed=42):
    """Normalize once; shuffle and augment training images only."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # Arrays are already in memory, so another full cache is unnecessary here.
    if training:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training and augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.03, fill_mode="constant", seed=seed),
            tf.keras.layers.RandomZoom(0.05, fill_mode="constant", seed=seed + 1),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed + 2),
        ])
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)
