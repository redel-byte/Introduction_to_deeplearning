"""Small integration checks; these do not train or evaluate the real dataset.

Run: uv run python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Save plots without opening windows during automated checks.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from introduction_to_deeplearning.config import ROOT_DIR, CLASS_NAMES
from introduction_to_deeplearning.data import read_image_bytes, load_images, save_splits, load_split, make_dataset
from introduction_to_deeplearning.model import build_cnn, compile_model
from introduction_to_deeplearning.predict import load_predictor, predict_bytes, predict_image
from introduction_to_deeplearning.explain import make_gradcam
from introduction_to_deeplearning.evaluate import evaluate_split, plot_examples


class WorkflowTests(unittest.TestCase):
    def test_project_root(self):
        self.assertTrue((ROOT_DIR / "pyproject.toml").is_file())

    def test_invalid_images(self):
        for contents in [b"", b"not an image"]:
            with self.assertRaises(ValueError):
                read_image_bytes(contents)

    def test_loader_duplicates_splits_and_normalization(self):
        # Synthetic images let us check the loader without touching real MRI data.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            for name in CLASS_NAMES:
                (root / name).mkdir()
                for i in range(10):
                    image = rng.integers(0, 256, (30, 40), dtype=np.uint8)
                    cv2.imwrite(str(root / name / f"{i}.png"), image)
            original = root / CLASS_NAMES[0] / "0.png"
            (root / CLASS_NAMES[0] / "copy.png").write_bytes(original.read_bytes())
            (root / CLASS_NAMES[0] / "broken.jpg").write_bytes(b"broken")
            images, labels, records, rejected = load_images(root)
            self.assertEqual(len(images), 40)
            self.assertEqual(len(rejected), 2)
            output = root / "processed"
            save_splits(images, labels, records, output)
            hashes = []
            for name in ["train", "val", "test"]:
                x, y = load_split(output, name)
                import pandas as pd
                hashes.extend(pd.read_csv(output / f"{name}_manifest.csv")["pixel_hash"])
                batch, batch_labels = next(iter(make_dataset(x, y, batch_size=4)))
                np.testing.assert_allclose(batch.numpy(), x[:4].astype(np.float32) / 255, atol=1e-7)
                np.testing.assert_array_equal(batch_labels.numpy(), y[:4])
            self.assertEqual(len(hashes), len(set(hashes)))
            # A copy under another class must be rejected as a label conflict.
            (root / CLASS_NAMES[1] / "conflict.png").write_bytes(original.read_bytes())
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                load_images(root)

    def test_model_save_prediction_heatmap_and_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            tf.keras.utils.set_random_seed(42)
            model = build_cnn()
            compile_model(model)
            rng = np.random.default_rng(42)
            images = rng.integers(0, 256, (4, 224, 224, 1), dtype=np.uint8)
            labels = np.eye(4, dtype=np.float32)
            # One update checks that gradients/loss work; it makes no accuracy claim.
            metrics = model.train_on_batch(images.astype(np.float32) / 255, labels)
            self.assertTrue(np.isfinite(metrics).all())
            model.save(run / "best.keras")
            (run / "preprocessing_config.json").write_text(json.dumps({"class_names": list(CLASS_NAMES)}))
            loaded, names = load_predictor(run)
            _, encoded = cv2.imencode(".png", images[0, ..., 0])
            result = predict_bytes(encoded.tobytes(), loaded, names)
            self.assertIn(result["class"], CLASS_NAMES)
            self.assertAlmostEqual(sum(result["scores"].values()), 1.0, places=5)
            path = run / "example.png"
            path.write_bytes(encoded.tobytes())
            self.assertEqual(predict_image(path, run)["class"], result["class"])
            with self.assertRaises(ValueError):
                predict_image(run / "missing.png", run)
            heatmap = make_gradcam(result["image"], loaded)
            self.assertEqual(heatmap.shape, (224, 224))
            self.assertTrue(np.isfinite(heatmap).all())
            self.assertGreaterEqual(heatmap.min(), 0)
            self.assertLessEqual(heatmap.max(), 1.00001)
            # Evaluation and file prediction must agree on input normalization and class order.
            scores, _ = evaluate_split(loaded, images, labels, names, run)
            np.testing.assert_allclose(scores[0], list(result["scores"].values()), atol=1e-6)
            plot_examples(images, labels, scores, names, run)
            self.assertTrue((run / "validation_report.json").is_file())
            self.assertTrue((run / "validation_examples.png").is_file())
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
