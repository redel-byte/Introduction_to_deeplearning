"""Check Colab setup using a temporary fake Drive, without account access."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def code_cells(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


class ColabNotebookTests(unittest.TestCase):
    def test_drive_archive_paths_and_gpu_message(self):
        # Only the Drive mount and installer are mocked. Image extraction and saving are real.
        with tempfile.TemporaryDirectory() as folder:
            temporary_root = Path(folder)
            project = temporary_root / "MyDrive" / "NeuroVisionAI"
            project.mkdir(parents=True)
            shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
            archive_path = temporary_root / "MyDrive" / "data.zip"
            fake_colab = types.ModuleType("google.colab")
            fake_colab.drive = types.SimpleNamespace(mount=Mock())

            def temporary_paths(source):
                # Redirect notebook paths to the temporary test directory on this computer.
                return (source.replace("/content/drive/MyDrive/NeuroVisionAI", project.as_posix())
                        .replace("/content/drive/MyDrive/data.zip", archive_path.as_posix())
                        .replace("dir='/content'", f"dir={temporary_root.as_posix()!r}"))

            preprocessing = code_cells("preprocessing.ipynb")
            training = code_cells("training.ipynb")
            namespace = {"__name__": "__main__"}
            try:
                with patch.dict(sys.modules, {"google.colab": fake_colab}):
                    setup = next(s for s in preprocessing if "drive.mount" in s)
                    exec(temporary_paths(setup), namespace)
                fake_colab.drive.mount.assert_called_once_with("/content/drive")
                dependency_setup = next(s for s in preprocessing if "subprocess.check_call" in s)
                with patch("subprocess.check_call") as install:
                    exec(dependency_setup, namespace)
                arguments = install.call_args.args[0]
                self.assertNotIn("tensorflow", arguments)

                for nesting in ["", "Data/"]:
                    with self.subTest(archive_structure=nesting or "direct class folders"):
                        rng = np.random.default_rng(42)
                        with zipfile.ZipFile(archive_path, "w") as archive:
                            for name in ["glioma", "meningioma", "notumor", "pituitary"]:
                                for index in range(10):
                                    image = rng.integers(0, 256, (32, 32), dtype=np.uint8)
                                    _, encoded = cv2.imencode(".png", image)
                                    archive.writestr(f"{nesting}{name}/{index}.png", encoded.tobytes())
                        archive_setup = next(s for s in preprocessing if "ZIP_PATH =" in s)
                        exec(temporary_paths(archive_setup), namespace)
                        self.assertTrue((namespace["DATA_PATH"] / "glioma").is_dir())
                        from introduction_to_deeplearning.data import load_images, save_splits
                        images, labels, records, _ = load_images(namespace["DATA_PATH"])
                        save_splits(images, labels, records, namespace["OUTPUT_PATH"])
                        self.assertEqual(namespace["OUTPUT_PATH"], project / "data" / "processed")

                imports = next(s for s in training if "import hashlib" in s)
                exec(imports, namespace)
                settings = next(s for s in training if "EXPERIMENT_NAME =" in s)
                with patch.object(tf.config, "list_physical_devices", return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, "Enable a GPU"):
                        exec(settings, namespace)
                # The GPU check has been exercised; load data without running a training job.
                load_data = next(s for s in training if "X_train, y_train = load_split" in s)
                exec(load_data, namespace)
                self.assertEqual(namespace["MODELS_DIR"], project / "models")
                self.assertEqual(namespace["X_train"].shape, (28, 224, 224, 1))
                self.assertEqual(namespace["X_val"].shape, (6, 224, 224, 1))
                archive_path.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "Dataset archive not found"):
                    exec(temporary_paths(archive_setup), namespace)
            finally:
                sys.path.remove(str(project / "src"))


if __name__ == "__main__":
    unittest.main()
