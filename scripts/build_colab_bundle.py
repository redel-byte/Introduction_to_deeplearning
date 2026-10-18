"""Package the notebooks and helper code for a simple Google Drive folder upload.

Run: uv run python scripts/build_colab_bundle.py
Images, processed arrays and model weights are deliberately left out.
"""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "dist" / "neurovision_colab.zip"
output.parent.mkdir(exist_ok=True)

files = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "uv.lock"]
files += sorted((ROOT / "src" / "introduction_to_deeplearning").glob("*.py"))
files += sorted((ROOT / "notebooks").glob("*.ipynb"))
files += sorted((ROOT / "docs").glob("*.*"))
files += [ROOT / "app" / "streamlit_app.py"]
files += sorted((ROOT / "tests").glob("*.py"))
files += [ROOT / "scripts" / "build_colab_bundle.py"]

with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        # A single outer folder makes the required Drive folder structure unambiguous.
        archive.write(path, Path("NeuroVisionAI") / path.relative_to(ROOT))

print(f"Upload bundle: {output}")
print("Extract it, then upload the NeuroVisionAI folder to My Drive.")
