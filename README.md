# NeuroVision AI

A beginner CNN project that classifies grayscale brain MRI images as `glioma`,
`meningioma`, `notumor` or `pituitary`. Educational prototype, not validated for diagnosis.

## Train on Google Colab with Google Drive

1. Extract `dist/neurovision_colab.zip` on your computer. Upload the extracted
   **NeuroVisionAI** folder to **My Drive**. It contains the notebooks and their
   required `src` helper code; uploading only a notebook is not enough.
2. Keep your dataset at **My Drive/data.zip**, matching the original EDA setup.
   Change ZIP_PATH in preprocessing if your file has a different path.
3. In Google Drive, open `NeuroVisionAI/notebooks/preprocessing.ipynb` with Colab.
   Run the cells and authorize the Drive mount. The archive may contain the four
   class folders directly or inside a folder such as `Data`.
4. Open `NeuroVisionAI/notebooks/training.ipynb` with Colab. Choose
   **Runtime > Change runtime type > GPU**, then run cells in order.
5. Fixed splits are saved to `My Drive/NeuroVisionAI/data/processed`. Models,
   interruption backups, plots and reports are saved to `My Drive/NeuroVisionAI/models`.
   Both notebooks print their exact paths.

The setup installs small supporting libraries and uses Colab's preinstalled
TensorFlow. It checks that a GPU is available before training. Availability
depends on your Colab allocation; the notebook reports a clear message if none is assigned.
Preprocessing copies the ZIP to Colab's temporary disk before extracting it,
following the [Colab guidance for Drive data](https://research.google.com/colaboratory/faq.html).
Processed arrays and results are written back to Drive for later sessions.

If you already have processed arrays you want to reuse, upload the full existing
`data/processed` folder to `My Drive/NeuroVisionAI/data/processed` and skip
preprocessing. Otherwise prepare the splits once, then reuse them for comparisons.
For an already extracted dataset in Drive, set DRIVE_DATA_FOLDER in preprocessing.
The dataset and trained models are excluded from the small upload bundle.

To recover after a disconnect, remount Drive and set RESUME_RUN_DIR to the saved
run folder inside `ROOT / 'models'`. Keep the same data and experiment settings.

## Local installation (optional)

From the project folder:

```powershell
uv sync --locked
```

Select `.venv/Scripts/python.exe` as your VS Code notebook kernel. Python 3.12 is
used here; dependencies are locked in uv.lock. Training prints available GPUs.
CPU works too, but full training can take longer.

## Start here

1. Open `notebooks/training.ipynb`. It works with your existing `data/processed` files.
2. Run sections in order. RUN_SMALL_CHECK optionally tests memorization of 32 training
   images without augmentation/dropout. It does not estimate generalization.
3. Run the default baseline at learning rate 0.0003. Inspect validation curves,
   macro-F1, per-class recall, prediction errors and Grad-CAM.
4. Compare augmentation off, then extra convolution on, in separate fresh runs.
   Change one setting at a time. Keep the same dataset. Fill docs/experiments.csv.
5. Choose using validation. Only then enable RUN_FINAL_TEST and set FINAL_RUN_DIR
   to the chosen experiment folder. An existing test report blocks accidental repeats.
6. Fill docs/evaluation.md with the actual results and limitations.

No full training run or final test evaluation is performed by installation.
New notebook sections and helper functions have explanatory comments.

## Prepare data when needed

Place images under `data/raw/glioma`, `data/raw/meningioma`, `data/raw/notumor` and
`data/raw/pituitary`. Run `notebooks/preprocessing.ipynb` to load with OpenCV,
reject unreadable files, remove exact resized-image duplicates, and create fixed
stratified 70/15/15 splits with manifests and a configuration file.

**The save cell replaces files in data/processed.** Choose another OUTPUT_PATH
to preserve an earlier dataset, and set PROCESSED_PATH in training to match.
Do not rebuild splits while comparing models. Store images as uint8; the dataset
pipeline divides by 255 once. Only training is augmented. File prediction and
uploads use the same grayscale conversion, resize and normalization.

Patient independence is not established: exact hashes do not identify shared patients
or near-duplicates. If patient IDs become available, split by patient instead.
Document dataset source and license in docs/evaluation.md.

EDA.ipynb provides additional exploration; its opening cell is for Colab. Replace
that cell with a local DATA_PATH to run locally. demo.ipynb is an older MNIST exercise.

## Saved experiments

Each run creates `models/<timestamp>_<experiment>/` with:

- best.keras: model with the lowest validation loss.
- settings.json: hyperparameters, sizes, data fingerprints, TensorFlow version and duration.
- preprocessing_config.json: preprocessing details and class order.
- epochs.csv, history.json, architecture summary and learning curves.
- Validation reports, confusion matrix, scores, prediction examples and Grad-CAM.
- Test reports only after explicit final evaluation.

BackupAndRestore keeps epoch progress under the run's backup folder. To resume,
set RESUME_RUN_DIR to the same run directory and reuse its model settings and data;
follow training section 8.
A new directory starts a new experiment. Callback patience counters restart across
sessions; epochs.csv retains completed epochs. Optional architecture pictures need
Graphviz installed separately; the text summary always works.

## Prediction demo

After training:

```powershell
uv run streamlit run app/streamlit_app.py
```

Choose a run and upload JPG/JPEG/PNG/BMP. The app shows the input, predicted class,
scores and optional Grad-CAM. Unreadable files get a clear error.
Softmax scores are not calibrated medical confidence; Grad-CAM is not segmentation.

From Python:

```python
from introduction_to_deeplearning.predict import predict_image
result = predict_image('scan.jpg', 'models/YOUR_RUN_FOLDER')
print(result['class'])
print(result['scores'])
```

## Checks

```powershell
uv run python -m unittest discover -s tests -v
```

These use temporary synthetic images to check loading, rejection, deduplication,
splitting, normalization, one training update, saving/reloading, prediction,
reports and Grad-CAM. They do not measure MRI classification accuracy.

## Files

- notebooks/preprocessing.ipynb: guided data preparation.
- notebooks/training.ipynb: guided diagnostic, training and evaluation.
- src/introduction_to_deeplearning/config.py: shared paths and defaults.
- src/introduction_to_deeplearning/data.py: loader, splits and tf.data.
- src/introduction_to_deeplearning/model.py: commented CNN and compilation.
- src/introduction_to_deeplearning/evaluate.py: metrics, curves and examples.
- src/introduction_to_deeplearning/predict.py: file/upload prediction.
- src/introduction_to_deeplearning/explain.py: optional Grad-CAM.
- app/streamlit_app.py: demonstration interface.
- tests/test_workflow.py: synthetic integration checks.
- docs/roadmap.md: remaining steps.
- docs/experiments.csv: comparison table to fill.
- docs/evaluation.md: findings and limitations to fill.
- docs/jira_tasks.csv: local plan to import into Jira manually; no tickets are created.

Generated datasets and trained models are excluded from new Git tracking.

## References

- [Keras callbacks](https://keras.io/api/callbacks/)
- [Keras Grad-CAM example](https://keras.io/examples/vision/grad_cam/)
- [Streamlit documentation](https://docs.streamlit.io/)
