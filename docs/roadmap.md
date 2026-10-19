# Beginner roadmap

1. Extract dist/neurovision_colab.zip and upload its NeuroVisionAI folder to My Drive.
   Keep the dataset ZIP at My Drive/data.zip, or edit ZIP_PATH in preprocessing.
2. Open notebooks/preprocessing.ipynb in Google Colab. Mount Drive and prepare
   splits once. They are saved in My Drive/NeuroVisionAI/data/processed.
   To preserve previous splits, upload your existing processed folder there instead.
3. Keep the same processed dataset during model comparison. Running the preprocessing
   save cell again replaces those arrays. EDA.ipynb is optional additional exploration.
4. Open notebooks/training.ipynb in Colab, choose Runtime > Change runtime type > GPU,
   and run its setup cells. Models and reports go to My Drive/NeuroVisionAI/models.
   First enable RUN_SMALL_CHECK if you want to
   investigate the pipeline on 32 training images. This is a diagnostic, not a model score.
5. Train the default baseline at learning rate 0.0003. Inspect curves and errors.
6. Start a fresh run with augmentation off; compare validation metrics. Then try
   EXTRA_CONVOLUTION=True while holding the other settings fixed. Write one row
   per run in experiments.csv. Do not choose models using test results.
7. Choose one run, enable final evaluation once, and fill in evaluation.md.
8. Run the Streamlit demo and try a readable image, a corrupt file and an empty file.
9. Complete dataset provenance/license, explain limitations, and import jira_tasks.csv
   into Jira if required by your course. The CSV is a local plan, not imported tickets.

Recovery: set RESUME_RUN_DIR to the same experiment directory and use unchanged settings for
BackupAndRestore. Do not create a new directory for a resumed fit. Callback patience
counters restart in a new Python session; the epoch CSV records all completed epochs.

Useful references:
- https://keras.io/api/callbacks/
- https://keras.io/examples/vision/grad_cam/
- https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
