# Write your findings after running the notebooks

## Data
Dataset source/link and license: TODO

Images kept, rejected and deduplicated: TODO

Train / validation / test sizes: TODO

Split unit: unique resized image, not verified independent patients. Exact pixel
hashes do not detect near-duplicates. If patient identifiers become available,
rebuild the splits by patient before drawing conclusions.

## Baseline supplied before these changes
Best validation loss: 1.0655 at epoch 2. Reported accuracy: about 55%, macro-F1:
about 0.53. The model predicted 161/261 pituitary images as notumor and 109/243
glioma images as meningioma. Training improved while validation loss worsened.
The reported split identity was not independently established; current evaluation
code uses validation. These results belong to the old run at learning rate 0.001.

## Compare your new experiments
Fill experiments.csv using each run's settings.json, epochs.csv and
validation_report.json. Compare only runs with matching dataset fingerprints.

Why was the final run chosen using validation? TODO

Which classes still have poor recall? TODO

What do the incorrect examples and Grad-CAM suggest? TODO

## Final test
Chosen model folder: TODO

Test accuracy, macro-F1, and recall per class: TODO

Was the test set previously used to select settings? TODO (report this honestly).

## Limitations
This is an educational prototype, not a diagnostic device. Dataset provenance,
patient independence and external generalization remain to be established.
Softmax scores are not calibrated medical confidence. Grad-CAM highlights
influential regions; it is not a tumor segmentation or proof of clinical reasoning.
