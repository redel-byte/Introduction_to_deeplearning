"""Plots and metrics used on validation, then once on the final test set."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def plot_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.history["loss"]) + 1)
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history.history[metric], label="train")
        ax.plot(epochs, history.history[f"val_{metric}"], label="validation")
        ax.set(xlabel="Epoch", ylabel=metric, title=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "learning_curves.png")
    return fig


def evaluate_split(model, images, labels, class_names, output_dir, split_name="validation"):
    from .data import make_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # No shuffle or augmentation: prediction order matches labels and source images.
    dataset = make_dataset(images, labels)
    scores = model.predict(dataset, verbose=0)
    true = np.argmax(labels, axis=1)
    predicted = np.argmax(scores, axis=1)
    class_ids = np.arange(len(class_names))
    text = classification_report(true, predicted, labels=class_ids, target_names=class_names, zero_division=0)
    report = classification_report(true, predicted, labels=class_ids, target_names=class_names, zero_division=0, output_dict=True)
    print(text)
    print(f"Macro-F1: {report['macro avg']['f1-score']:.3f}")
    (output_dir / f"{split_name}_report.txt").write_text(text, encoding="utf-8")
    (output_dir / f"{split_name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    matrix = confusion_matrix(true, predicted, labels=class_ids)
    np.save(output_dir / f"{split_name}_confusion.npy", matrix)
    np.save(output_dir / f"{split_name}_scores.npy", scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{split_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_confusion.png")
    return scores, fig


def plot_examples(images, labels, scores, class_names, output_dir, split_name="validation"):
    true, predicted = labels.argmax(axis=1), scores.argmax(axis=1)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for row, correct in enumerate([True, False]):
        # Fixed order makes it easy to compare examples between experiments.
        indices = np.flatnonzero((true == predicted) == correct)[:4]
        for column, ax in enumerate(axes[row]):
            ax.axis("off")
            if column < len(indices):
                i = indices[column]
                ax.imshow(images[i, ..., 0], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"Index {i} | true: {class_names[true[i]]}\npred: {class_names[predicted[i]]} ({scores[i].max():.2f})", fontsize=9)
    fig.suptitle("Top: correct predictions | Bottom: incorrect predictions")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"{split_name}_examples.png")
    return fig
