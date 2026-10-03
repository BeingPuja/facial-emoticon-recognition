"""
evaluate.py

Loads a trained model and produces:
- Overall test accuracy
- Confusion matrix heatmap
- Per-class precision / recall / F1 (classification report)

Usage:
    python evaluate.py --model outputs/custom_cnn.keras
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

from data_preprocessing import load_fer2013, EMOTION_LABELS

OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved .keras model")
    args = parser.parse_args()

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    data = load_fer2013()
    model = load_model(args.model)

    y_true = np.argmax(data["y_test"], axis=1)
    y_pred_probs = model.predict(data["X_test"], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    labels_ordered = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]

    test_acc = (y_true == y_pred).mean()
    print(f"\n{model_name} — Test Accuracy: {test_acc:.4f}\n")

    report = classification_report(
        y_true, y_pred, target_names=labels_ordered, digits=3
    )
    print(report)
    report_path = os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_ordered, yticklabels=labels_ordered, ax=axes[0])
    axes[0].set_title(f"{model_name} — Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels_ordered, yticklabels=labels_ordered, ax=axes[1])
    axes[1].set_title(f"{model_name} — Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    main()
