"""
train.py

Trains either the custom CNN or the MobileNetV2 transfer model on FER-2013.

Usage:
    python train.py --model custom --epochs 40
    python train.py --model mobilenet --epochs 20
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from data_preprocessing import load_fer2013, get_class_weights, get_augmentation_generator
from models import build_custom_cnn, build_mobilenet_transfer

OUTPUT_DIR = "outputs"


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "mobilenet"], required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_fer2013()
    class_weights = get_class_weights(data["raw_labels_train"])

    if args.model == "custom":
        model = build_custom_cnn()
        lr = 1e-3
        model_path = os.path.join(OUTPUT_DIR, "custom_cnn.keras")
    else:
        model = build_mobilenet_transfer()
        lr = 1e-4  # smaller LR: fine-tuning a pretrained model
        model_path = os.path.join(OUTPUT_DIR, "mobilenet_transfer.keras")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    datagen = get_augmentation_generator()
    train_flow = datagen.flow(
        data["X_train"], data["y_train"], batch_size=args.batch_size
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print(f"\nTraining {args.model} model for up to {args.epochs} epochs...\n")
    history = model.fit(
        train_flow,
        validation_data=(data["X_val"], data["y_val"]),
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    history_path = os.path.join(OUTPUT_DIR, f"{args.model}_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)

    plot_history(history, os.path.join(OUTPUT_DIR, f"{args.model}_training_curves.png"))

    test_loss, test_acc = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    print(f"\nFinal test accuracy ({args.model}): {test_acc:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
