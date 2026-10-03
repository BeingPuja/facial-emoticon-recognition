"""
gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing which
regions of a face image most influenced the model's predicted emotion.

Method (for viva): computes the gradient of the predicted class score with
respect to the feature maps of the last convolutional layer, global-average
pools those gradients to get per-channel importance weights, then takes a
weighted sum of the feature maps followed by ReLU -- producing a coarse
heatmap over the spatial regions that most increased the predicted class
score.

Usage:
    python gradcam.py --model outputs/custom_cnn.keras --num_samples 8
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

from data_preprocessing import load_fer2013, EMOTION_LABELS

OUTPUT_DIR = "outputs"


def find_last_conv_layer(model):
    """Finds the last Conv2D layer's name automatically. Handles both the
    custom CNN and models with a nested pretrained base (MobileNetV2)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):  # nested sub-model (e.g., MobileNetV2)
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer.name  # use the sub-model as the target layer
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(img_gray_uint8, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (img_gray_uint8.shape[1], img_gray_uint8.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_color = cv2.cvtColor(img_gray_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_color, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    model = load_model(args.model)
    last_conv_layer_name = find_last_conv_layer(model)
    print(f"Using last conv layer: {last_conv_layer_name}")

    data = load_fer2013()
    y_true = np.argmax(data["y_test"], axis=1)

    # Pick one sample per emotion class (up to num_samples classes) for a
    # clean, presentation-ready grid.
    n = min(args.num_samples, len(EMOTION_LABELS))
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))

    for i in range(n):
        idx = np.where(y_true == i)[0][0]
        img = data["X_test"][idx]
        img_batch = np.expand_dims(img, axis=0)

        heatmap, pred_class = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)
        img_uint8 = (img.squeeze() * 255).astype("uint8")
        overlay = overlay_heatmap(img_uint8, heatmap)

        axes[0, i].imshow(img_uint8, cmap="gray")
        axes[0, i].set_title(f"True: {EMOTION_LABELS[i]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        correct = "✓" if pred_class == i else "✗"
        axes[1, i].set_title(f"Pred: {EMOTION_LABELS[pred_class]} {correct}", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle(f"{model_name} — Grad-CAM: original (top) vs attention heatmap (bottom)")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_gradcam.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved Grad-CAM visualization to {out_path}")


if __name__ == "__main__":
    main()
