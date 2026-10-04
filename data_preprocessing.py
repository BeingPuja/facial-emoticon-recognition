"""
data_preprocessing.py

Loads the FER-2013 dataset and prepares train/validation/test splits with
classical image-processing preprocessing (CLAHE contrast enhancement +
normalization) and data augmentation.

Supports TWO dataset layouts automatically (whichever you downloaded from
Kaggle):

  1. CSV format:     data/fer2013.csv  (columns: emotion, pixels, Usage)
  2. Folder format:  data/train/<emotion_name>/*.jpg
                      data/test/<emotion_name>/*.jpg
     (this is the "Face expression recognition dataset" / jonathanoheix
     layout on Kaggle, with one subfolder per emotion)

FER-2013 label mapping (fixed by the dataset):
    0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy,
    4 = Sad,   5 = Surprise, 6 = Neutral
"""

import os
import glob
import numpy as np
import pandas as pd
# TensorFlow is imported before cv2 deliberately: on Linux, importing
# OpenCV first and TensorFlow second can trigger a native segmentation
# fault from conflicting bundled libraries. See the same note in
# streamlit_app.py.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "outputs"
IMG_SIZE = 48
NUM_CLASSES = 7

EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# Maps a lowercased folder name to our fixed emotion id, so it doesn't
# matter what order the folders happen to be in on disk.
_NAME_TO_ID = {name.lower(): idx for idx, name in EMOTION_LABELS.items()}
# A couple of common spelling variants seen across different Kaggle mirrors
_NAME_ALIASES = {"suprise": 5, "surprised": 5, "neutrality": 6}


def apply_clahe(img_uint8):
    """
    Apply Contrast Limited Adaptive Histogram Equalization to a single
    grayscale uint8 image. CLAHE is preferred over global histogram
    equalization because it operates on local tiles and clips the contrast
    amplification, preventing noise over-amplification in near-flat regions
    (common on cheeks/forehead in low-res face crops).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def _load_from_csv(csv_path, apply_preprocessing):
    print(f"Found CSV dataset. Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    def decode_pixels(pixel_str):
        arr = np.array(pixel_str.split(), dtype="uint8")
        return arr.reshape(IMG_SIZE, IMG_SIZE)

    images = np.stack(df["pixels"].apply(decode_pixels).values)
    labels = df["emotion"].values
    usage = df["Usage"].values

    if apply_preprocessing:
        print("Applying CLAHE preprocessing to all images ...")
        images = np.stack([apply_clahe(img) for img in images])

    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)  # (N, 48, 48, 1)
    labels_onehot = to_categorical(labels, NUM_CLASSES)

    train_mask = usage == "Training"
    val_mask = usage == "PublicTest"
    test_mask = usage == "PrivateTest"

    return {
        "X_train": images[train_mask], "y_train": labels_onehot[train_mask],
        "X_val": images[val_mask], "y_val": labels_onehot[val_mask],
        "X_test": images[test_mask], "y_test": labels_onehot[test_mask],
        "raw_labels_train": labels[train_mask],
    }


def _emotion_id_from_folder_name(folder_name):
    key = folder_name.strip().lower()
    if key in _NAME_TO_ID:
        return _NAME_TO_ID[key]
    if key in _NAME_ALIASES:
        return _NAME_ALIASES[key]
    raise ValueError(
        f"Unrecognized emotion folder name '{folder_name}'. Expected one of "
        f"{list(_NAME_TO_ID.keys())}. Rename the folder to match, or tell "
        f"me the exact folder names you have and I'll add them as aliases."
    )


def _load_images_from_split_dir(split_dir, apply_preprocessing):
    """Reads every image under split_dir/<emotion_name>/*.{jpg,png,jpeg}."""
    emotion_folders = sorted(
        d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )
    if not emotion_folders:
        raise FileNotFoundError(f"No emotion subfolders found inside {split_dir}")

    images, labels = [], []
    for folder_name in emotion_folders:
        emotion_id = _emotion_id_from_folder_name(folder_name)
        folder_path = os.path.join(split_dir, folder_name)
        file_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            file_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        print(f"  {folder_name:>10s} -> {len(file_paths)} images")
        for fp in file_paths:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # skip unreadable/corrupt files rather than crash
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            if apply_preprocessing:
                img = apply_clahe(img)
            images.append(img)
            labels.append(emotion_id)

    images = np.stack(images).astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels)
    return images, labels


def _load_from_folders(train_dir, test_dir, apply_preprocessing, val_split=0.1):
    print(f"Found folder-based dataset. Loading train images from {train_dir} ...")
    X_train_full, y_train_full = _load_images_from_split_dir(train_dir, apply_preprocessing)

    print(f"Loading test images from {test_dir} ...")
    X_test, y_test = _load_images_from_split_dir(test_dir, apply_preprocessing)

    # The folder layout has no separate validation split, so carve one out
    # of the training set (stratified, so class proportions are preserved).
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_split,
        stratify=y_train_full, random_state=42
    )

    return {
        "X_train": X_train, "y_train": to_categorical(y_train, NUM_CLASSES),
        "X_val": X_val, "y_val": to_categorical(y_val, NUM_CLASSES),
        "X_test": X_test, "y_test": to_categorical(y_test, NUM_CLASSES),
        "raw_labels_train": y_train,
    }


def load_fer2013(data_dir=DATA_DIR, apply_preprocessing=True):
    """
    Auto-detects which FER-2013 layout you downloaded and loads it:
    - data/fer2013.csv                         -> CSV format
    - data/train/<emotion>/*.jpg + data/test/.. -> folder format

    Returns dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
    Images are returned as float32 arrays in [0, 1], shape (N, 48, 48, 1).
    """
    csv_path = os.path.join(data_dir, "fer2013.csv")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if os.path.exists(csv_path):
        data = _load_from_csv(csv_path, apply_preprocessing)
    elif os.path.isdir(train_dir) and os.path.isdir(test_dir):
        data = _load_from_folders(train_dir, test_dir, apply_preprocessing)
    else:
        raise FileNotFoundError(
            f"Could not find a dataset in '{data_dir}'. Expected either "
            f"'{csv_path}', or both '{train_dir}' and '{test_dir}' folders "
            f"(each containing one subfolder per emotion). See README.md."
        )

    print(f"Train: {data['X_train'].shape[0]} | "
          f"Val: {data['X_val'].shape[0]} | "
          f"Test: {data['X_test'].shape[0]}")

    return data


def get_class_weights(raw_labels_train):
    """
    Computes class weights inversely proportional to class frequency, to
    counter FER-2013's severe class imbalance (e.g. 'Disgust' has roughly
    15x fewer samples than 'Happy'). Used in model.fit(class_weight=...)
    rather than oversampling, to avoid overfitting on duplicated minority
    samples.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(raw_labels_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=raw_labels_train
    )
    return dict(zip(classes, weights))


def get_augmentation_generator():
    """
    Data augmentation for training only (never applied to val/test).
    Mild transforms chosen to reflect realistic face-image variation
    without distorting expression-relevant features.
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )


def _save_sample_grid(data, out_path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 7, figsize=(14, 4))
    raw_labels = data["raw_labels_train"]
    for class_id in range(NUM_CLASSES):
        idx = np.where(raw_labels == class_id)[0][0]
        img = data["X_train"][idx].squeeze()
        axes[0, class_id].imshow(img, cmap="gray")
        axes[0, class_id].set_title(EMOTION_LABELS[class_id], fontsize=9)
        axes[0, class_id].axis("off")
        axes[1, class_id].hist(img.ravel(), bins=20, color="gray")
        axes[1, class_id].set_xticks([]); axes[1, class_id].set_yticks([])
    axes[0, 0].set_ylabel("CLAHE image", fontsize=8)
    axes[1, 0].set_ylabel("Pixel histogram", fontsize=8)
    plt.suptitle("Sample preprocessed images per emotion class (top) "
                 "and intensity distribution (bottom)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved sample grid to {out_path}")


if __name__ == "__main__":
    data = load_fer2013()
    weights = get_class_weights(data["raw_labels_train"])
    print("Computed class weights:", weights)
    _save_sample_grid(data, os.path.join(OUTPUT_DIR, "sample_preprocessed.png"))
