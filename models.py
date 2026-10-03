"""
models.py

Defines two architectures for FER-2013 emotion classification:
1. build_custom_cnn()  -- a compact CNN trained from scratch
2. build_mobilenet_transfer() -- MobileNetV2 pretrained on ImageNet, fine-tuned

Both take 48x48 grayscale input and output a 7-way softmax over emotions.
"""

from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = 48
NUM_CLASSES = 7


def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Custom CNN: 3 convolutional blocks (increasing filters: 32 -> 64 -> 128),
    each with BatchNorm + ReLU + MaxPool + Dropout, followed by a dense
    classifier head.

    Design choices to justify in viva:
    - BatchNorm after each conv: stabilizes and speeds up training by
      reducing internal covariate shift.
    - Dropout increasing with depth (0.25 -> 0.5): deeper layers have more
      parameters and are more prone to overfitting on a dataset this size.
    - Two conv layers per block before pooling: increases receptive field
      before downsampling, preserving more spatial detail from these
      already-small 48x48 inputs.
    """
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation=None, name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    return model


def build_mobilenet_transfer(input_shape=(IMG_SIZE, IMG_SIZE, 1),
                              num_classes=NUM_CLASSES,
                              fine_tune_layers=20):
    """
    Transfer learning with MobileNetV2 (ImageNet weights).

    MobileNetV2 expects 3-channel input and performs best at >=96x96, so:
    1. Grayscale (1-channel) input is replicated to 3 channels
    2. Images are upsampled from 48x48 to 96x96 inside the model graph
       (so preprocessing stays consistent with the custom CNN's data
       pipeline -- no need to duplicate data on disk)

    fine_tune_layers: number of layers at the end of the base model to
    unfreeze for fine-tuning; earlier layers stay frozen since they encode
    generic edge/texture/color features that transfer well regardless of
    the target task.
    """
    inputs = Input(shape=input_shape)
    x = layers.Concatenate()([inputs, inputs, inputs])  # 1 -> 3 channels
    x = layers.Resizing(96, 96)(x)

    base_model = MobileNetV2(
        input_shape=(96, 96, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenet_transfer")
    return model


if __name__ == "__main__":
    m1 = build_custom_cnn()
    m1.summary()
    print("\n" + "=" * 80 + "\n")
    m2 = build_mobilenet_transfer()
    m2.summary()
