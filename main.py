# main.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- CONFIG ------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "anime_dataset"  # adjust as needed
# --------------------------------


def load_datasets():
    train_ds = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"),
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    test_ds = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE)
    )


def build_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history(history):
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History")
    plt.legend()
    plt.show()


def predict_image(model, path):
    img = image.load_img(path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)[0][0]
    label = "AI-generated" if prediction > 0.5 else "Authentic"
    print(f"{os.path.basename(path)} → {label} ({prediction:.2f})")


if __name__ == "__main__":
    # 1. Load dataset
    train_ds, val_ds, test_ds = load_datasets()

    # 2. Build and train model
    model = build_model()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # 3. Evaluate on test set
    loss, acc = model.evaluate(test_ds)
    print(f"\n✅ Final Test Accuracy: {acc:.4f}")

    # 4. Plot training history
    plot_history(history)

    # 5. Try predicting one image
    # Make sure this path is valid
    predict_image(model, "test_images/sample1.jpg")
