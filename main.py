# main.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ==== CONFIG ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "anime_dataset"
CLASS_NAMES = ["authentic", "generated"]
# ===============


def load_datasets():
    train_ds = image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    test_ds = image_dataset_from_directory(
        os.path.join(DATASET_DIR, "test"),
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
    )


def build_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze the base model

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_image(model, image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]
    label = "AI-Generated" if prediction > 0.5 else "Authentic"
    confidence = round(float(prediction), 2)
    print(f"{os.path.basename(image_path)} → {label} (Confidence: {confidence})")


if __name__ == "__main__":
    print("🚀 Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets()

    print("🔧 Building model...")
    model = build_model()

    print("🏋️‍♂️ Training model...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    print("📊 Evaluating on test set...")
    loss, acc = model.evaluate(test_ds)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    plot_history(history)

    print("🧪 Generating classification report...")

    # Collect all test images and true labels
    y_true = []
    y_pred = []

for batch_images, batch_labels in test_ds:
    preds = model.predict(batch_images)
    preds = (preds > 0.5).astype(int).flatten()
    y_pred.extend(preds)
    y_true.extend(batch_labels.numpy().astype(int))

    # Print classification report
    print("\n🔍 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # predict_image(model, "anime_dataset/test/authentic/your_test_image.jpg")
