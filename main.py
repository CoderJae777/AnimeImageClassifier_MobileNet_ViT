# main.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dropout, Dense, LayerNormalization,
    MultiHeadAttention, Add, Reshape, Input, GlobalAveragePooling1D
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==== CONFIG ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "anime_dataset"
CLASS_NAMES = ["authentic", "generated"]

# for the transformer specifically:
NUM_HEADS = 8
KEY_DIM = 64
FF_DIM = 256
NUM_TRANSFORMER_BLOCKS = 2
DROPOUT_RATE = 0.1
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

# Some link for reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
# https://keras.io/examples/vision/image_classification_using_global_context_vision_transformer
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
# https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras
# https://medium.com/%40prudhviraju.srivatsavaya/implementing-multiheaded-attention-a321dcb5aab8 

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate
        )
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dropout(dropout_rate),
            Dense(key_dim * num_heads) # return to original dimension 
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

def build_hybrid_model():
    # Recap of what we are doing here:
    # - Mobilenet to extract features
    # - Reshape features to sequence of patches
    # - Transformer blocks to apply self-attention 
    # - Dense layers for classification
    
    inputs = Input(shape=(224, 224, 3))
    
    # ----First step: MobileNet feature extractor
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights="imagenet"
    )
    base_model.trainable = False 
    
    # Known features size: (7x7x1280)
    features = base_model(inputs, training=False)
    
    # -----Next step: convert features to sequence
    # (batch_size, 7, 7, 1280) -> (batch_size, 49, 1280)
    h, w, c = 7, 7, 1280 
    sequence = Reshape((h * w, c))(features)  # Now we have: (batch_size, 49, 1280)
    
    # Feature projection (not necessary with our default config)
    transformer_dim = NUM_HEADS * KEY_DIM
    if c != transformer_dim:
        sequence = Dense(transformer_dim, name="feature_projection")(sequence)
    
    # -----Next step: Transformer blocks
    x = sequence
    for i in range(NUM_TRANSFORMER_BLOCKS):
        x = TransformerBlock(
            num_heads=NUM_HEADS,
            key_dim=KEY_DIM,
            ff_dim=FF_DIM,
            dropout_rate=DROPOUT_RATE,
            name=f"transformer_block_{i}"
        )(x)
    
    # -----Final step: classification
    x = GlobalAveragePooling1D()(x)  # Now: (batch_size, transformer_dim)
    
    # Usual classification layers
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu", name="fc1")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu", name="fc2")(x)
    x = Dropout(0.1)(x)
    
    outputs = Dense(1, activation="sigmoid", name="classification")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="MobileNet_ViT_Hybrid")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def build_original_mobilenet():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze the base model
    
    model = tf.keras.Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# I don't know if you were using a notebook, but I couldn't display the plots, so I saved them instead
def plot_history(history, model_name="Hybrid"):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker='o')
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss", marker='o')
    plt.plot(history.history["val_loss"], label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

    plt.savefig(f'training_history_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"📊 Training history saved as 'training_history_{model_name.lower().replace(' ', '_')}.png'")

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
    
    print("🔧 Building Hybrid MobileNet + Vision Transformer model...")
    model = build_hybrid_model()

    print("\n📋 Model Architecture:")
    model.summary()
    
    print("\n🏋️‍♂️ Training hybrid model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    print("📊 Evaluating on test set...")
    loss, acc = model.evaluate(test_ds)
    print(f"\n✅ Hybrid Model Test Accuracy: {acc:.4f}")
    
    plot_history(history, "MobileNet + ViT Hybrid")
    
    print("🧪 Generating classification report...")
    
    # Collect all test images and true labels
    y_true = []
    y_pred = []
    
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        preds = (preds > 0.5).astype(int).flatten()
        y_pred.extend(preds)
        y_true.extend(batch_labels.numpy().astype(int))
    
    # Print classification report
    print(f"\n🔍 Classification Report (Hybrid Model):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
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
    plt.title("Confusion Matrix - Hybrid MobileNet + ViT")
    plt.tight_layout()
    plt.savefig('confusion_matrix_hybrid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Confusion matrix saved as 'confusion_matrix_hybrid.png'")
    
    print("\n🔄 Training original MobileNet for comparison...")
    original_model = build_original_mobilenet()
    original_history = original_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    original_loss, original_acc = original_model.evaluate(test_ds)
    print(f"📊 Original MobileNet Test Accuracy: {original_acc:.4f}")
    print(f"🆚 Improvement: {((acc - original_acc) / original_acc * 100):+.2f}%")
    
    # predict_image(model, "anime_dataset/test/authentic/your_test_image.jpg")