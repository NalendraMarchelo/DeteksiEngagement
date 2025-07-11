# utils/models_utils.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

def build_engagement_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=1e-4):
    """
    Membangun model klasifikasi. Disesuaikan untuk bisa menangani biner atau multi-kelas.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # --- LOGIKA BARU UNTUK MENYESUAIKAN OUTPUT ---
    if num_classes == 2:
        # Untuk klasifikasi biner, gunakan 1 neuron dengan aktivasi sigmoid
        output = Dense(1, activation='sigmoid', name='engagement_output')(x)
        loss_function = 'binary_crossentropy'
    else:
        # Untuk klasifikasi multi-kelas (jika nanti dibutuhkan)
        output = Dense(num_classes, activation='softmax', name='engagement_output')(x)
        loss_function = 'sparse_categorical_crossentropy'
    # ---------------------------------------------
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, train_datagen, batch_size=32, epochs=10):
    """
    Melatih model dengan data yang diberikan.
    Fungsi ini tidak berubah dari versi refaktor sebelumnya.
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_model.keras')

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {0: 1.0, 1: 2.2}
    print(f"Menggunakan Class Weights MANUAL: {class_weights_dict}")

    callbacks = [
        ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    steps_per_epoch = len(X_train) // batch_size

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )
    return history

def plot_training_history(history, save_path: str):
    """
    Membuat plot loss dan accuracy dari history training dan menyimpannya ke file.
    Fungsi ini tidak berubah dari versi refaktor sebelumnya.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot history training disimpan di: {save_path}")

def plot_confusion_matrix(cm, save_path: str, class_names=None):
    """
    Membuat plot confusion matrix dan menyimpannya ke file.
    Fungsi ini tidak berubah dari versi refaktor sebelumnya.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot confusion matrix disimpan di: {save_path}")

def evaluate_model_metrics(model, X_test, y_true, save_path: str):
    """
    Mengevaluasi model pada data tes, mencetak, dan menyimpan metrik.
    Fungsi ini tidak berubah dari versi refaktor sebelumnya.
    """
    print("\n--- Mengevaluasi Model pada Data Uji ---")
    
    # Untuk model biner, outputnya (0,1), jadi perlu dibulatkan
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype("int32") if model.output_shape[-1] == 1 else np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics_summary = (
        f"Akurasi: {accuracy:.4f}\n"
        f"Presisi (macro): {precision:.4f}\n"
        f"Recall (macro): {recall:.4f}\n"
        f"F1-Score (macro): {f1:.4f}\n\n"
        f"Confusion Matrix:\n{cm}"
    )
    print(metrics_summary)
    with open(save_path, "w") as f:
        f.write(metrics_summary)
    print(f"Hasil evaluasi disimpan di: {save_path}")

    cm_plot_path = save_path.replace('.txt', '_confusion_matrix.png')
    plot_confusion_matrix(cm, save_path=cm_plot_path, class_names=['Tidak Bosan', 'Bosan'])