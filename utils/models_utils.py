import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

def build_engagement_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=1e-4):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    # Lakukan fine-tuning hanya pada lapisan-lapisan atas
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Menyesuaikan lapisan output dan fungsi loss berdasarkan jumlah kelas
    if num_classes == 2 or num_classes == 1:
        output = Dense(1, activation='sigmoid', name='engagement_output')(x)
        loss_function = 'binary_crossentropy'
    else:
        output = Dense(num_classes, activation='softmax', name='engagement_output')(x)
        loss_function = 'sparse_categorical_crossentropy'
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
    
    return model

def train_model(model, train_dataset, validation_dataset, epochs=10, class_weight=None):
    """
    Melatih model Keras dengan dataset yang diberikan.
    Menerima parameter class_weight opsional.
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'cnn_best_model.keras')

    if class_weight:
        print(f"INFO: Training menggunakan Class Weights: {class_weight}")
    else:
        print("INFO: Training tanpa class weights.")

    callbacks = [
        ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    return history

def plot_training_history(history, save_path: str):
    """Membuat plot loss dan accuracy dari history training dan menyimpannya ke file."""
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
    print(f"INFO: Plot history training disimpan di: {save_path}")

def evaluate_and_plot_cm(model, test_dataset, save_path: str, class_names: list):
    """
    Mengevaluasi model, mencetak laporan, dan membuat plot confusion matrix.
    Fungsi ini dirancang untuk bekerja dengan objek tf.data.Dataset.
    """
    print("\n--- Mengevaluasi Model pada Data Uji ---")
    
    # Ekstrak label asli dari dataset
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    
    # Dapatkan prediksi dari model
    y_pred_probs = model.predict(test_dataset, verbose=0)
    
    # Konversi probabilitas ke kelas prediksi (mendukung biner dan multi-kelas)
    if model.output_shape[-1] == 1: # Output biner (sigmoid)
        y_pred = (y_pred_probs > 0.5).astype("int32")
    else: # Output multi-kelas (softmax)
        y_pred = np.argmax(y_pred_probs, axis=1)

    # Tampilkan dan simpan laporan klasifikasi
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"INFO: Laporan evaluasi disimpan di: {save_path}")

    # Buat dan simpan plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    cm_plot_path = save_path.replace('.txt', '_cm.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"INFO: Plot confusion matrix disimpan di: {cm_plot_path}")