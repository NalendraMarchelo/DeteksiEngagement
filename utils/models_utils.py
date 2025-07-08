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


def build_engagement_model(input_shape=(224, 224, 3), fine_tune_at=100, num_classes=4):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', name='confusion_output')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, train_datagen, batch_size=32, epochs=10, user_id="user_001"):
    os.makedirs(f"models/{user_id}", exist_ok=True)

    # Hitung class weight otomatis berdasarkan distribusi label
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    callbacks = [
        ModelCheckpoint(
            filepath=f'models/{user_id}/confusion_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
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


def predict_engagement(model, frames):
    preds = model.predict(frames, verbose=0)
    avg_confusion_level = float(np.mean(np.argmax(preds, axis=1)))
    return {'confusion_level': avg_confusion_level}


def plot_training_history(history, user_id="user_001"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (Confusion Level)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy (Confusion Level)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_dir = f"data/results/{user_id}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/training_history.png")
    plt.close()
    print(f"Training history saved to {output_dir}/training_history.png")


def plot_confusion_matrix(cm, user_id="user_001", class_names=None):
    if class_names is None:
        class_names = ["0", "1", "2", "3"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    output_dir = f"data/results/{user_id}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")


def evaluate_model_metrics(model, X_test, y_true, user_id="user_001"):
    print(f"\n--- Evaluating Model on Validation Data (User: {user_id}) ---")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    metrics_results = {
        "user_id": user_id,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_score_macro": f1,
        "confusion_matrix": cm.tolist()
    }

    output_dir = f"data/results/{user_id}"
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_file}")

    plot_confusion_matrix(cm, user_id=user_id)


def predict_and_save_results(model, user_frames, user_id="user_001"):
    print("\n=== Predicting Confusion Level on User Video ===")
    predictions = predict_engagement(model, user_frames)

    results = {
        "user_id": user_id,
        "confusion_level_avg": predictions['confusion_level']
    }

    output_dir = f"data/results/{user_id}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "user_prediction_results.json")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"User video prediction results saved to {results_file}:")
    print(json.dumps(results, indent=2))

    return results
