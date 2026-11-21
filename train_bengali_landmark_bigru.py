"""Improved real run for the Bengali Sign Language alphabet dataset.

Compared with the first run this uses 126 features per sample (wrist-relative
coordinates plus scale-normalized coordinates), a Bidirectional GRU stack,
small landmark jitter augmentation, and a stratified validation split.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
FEATURES = 126


def extract_features(image_path: Path, landmarker) -> np.ndarray | None:
    image = mp.Image.create_from_file(str(image_path))
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None
    points = np.asarray([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
    relative = points - points[0:1]
    scale = np.max(np.linalg.norm(relative[:, :2], axis=1))
    normalized = relative / scale if scale > 1e-6 else relative
    return np.concatenate([relative.reshape(-1), normalized.reshape(-1)]).astype(np.float32)


def load_split(root: Path, split_name: str, classes: list[str], landmarker):
    X, y = [], []
    for index, name in enumerate(classes):
        for path in sorted((root / split_name / name).glob('*.jpg')):
            features = extract_features(path, landmarker)
            if features is not None:
                X.append(features)
                y.append(index)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def build_model(class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(1, FEATURES), name='rich_mediapipe_landmarks')
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(192, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(96))(x)
    x = tf.keras.layers.Dropout(.22)(x)
    x = tf.keras.layers.Dense(96, activation='gelu')(x)
    x = tf.keras.layers.Dropout(.16)(x)
    outputs = tf.keras.layers.Dense(class_count, activation='softmax', name='alphabet_class')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/bengali'))
    parser.add_argument('--out', type=Path, default=Path('public/models/signaura-bengali-bigru-v2.pkl'))
    parser.add_argument('--epochs', type=int, default=24)
    args = parser.parse_args()
    started = time.time()
    train_root = args.data / 'RESIZED_DATASET'
    classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()], key=lambda x: int(x))
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='ml/hand_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=.45, min_hand_presence_confidence=.45,
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        X, y = load_split(args.data, 'RESIZED_DATASET', classes, landmarker)
        X_test, y_test = load_split(args.data, 'RESIZED_TESTING_DATA', classes, landmarker)
    print(f'landmarks extracted: train={len(X)} test={len(X_test)} features={X.shape[-1]}')
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=.15, stratify=y, random_state=SEED)
    noise = np.random.normal(0, .008, X_fit.shape).astype(np.float32)
    X_aug = np.concatenate([X_fit, X_fit + noise], axis=0)
    y_aug = np.concatenate([y_fit, y_fit], axis=0)
    X_aug = X_aug[:, None, :]
    X_val = X_val[:, None, :]
    X_test = X_test[:, None, :]
    model = build_model(len(classes))
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    ]
    history = model.fit(X_aug, y_aug, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=64, callbacks=callbacks, verbose=2)
    probabilities = model.predict(X_test, batch_size=128, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'signaura-pickle-v3', 'status': 'real-trained', 'dataset': 'muntakimrafi/bengali-sign-language-dataset',
        'dataset_kind': 'static alphabet images', 'architecture': 'BiGRU(192)->BiGRU(96)->Dense(96)->Dense(38)',
        'classes': classes, 'sequence_length': 1, 'landmarks_per_frame': FEATURES,
        'preprocessing': 'MediaPipe HandLandmarker; wrist-relative + scale-normalized; jitter augmentation',
        'weights': model.get_weights(), 'model_config': model.get_config(),
        'metrics': {'test_accuracy': accuracy, 'test_samples': len(y_test)},
    }
    with args.out.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metrics_path = args.out.with_name('signaura-bigru-metrics.json')
    metrics_path.write_text(json.dumps({
        'status': 'real-trained', 'dataset': 'Bengali Sign Language Dataset', 'dataset_source': 'https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset',
        'dataset_kind': 'static alphabet images', 'architecture': payload['architecture'], 'classes': len(classes),
        'train_samples': len(y_fit), 'augmented_train_samples': len(y_aug), 'validation_samples': len(y_val), 'test_samples': len(y_test),
        'test_accuracy': accuracy, 'test_accuracy_percent': round(accuracy * 100, 2),
        'final_train_accuracy': float(history.history['accuracy'][-1]), 'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]), 'final_val_loss': float(history.history['val_loss'][-1]),
        'epochs_requested': args.epochs, 'epochs_completed': len(history.history['loss']), 'elapsed_seconds': round(time.time() - started, 1),
        'class_names': classes, 'confusion_matrix': matrix, 'classification_report': report,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'test_accuracy_percent': round(accuracy * 100, 2), 'val_accuracy_percent': round(float(history.history['val_accuracy'][-1]) * 100, 2), 'train_samples': len(y_fit), 'test_samples': len(y_test), 'artifact': str(args.out), 'metrics': str(metrics_path)}))


if __name__ == '__main__':
    main()
