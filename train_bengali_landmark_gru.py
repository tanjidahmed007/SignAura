"""Real training run for the downloaded Bengali Sign Language image dataset.

The public Kaggle dataset is an image alphabet dataset (38 classes), not a video
corpus. We therefore train a GRU over a one-step MediaPipe hand-landmark sequence.
This is a real landmark classifier and produces honest image-set metrics; it is
not presented as sentence-level video translation.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
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


def extract_landmarks(image_path: Path, hands) -> np.ndarray | None:
    image = mp.Image.create_from_file(str(image_path))
    result = hands.detect(image)
    if not result.hand_landmarks:
        return None
    points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
    # Wrist-relative and scale-normalized so the classifier focuses on structure.
    points = points - points[0:1]
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points = points / scale
    return points.reshape(-1)


def load_split(root: Path, split_name: str, class_names: list[str], hands, limit: int | None = None):
    X, y, paths = [], [], []
    split_root = root / split_name
    for class_index, class_name in enumerate(class_names):
        files = sorted((split_root / class_name).glob('*.jpg'))
        if limit:
            files = files[:limit]
        for image_path in files:
            vector = extract_landmarks(image_path, hands)
            if vector is not None:
                X.append(vector)
                y.append(class_index)
                paths.append(str(image_path))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), paths


def build_model(class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(1, 63), name='mediapipe_landmarks')
    x = tf.keras.layers.GRU(128, return_sequences=True, name='gru_1')(inputs)
    x = tf.keras.layers.GRU(64, name='gru_2')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation='softmax', name='alphabet_class')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/bengali'))
    parser.add_argument('--out', type=Path, default=Path('public/models/signaura-bengali-gru-v1.pkl'))
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--limit-per-class', type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    train_root = args.data / 'RESIZED_DATASET'
    test_root = args.data / 'RESIZED_TESTING_DATA'
    class_names = sorted([p.name for p in train_root.iterdir() if p.is_dir()], key=lambda x: int(x))
    limit = args.limit_per_class or None
    print(f'classes={len(class_names)} train_root={train_root}')

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='ml/hand_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
    )
    with vision.HandLandmarker.create_from_options(options) as hands:
        X_train, y_train, train_paths = load_split(args.data, 'RESIZED_DATASET', class_names, hands, limit)
        X_test, y_test, test_paths = load_split(args.data, 'RESIZED_TESTING_DATA', class_names, hands)
    print(f'landmarks extracted: train={len(X_train)} test={len(X_test)}')
    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit('No landmarks extracted. Check the dataset mount and MediaPipe detection.')

    X_train = X_train[:, None, :]
    X_test = X_test[:, None, :]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )
    model = build_model(len(class_names))
    history = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=64, verbose=2)
    probabilities = model.predict(X_test, batch_size=128, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'signaura-pickle-v2', 'status': 'real-trained', 'dataset': 'muntakimrafi/bengali-sign-language-dataset',
        'dataset_kind': 'static alphabet images', 'architecture': 'GRU(128)->GRU(64)->Dense', 'classes': class_names,
        'sequence_length': 1, 'landmarks_per_frame': 63, 'preprocessing': 'MediaPipe Hands; wrist-relative; scale-normalized',
        'weights': model.get_weights(), 'model_config': model.get_config(), 'metrics': {'test_accuracy': accuracy, 'test_samples': len(y_test)},
    }
    with args.out.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report_path = args.out.with_name('signaura-real-metrics.json')
    report_path.write_text(json.dumps({
        'status': 'real-trained', 'dataset': 'Bengali Sign Language Dataset', 'dataset_source': 'https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset',
        'dataset_kind': 'static alphabet images', 'classes': len(class_names), 'train_samples': len(y_fit), 'validation_samples': len(y_val), 'test_samples': len(y_test),
        'test_accuracy': accuracy, 'test_accuracy_percent': round(accuracy * 100, 2), 'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]), 'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]), 'epochs': args.epochs, 'elapsed_seconds': round(time.time() - started, 1),
        'class_names': class_names, 'confusion_matrix': matrix, 'classification_report': report,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'test_accuracy_percent': round(accuracy * 100, 2), 'train_samples': len(y_fit), 'validation_samples': len(y_val), 'test_samples': len(y_test), 'artifact': str(args.out), 'metrics': str(report_path)}))


if __name__ == '__main__':
    main()
