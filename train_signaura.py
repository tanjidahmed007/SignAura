"""Train a landmark sequence model for Bengali Sign Language.

Expected mounted dataset shape:
  data/<label>/*.mp4                    (raw video clips), or
  arrays/<split>_X.npy + <split>_y.npy  (pre-extracted sequences)

This script keeps extraction and modeling together so the same landmark schema is
used in the web artifact metadata and in model training. It is intentionally kept
outside the browser bundle; run it in a Python environment with the dataset mounted.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


SEQ_LEN = 32
LANDMARKS_PER_FRAME = 63  # 21 hand landmarks x/y/z; extend with pose when available.


def frame_landmarks(video_path: Path, max_frames: int = SEQ_LEN) -> np.ndarray:
    """Sample a clip into a fixed sequence of normalized hand landmarks."""
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            values = np.array([[point.x, point.y, point.z] for point in hand.landmark], dtype=np.float32).reshape(-1)
        else:
            values = np.zeros(LANDMARKS_PER_FRAME, dtype=np.float32)
        frames.append(values)
    capture.release()
    hands.close()
    if not frames:
        return np.zeros((max_frames, LANDMARKS_PER_FRAME), dtype=np.float32)
    sequence = np.stack(frames)
    if len(sequence) < max_frames:
        sequence = np.pad(sequence, ((0, max_frames - len(sequence)), (0, 0)))
    return sequence


def iter_video_samples(root: Path) -> Iterable[tuple[np.ndarray, str]]:
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for video in sorted(label_dir.glob('*.mp4')):
            yield frame_landmarks(video), label_dir.name


def build_model(classes: int, architecture: str = 'gru') -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEQ_LEN, LANDMARKS_PER_FRAME), name='landmark_sequence')
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    if architecture == 'bilstm':
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:
        x = tf.keras.layers.GRU(128, return_sequences=True)(x)
        x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(classes, activation='softmax', name='sign_class')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='Dataset root with one folder per sign label')
    parser.add_argument('--out', type=Path, default=Path('artifacts/signaura-gru.pkl'))
    parser.add_argument('--architecture', choices=('gru', 'bilstm'), default='gru')
    parser.add_argument('--epochs', type=int, default=24)
    args = parser.parse_args()

    samples = list(iter_video_samples(args.data))
    if len(samples) < 4:
        raise SystemExit('Need at least four labeled clips before training.')
    X = np.stack([sample[0] for sample in samples])
    encoder = LabelEncoder().fit([sample[1] for sample in samples])
    y = encoder.transform([sample[1] for sample in samples])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = build_model(len(encoder.classes_), args.architecture)
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=args.epochs, batch_size=32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'signaura-pickle-v1',
        'architecture': args.architecture,
        'sequence_length': SEQ_LEN,
        'landmarks_per_frame': LANDMARKS_PER_FRAME,
        'labels': encoder.classes_.tolist(),
        'weights': model.get_weights(),
        'model_config': model.get_config(),
    }
    with args.out.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Exported {args.out} with {len(encoder.classes_)} labels')


if __name__ == '__main__':
    main()
