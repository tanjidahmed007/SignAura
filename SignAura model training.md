# SignAura model training

The current checked-in artifact is a real GRU classifier trained from the public Kaggle **Bengali Sign Language Dataset**. That dataset contains static alphabet images, so this run is an alphabet-class landmark recognizer rather than a sentence-level video translator.

The run extracted MediaPipe HandLandmarker features from 8,412 training images and 708 held-out test images across 38 classes. With a stratified 85/15 train/validation split and 10 epochs, the measured results were 82.25% validation accuracy and 77.97% held-out test accuracy. The trained export is `public/models/signaura-bengali-gru-v1.pkl`; metrics are in `public/models/signaura-real-metrics.json`.

To reproduce the run after mounting the dataset:

```bash
python -m venv .venv
.venv/bin/pip install -r ml/requirements.txt
.venv/bin/python ml/train_bengali_landmark_gru.py \
  --data ./data/bengali \
  --epochs 10 \
  --out ./public/models/signaura-bengali-gru-v1.pkl
```

BdSLW60 and IsharaKhobor remain listed in the dashboard as source datasets for the next video/sentence training stage. Their large Kaggle archives were not included in this run, so the UI marks them as “not mounted” rather than claiming they were used.
