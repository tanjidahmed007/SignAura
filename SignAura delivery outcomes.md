# SignAura delivery outcomes

## Live sign-to-text workspace

- [x] The website presents a desktop/mobile-friendly camera workspace with a clearly labeled local capture surface and Bengali alphabet-class output from the real trained landmark model. Sentence-level video translation remains a separate next-stage dataset run.
- [x] The interface includes start/stop capture, confidence, frame-rate, landmark-count, clear sentence, and assistive status controls.

## Landmark and sequence-model pipeline

- [ ] The website visualizes MediaPipe-style hand/body landmark extraction with a moving 3D hand-pose interpretation and a readable pipeline from frames to semantic text.
- [x] The training view identifies the GRU configuration, Bengali dataset source, 10 epochs, 77.97% held-out test accuracy, 82.25% validation accuracy, 0.591 validation loss, and trained model readiness.

## Dataset and artifact workflow

- [x] The dataset workspace references BdSLW60, IsharaKhobor, and Bengali Sign Language dataset sources and shows that the Bengali image dataset was used while the larger video/sentence archives are not mounted.
- [x] The project includes an end-to-end MediaPipe Tasks + GRU training script for the mounted Bengali dataset and a `.pkl` export path.
- [x] The UI exposes the real model version, artifact size, schema, measured test accuracy, and a download action for the production-trained pickle artifact.

## Accessibility and visual quality

- [x] Bengali is used for primary user-facing recognition output and key workflow labels, with an English/Bangla switch, a separate validation icon/drawer, high contrast, large readable text, keyboard-visible controls, and reduced-motion support.
- [x] The dashboard has an intentional dark observatory visual system with poster-like 3D depth, responsive layout, clear status messaging, and no generic placeholder imagery.
