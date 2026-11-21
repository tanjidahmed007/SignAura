# SignAura — Implementation Plan

## Product scope

SignAura is a Bengali-first assistive sign-to-text workspace. The shipped web experience is a polished prototype of the full pipeline: structural video frames → MediaPipe landmarks → Bi-LSTM/GRU sequence model → Bengali semantic text, with transparent dataset, training, and `.pkl` artifact states. The live camera surface is intentionally labeled as demo/simulation until a real dataset and camera inference service are connected.

## Design direction

### Candidate styles
- **Clinical glass observatory** (0.06): dark scientific control room with luminous teal data traces and a warm amber signal accent.
- **Soft civic utility** (0.08): bright public-service interface with rounded cards and friendly primary colors.
- **Editorial research notebook** (0.07): paper-like layout with annotated charts and code fragments.

### Chosen style: Clinical glass observatory

- **Design movement:** mission-control UI meets modern accessibility tooling, with a restrained glass surface over an almost-black blue canvas.
- **Core principles:** signal before decoration; Bengali language always remains primary; every ML claim has a visible status; controls feel calm and reversible.
- **Color philosophy:** deep navy provides low-glare contrast, mint/teal communicates active perception and trust, and amber is reserved for human attention and model readiness.
- **Layout paradigm:** asymmetric dashboard with a wide capture bay, narrow telemetry rail, and a bottom pipeline ribbon; avoid a generic centered marketing layout.
- **Signature elements:** numbered pipeline nodes, thin grid-line overlays, and a floating 3D hand skeleton with orbiting landmark particles.
- **Interaction philosophy:** actions update a clear state pill and supporting microcopy; no hidden magic; demo mode is explicit.
- **Animation:** slow scanline drift, pulsing landmark points, gentle card rise on hover, and a 3D wrist orbit that never competes with the text output.
- **Typography:** Space Grotesk for interface headings and IBM Plex Mono for metrics/metadata, with Noto Sans Bengali fallback for Bangla copy.
- **Brand essence:** A calm, transparent sign-to-text copilot for Bengali communication — precise, humane, explainable.
- **Brand voice:** direct, optimistic, and technical without being cold. Example: “Landmarks are stable. Sentence assembly is ready.” / “Train from your arrays when the dataset is mounted.”
- **Wordmark/mark:** a compact “SA” monogram inside a rounded signal frame, paired with the SignAura wordmark.
- **Signature brand color:** signal mint `#66F3C1`.

## Implementation

- Vite + React + TypeScript single-page application.
- No external image dependency; the primary visual is a CSS/SVG 3D landmark stage designed for the product itself.
- Client-side state machine for capture, alphabet-class output, real run metrics, dataset selection, and artifact export.
- `public/models/signaura-bengali-gru-v1.pkl` is a real pickle export containing GRU weights, the 38-class label map, model config, and MediaPipe schema. It was trained from the downloaded Bengali Sign Language image dataset.
- `ml/train_bengali_landmark_gru.py` performs the real MediaPipe Tasks + TensorFlow GRU run and writes both the pickle and `signaura-real-metrics.json` report. The dashboard keeps BdSLW60 and IsharaKhobor visible as the next video/sentence training sources, but does not claim they were used in this run.
- `public/manus-routes.json` declares the single `/` route.

## Project structure

- `src/App.tsx`: dashboard composition and interactive state.
- `src/main.tsx`: React entrypoint.
- `src/styles.css`: full visual system, responsive layout, motion, and accessible states.
- `public/models/`: downloadable artifact.
- `ml/`: reproducible dataset/training/export script and dependency notes.
- `public/manus-routes.json`: managed route manifest.
