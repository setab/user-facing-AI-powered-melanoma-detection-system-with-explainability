# Copilot Context: Melanoma-detection

Updated: 2025-11-15

## Snapshot
- OS/Shell: Linux, bash
- Python env (terminal): `/home/the/miniconda/envs/ml2/bin/python`
- Torch/vision detected: torch 2.8.0+cu128, torchvision 0.23.0+cu128
- GPU use: auto-detected via `cuda` if available

## Goals and scope
- Build an automated melanoma detector in PyTorch with explainability (Grad-CAM).
- Make it thesis-grade reliable via temperature calibration and medical operating thresholds.
- Provide two inference paths: a CLI (with optional Q&A) and a Gradio web UI.

## Data and labels
- Metadata CSV: `data/HAM10000_metadata.csv`
- Images: `data/ds/img/*.jpg`
- Labels are taken from the `dx` column and mapped into an integer `label`.
- Label map saved at `models/label_maps/label_map_nb.json`.

## Model and artifacts
- Checkpoint (ResNet50): `models/checkpoints/melanoma_resnet50_nb.pth`
- Temperature calibration: `models/checkpoints/temperature.json`
  - Current value: `1.8653918504714966`
- Operating thresholds: `models/checkpoints/operating_points.json`
  - Content: `{ "class_index": 5, "use_calibrated": true, "thresholds": { "melanoma_spec95": 0.7238592505455017, "melanoma_spec90": 0.614717423915863 } }`
- Reliability plot saved at `experiments/calibration/reliability_pre_post.png`.

## Notebook
- Main pipeline notebook: `learning/day1.ipynb` includes:
  1) Setup and config
  2) Data split and datasets
  3) Loaders with class weighting and sampler
  4) Model + training/evaluation helpers
  5) Training with early stopping (or load checkpoint)
  6) Evaluation (ROC/PR, confusion)
  7) Grad-CAM visualizations
  8) Temperature calibration (ECE, Brier, reliability diagram)
  9) Operating thresholds for melanoma (e.g., spec≈95% and ≈90%)
  10) Sanity inference with calibrated probability and operating verdict

## Inference (CLI)
- Entry point: `src/inference/cli.py`
- Features:
  - Loads model and label map
  - Applies temperature scaling before softmax if `temperature.json` is provided
  - Uses `operating_points.json` to compute a calibrated melanoma verdict at a chosen operating key (default: `melanoma_spec95`)
  - Saves a Grad-CAM overlay image; optional, and robust to torchcam shape differences
  - Optional follow-up Q&A if melanoma probability is below threshold

Example run (used successfully):
```bash
python -m src.inference.cli \
  --image data/ds/img/ISIC_0027990.jpg \
  --weights models/checkpoints/melanoma_resnet50_nb.pth \
  --label-map models/label_maps/label_map_nb.json \
  --temperature-json models/checkpoints/temperature.json \
  --operating-json models/checkpoints/operating_points.json \
  --operating-key melanoma_spec95 \
  --out experiments/overlay_sanity.jpg
```
Sample output:
```
Prediction: benign keratosis-like lesions | p=0.956
Calibrated melanoma p=0.029 | threshold=0.724 → VERDICT: non-melanoma
Grad-CAM overlay saved to experiments/overlay_sanity.jpg
```

## Inference (Gradio)
- App: `src/serve_gradio.py`
- Reads weights/label map via environment variables (defaults target local files):
  - `WEIGHTS_PATH` (default: `melanoma_resnet50.pth`) → set to `models/checkpoints/melanoma_resnet50_nb.pth`
  - `LABEL_MAP` (default: `label_map.json`) → set to `models/label_maps/label_map_nb.json`
  - `TEMPERATURE_JSON` (default: `models/checkpoints/temperature.json`)
  - `OPERATING_JSON` (default: `models/checkpoints/operating_points.json`)

Run example:
```bash
export WEIGHTS_PATH=models/checkpoints/melanoma_resnet50_nb.pth
export LABEL_MAP=models/label_maps/label_map_nb.json
export TEMPERATURE_JSON=models/checkpoints/temperature.json
export OPERATING_JSON=models/checkpoints/operating_points.json
python src/serve_gradio.py
```
Then open the printed URL (default http://127.0.0.1:7860/). The UI shows:
- Grad-CAM heatmap
- Predicted class
- Probability JSON (calibrated)
- "Melanoma Decision (calibrated)" string using the chosen operating point

## Implementation notes
- Model: `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`, final `fc` replaced with `num_classes`.
- Calibration: Temperature scaling fitted on val logits; applied as logits / T before softmax.
- Thresholding: melanoma vs rest ROC, picking thresholds at ≥target specificity (95% and 90%).
- Grad-CAM: torchcam with robust overlay handling; hooks are removed after use to avoid interference.
- Files centralized under `models/checkpoints` to avoid duplicates.

## Tests
- Smoke test: `tests/test_smoke_inference.py`
  - Loads checkpoint + label map
  - Applies temperature
  - Runs single forward on a sample image
  - Asserts probability shape and melanoma index

Run:
```bash
python -m unittest -q tests/test_smoke_inference.py
```

## Dependencies
- Training-oriented: `requirements-train.txt`
- Serving-oriented: `models/requirements-serving.txt`

Pinned highlights:
- torch, torchvision, scikit-learn, pandas, numpy, pillow, tqdm, torchcam, gradio

## Known good results (examples)
- Temperature T ≈ 1.865 → ECE improved significantly (see reliability plot)
- Melanoma operating thresholds:
  - spec95 ≈ 0.724
  - spec90 ≈ 0.615
- Example CLI verdict: `melanoma p=0.029 vs thr=0.724 → non-melanoma`

## Next steps (optional)
- Add README HOWTO snippets (CLI + Gradio usage) or merge this file content into `README.md`.
- Add switch `--no-ask` to CLI to disable follow-up Q&A for batch use.
- Consider uncertainty estimation, cross-validation, ablations, and bias/robustness checks for thesis completeness.
- Lock exact package versions for reproducibility.
