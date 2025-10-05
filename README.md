# Melanoma Detection (CLI + XAI)

This repository trains a ResNet50 classifier on ISIC-style skin lesion images and provides a terminal-based prediction CLI with optional follow-up questions and Grad-CAM explanations.

## Quickstart

1) Install dependencies (training + CLI)

	Optional commands:
	- pip install -r requirements-train.txt

2) Build metadata CSV from annotations

	Optional commands:
	- python scripts/build_metadata.py --ann-dir ds/ann --out HAM10000_metadata.csv

3) Train the model (saves best weights and label map)

	Optional commands:
	- python train.py --metadata HAM10000_metadata.csv --img-dir ds/img --epochs 30 --batch-size 32 --img-size 224 --out-weights melanoma_resnet50.pth --label-map label_map.json

4) Predict from terminal with Grad-CAM and follow-up Q&A if low confidence

	Optional commands:
	- python predict_cli.py --image path/to/ISIC_XXXX.jpg --weights melanoma_resnet50.pth --label-map label_map.json --out gradcam_overlay.jpg --ask-threshold 0.6

Notes:
- Grad-CAM requires `torchcam`. If not installed, overlay generation is skipped gracefully.
- Follow-up Q&A is a simple illustrative heuristic. For a thesis, consider calibrated fusion via logistic regression with answers encoded as features.
