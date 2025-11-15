# Melanoma Detection with XAI

This repository provides an automated melanoma detection system in PyTorch with:
- ResNet50 baseline trained on HAM10000-style skin lesion images
- Temperature calibration for reliable probability estimates
- Medical operating thresholds (e.g., melanoma at ~95% specificity)
- Grad-CAM explanations to visualize model decisions
- Two inference modes: CLI (batch-ready) and Gradio web UI

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements-train.txt
```

Or for serving only:
```bash
pip install -r models/requirements-serving.txt
```

### 2. Configure environment (important for security!)

```bash
# Copy template and edit
cp .env.example .env
nano .env
```

**Required for remote access:**
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# Set these for authentication!
GRADIO_USERNAME=your_username
GRADIO_PASSWORD=your_secure_password
```

See `.env.example` for all available options.

### 3. Project Structure

```
├── data/                      # Images and metadata (gitignored)
│   ├── HAM10000_metadata.csv
│   └── ds/img/
├── models/
│   ├── checkpoints/           # Trained weights and calibration artifacts
│   │   ├── melanoma_resnet50_nb.pth
│   │   ├── temperature.json
│   │   └── operating_points.json
│   └── label_maps/
│       └── label_map_nb.json
├── src/
│   ├── inference/
│   │   ├── cli.py            # Command-line inference
│   │   └── xai.py            # Shared inference and XAI utilities
│   └── serve_gradio.py       # Gradio web UI
├── learning/
│   └── day1.ipynb            # End-to-end training notebook
├── notebooks/                 # Exploratory notebooks
├── tests/
│   └── test_smoke_inference.py
├── experiments/               # Outputs (plots, overlays; gitignored)
├── docs/                      # Project documentation
└── copilot.md                 # Full context for AI pair programming

```

### 3. Training (optional)

Open `learning/day1.ipynb` and run cells in order:
1. Setup and config
2. Load metadata and split
3. Dataset and transforms
4. Model + helpers
5. Train with early stopping (or load checkpoint)
6. Evaluation (ROC/PR, confusion matrix)
7. Grad-CAM visualizations
8. Temperature calibration
9. Operating thresholds for melanoma
10. Sanity inference with calibrated verdict

The notebook saves:
- `models/checkpoints/melanoma_resnet50_nb.pth`
- `models/label_maps/label_map_nb.json`
- `models/checkpoints/temperature.json` (T ≈ 1.865)
- `models/checkpoints/operating_points.json` (spec95 ≈ 0.724, spec90 ≈ 0.615)
- `experiments/calibration/reliability_pre_post.png`

### 4. CLI Inference

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

**Options:**
- `--no-ask`: Disable interactive follow-up Q&A (useful for batch runs or CI)
- `--operating-key`: Choose `melanoma_spec95` (default, ~95% specificity) or `melanoma_spec90` (~90% specificity)

**Example output:**
```
Prediction: benign keratosis-like lesions | p=0.956
Calibrated melanoma p=0.029 | threshold=0.724 → VERDICT: non-melanoma
Grad-CAM overlay saved to experiments/overlay_sanity.jpg
```

### 5. Gradio Web UI

**For local testing:**
```bash
python src/serve_gradio.py
```

**For remote access (Ubuntu server → Mac client):**

1. Configure `.env`:
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_USERNAME=your_username
GRADIO_PASSWORD=your_password
```

2. Launch server:
```bash
python src/serve_gradio.py
```

3. On Mac, open: `http://<server-ip>:7860`

See [docs/SERVER_DEPLOYMENT.md](docs/SERVER_DEPLOYMENT.md) for complete server setup guide.

Then open the printed URL (default: http://127.0.0.1:7860/).

The UI shows:
- Grad-CAM heatmap overlay
- Predicted class
- Calibrated class probabilities (JSON)
- Melanoma decision string (probability vs threshold)

## Key Features

### Temperature Calibration
- Fits a temperature parameter T on validation logits to minimize cross-entropy
- Divides logits by T before softmax → better-calibrated probabilities
- Metrics: ECE (Expected Calibration Error), Brier score, reliability diagram

### Operating Thresholds
- Computes ROC for melanoma vs rest
- Selects thresholds at target specificity (e.g., ≥95%, ≥90%)
- CLI and Gradio use these thresholds for binary melanoma decisions

### Grad-CAM Explanations
- Highlights image regions influencing the prediction
- Uses torchcam library with layer4 as the target layer
- Heatmap overlayed on original image at 50% alpha

## Testing

```bash
python -m unittest tests/test_smoke_inference.py
```

Smoke test validates:
- Model and label map loading
- Temperature scaling application
- Operating point JSON parsing
- Single forward pass with calibrated probabilities

## Notes

- **Data:** Place `HAM10000_metadata.csv` and images in `data/ds/img/`. The CSV should have columns `image_id` and `dx` (diagnosis label).
- **Security:** Always set `GRADIO_USERNAME` and `GRADIO_PASSWORD` in `.env` when deploying on a network. Never commit `.env` to git!
- **Server Setup:** See [docs/SERVER_DEPLOYMENT.md](docs/SERVER_DEPLOYMENT.md) for Ubuntu GPU server + Mac client architecture.
- **Reproducibility:** Set `SEED=42` in the notebook; consider pinning exact package versions in requirements.
- **Thesis:** Include reliability plot, report ECE/Brier pre/post calibration, document chosen operating point (spec95/spec90), and discuss Grad-CAM faithfulness.

## License

See `LICENSE.md`.

## References

- HAM10000 dataset (example): https://doi.org/10.1038/sdata.2018.161
- Temperature scaling: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations..." (ICCV 2017)
