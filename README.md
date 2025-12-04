# Melanoma Detection with Explainable AI

Automated melanoma detection system built with PyTorch. Includes ResNet50 baseline trained on HAM10000 skin lesion images, temperature calibration for reliable probabilities, medical operating thresholds, Grad-CAM explanations, and a web interface with interactive chat for uncertain cases.

Features:
- Model comparison framework for evaluating multiple architectures
- CLI for batch inference and Gradio web UI with chat
- Temperature calibration and clinical thresholds (95% specificity for melanoma)
- Grad-CAM visualizations to explain model decisions

## Quick Start

### 1. Install dependencies

```bash
# For training and development
pip install -r requirements/requirements-train.txt

# For serving only
pip install -r requirements/requirements-serve.txt

# Core dependencies only
pip install -r requirements/requirements-base.txt
```

### 2. Configure environment

Important for remote access and security:

```bash
cp .env.example .env
nano .env
```

Required settings for remote access:
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# Authentication (important!)
GRADIO_USERNAME=your_username
GRADIO_PASSWORD=your_secure_password
```

See `.env.example` for all options.

### 3. Project Structure

```
├── data/                      # Images and metadata (gitignored)
│   ├── HAM10000_metadata.csv
│   └── ds/img/
├── models/
│   ├── checkpoints/           # Trained weights and calibration files
│   │   ├── melanoma_resnet50_nb.pth
│   │   ├── temperature.json
│   │   └── operating_points.json
│   └── label_maps/
│       └── label_map_nb.json
├── src/
│   ├── inference/
│   │   ├── cli.py            # Command-line inference
│   │   └── xai.py            # Inference and XAI utilities
│   ├── training/
│   │   ├── train.py          # Training utilities
│   │   ├── compare_models.py # Multi-architecture comparison
│   │   └── visualize_comparison.py # Generate plots
│   ├── config.py             # Config loader
│   └── serve_gradio.py       # Gradio web UI with chat
├── notebooks/                 # Jupyter notebooks (numbered workflow)
│   ├── 01_train_baseline.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── tests/
│   ├── test_smoke_inference.py
│   └── test_gradio_chat.py
├── experiments/               # Outputs (gitignored)
├── docs/                      # Documentation
│   ├── THESIS_ROADMAP.md
│   ├── ML_ROADMAP.md
│   ├── MEDICAL_BACKGROUND.md
│   ├── ARCHITECTURE.md
│   ├── MODEL_COMPARISON_GUIDE.md
│   └── SERVER_DEPLOYMENT.md
├── scripts/
│   ├── start_server.sh
│   └── setup_experiments.sh
└── requirements/
    ├── requirements-base.txt
    ├── requirements-train.txt
    └── requirements-serve.txt
```
│   ├── MEDICAL_BACKGROUND.md # Essential clinical knowledge
│   ├── ARCHITECTURE.md       # System design
│   ├── MODEL_COMPARISON_GUIDE.md # Experiment guide
│   └── SERVER_DEPLOYMENT.md  # Deployment guide
├── scripts/                   # Executable scripts
│   ├── start_server.sh       # Launch Gradio UI
│   └── setup_experiments.sh  # Setup experiments
└── requirements/              # Dependencies (organized)
    ├── requirements-base.txt  # Core packages
    ├── requirements-train.txt # Training-specific
    └── requirements-serve.txt # Serving-specific
```

### 4. Training (optional)

Open `notebooks/01_train_baseline.ipynb` and run the cells in order. The notebook walks through:
1. Setup and configuration
2. Loading metadata and creating train/val/test splits
3. Dataset class and transforms
4. Model definition and training helpers
5. Training with early stopping (or load existing checkpoint)
6. Evaluation with ROC/PR curves and confusion matrix
7. Grad-CAM visualizations
8. Temperature calibration
9. Operating thresholds for melanoma
10. Sanity check inference with calibrated verdict

The notebook saves these files:
- `models/checkpoints/melanoma_resnet50_nb.pth`
- `models/label_maps/label_map_nb.json`
- `models/checkpoints/temperature.json` (T ≈ 1.865)
- `models/checkpoints/operating_points.json` (spec95 ≈ 0.724, spec90 ≈ 0.615)
- `experiments/calibration/reliability_pre_post.png`

### 5. CLI Inference

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

Options:
- `--no-ask`: Disable interactive Q&A (useful for batch runs)
- `--operating-key`: Choose `melanoma_spec95` (default, ~95% specificity) or `melanoma_spec90` (~90% specificity)

Example output:
```
Prediction: benign keratosis-like lesions | p=0.956
Calibrated melanoma p=0.029 | threshold=0.724 → VERDICT: non-melanoma
Grad-CAM overlay saved to experiments/overlay_sanity.jpg
```

### 6. Gradio Web UI

For local testing:
```bash
bash scripts/start_server.sh
# or directly:
python src/serve_gradio.py
```

For remote access (e.g., Ubuntu server to Mac client):

1. Configure `.env`:
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_USERNAME=your_username
GRADIO_PASSWORD=your_password
```

2. Launch server:
```bash
bash scripts/start_server.sh
```

3. On client machine, open: `http://<server-ip>:7860`

See `docs/SERVER_DEPLOYMENT.md` for complete server setup instructions.

The UI provides:
- Grad-CAM heatmap overlay
- Predicted class
- Calibrated class probabilities (JSON)
- Melanoma decision (probability vs threshold)
- Interactive chat Q&A when melanoma probability is uncertain

The chat feature appears when melanoma probability is within ±0.15 of the operating threshold. It asks 3 clinical questions about the lesion (size changes, diameter, border/color irregularity) and provides educational context based on ABCDE criteria.

### 7. Model Comparison Experiments

Compare multiple architectures on HAM10000:

```bash
# Train and evaluate 4 architectures
python src/training/compare_models.py \
  --metadata data/HAM10000_metadata.csv \
  --img-dir data/ds/img \
  --output-dir experiments/model_comparison \
  --architectures resnet50 efficientnet_b3 densenet121 vit_b_16 \
  --epochs 20 --batch-size 32

# Generate visualizations
python src/training/visualize_comparison.py \
  --results experiments/model_comparison/comparison_results.json \
  --output-dir experiments/model_comparison/visualizations
```

Outputs:
- Comparison tables (CSV and LaTeX format)
- Training curves, metrics charts, calibration plots
- Confusion matrices, inference time comparisons
- Summary report with model rankings

See `docs/MODEL_COMPARISON_GUIDE.md` for details.

## Key Features

**Interactive Chat Q&A**
- Appears when melanoma probability is uncertain
- Asks clinical questions based on ABCDE criteria
- Provides educational context
- Helps refine risk assessment

**Temperature Calibration**
- Fits temperature parameter T on validation data to improve probability calibration
- Divides logits by T before softmax
- Evaluated with ECE (Expected Calibration Error), Brier score, reliability diagrams

**Operating Thresholds**
- Computes ROC for melanoma detection
- Selects thresholds at target specificity (95% or 90%)
- Used for binary melanoma decisions in CLI and web UI

**Grad-CAM Explanations**
- Highlights image regions influencing predictions
- Uses torchcam library with layer4 as target
- Overlays heatmap on original image at 50% alpha

**Model Comparison Framework**
- Supports ResNet50, EfficientNet-B3, DenseNet121, ViT-B/16
- Evaluates accuracy, AUC, calibration, inference speed
- Generates plots and LaTeX tables
- Rankings by different criteria

## Testing

```bash
python -m unittest tests/test_smoke_inference.py
```

Validates model loading, temperature scaling, operating points, and inference.

## Notes

- **Data**: Place `HAM10000_metadata.csv` and images in `data/ds/img/`. CSV needs columns `image_id` and `dx` (diagnosis label).
- **Security**: Always set `GRADIO_USERNAME` and `GRADIO_PASSWORD` in `.env` when deploying on a network. Never commit `.env`!
- **Server Setup**: See `docs/SERVER_DEPLOYMENT.md` for Ubuntu GPU server + Mac client setup.
- **Reproducibility**: Set `SEED=42` in notebooks. Consider pinning exact package versions in requirements.
- **Thesis:** Include reliability plot, report ECE/Brier pre/post calibration, document chosen operating point (spec95/spec90), and discuss Grad-CAM faithfulness.

## License

See `LICENSE.md`.

## References

- HAM10000 dataset (example): https://doi.org/10.1038/sdata.2018.161
- Temperature scaling: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations..." (ICCV 2017)
