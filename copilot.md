# Copilot Context: Melanoma-detection

**Last Updated**: 2025-11-15  
**Status**: ✅ Production-Ready with Security Hardening

---

## 🏗️ Architecture Overview

**Deployment Model**: Ubuntu GPU Server (RTX 5060 Ti 16GB) + Mac Client

```
┌──────────────────────┐         Network         ┌──────────────────────┐
│   Ubuntu Server      │◄────────────────────────►│   Mac Client         │
│   (GPU: RTX 5060Ti)  │   SSH / HTTP            │   (Web Browser)      │
│                      │                          │                      │
│  • Training          │                          │  • Access Gradio UI  │
│  • Inference (GPU)   │                          │  • Upload images     │
│  • Gradio server     │                          │  • View results      │
└──────────────────────┘                          └──────────────────────┘
```

---

## 🔒 Security & Configuration

### Environment Variables (.env)
**CRITICAL**: Never commit `.env` to git!

```bash
# .env (gitignored)
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_USERNAME=your_username       # SET THIS!
GRADIO_PASSWORD=your_password       # SET THIS!
WEIGHTS_PATH=models/checkpoints/melanoma_resnet50_nb.pth
LABEL_MAP_PATH=models/label_maps/label_map_nb.json
TEMPERATURE_JSON_PATH=models/checkpoints/temperature.json
OPERATING_JSON_PATH=models/checkpoints/operating_points.json
```

### Files Protected by .gitignore
- `.env`, `.env.local`, `.env.*.local`
- `secrets/`, `credentials/`
- `*.key`, `*.pem`, `*.crt` (SSL certs)
- `id_rsa`, `*.ssh/` (SSH keys)
- `config.local.py`
- `__pycache__/`, `*.pyc`
- `data/` (images)
- `models/checkpoints/*.pth` (model weights)
- `experiments/` (outputs)

### Configuration Management
- **src/config.py**: Centralized config loader with validation
- Loads from `.env` using python-dotenv
- Safe defaults for all settings
- Validates critical paths on startup

---

## 📊 Project Status: Completed Features

### ✅ Core ML Pipeline
- ResNet50 training with class weighting and early stopping
- Temperature calibration (T ≈ 1.865)
- Operating thresholds: melanoma_spec95 (0.724), melanoma_spec90 (0.615)
- Grad-CAM explanations with robust hook management

### ✅ Inference Modes
1. **CLI** (`src/inference/cli.py`):
   - `--no-ask` flag for batch/CI runs
   - Temperature scaling integration
   - Operating threshold verdicts
   - Grad-CAM overlay export

2. **Gradio UI** (`src/serve_gradio.py`):
   - Remote access ready (0.0.0.0 binding)
   - Optional authentication (username/password)
   - Calibrated probabilities
   - Melanoma decision display
   - Hardened Grad-CAM (shape handling + hook cleanup)

### ✅ Testing & Quality
- Smoke test: `tests/test_smoke_inference.py`
- Verified on: torch 2.8.0+cu128, torchvision 0.23.0+cu128
- GPU: RTX 5060 Ti 16GB

### ✅ Documentation
- **README.md**: Complete usage guide
- **docs/SERVER_DEPLOYMENT.md**: Ubuntu server + Mac client setup
- **copilot.md**: Living context (this file)
- **.env.example**: Configuration template

---

## 🗂️ Project Structure

```
melanoma-detection/
├── .env.example           # Config template (copy to .env)
├── .gitignore            # Security: ignores .env, secrets, keys
├── README.md             # Main documentation
├── copilot.md            # Living context (AI pair programming)
├── requirements-train.txt
│
├── data/                 # Images & metadata (gitignored)
│   ├── HAM10000_metadata.csv
│   └── ds/img/
│
├── models/
│   ├── checkpoints/      # Model weights + calibration
│   │   ├── melanoma_resnet50_nb.pth (gitignored)
│   │   ├── temperature.json
│   │   └── operating_points.json
│   ├── label_maps/
│   │   └── label_map_nb.json
│   └── requirements-serving.txt
│
├── src/
│   ├── config.py         # ⭐ Config loader (reads .env)
│   ├── serve_gradio.py   # Gradio UI with auth
│   ├── inference/
│   │   ├── cli.py        # CLI with --no-ask
│   │   └── xai.py        # Shared utilities
│   └── training/
│
├── learning/
│   └── day1.ipynb        # End-to-end training notebook
│
├── notebooks/            # Exploratory
│   ├── archive/
│   ├── main.ipynb
│   └── melanomaDetection.ipynb
│
├── tests/
│   └── test_smoke_inference.py
│
├── experiments/          # Outputs (gitignored)
│   └── calibration/
│       └── reliability_pre_post.png
│
└── docs/
    ├── SERVER_DEPLOYMENT.md  # Server setup guide
    ├── steps/
    └── markdown/
```

---

## 🚀 Quick Start Commands

### On Ubuntu Server

```bash
# 1. Setup environment
cd /home/the/Codes/Melanoma-detection
cp .env.example .env
nano .env  # Set GRADIO_USERNAME and GRADIO_PASSWORD!

# 2. Install python-dotenv
/home/the/miniconda/envs/ml2/bin/pip install python-dotenv

# 3. Configure firewall (local network only)
sudo ufw allow from 192.168.1.0/24 to any port 7860

# 4. Launch Gradio (persistent with tmux)
tmux new -s melanoma
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
# Detach: Ctrl+B then D

# 5. CLI inference (batch mode)
/home/the/miniconda/envs/ml2/bin/python -m src.inference.cli \
  --image data/ds/img/ISIC_0027990.jpg \
  --weights models/checkpoints/melanoma_resnet50_nb.pth \
  --label-map models/label_maps/label_map_nb.json \
  --temperature-json models/checkpoints/temperature.json \
  --operating-json models/checkpoints/operating_points.json \
  --operating-key melanoma_spec95 \
  --out experiments/result.jpg \
  --no-ask
```

### On Mac Client

```bash
# Find server IP (run on Ubuntu)
ip addr show | grep "inet " | grep -v 127.0.0.1

# Open in browser on Mac
open http://192.168.1.100:7860
```

---

## 🔬 Technical Details

### Model & Artifacts
- **Checkpoint**: `melanoma_resnet50_nb.pth` (ResNet50, 7 classes)
- **Temperature**: 1.8653918504714966
- **Operating Points**:
  - class_index: 5 (melanoma)
  - melanoma_spec95: 0.7238592505455017
  - melanoma_spec90: 0.614717423915863

### Calibration Metrics
- **ECE pre**: 0.1695 → **post**: 0.0509 (70% reduction)
- **Brier pre**: 0.0647 → **post**: 0.0575 (11% reduction)
- Reliability diagram: `experiments/calibration/reliability_pre_post.png`

### Hardware
- **Server**: Ubuntu with RTX 5060 Ti 16GB
- **Python**: `/home/the/miniconda/envs/ml2/bin/python`
- **CUDA**: torch 2.8.0+cu128, torchvision 0.23.0+cu128

---

## 🔐 Security Checklist

- [x] `.env` in `.gitignore`
- [x] Secrets/keys patterns in `.gitignore`
- [x] Authentication support in Gradio (username/password)
- [x] Config validation on startup
- [x] Firewall rules documented
- [x] No hardcoded credentials in code
- [x] Server deployment guide with security notes

---

## 🧪 Testing

```bash
# Smoke test
/home/the/miniconda/envs/ml2/bin/python -m unittest tests/test_smoke_inference.py

# GPU check
/home/the/miniconda/envs/ml2/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Config validation
/home/the/miniconda/envs/ml2/bin/python -c "from src.config import Config; Config.validate(); print('✅ Config valid')"
```

---

## 📝 Maintenance Notes

### Before Git Push
```bash
# Verify no secrets in staged files
git diff --cached | grep -E "(password|api_key|secret|token)" && echo "⚠️ Found secrets!" || echo "✅ Safe"

# Check .env is ignored
git status | grep -q ".env" && echo "⚠️ .env tracked!" || echo "✅ .env ignored"
```

### Updates from Git
```bash
# Backup .env before pull
cp .env .env.backup
git pull
# Merge any new vars from .env.example into .env
```

### Monitor Server
```bash
# GPU usage
nvidia-smi -l 1

# Gradio logs (if using systemd)
sudo journalctl -u melanoma-gradio -f

# Network connections
sudo netstat -tulpn | grep 7860
```

---

## 🎯 Next Steps (Optional Enhancements)

### High Priority
- [ ] Add systemd service for auto-start on boot
- [ ] Setup HTTPS with nginx reverse proxy + Let's Encrypt
- [ ] Add rate limiting for public deployments

### Medium Priority
- [ ] CI/CD with GitHub Actions (run tests on push)
- [ ] Docker/Podman container for portable deployment
- [ ] Prometheus metrics export for monitoring

### Research/Thesis Enhancements
- [ ] Cross-validation for robust performance estimates
- [ ] Uncertainty estimation (MC Dropout or ensembles)
- [ ] Subgroup analysis (if demographics available)
- [ ] XAI faithfulness tests (deletion/insertion curves)
- [ ] Bias/robustness analysis (perturbation tests)

---

## 📚 References

- HAM10000 dataset: https://doi.org/10.1038/sdata.2018.161
- Temperature scaling: Guo et al., ICML 2017
- Grad-CAM: Selvaraju et al., ICCV 2017

---

## 🤖 AI Context Notes

This file serves as the **living context** for AI pair programming (GitHub Copilot, cursor, etc.). Always update this when:
- Architecture changes (new components, deployment model)
- Security changes (new secrets, auth methods)
- Configuration changes (new env vars, paths)
- Feature additions (new endpoints, modes)
- Bug fixes that affect workflow

Keep it concise but complete. Date every major update.
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
