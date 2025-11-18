# System Architecture Overview

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MELANOMA DETECTION SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐
│   DATA PREPARATION      │
├─────────────────────────┤
│ HAM10000_metadata.csv   │
│ Image dataset (ds/img/) │
│ Train/val split         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  ResNet-50   │  │ EfficientNet │  │ DenseNet-121 │  │  ViT-B/16    │  │
│  │              │  │     -B3      │  │              │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │                 │            │
│         └─────────────────┴─────────────────┴─────────────────┘            │
│                               │                                             │
│                               ▼                                             │
│                    ┌──────────────────────┐                                │
│                    │  Training Loop       │                                │
│                    │  - Data Augmentation │                                │
│                    │  - Early Stopping    │                                │
│                    │  - Best Model Save   │                                │
│                    └──────────┬───────────┘                                │
│                               │                                             │
│                               ▼                                             │
│                    ┌──────────────────────┐                                │
│                    │ Temperature          │                                │
│                    │ Calibration          │                                │
│                    │ (LBFGS on val set)   │                                │
│                    └──────────┬───────────┘                                │
│                               │                                             │
│                               ▼                                             │
│                    ┌──────────────────────┐                                │
│                    │ Operating Threshold  │                                │
│                    │ Computation          │                                │
│                    │ (95% Specificity)    │                                │
│                    └──────────┬───────────┘                                │
│                               │                                             │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL ARTIFACTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ✓ model_checkpoint.pth     (weights + architecture info)                  │
│  ✓ temperature.json          (optimal T for calibration)                    │
│  ✓ operating_points.json     (thresholds at spec95/spec90)                 │
│  ✓ label_map.json            (class name → index mapping)                  │
└────────┬────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE MODES                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐                    ┌───────────────────────────┐
│   CLI INFERENCE         │                    │  GRADIO WEB UI            │
├─────────────────────────┤                    ├───────────────────────────┤
│                         │                    │                           │
│  1. Load image          │                    │  ┌─────────────────────┐ │
│  2. Preprocess          │                    │  │  Image Upload       │ │
│  3. Forward pass        │                    │  └──────────┬──────────┘ │
│  4. Apply temperature   │                    │             │            │
│  5. Get probabilities   │                    │             ▼            │
│  6. Apply threshold     │                    │  ┌─────────────────────┐ │
│  7. Generate Grad-CAM   │                    │  │  Prediction         │ │
│  8. Save overlay        │                    │  │  - Grad-CAM overlay │ │
│                         │                    │  │  - Class probs      │ │
│  Optional:              │                    │  │  - Melanoma verdict │ │
│  9. Ask follow-up Q&A   │                    │  └──────────┬──────────┘ │
│     (if uncertain)      │                    │             │            │
│                         │                    │             ▼            │
│  Flags:                 │                    │  ┌─────────────────────┐ │
│  --no-ask (disable Q&A) │                    │  │  Chat Q&A           │ │
│  --operating-key        │                    │  │  (if uncertain)     │ │
│                         │                    │  │                     │ │
└─────────────────────────┘                    │  │  Q1: Size change?   │ │
                                               │  │  Q2: Diameter >6mm? │ │
                                               │  │  Q3: Irregular?     │ │
                                               │  │                     │ │
                                               │  │  → Refine prob      │ │
                                               │  │  → Final verdict    │ │
                                               │  └─────────────────────┘ │
                                               │                           │
                                               └───────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION & COMPARISON                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  compare_models.py → comparison_results.json                                │
│                                                                              │
│  visualize_comparison.py ↓                                                  │
│                                                                              │
│  ┌───────────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  Comparison Tables    │  │  Training Curves │  │  Metrics Charts    │  │
│  │  - CSV format         │  │  - Loss/Accuracy │  │  - Bar comparisons │  │
│  │  - LaTeX format       │  │  - Per model     │  │  - All models      │  │
│  └───────────────────────┘  └──────────────────┘  └────────────────────┘  │
│                                                                              │
│  ┌───────────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  Calibration Plots    │  │  Confusion       │  │  Summary Report    │  │
│  │  - ECE comparison     │  │  Matrices        │  │  - Rankings        │  │
│  │  - Brier scores       │  │  - All models    │  │  - Recommendations │  │
│  └───────────────────────┘  └──────────────────┘  └────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT OPTIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Local Development:                                                         │
│  └─ python src/serve_gradio.py (localhost:7860)                            │
│                                                                              │
│  Remote Server (Ubuntu GPU + Mac client):                                  │
│  └─ Configure .env → Launch on 0.0.0.0:7860 → Access from Mac browser     │
│                                                                              │
│  Production (Future):                                                       │
│  └─ Docker container → Cloud deployment → CI/CD pipeline                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Flow
```
HAM10000 Images → Preprocessing → Augmentation → Model Training → 
  → Best Checkpoint → Temperature Calibration → Operating Thresholds → 
    → Model Artifacts (ready for inference)
```

### Inference Flow (CLI)
```
Input Image → Preprocess → Model Forward Pass → Apply Temperature → 
  → Get Probabilities → Compare to Threshold → Melanoma Decision → 
    → Grad-CAM Generation → Save Overlay → 
      → [Optional] Ask Q&A if uncertain → Refine Probability
```

### Inference Flow (Web UI)
```
User Uploads Image → Predict Button Click → 
  → Forward Pass (temp-scaled) → Display Results:
    ├─ Grad-CAM Overlay
    ├─ Class Probabilities
    └─ Melanoma Verdict
  
  IF probability uncertain (within ±0.15 of threshold):
    → Show Chat Interface → Ask Q1 → Get Answer → Update Prob → 
      → Ask Q2 → Get Answer → Update Prob → 
        → Ask Q3 → Get Answer → Update Prob → 
          → Final Assessment
```

## Key Decision Points

### 1. Temperature Calibration
```
Logits → Divide by Temperature T → Softmax → Calibrated Probabilities
```
- T > 1: Softens (flattens) distribution → less confident
- T < 1: Sharpens (peaks) distribution → more confident  
- T = 1: No change (standard softmax)

### 2. Operating Threshold
```
Melanoma Probability ≥ Threshold → MELANOMA
Melanoma Probability < Threshold → NON-MELANOMA
```
- Threshold at 95% specificity: ~0.724 (fewer false positives, clinical safety)
- Threshold at 90% specificity: ~0.615 (higher sensitivity, screening mode)

### 3. Chat Activation
```
|melanoma_prob - threshold| ≤ 0.15 → Show Chat Q&A
```
Example: If threshold = 0.50, chat shows for probs in [0.35, 0.65]

### 4. Q&A Probability Refinement
```
Base Prob → Answer Q1 → Adjust ±8%/3% → Answer Q2 → Adjust ±8%/3% → 
  → Answer Q3 → Adjust ±8%/3% → Final Refined Prob
```
- "yes" (risk factor present) → +8%
- "no" (risk factor absent) → -3%

## File Organization

```
melanoma-detection/
├── data/                         # Raw data (gitignored)
├── models/
│   ├── checkpoints/              # Trained weights (gitignored)
│   └── label_maps/               # Class mappings
├── src/
│   ├── config.py                 # Configuration loader
│   ├── serve_gradio.py           # Web UI with chat
│   ├── inference/
│   │   ├── cli.py                # Command-line interface
│   │   └── xai.py                # XAI utilities (Grad-CAM)
│   └── training/
│       ├── train.py              # Training utilities
│       ├── compare_models.py     # Multi-arch comparison
│       └── visualize_comparison.py  # Plot generation
├── experiments/                  # Experiment outputs (gitignored)
│   └── model_comparison/
│       ├── comparison_results.json
│       ├── *_checkpoint.pth      # All model checkpoints
│       └── visualizations/       # Thesis-ready figures
├── docs/                         # Documentation
│   ├── MODEL_COMPARISON_GUIDE.md
│   ├── SERVER_DEPLOYMENT.md
│   ├── PRE_GITHUB_CHECKLIST.md
│   └── RECENT_UPDATES.md
├── tests/
│   ├── test_smoke_inference.py
│   └── test_gradio_chat.py       # Chat Q&A validation
├── .env.example                  # Configuration template
├── .gitignore                    # Security + large files
└── README.md                     # Main documentation
```

## Technology Stack

### Core ML
- **PyTorch 2.8.0** - Deep learning framework
- **torchvision 0.23.0** - Pretrained models + transforms
- **torchcam** - Grad-CAM implementation

### Web UI
- **Gradio** - Interactive web interface with chat

### Evaluation
- **scikit-learn** - Metrics (ROC, confusion matrix, calibration)
- **matplotlib + seaborn** - Visualization

### Configuration
- **python-dotenv** - Environment variable management

### Deployment
- **Ubuntu 22.04** - Server OS (GPU: RTX 5060 Ti 16GB)
- **UFW** - Firewall (port 7860)
- **tmux** - Session management
- **systemd** (optional) - Service management

## Security Model

```
.env (local only, gitignored)
  ├─ GRADIO_USERNAME
  ├─ GRADIO_PASSWORD
  ├─ Server binding (0.0.0.0 for remote)
  └─ Port configuration

src/config.py
  ├─ Loads .env variables
  ├─ Validates paths
  └─ Provides auth helpers

Gradio
  ├─ Basic HTTP auth (if configured)
  ├─ Bound to 0.0.0.0 for remote access
  └─ Firewall restricts to local subnet

GitHub
  ├─ .gitignore blocks .env, secrets/, *.key
  ├─ Only .env.example committed
  └─ Model weights NOT pushed (too large)
```

## Performance Characteristics

### Model Sizes (Approximate)
- ResNet-50: ~25M parameters, ~100MB
- EfficientNet-B3: ~12M parameters, ~50MB  
- DenseNet-121: ~8M parameters, ~32MB
- ViT-B/16: ~86M parameters, ~350MB

### Inference Speed (RTX 5060 Ti, batch=1)
- ResNet-50: ~20-30ms
- EfficientNet-B3: ~25-35ms
- DenseNet-121: ~15-25ms  
- ViT-B/16: ~40-60ms

### Training Time (20 epochs, HAM10000)
- Per model: 2-4 hours
- Full comparison (4 models): 8-16 hours

---

**Created:** November 15, 2025  
**Purpose:** System architecture documentation for thesis and development
