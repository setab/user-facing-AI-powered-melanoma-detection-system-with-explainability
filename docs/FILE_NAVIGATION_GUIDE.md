# File Navigation Guide

Complete map of the melanoma detection project structure. This guide tells you where everything is and what each file does.

## Quick Reference

- **Need to train a model?** → `notebooks/01_train_baseline.ipynb`
- **Want to run inference on an image?** → `src/inference/cli.py` or `src/serve_gradio.py`
- **Looking for trained models?** → `models/checkpoints/` and `experiments/model_comparison_full/`
- **Need the thesis?** → `thesis/MAIN_THESIS.md` or individual sections in `thesis/sections/`
- **Want experiment results?** → `experiments/model_comparison_full/comparison_results.json`
- **Study materials?** → `docs/` directory (see Study Materials section below)

## Directory Structure

### `/data/` - Dataset Files
Where all the image data and metadata lives (gitignored to avoid committing large files).

```
data/
├── HAM10000_metadata.csv          # Original metadata with image_id, dx, lesion_id, etc.
├── build_metadata.py              # Script to process/filter metadata
├── splits.json                    # Train/val/test split indices (if saved)
└── ds/
    ├── img/                       # Raw dermoscopic images (ISIC_*.jpg)
    └── ann/                       # Annotations (if any)
```

**Key files:**
- `HAM10000_metadata.csv`: Original dataset with 10,015 rows (7 diagnostic categories)
- `splits.json`: Saved train/val/test indices for reproducibility
- `ds/img/`: Contains ~10,000 JPEG images named ISIC_0024306.jpg, etc.

### `/models/` - Trained Models and Configs

```
models/
├── checkpoints/
│   ├── melanoma_resnet50_nb.pth   # ResNet-50 trained from notebook 01
│   ├── temperature.json           # Temperature value (T ≈ 1.865) for calibration
│   └── operating_points.json      # Melanoma thresholds (spec95: 0.724, spec90: 0.615)
└── label_maps/
    └── label_map_nb.json          # Maps class names to indices (7 classes)
```

**What these do:**
- `.pth` files: PyTorch model weights (state_dict)
- `temperature.json`: Contains single float for temperature scaling
- `operating_points.json`: Contains melanoma class index and thresholds for different specificity levels
- `label_map_nb.json`: Maps diagnosis names (e.g., "melanoma", "nevus") to integer indices

### `/experiments/` - Experiment Results

```
experiments/
├── model_comparison/              # Initial comparison experiment (4 models, 10 epochs)
│   ├── comparison_results.json
│   ├── resnet50_checkpoint.pth
│   ├── train_split.csv
│   ├── val_split.csv
│   └── visualizations/            # 8 PNG plots
└── model_comparison_full/         # Full training (4 models, 20 epochs) ← USE THIS
    ├── comparison_results.json    # Complete metrics for all 4 architectures
    ├── resnet50_checkpoint.pth    # ResNet-50 weights (91 MB)
    ├── efficientnet_b3_checkpoint.pth  # EfficientNet-B3 weights (42 MB) ← BEST MODEL
    ├── densenet121_checkpoint.pth      # DenseNet-121 weights (28 MB)
    ├── vit_b_16_checkpoint.pth         # ViT-B/16 weights (328 MB)
    ├── train_split.csv            # Training data used
    ├── val_split.csv              # Validation data used
    └── visualizations/            # 8 figures for thesis
        ├── accuracy_comparison.png
        ├── auc_comparison.png
        ├── calibration_comparison.png
        ├── training_curves.png
        ├── confusion_matrices.png
        ├── inference_time.png
        ├── calibration_curves.png
        └── metrics_table.png
```

**Key results (from `model_comparison_full/comparison_results.json`):**
- **EfficientNet-B3**: 89.22% accuracy, 95.34% melanoma AUC (BEST)
- **ResNet-50**: 87.27% accuracy, 94.18% melanoma AUC
- **DenseNet-121**: 86.12% accuracy, 93.89% melanoma AUC
- **ViT-B/16**: 84.47% accuracy, 92.76% melanoma AUC

### `/src/` - Source Code

```
src/
├── __init__.py
├── config.py                      # Configuration loader (reads .env)
├── serve_gradio.py                # Gradio web UI with chat feature
├── inference/
│   ├── __init__.py
│   ├── cli.py                     # Command-line inference script
│   └── xai.py                     # Shared inference utilities (Grad-CAM, calibration)
└── training/
    ├── __init__.py
    ├── train.py                   # Training utilities (Dataset, train loop, etc.)
    ├── compare_models.py          # Multi-architecture comparison script
    └── visualize_comparison.py    # Generate plots from comparison results
```

**What each file does:**

**`config.py`**
- Loads configuration from `.env` file
- Defines paths to models, data, checkpoints
- Provides `Config` class with all settings

**`serve_gradio.py`**
- Main web interface using Gradio
- Loads trained model and runs inference
- Shows Grad-CAM heatmap, probabilities, melanoma decision
- Interactive chat Q&A when melanoma probability is uncertain
- Run with: `python src/serve_gradio.py`

**`inference/cli.py`**
- Command-line inference on single images
- Supports calibration and operating thresholds
- Generates Grad-CAM overlays
- Interactive Q&A follow-up (use `--no-ask` to disable)
- Run with: `python -m src.inference.cli --image <path>`

**`inference/xai.py`**
- Core inference functions
- `load_temperature()`, `load_operating_points()`: Load calibration configs
- `apply_temperature()`: Apply temperature scaling to logits
- `predict_with_gradcam()`: Run inference and generate Grad-CAM
- Shared by both CLI and web UI

**`training/train.py`**
- `HAM10000Dataset`: PyTorch Dataset class
- `build_model()`: Create model architecture
- `train_one_epoch()`, `validate()`: Training loops
- `fit_temperature()`: Temperature calibration
- `compute_operating_thresholds()`: Find thresholds for target specificity
- Used by notebooks and comparison script

**`training/compare_models.py`**
- Trains multiple architectures (ResNet50, EfficientNet-B3, DenseNet121, ViT-B/16)
- Identical training setup for fair comparison
- Evaluates all metrics: accuracy, AUC, ECE, Brier score, inference time
- Saves checkpoints and results JSON
- Run with: `python src/training/compare_models.py`

**`training/visualize_comparison.py`**
- Loads comparison results JSON
- Generates 8 publication-ready plots
- Creates LaTeX tables for thesis
- Run with: `python src/training/visualize_comparison.py --results <path>`

### `/notebooks/` - Jupyter Notebooks

```
notebooks/
├── 01_train_baseline.ipynb        # Main training notebook (start here)
├── 02_exploratory_analysis.ipynb  # Data exploration and statistics
├── 03_model_evaluation.ipynb      # Model testing and analysis
├── archive/
│   └── printImages.ipynb          # Old notebook for viewing images
└── models/
    └── checkpoints/               # Models saved from notebooks
```

**Workflow:**
1. **01_train_baseline.ipynb**: Complete training pipeline
   - Load HAM10000 data and create splits
   - Train ResNet-50 from scratch
   - Evaluate with ROC curves, confusion matrix
   - Generate Grad-CAM visualizations
   - Calibrate with temperature scaling
   - Compute operating thresholds
   - Save all artifacts to `models/checkpoints/`

2. **02_exploratory_analysis.ipynb**: Understand the data
   - Class distribution (67% nevi, 11% melanoma)
   - Image statistics
   - Patient demographics
   - Visualize sample images

3. **03_model_evaluation.ipynb**: Test and analyze models
   - Load trained model
   - Detailed performance analysis
   - Error analysis
   - Grad-CAM examples

### `/thesis/` - Thesis Documents

```
thesis/
├── MAIN_THESIS.md                 # Main document with table of contents
├── README.md                      # Navigation guide for thesis
├── THESIS_COMPLETE_SUMMARY.md     # 2-page overview
├── THESIS_PROGRESS_TRACKER_COMPLETE.md  # Completion checklist
├── THESIS_PROGRESS_TRACKER.md     # Ongoing progress tracker
├── sections/
│   ├── 00_abstract.md             # Abstract (400 words)
│   ├── 01_introduction.md         # Introduction (~2,500 words)
│   ├── 02_background.md           # Background & Related Work (~4,500 words)
│   ├── 03_methodology.md          # Methodology (~5,000 words)
│   ├── 04_results.md              # Results (~4,000 words)
│   ├── 05_discussion.md           # Discussion (~5,500 words)
│   ├── 06_conclusion.md           # Conclusion (~2,500 words)
│   ├── 07_references.md           # References (30 citations)
│   └── 08_appendices.md           # Appendices (technical details)
├── figures/                       # Figures for thesis (empty - copy from experiments/)
└── references/                    # Reference materials (empty)
```

**Total**: ~25,000 words, 100% complete

**How to read:**
1. Start with `MAIN_THESIS.md` for structure
2. Read sections in order (00 through 08)
3. Figures referenced are in `experiments/model_comparison_full/visualizations/`

### `/docs/` - Documentation (Study Materials)

```
docs/
├── ARCHITECTURE.md                # System design and data flow
├── AI_EXPLANATION_FEATURE.md      # How AI explanations work
├── AI_EXPLANATION_QUICK_REF.md    # Quick reference for explanations
├── CHAT_SYSTEM_EXPLAINED.md       # Interactive chat feature details
├── COMPLETE_CODE_WALKTHROUGH.md   # Line-by-line code explanation
├── DATA_FLOW_DIAGRAM.md           # Visual system architecture
├── MEDICAL_BACKGROUND.md          # Melanoma clinical knowledge
├── ML_ROADMAP.md                  # Machine learning learning path
├── MODEL_COMPARISON_GUIDE.md      # How to run comparison experiments
├── PROJECT_RESTRUCTURE.md         # Project organization history
├── SERVER_DEPLOYMENT.md           # Deployment instructions
├── THESIS_ROADMAP.md              # Thesis writing plan
└── archive/                       # Old documentation
    ├── copilot.md
    ├── High_level_plan.md
    ├── latex_thesis_ready.md
    └── ...
```

**Study materials by topic:**

**Understanding the System:**
- `ARCHITECTURE.md`: Overall system design
- `DATA_FLOW_DIAGRAM.md`: How data flows through the system (7 layers)
- `COMPLETE_CODE_WALKTHROUGH.md`: Detailed code explanation

**Medical Knowledge:**
- `MEDICAL_BACKGROUND.md`: Melanoma, ABCDE criteria, dermoscopy basics

**Machine Learning:**
- `ML_ROADMAP.md`: Learning path for ML concepts
- `MODEL_COMPARISON_GUIDE.md`: How to train and compare models

**Features:**
- `AI_EXPLANATION_FEATURE.md`: How AI generates explanations
- `CHAT_SYSTEM_EXPLAINED.md`: Interactive Q&A feature
- `SERVER_DEPLOYMENT.md`: Deployment guide

**Thesis:**
- `THESIS_ROADMAP.md`: Complete thesis writing plan

### `/scripts/` - Helper Scripts

```
scripts/
├── start_server.sh                # Launch Gradio web UI
├── setup_experiments.sh           # Setup experiment directories
├── save_splits.py                 # Save train/val/test splits to JSON
└── monitor_training.sh            # Monitor GPU during training
```

**Usage:**
- `bash scripts/start_server.sh`: Start web interface
- `bash scripts/setup_experiments.sh`: Create experiment directories
- `python scripts/save_splits.py`: Save split indices for reproducibility

### `/tests/` - Unit Tests

```
tests/
├── test_smoke_inference.py        # Basic inference test
└── test_gradio_chat.py            # Chat feature test
```

**Run tests:**
```bash
python -m unittest tests/test_smoke_inference.py
```

### `/requirements/` - Dependencies

```
requirements/
├── requirements-base.txt          # Core dependencies (torch, numpy, PIL)
├── requirements-train.txt         # Training dependencies (includes base)
└── requirements-serve.txt         # Serving dependencies (includes base)
```

**Install:**
- Training: `pip install -r requirements/requirements-train.txt`
- Serving only: `pip install -r requirements/requirements-serve.txt`
- Core only: `pip install -r requirements/requirements-base.txt`

## Root Files

```
.env                               # Environment configuration (NEVER commit!)
.env.example                       # Template for .env
.gitignore                         # Git ignore rules
setup.py                           # Package setup
LICENSE.md                         # MIT License
README.md                          # Project README
QUICK_START.md                     # Quick start guide
COMPLETE_REPRODUCTION_GUIDE.md     # Step-by-step reproduction
CONTRIBUTING.md                    # Contribution guidelines
PROJECT_SUMMARY.txt                # Project summary
TODO.md                            # TODO list
TASKS_COMPLETED.md                 # Completed tasks
TRAINING_IN_PROGRESS.md            # Training status
RESTRUCTURE_COMPLETE.md            # Restructuring notes
training_log_20251121_151106.txt  # Training log from Nov 21
```

## Common Tasks

### I want to train a model
1. Open `notebooks/01_train_baseline.ipynb`
2. Run cells in order
3. Model saves to `models/checkpoints/melanoma_resnet50_nb.pth`

### I want to run inference on an image
**CLI:**
```bash
python -m src.inference.cli --image data/ds/img/ISIC_0024306.jpg
```

**Web UI:**
```bash
bash scripts/start_server.sh
```

### I want to compare multiple models
```bash
python src/training/compare_models.py \
  --metadata data/HAM10000_metadata.csv \
  --img-dir data/ds/img \
  --output-dir experiments/my_comparison \
  --architectures resnet50 efficientnet_b3 densenet121 vit_b_16 \
  --epochs 20
```

### I want to see experiment results
Look at: `experiments/model_comparison_full/comparison_results.json`

Best model: EfficientNet-B3 (89.22% accuracy)

### I want to read the thesis
Start with: `thesis/MAIN_THESIS.md`

Or read sections individually in `thesis/sections/`

### I want to deploy the web interface
1. Configure `.env` with authentication
2. Run `bash scripts/start_server.sh`
3. See `docs/SERVER_DEPLOYMENT.md` for full instructions

### I want to understand how the code works
Read: `docs/COMPLETE_CODE_WALKTHROUGH.md`

### I want to understand the medical concepts
Read: `docs/MEDICAL_BACKGROUND.md`

## File Sizes Reference

**Large files (gitignored):**
- `data/ds/img/`: ~2.5 GB (10,000 images)
- `experiments/model_comparison_full/*.pth`: 
  - ResNet-50: 91 MB
  - EfficientNet-B3: 42 MB
  - DenseNet-121: 28 MB
  - ViT-B/16: 328 MB
- `models/checkpoints/melanoma_resnet50_nb.pth`: 91 MB

**Small files (tracked in git):**
- Code files: ~200 KB total
- Documentation: ~500 KB total
- Configuration files: ~10 KB total

## Version Control

**Tracked in git:**
- All source code (`src/`, `scripts/`, `tests/`)
- Documentation (`docs/`, `thesis/`)
- Configuration templates (`.env.example`)
- Requirements files
- README files

**Ignored (.gitignored):**
- `data/` - too large
- `experiments/` - regenerable
- `models/checkpoints/*.pth` - too large
- `.env` - contains secrets
- `__pycache__/` - Python cache
- `.ipynb_checkpoints/` - Jupyter cache

## Where Things Get Saved

**When you train in notebook 01:**
- Model: `models/checkpoints/melanoma_resnet50_nb.pth`
- Label map: `models/label_maps/label_map_nb.json`
- Temperature: `models/checkpoints/temperature.json`
- Thresholds: `models/checkpoints/operating_points.json`
- Calibration plot: `experiments/calibration/reliability_pre_post.png`

**When you run compare_models.py:**
- Results: `experiments/model_comparison_full/comparison_results.json`
- Checkpoints: `experiments/model_comparison_full/*_checkpoint.pth`
- Splits: `experiments/model_comparison_full/train_split.csv`, `val_split.csv`

**When you run visualize_comparison.py:**
- Plots: `experiments/model_comparison_full/visualizations/*.png`
- Tables: `experiments/model_comparison_full/visualizations/*.csv`

**When you run CLI inference:**
- Overlay: Path specified by `--out` argument (e.g., `experiments/overlay.jpg`)

## Quick File Finder

Need to find something? Use these commands:

```bash
# Find all Python files
find . -name "*.py" -type f

# Find all Jupyter notebooks
find . -name "*.ipynb" -type f

# Find all markdown documentation
find docs/ -name "*.md" -type f

# Find trained models
find . -name "*.pth" -type f

# Find JSON config files
find . -name "*.json" -type f

# Find all thesis sections
ls thesis/sections/
```

## Summary

**Start here for:**
- Training: `notebooks/01_train_baseline.ipynb`
- Inference: `src/inference/cli.py` or `src/serve_gradio.py`
- Results: `experiments/model_comparison_full/comparison_results.json`
- Thesis: `thesis/MAIN_THESIS.md`
- Documentation: `docs/` (especially `COMPLETE_CODE_WALKTHROUGH.md`)
- Study: `docs/MEDICAL_BACKGROUND.md` and `docs/ML_ROADMAP.md`

**Best model:** EfficientNet-B3 at `experiments/model_comparison_full/efficientnet_b3_checkpoint.pth`
- 89.22% accuracy
- 95.34% melanoma AUC
- 10.06ms inference time
