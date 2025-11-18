# Project Restructure Guide

**Goal**: Clean, thesis-ready project structure with clear organization

---

## 📁 Proposed New Structure

```
Melanoma-detection/
│
├── README.md                          # Main project overview
├── QUICK_START.md                     # How to run (keep)
├── LICENSE.md                         # License (keep)
├── .gitignore                         # Git ignore (keep)
├── .env                               # Secrets (keep, don't commit)
├── .env.example                       # Template (keep)
│
├── 📚 docs/                           # ← CONSOLIDATED DOCUMENTATION
│   ├── THESIS_ROADMAP.md              # Full academic plan (moved from root)
│   ├── ML_ROADMAP.md                  # ML/software guide (moved from root)
│   ├── MEDICAL_BACKGROUND.md          # Medical essentials (moved from root)
│   ├── ARCHITECTURE.md                # System design (keep)
│   ├── MODEL_COMPARISON_GUIDE.md      # Experiment guide (keep)
│   ├── SERVER_DEPLOYMENT.md           # Deployment (keep)
│   └── archive/                       # Old docs (moved, not needed daily)
│       ├── all_Pip_installed.md
│       ├── PRE_GITHUB_CHECKLIST.md
│       ├── RECENT_UPDATES.md
│       ├── High_level_plan.md
│       ├── latex_thesis_ready.md
│       └── web_based_melanomaDetection.md
│
├── 📊 data/                           # ← DATA ONLY
│   ├── build_metadata.py              # Data preprocessing script (keep)
│   ├── HAM10000_metadata.csv          # Metadata (keep)
│   └── ds/                            # Dataset (keep)
│       ├── img/                       # Images
│       └── ann/                       # Annotations
│
├── 💻 src/                            # ← SOURCE CODE
│   ├── config.py                      # Configuration (keep)
│   ├── serve_gradio.py                # Web UI (keep)
│   ├── training/                      # Training scripts (keep)
│   │   ├── train.py
│   │   ├── compare_models.py
│   │   └── visualize_comparison.py
│   └── inference/                     # Inference scripts (keep)
│       ├── cli.py
│       └── xai.py
│
├── 🧪 notebooks/                      # ← JUPYTER NOTEBOOKS (CONSOLIDATED)
│   ├── 01_train_baseline.ipynb        # Renamed from learning/day1.ipynb
│   ├── 02_exploratory_analysis.ipynb  # Renamed from main.ipynb
│   ├── 03_model_evaluation.ipynb      # Renamed from melanomaDetection.ipynb
│   └── archive/                       # Old experiments
│       └── printImages.ipynb
│
├── 🎯 models/                         # ← TRAINED MODELS
│   ├── checkpoints/                   # Model weights (keep)
│   │   ├── melanoma_resnet50_nb.pth
│   │   ├── temperature.json
│   │   └── operating_points.json
│   ├── label_maps/                    # Label mappings (keep)
│   │   └── label_map_nb.json
│   └── requirements-serving.txt       # Serving deps (keep)
│
├── 📈 experiments/                    # ← EXPERIMENT RESULTS
│   └── model_comparison/              # Comparison outputs (auto-generated)
│       ├── comparison_results.json
│       ├── comparison_table.tex
│       ├── training_curves.png
│       └── ...
│
├── 🧪 tests/                          # ← UNIT TESTS
│   ├── test_gradio_chat.py            # Chat tests (keep)
│   └── test_smoke_inference.py        # Inference tests (keep)
│
├── 🚀 scripts/                        # ← EXECUTABLE SCRIPTS (NEW)
│   ├── start_server.sh                # Start web UI (moved from root)
│   └── setup_experiments.sh           # Setup experiments (moved from root)
│
└── 📦 requirements/                   # ← DEPENDENCIES (NEW, ORGANIZED)
    ├── requirements-base.txt          # Core dependencies
    ├── requirements-train.txt         # Training-specific (moved from root)
    └── requirements-serve.txt         # Serving-specific (moved from models/)
```

---

## 🗑️ Files to DELETE (Clutter)

### Root Level (Too Many MD Files)
- [x] `copilot.md` → Merge into `docs/archive/` (AI context, not user-facing)
- [x] `THESIS_ROADMAP.md` → Move to `docs/`
- [x] `ML_ROADMAP.md` → Move to `docs/`
- [x] `MEDICAL_BACKGROUND.md` → Move to `docs/`

### Docs Folder (Outdated)
- [x] `docs/all_Pip_installed.md` → Delete or archive (snapshot, not needed)
- [x] `docs/PRE_GITHUB_CHECKLIST.md` → Archive (one-time task)
- [x] `docs/RECENT_UPDATES.md` → Delete (use git log instead)
- [x] `docs/markdown/` → Archive entire folder (outdated drafts)
- [x] `docs/steps/` → Archive entire folder (superseded by roadmaps)

### Notebooks Folder (Confusing Names)
- [x] Rename `main.ipynb` → `02_exploratory_analysis.ipynb`
- [x] Rename `melanomaDetection.ipynb` → `03_model_evaluation.ipynb`
- [x] Move `learning/day1.ipynb` → `notebooks/01_train_baseline.ipynb`
- [x] Delete `learning/` folder after move

### Root Scripts (Cluttered)
- [x] `setup_experiments.sh` → Move to `scripts/`
- [x] `start_server.sh` → Move to `scripts/`

---

## 📋 Restructuring Commands (Run These)

### Step 1: Create New Directories
```bash
cd /home/the/Codes/Melanoma-detection
mkdir -p docs/archive
mkdir -p notebooks/archive
mkdir -p scripts
mkdir -p requirements
```

### Step 2: Move Documentation
```bash
# Move roadmaps to docs/
mv THESIS_ROADMAP.md docs/
mv ML_ROADMAP.md docs/
mv MEDICAL_BACKGROUND.md docs/

# Archive old docs
mv docs/all_Pip_installed.md docs/archive/
mv docs/PRE_GITHUB_CHECKLIST.md docs/archive/
mv docs/RECENT_UPDATES.md docs/archive/
mv docs/markdown/High_level_plan.md docs/archive/
mv "docs/markdown/latex thesis ready.md" docs/archive/latex_thesis_ready.md
mv docs/markdown/web_based_melanomaDetection.md docs/archive/
rmdir docs/markdown
rmdir docs/steps

# Archive copilot.md (AI context, not user-facing)
mv copilot.md docs/archive/
```

### Step 3: Reorganize Notebooks
```bash
# Move and rename for clear ordering
mv learning/day1.ipynb notebooks/01_train_baseline.ipynb
mv notebooks/main.ipynb notebooks/02_exploratory_analysis.ipynb
mv notebooks/melanomaDetection.ipynb notebooks/03_model_evaluation.ipynb

# Remove empty learning folder
rmdir learning
```

### Step 4: Organize Scripts
```bash
mv start_server.sh scripts/
mv setup_experiments.sh scripts/
```

### Step 5: Organize Requirements
```bash
# Split requirements by purpose
mv requirements-train.txt requirements/requirements-train.txt
mv models/requirements-serving.txt requirements/requirements-serve.txt

# Create base requirements (common to all)
cat > requirements/requirements-base.txt << 'EOF'
# Core dependencies for all tasks
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
EOF
```

### Step 6: Update README.md References
```bash
# Update README to point to new locations
# (Manual edit needed - see section below)
```

---

## 📝 Files to UPDATE (Path References)

### 1. README.md
**Update paths:**
```markdown
## 📚 Documentation
- [Quick Start](QUICK_START.md) - Get running in 5 minutes
- [Thesis Roadmap](docs/THESIS_ROADMAP.md) - Complete academic plan
- [ML Roadmap](docs/ML_ROADMAP.md) - ML/software focus
- [Medical Background](docs/MEDICAL_BACKGROUND.md) - Essential clinical knowledge
- [Architecture](docs/ARCHITECTURE.md) - System design
- [Model Comparison](docs/MODEL_COMPARISON_GUIDE.md) - Experiment guide

## 🚀 Quick Start
```bash
# Start web UI
bash scripts/start_server.sh

# Or run directly
python src/serve_gradio.py
```

## 🧪 Training
See `notebooks/01_train_baseline.ipynb` for baseline model training.
```

### 2. QUICK_START.md
**Update commands:**
```markdown
## Start the Server
```bash
bash scripts/start_server.sh
```

## Train Your First Model
Open `notebooks/01_train_baseline.ipynb` in Jupyter and run all cells.
```

### 3. docs/THESIS_ROADMAP.md
**Update notebook reference:**
```markdown
## Week 1: Train Baseline Model
1. Open `notebooks/01_train_baseline.ipynb`
2. Run all cells (30-60 min)
```

### 4. docs/ML_ROADMAP.md
**Update notebook reference:**
```markdown
### Step 1: Train Baseline Model
- **Notebook**: `notebooks/01_train_baseline.ipynb`
- **Time**: 30-60 min (GPU)
```

### 5. scripts/start_server.sh
**Update if needed:**
```bash
#!/bin/bash
cd "$(dirname "$0")/.."  # Go to project root
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
```

### 6. src/serve_gradio.py
**Check imports** (should already use relative imports, but verify):
```python
# Should work from project root
from src.config import ...
```

---

## ✅ Clean Structure Benefits

### Before (Cluttered):
```
❌ 8 .md files in root (confusing)
❌ 3 notebooks in 2 folders (learning/, notebooks/)
❌ Scripts scattered in root
❌ Outdated docs mixed with current
❌ Requirements in 2 locations
```

### After (Clean):
```
✅ 3 essential .md files in root (README, QUICK_START, LICENSE)
✅ All documentation in docs/
✅ All notebooks in notebooks/ with clear numbering
✅ All scripts in scripts/
✅ All requirements in requirements/
✅ Old docs archived, not deleted
✅ Clear naming: 01_, 02_, 03_ for notebooks
```

---

## 🎯 Post-Restructure Verification

**Run these checks:**

```bash
# 1. Server still works
bash scripts/start_server.sh

# 2. Imports work
python -c "from src.config import *; print('✅ Imports OK')"

# 3. Training notebook exists
ls notebooks/01_train_baseline.ipynb

# 4. Requirements exist
ls requirements/*.txt

# 5. Git status clean
git status
```

---

## 📦 Final Root Directory (After Cleanup)

```
Melanoma-detection/
├── README.md              ← Main entry point
├── QUICK_START.md         ← Get started fast
├── LICENSE.md             ← Legal
├── .gitignore             ← Git
├── .env                   ← Secrets
├── .env.example           ← Template
├── data/                  ← Dataset
├── docs/                  ← All documentation
├── experiments/           ← Results
├── models/                ← Trained weights
├── notebooks/             ← Jupyter notebooks (numbered)
├── requirements/          ← Dependencies (organized)
├── scripts/               ← Executable scripts
├── src/                   ← Source code
└── tests/                 ← Unit tests
```

**Only 6 items in root + 9 folders** (down from 20+ items!)

---

## 🚀 Run All Commands (Copy-Paste)

```bash
cd /home/the/Codes/Melanoma-detection

# Create directories
mkdir -p docs/archive notebooks/archive scripts requirements

# Move docs
mv THESIS_ROADMAP.md ML_ROADMAP.md MEDICAL_BACKGROUND.md docs/
mv docs/all_Pip_installed.md docs/PRE_GITHUB_CHECKLIST.md docs/RECENT_UPDATES.md docs/archive/
mv "docs/markdown/latex thesis ready.md" docs/archive/latex_thesis_ready.md
mv docs/markdown/*.md docs/archive/
mv docs/steps/*.md docs/archive/
rmdir docs/markdown docs/steps
mv copilot.md docs/archive/

# Move notebooks
mv learning/day1.ipynb notebooks/01_train_baseline.ipynb
mv notebooks/main.ipynb notebooks/02_exploratory_analysis.ipynb
mv notebooks/melanomaDetection.ipynb notebooks/03_model_evaluation.ipynb
rmdir learning

# Move scripts
mv start_server.sh setup_experiments.sh scripts/

# Move requirements
mv requirements-train.txt requirements/
mv models/requirements-serving.txt requirements/requirements-serve.txt

# Create base requirements
cat > requirements/requirements-base.txt << 'EOF'
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
EOF

echo "✅ Restructure complete! Now update file references in README.md and docs/"
```

---

## 📝 Next Steps After Restructure

1. **Test server**: `bash scripts/start_server.sh`
2. **Update README.md** with new paths (see section above)
3. **Update QUICK_START.md** with new paths
4. **Update docs/THESIS_ROADMAP.md** notebook references
5. **Update docs/ML_ROADMAP.md** notebook references
6. **Commit changes**: 
   ```bash
   git add .
   git commit -m "Restructure project for thesis clarity"
   ```

---

## 🎓 Why This Structure?

**For Thesis:**
- Clear separation: data, code, experiments, docs
- Numbered notebooks show workflow progression
- All documentation in one place
- Easy to zip and submit

**For Development:**
- Scripts isolated in `scripts/`
- Requirements split by purpose (train/serve/base)
- Source code in `src/` (importable)
- Tests in `tests/` (pytest discoverable)

**For GitHub:**
- Clean root directory (not overwhelming)
- Clear README with links to docs
- Archive folder preserves history without clutter

---

**Ready to restructure?** Run the commands above! 🚀
