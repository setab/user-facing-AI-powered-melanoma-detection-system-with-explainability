# ML & Software Engineering Roadmap for Thesis

**Target**: University thesis on melanoma detection with calibration & XAI  
**Background**: Software/ML focus (no medical expertise required)  
**Timeline**: 8-10 weeks

---

## 🎯 Your Thesis Unique Selling Points

1. **Temperature Calibration** - Make probabilities reliable
2. **Multi-Model Comparison** - Find the best architecture
3. **Explainable AI** - Grad-CAM visualizations
4. **Practical Deployment** - Web UI with authentication

**You don't need medical expertise!** Focus on the ML/engineering quality.

---

## 📚 What to Learn (ML/Software Only)

### Week 1: Fundamentals Review

**1. Deep Learning Basics (if rusty)**
- CNNs: Convolutional layers, pooling, fully connected layers
- Transfer learning: Why pretrained models work
- Loss functions: Cross-entropy for classification
- Optimizers: Adam, SGD
- Training: Epochs, batches, validation

**Resources:**
- Stanford CS231n lectures (YouTube) - Lectures 5-9
- PyTorch tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html

**2. Model Architectures (understand at high level)**
- **ResNet-50**: Skip connections solve vanishing gradient
- **EfficientNet**: Balanced scaling (width, depth, resolution)
- **DenseNet**: Every layer connected to every other
- **Vision Transformer**: Attention mechanism on image patches

**Resources:**
- Papers With Code: https://paperswithcode.com/methods
- Just read abstracts + look at diagrams (don't need full papers)

**3. Evaluation Metrics**
- Accuracy, Precision, Recall, F1
- ROC curve, AUC
- Confusion matrix
- Sensitivity = Recall (True Positive Rate)
- Specificity (True Negative Rate)

**Tool:** Scikit-learn metrics documentation

---

### Week 2: Your Thesis Core - Calibration ⭐

**What is Temperature Scaling?**

Simple concept:
```
Before: prob = softmax(logits)
After:  prob = softmax(logits / T)

Where T > 1 "softens" (less confident)
      T < 1 "sharpens" (more confident)
```

**Why it matters:**
- Neural networks often overconfident (say 99% when should say 70%)
- Calibration fixes this → reliable probabilities
- Essential for medical applications

**What to learn:**
1. Read paper: Guo et al. "On Calibration of Modern Neural Networks"
   - https://arxiv.org/abs/1706.04599
   - **Skip medical parts**, focus on Section 3 (Temperature Scaling)
   - Just understand: They fit T on validation set to minimize NLL loss

2. Calibration metrics:
   - **ECE (Expected Calibration Error)**: How far predictions are from true frequencies
   - **Brier Score**: Mean squared error of probabilities
   - **Reliability Diagram**: Visual plot of calibration

3. **Implementation (already done in your code!):**
   ```python
   # Find optimal T
   temperature = nn.Parameter(torch.ones(1) * 1.5)
   optimizer = optim.LBFGS([temperature], lr=0.01)
   
   # Apply during inference
   calibrated_logits = logits / temperature
   probs = torch.softmax(calibrated_logits, dim=1)
   ```

**Your thesis contribution:**
- "We apply temperature scaling to melanoma detection on HAM10000"
- Show before/after reliability diagrams
- Report ECE improvement (e.g., 0.15 → 0.05)

**Time:** 1-2 days to understand, already implemented ✅

---

### Week 3: Reproducible Experiments

**Software Engineering Best Practices:**

**1. Random Seeds**
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

**2. Version Control**
```bash
# Commit your exact environment
pip freeze > requirements-exact.txt
git add requirements-exact.txt
git commit -m "Pin dependency versions for reproducibility"
```

**3. Experiment Logging**
```python
# Save all hyperparameters
config = {
    'model': 'resnet50',
    'lr': 1e-4,
    'batch_size': 32,
    'seed': 42
}
with open('experiments/config.json', 'w') as f:
    json.dump(config, f)
```

**4. Data Split Documentation**
```python
# Save train/val split for reproducibility
with open('experiments/data_split.json', 'w') as f:
    json.dump({
        'train_indices': train_idx.tolist(),
        'val_indices': val_idx.tolist()
    }, f)
```

---

## 🚀 Practical Steps (What to Actually Do)

### Step 1: Train Baseline Model (This Week)

```bash
# Open training notebook
cd /home/the/Codes/Melanoma-detection
jupyter notebook learning/day1.ipynb
```

**In notebook, run all cells:**
1. Data loading → see HAM10000 images
2. Model setup → ResNet50 pretrained
3. Training loop → 30-60 min
4. Calibration → fit temperature
5. Evaluation → metrics + plots

**What you'll learn:**
- How to load image datasets
- How to use pretrained models
- How training loops work
- How to evaluate models
- How calibration is applied

**Output:**
- `melanoma_resnet50_nb.pth` (trained model)
- `temperature.json` (calibration parameter)
- `operating_points.json` (thresholds)

### Step 2: Multi-Model Comparison (Week 4)

```bash
# Train 4 different architectures
python src/training/compare_models.py \
  --metadata data/HAM10000_metadata.csv \
  --img-dir data/ds/img \
  --output-dir experiments/model_comparison \
  --architectures resnet50 efficientnet_b3 densenet121 vit_b_16 \
  --epochs 20 \
  --batch-size 32 \
  --seed 42
```

**This will take 8-16 hours on your GPU** (or 2-4 hours on Azure)

**What you'll learn:**
- Comparing different architectures
- How to evaluate multiple models fairly
- What makes one architecture better than another

**Output:**
- 4 trained models
- Comparison metrics for all models
- JSON with all results

### Step 3: Generate Thesis Results (Week 4)

```bash
# Create all visualizations
python src/training/visualize_comparison.py \
  --results experiments/model_comparison/comparison_results.json \
  --output-dir experiments/model_comparison/visualizations
```

**Output (ready for thesis):**
- `comparison_table.tex` → Copy directly into LaTeX
- `training_curves.png` → Figure 1
- `metrics_comparison.png` → Figure 2
- `calibration_comparison.png` → Figure 3
- `confusion_matrices.png` → Figure 4
- `summary_report.txt` → Use for discussion

### Step 4: Test Deployment (Week 5)

```bash
# Start web server
bash start_server.sh

# Access from Mac: http://SERVER_IP_HIDDEN:7860
# Test with real trained model
```

**What you'll learn:**
- How to deploy ML models
- Web UI development with Gradio
- Authentication and security

---

## 📊 Thesis Structure (Software/ML Focus)

### Chapter 1: Introduction (2-3 pages)
- Problem: Melanoma detection needs reliable AI
- Gap: Current models overconfident
- Solution: Temperature calibration + comparison
- Contributions: List your 4-5 contributions

### Chapter 2: Background & Related Work (5-7 pages)

**Section 2.1: Deep Learning for Image Classification**
- CNNs basics (you understand this)
- Transfer learning
- Common architectures (ResNet, EfficientNet, etc.)

**Section 2.2: Model Calibration ⭐**
- What is calibration
- Temperature scaling method
- Why it matters

**Section 2.3: Explainable AI**
- Need for interpretability
- Grad-CAM method
- How it works (gradient-based attention)

**Section 2.4: Related Work**
- Other melanoma detection papers
- What they did vs what you do
- Your improvements

### Chapter 3: Methodology (10-12 pages)

**Section 3.1: Dataset**
- HAM10000 overview
- 7 classes, 10,000+ images
- Train/val split (80/20)

**Section 3.2: Model Architectures**
- ResNet-50 (baseline)
- EfficientNet-B3 (efficient)
- DenseNet-121 (dense connections)
- ViT-B/16 (transformer)
- Why these four?

**Section 3.3: Training Procedure**
- Transfer learning approach
- Data augmentation (flip, rotate, color jitter)
- Hyperparameters (lr=1e-4, batch=32, etc.)
- Early stopping
- Optimizer (Adam)

**Section 3.4: Temperature Calibration ⭐**
- Theory: minimize NLL on validation set
- Implementation: LBFGS optimizer
- Metrics: ECE, Brier score
- Reliability diagrams

**Section 3.5: Operating Threshold Selection**
- ROC analysis
- Choose threshold at 95% specificity
- Trade-off: sensitivity vs specificity

**Section 3.6: Explainable AI**
- Grad-CAM implementation
- Visual heatmaps
- Clinical interpretability

**Section 3.7: Evaluation Metrics**
- Accuracy, AUC, F1
- Calibration (ECE, Brier)
- Inference time

### Chapter 4: Experiments & Results (8-10 pages)

**Section 4.1: Experimental Setup**
- Hardware (RTX 5060 Ti)
- Software (PyTorch 2.8, Python 3.12)
- Reproducibility (seed=42)

**Section 4.2: Single Model Results**
- ResNet-50 baseline
- Before calibration: ECE, accuracy
- After calibration: ECE, accuracy
- Reliability diagrams

**Section 4.3: Multi-Model Comparison**
- Table 1: All metrics for all models
- Figure: Training curves
- Figure: Metrics comparison
- Figure: Calibration comparison

**Section 4.4: Best Model Analysis**
- Which model won? Why?
- Speed vs accuracy trade-off
- Calibration quality comparison

**Section 4.5: Explainability Analysis**
- Grad-CAM examples
- Do heatmaps make sense?
- Compare across models

### Chapter 5: Discussion (5-7 pages)

**Section 5.1: Interpretation**
- Why did [best model] perform best?
- Impact of calibration
- Importance of architecture choice

**Section 5.2: Practical Implications**
- Deployment considerations
- Inference speed matters
- Calibration essential for trust

**Section 5.3: Limitations**
- Single dataset (HAM10000)
- No external validation
- GPU requirements
- Class imbalance

**Section 5.4: Future Work**
- Test on other datasets
- Ensemble methods
- Better XAI techniques
- User studies

### Chapter 6: Conclusion (1-2 pages)
- Summary of contributions
- Key findings
- Final thoughts

---

## 🎓 How to Get Good Marks

### Technical Excellence (40%)

✅ **Reproducibility** (Critical!)
- Random seeds documented
- Exact package versions
- Data splits saved
- All configs logged

✅ **Fair Comparison**
- Same data splits for all models
- Same hyperparameters
- Same calibration method
- Same evaluation metrics

✅ **Best Practices**
- Version control (Git)
- Clean code structure
- Documentation
- Testing

### Innovation (30%)

✅ **Novel Contribution**: Temperature calibration for melanoma
- Show it's not widely applied in this domain
- Demonstrate improvement
- Explain why it matters

✅ **Comprehensive Study**: 4 architectures compared
- ResNet (classic)
- EfficientNet (modern)
- DenseNet (alternative)
- ViT (transformer)

✅ **Practical System**: Working web deployment
- Authentication
- User-friendly UI
- Real-time inference

### Presentation (20%)

✅ **Clear Writing**
- Explain concepts simply
- Good figures and tables
- Logical flow

✅ **Professional Results**
- LaTeX-formatted tables
- High-quality plots
- Consistent formatting

✅ **Complete Documentation**
- README for reproduction
- Code comments
- Setup instructions

### Understanding (10%)

✅ **Defend Your Choices**
- Why temperature scaling?
- Why these architectures?
- Why these metrics?

✅ **Acknowledge Limitations**
- What could be better?
- What would you do with more time?

---

## 🐛 Fix: Chat Not Appearing

The chat only appears when melanoma probability is **uncertain** (within ±0.15 of threshold).

**Test it:**

1. **Train real model first** (dummy model gives random probabilities)
   ```bash
   jupyter notebook learning/day1.ipynb
   # Run all cells, wait for training
   ```

2. **Restart server with trained model**
   ```bash
   bash start_server.sh
   ```

3. **Upload test image** and check melanoma probability

4. **Chat appears when:** `|melanoma_prob - threshold| <= 0.15`
   - Example: If threshold=0.724, chat shows for probs between 0.574-0.874

**Quick fix to always show chat (for testing):**

Edit `src/serve_gradio.py`, line ~110:

```python
# Change from:
if abs(mel_prob - threshold) <= 0.15 + 1e-9:

# To (always show):
if True:  # Always show chat for testing
```

Then restart server.

---

## ⏱️ Time Estimates

| Task | Time | When |
|------|------|------|
| Understand calibration | 2-3 hours | Week 2 |
| Setup reproducibility | 1-2 hours | Week 3 |
| Train baseline model | 1 hour (30-60 min GPU) | Week 3 |
| Run model comparison | 8-16 hours (background) | Week 4 |
| Generate visualizations | 10 minutes | Week 4 |
| Analyze results | 3-4 hours | Week 4-5 |
| Write methodology | 6-8 hours | Week 5-6 |
| Write results | 4-6 hours | Week 6 |
| Write intro/discussion | 6-8 hours | Week 7 |
| Revisions | 8-10 hours | Week 8 |

**Total active time: ~40-50 hours** (plus GPU training time in background)

---

## 🚀 Start NOW

**Today (30 minutes):**
1. Open `learning/day1.ipynb`
2. Run cells 1-5 (setup, see data)
3. Start training (cell 6) - let it run
4. Come back in 1 hour to see results

**This week:**
- Complete baseline training
- Understand calibration concept
- Read temperature scaling paper (just Section 3)

**Next week:**
- Run model comparison
- Generate thesis figures
- Start writing methodology

**You're already 60% done!** You have working code, just need to:
1. Run experiments
2. Analyze results  
3. Write it up

---

## 📌 Quick Reference

**Train baseline:**
```bash
jupyter notebook learning/day1.ipynb
```

**Run comparison:**
```bash
python src/training/compare_models.py --epochs 20
```

**Generate plots:**
```bash
python src/training/visualize_comparison.py \
  --results experiments/model_comparison/comparison_results.json
```

**Start web UI:**
```bash
bash start_server.sh
# Access: http://SERVER_IP_HIDDEN:7860
```

**Test chat (after training):**
Upload image → If melanoma prob ≈ 0.72 (near threshold), chat appears

---

**Questions?** Ask when you:
- Start training (to understand what's happening)
- Get results (to interpret them)
- Write thesis (for structure help)

**Start training NOW!** Everything else follows from having trained models.
