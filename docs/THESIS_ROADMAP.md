# Thesis Roadmap: Melanoma Detection with XAI

## 🎯 Thesis Objectives

1. **Primary Goal**: Develop an accurate, calibrated melanoma detection system with explainability
2. **Novelty**:
   - Temperature calibration for reliable probability estimates
   - Operating threshold optimization for clinical deployment
   - Interactive Q&A for uncertainty resolution
   - Comprehensive multi-architecture comparison
3. **Target**: University thesis with reproducible experiments

---

## 📚 Phase 1: Literature Review (Weeks 1-2)

### Required Reading

**Medical Domain:**
- [ ] Tschandl et al. "The HAM10000 dataset..." (Nature Scientific Data 2018)
- [ ] Melanoma clinical guidelines (ABCDE criteria)
- [ ] Dermoscopy basics and imaging considerations

**Deep Learning:**
- [ ] He et al. "Deep Residual Learning..." (ResNet, CVPR 2016)
- [ ] Tan & Le "EfficientNet: Rethinking Model Scaling..." (ICML 2019)
- [ ] Huang et al. "Densely Connected Networks" (DenseNet, CVPR 2017)
- [ ] Dosovitskiy et al. "An Image is Worth 16x16 Words..." (ViT, ICLR 2021)

**Calibration (KEY NOVELTY):**
- [ ] **Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)** ⭐
- [ ] Platt "Probabilistic Outputs for SVMs..." (1999)
- [ ] Niculescu-Mizil & Caruana "Predicting Good Probabilities..." (ICML 2005)

**Explainable AI:**
- [ ] **Selvaraju et al. "Grad-CAM: Visual Explanations..." (ICCV 2017)** ⭐
- [ ] Adebayo et al. "Sanity Checks for Saliency Maps" (NeurIPS 2018)
- [ ] Rudin "Stop Explaining Black Box Models..." (Nature MI 2019)

**Clinical Deployment:**
- [ ] Liu et al. "A comparison of deep learning performance..." (Nature Medicine 2019)
- [ ] Esteva et al. "Dermatologist-level classification..." (Nature 2017)
- [ ] Brinker et al. "Deep learning outperformed..." (European J. Cancer 2019)

### Key Concepts to Master

- **Calibration Metrics**: ECE, Brier score, reliability diagrams
- **ROC Analysis**: AUC, sensitivity, specificity, PPV, NPV
- **Operating Points**: Clinical threshold selection
- **Transfer Learning**: Feature extraction vs fine-tuning
- **Medical AI Ethics**: Bias, fairness, transparency

---

## 🔬 Phase 2: Reproducible Experiment Setup (Week 3)

### 2.1 Environment Setup

**Create requirements file with exact versions:**

```bash
# Create pinned requirements for reproducibility
pip freeze > requirements-exact.txt

# Or manually create requirements-reproducible.txt with key versions:
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchcam==0.4.0
scikit-learn==1.3.0
pandas==2.1.0
matplotlib==3.7.2
seaborn==0.12.2
gradio==4.0.0
python-dotenv==1.0.0
jupyter==1.0.0
```

**Document environment:**
- Python version: 3.12.11
- CUDA version: 12.8
- GPU: RTX 5060 Ti 16GB
- OS: Ubuntu 22.04

### 2.2 Random Seed Management

**Add to training notebook:**

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Call at start of notebook
SEED = 42
set_seed(SEED)
```

### 2.3 Data Split Documentation

**Save data splits for reproducibility:**

```python
# In training notebook, after creating train/val split
import json

# Save split indices
split_info = {
    'seed': SEED,
    'train_size': len(train_dataset),
    'val_size': len(val_dataset),
    'train_indices': train_indices.tolist(),  # If using IndexSampler
    'val_indices': val_indices.tolist(),
    'date': '2025-11-17',
    'split_ratio': 0.8
}

with open('experiments/data_split.json', 'w') as f:
    json.dump(split_info, f, indent=2)
```

### 2.4 Experiment Tracking

**Log all hyperparameters:**

```python
config = {
    'model_architecture': 'resnet50',
    'pretrained': True,
    'num_classes': 7,
    'img_size': 224,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'epochs': 50,
    'early_stopping_patience': 5,
    'data_augmentation': {
        'random_flip': True,
        'random_rotation': 20,
        'color_jitter': {'brightness': 0.2, 'contrast': 0.2}
    },
    'seed': SEED,
    'gpu': 'RTX 5060 Ti 16GB',
    'date': '2025-11-17'
}

# Save config
with open('experiments/training_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

---

## ☁️ Phase 3: Azure ML Training (Week 3-4)

### 3.1 Azure Setup

**Why Azure ML?**
- ✅ Faster training with powerful GPUs (V100, A100)
- ✅ Experiment tracking built-in
- ✅ Easy scaling to multiple architectures
- ✅ Cost-effective for thesis (student credits available)

**Prerequisites:**

1. **Azure Account**:
   - Sign up: https://azure.microsoft.com/free/students/
   - Get $100-200 free credits for students
   - Or use pay-as-you-go (GPU compute ~$1-3/hour)

2. **Install Azure ML SDK**:
   ```bash
   pip install azure-ai-ml azure-identity
   ```

3. **Create Azure ML Workspace**:
   - Go to portal.azure.com
   - Create Resource Group: "melanoma-thesis"
   - Create ML Workspace: "melanoma-detection"
   - Note: Region (choose one with GPU quota, e.g., East US, West Europe)

### 3.2 Training Script Adaptation

**Create `src/training/train_azure.py`:**

```python
"""
Training script for Azure ML
Adapted from learning/day1.ipynb for cloud execution
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import mlflow

# Reproducibility
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Enable MLflow logging
    mlflow.start_run()
    mlflow.log_params({
        'architecture': 'resnet50',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed
    })
    
    # Your training code here...
    # (Adapt from day1.ipynb)
    
    # Log metrics
    mlflow.log_metrics({
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_auc': val_auc
    })
    
    # Save model
    torch.save(model.state_dict(), f"{args.output_path}/model.pth")
    mlflow.log_artifact(f"{args.output_path}/model.pth")
    
    mlflow.end_run()

if __name__ == '__main__':
    main()
```

### 3.3 Azure ML Job Submission

**Create `azure_submit_job.py`:**

```python
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="YOUR_SUBSCRIPTION_ID",
    resource_group_name="melanoma-thesis",
    workspace_name="melanoma-detection"
)

# Define job
job = command(
    code="./src/training",
    command="python train_azure.py --data-path ${{inputs.data}} --output-path ${{outputs.model}}",
    inputs={
        "data": Input(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/ham10000")
    },
    outputs={
        "model": Output(type="uri_folder")
    },
    environment="azureml://registries/azureml/environments/pytorch-2.0/labels/latest",
    compute="gpu-cluster",  # Create this in Azure ML
    instance_type="Standard_NC6s_v3",  # V100 GPU
    display_name="melanoma-training-resnet50"
)

# Submit
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
```

### 3.4 Cost Optimization

**GPU Instance Options:**

| Instance | GPU | vCPU | RAM | Cost/hour | Best For |
|----------|-----|------|-----|-----------|----------|
| Standard_NC6s_v3 | V100 (16GB) | 6 | 112GB | ~$3 | Single model training |
| Standard_NC12s_v3 | 2x V100 | 12 | 224GB | ~$6 | Model comparison |
| Standard_NC6 | K80 | 6 | 56GB | ~$1 | Budget option |

**Tips:**
- Use spot instances (up to 80% cheaper)
- Auto-shutdown after job completion
- Use low-priority compute for experimentation

---

## 🏋️ Phase 4: Training Execution (Week 4)

### 4.1 Single Model Training (Local/Azure)

**Steps:**

1. **Prepare data** (already done ✅):
   - HAM10000 images in `data/ds/img/`
   - Metadata CSV ready

2. **Run training**:
   ```bash
   # Local (RTX 5060 Ti) - 30-60 min
   jupyter notebook learning/day1.ipynb
   
   # Or Azure ML - 15-30 min with V100
   python azure_submit_job.py
   ```

3. **Verify outputs**:
   - Model checkpoint: `melanoma_resnet50_nb.pth` (~90MB)
   - Temperature JSON: `temperature.json`
   - Operating points: `operating_points.json`
   - Training curves saved

### 4.2 Model Comparison (Thesis Core)

**Run multi-architecture comparison:**

```bash
# Local (8-16 hours)
python src/training/compare_models.py \
  --metadata data/HAM10000_metadata.csv \
  --img-dir data/ds/img \
  --output-dir experiments/model_comparison \
  --architectures resnet50 efficientnet_b3 densenet121 vit_b_16 \
  --epochs 20 \
  --batch-size 32 \
  --seed 42

# Azure ML (2-4 hours with V100)
# Submit 4 parallel jobs (one per architecture)
```

### 4.3 Generate Thesis Results

```bash
# Create all visualizations
python src/training/visualize_comparison.py \
  --results experiments/model_comparison/comparison_results.json \
  --output-dir experiments/model_comparison/visualizations

# Outputs:
# - comparison_table.tex (for LaTeX)
# - training_curves.png
# - metrics_comparison.png
# - calibration_plots.png
# - confusion_matrices.png
# - summary_report.txt
```

---

## 📊 Phase 5: Evaluation & Analysis (Week 5)

### 5.1 Quantitative Metrics

**Calculate for each model:**

1. **Classification Performance**:
   - Overall accuracy
   - Per-class accuracy
   - Macro/micro F1-score
   - AUC-ROC (multi-class and melanoma-specific)

2. **Calibration Quality** ⭐ (YOUR NOVELTY):
   - ECE (Expected Calibration Error)
   - Brier Score
   - Reliability diagrams (pre/post calibration)
   - Optimal temperature value

3. **Clinical Metrics**:
   - Sensitivity @ 95% specificity
   - Sensitivity @ 90% specificity
   - PPV, NPV at operating points
   - False positive/negative rates

4. **Efficiency**:
   - Inference time (mean ± std)
   - Model size (MB)
   - FLOPs

### 5.2 Qualitative Analysis

**XAI Evaluation:**

1. **Grad-CAM Visualization**:
   - Generate for all test cases
   - Visual inspection for clinical relevance
   - Do heatmaps highlight lesion boundaries?
   - Are features clinically meaningful?

2. **Error Analysis**:
   - Identify failure cases
   - Categorize errors (confusion between which classes?)
   - Clinical interpretation of errors

3. **Uncertainty Quantification**:
   - When does chat Q&A trigger?
   - Does it improve confidence?
   - User study (if time permits)

### 5.3 Statistical Significance

**Compare models statistically:**

```python
from scipy import stats

# McNemar's test for paired predictions
# Bootstrap for confidence intervals
# Wilcoxon signed-rank for calibration metrics
```

---

## 📝 Phase 6: Thesis Writing (Weeks 6-8)

### Thesis Structure

**Chapter 1: Introduction**
- Melanoma clinical significance
- AI in medical diagnosis
- Problem statement
- Research objectives
- Thesis contributions

**Chapter 2: Literature Review**
- CNN architectures for medical imaging
- Calibration methods ⭐
- Explainable AI in healthcare
- Related work in melanoma detection

**Chapter 3: Methodology**
- Dataset (HAM10000)
- Preprocessing & augmentation
- Model architectures
- Temperature scaling ⭐
- Operating threshold optimization ⭐
- Grad-CAM implementation
- Interactive Q&A system ⭐

**Chapter 4: Experiments**
- Experimental setup (reproducible!)
- Training procedure
- Hyperparameters
- Evaluation metrics
- Multi-architecture comparison

**Chapter 5: Results**
- Quantitative results (tables from visualize_comparison.py)
- Calibration improvements (reliability diagrams)
- XAI visualizations (Grad-CAM examples)
- Statistical comparisons
- Model rankings

**Chapter 6: Discussion**
- Interpretation of results
- Clinical implications
- Calibration importance ⭐
- XAI trustworthiness
- Limitations
- Ethical considerations

**Chapter 7: Conclusion**
- Summary of contributions
- Future work
- Final remarks

---

## 🎯 Thesis Contributions (Highlight These!)

### 1. Temperature Calibration for Melanoma Detection ⭐
- **Novel**: Apply temperature scaling specifically to HAM10000
- **Show**: Reliability diagrams before/after
- **Quantify**: ECE improvement (e.g., from 0.15 → 0.05)
- **Argue**: Essential for clinical trust

### 2. Operating Threshold Optimization ⭐
- **Novel**: Clinical deployment-ready thresholds
- **Show**: ROC curve with marked operating points
- **Quantify**: Sensitivity/specificity at 90% and 95%
- **Argue**: Balances false positives vs false negatives

### 3. Interactive Q&A for Uncertainty ⭐
- **Novel**: Chat interface for borderline cases
- **Show**: Example interactions
- **Quantify**: Improvement in confidence
- **Argue**: Human-AI collaboration

### 4. Comprehensive Architecture Comparison
- **Novel**: All four SOTA architectures on HAM10000
- **Show**: Comparison tables and plots
- **Quantify**: Best model for each metric
- **Argue**: Informed deployment decision

### 5. Explainable AI Integration
- **Novel**: Grad-CAM + calibration + Q&A
- **Show**: Visual examples
- **Quantify**: Clinical relevance evaluation
- **Argue**: Trust and transparency

---

## ✅ Reproducibility Checklist

- [ ] All random seeds documented and set
- [ ] Exact package versions in requirements-exact.txt
- [ ] Data splits saved (train/val indices)
- [ ] Hyperparameters logged in JSON
- [ ] Training scripts version-controlled (Git)
- [ ] Model checkpoints saved with metadata
- [ ] Evaluation scripts documented
- [ ] Visualization code included
- [ ] README with step-by-step instructions
- [ ] Hardware specs documented (GPU model, etc.)

---

## 📅 Timeline Summary

| Week | Tasks | Deliverable |
|------|-------|-------------|
| 1-2 | Literature review | Annotated bibliography |
| 3 | Setup reproducibility | Environment + scripts |
| 3-4 | Azure setup (optional) | Cloud training ready |
| 4 | Training execution | Trained models |
| 5 | Evaluation & analysis | Results tables/plots |
| 6-8 | Thesis writing | Complete thesis draft |
| 9 | Revisions | Final thesis |
| 10 | Defense preparation | Presentation slides |

---

## 💰 Azure ML Cost Estimate

**For Thesis Project:**

| Task | Instance | Hours | Cost |
|------|----------|-------|------|
| Single model training | NC6s_v3 (V100) | 0.5 | $1.50 |
| 4-model comparison | NC6s_v3 | 2 | $6.00 |
| Experimentation/tuning | NC6 (K80) | 4 | $4.00 |
| **Total** | | | **~$12** |

**With student credits ($100-200)**, you can run many experiments!

---

## 🎓 Academic Integrity

- Cite all papers and datasets properly
- Acknowledge open-source libraries
- Document any external code used
- Ensure reproducibility for thesis defense
- Follow university ethics guidelines for AI research

---

## 🚀 Next Immediate Steps

1. **Today**: Run `jupyter notebook learning/day1.ipynb` to train ResNet50 baseline
2. **Tomorrow**: Set up reproducibility (seeds, logging, requirements-exact.txt)
3. **This week**: Decide on local vs Azure training
4. **Next week**: Run model comparison experiments
5. **Following weeks**: Analysis and thesis writing

---

**Ready to start?** Begin with Step 1: Train your baseline model! 🎯
