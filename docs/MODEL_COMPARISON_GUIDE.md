# Model Comparison Experiments Guide

This guide walks you through running comprehensive model comparison experiments for your melanoma detection thesis.

## Overview

The comparison framework evaluates multiple deep learning architectures on the HAM10000 dataset:
- **ResNet-50** (current baseline)
- **EfficientNet-B3** (efficient modern architecture)
- **DenseNet-121** (dense connections)
- **Vision Transformer (ViT-B/16)** (transformer-based)

All models are evaluated on:
- **Classification Performance**: Accuracy, AUC-ROC
- **Calibration Quality**: ECE, Brier Score, Temperature Scaling
- **Melanoma Detection**: Sensitivity/Specificity at operating thresholds
- **Computational Efficiency**: Inference time, model size
- **Explainability**: Grad-CAM visualization quality

## Prerequisites

```bash
# Install additional requirements for model comparison
pip install scikit-learn matplotlib seaborn pandas
```

## Quick Start

### 1. Run Model Comparison

Train and evaluate all architectures:

```bash
python src/training/compare_models.py \
    --metadata data/HAM10000_metadata.csv \
    --img-dir data/ds/img \
    --label-map models/label_maps/label_map_nb.json \
    --output-dir experiments/model_comparison \
    --architectures resnet50 efficientnet_b3 densenet121 vit_b_16 \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4
```

**Note**: This will take several hours depending on your GPU. Each model trains for up to 20 epochs with early stopping.

### 2. Generate Visualizations

After training completes, generate all plots and tables for your thesis:

```bash
python src/training/visualize_comparison.py \
    --results experiments/model_comparison/comparison_results.json \
    --label-map models/label_maps/label_map_nb.json \
    --output-dir experiments/model_comparison/visualizations
```

This creates:
- `comparison_table.csv` / `comparison_table.tex` - LaTeX-ready table
- `training_curves.png` - Training/validation loss and accuracy
- `metrics_comparison.png` - Bar charts of key metrics
- `calibration_comparison.png` - ECE and Brier score comparison
- `inference_time_comparison.png` - Speed comparison
- `confusion_matrices.png` - All confusion matrices side-by-side
- `summary_report.txt` - Comprehensive text report with rankings

## Output Structure

```
experiments/model_comparison/
├── comparison_results.json          # All metrics in JSON format
├── train_split.csv                  # Training set split (reproducible)
├── val_split.csv                    # Validation set split
├── resnet50_checkpoint.pth          # Model checkpoints
├── efficientnet_b3_checkpoint.pth
├── densenet121_checkpoint.pth
├── vit_b_16_checkpoint.pth
└── visualizations/
    ├── comparison_table.csv
    ├── comparison_table.tex
    ├── training_curves.png
    ├── metrics_comparison.png
    ├── calibration_comparison.png
    ├── inference_time_comparison.png
    ├── confusion_matrices.png
    └── summary_report.txt
```

## Customization

### Train Specific Architectures Only

```bash
python src/training/compare_models.py \
    --architectures resnet50 efficientnet_b3 \
    --epochs 30 \
    --batch-size 64
```

### Adjust Hyperparameters

```bash
python src/training/compare_models.py \
    --lr 5e-5 \               # Lower learning rate
    --epochs 50 \             # More epochs
    --batch-size 16 \         # Smaller batch (if GPU memory limited)
    --img-size 384 \          # Higher resolution (better for ViT)
    --val-split 0.15          # Different validation split
```

### Resume from Checkpoint

If training was interrupted, you can load a checkpoint and continue evaluation:

```python
import torch
checkpoint = torch.load('experiments/model_comparison/resnet50_checkpoint.pth')
model.load_state_dict(checkpoint['model_state'])
temperature = checkpoint['temperature']
```

## Key Metrics Explained

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **AUC (Macro)**: Average AUC across all classes
- **Melanoma AUC**: Binary melanoma vs. non-melanoma AUC

### Calibration Metrics
- **ECE (Expected Calibration Error)**: Measures reliability of predicted probabilities (lower = better calibrated)
- **Brier Score**: Mean squared error between predicted probabilities and true labels (lower = better)
- **Temperature**: Optimal temperature for probability calibration (closer to 1.0 = less correction needed)

### Clinical Metrics (Melanoma Detection)
- **Threshold @ 95% Specificity**: Operating point for clinical deployment
- **Sensitivity @ 95% Spec**: True positive rate at the operating threshold
- **PPV**: Positive predictive value (precision)
- **NPV**: Negative predictive value

### Efficiency Metrics
- **Inference Time**: Average time per image (important for deployment)

## Thesis Integration

### For Your Methods Section

```latex
\section{Model Comparison}
We evaluated four state-of-the-art deep learning architectures on the HAM10000 dataset:
ResNet-50 \cite{he2016deep}, EfficientNet-B3 \cite{tan2019efficientnet}, 
DenseNet-121 \cite{huang2017densely}, and Vision Transformer (ViT-B/16) \cite{dosovitskiy2020image}.

All models were trained using:
- Input resolution: 224×224 pixels
- Batch size: 32
- Optimizer: Adam with learning rate 1e-4
- Data augmentation: Random flips, rotation, color jitter
- Early stopping: Patience of 5 epochs

Post-training, we applied temperature scaling \cite{guo2017calibration} to improve 
probability calibration, and computed operating thresholds at 95\% specificity 
for melanoma detection.
```

### For Your Results Section

Use the generated `comparison_table.tex` directly:

```latex
\section{Results}
\subsection{Model Performance Comparison}

Table \ref{tab:model_comparison} presents the comprehensive evaluation results 
across all architectures.

\begin{table}[h]
\centering
\caption{Comparison of deep learning architectures for melanoma detection}
\label{tab:model_comparison}
\input{visualizations/comparison_table.tex}
\end{table}

As shown in Figure \ref{fig:metrics_comparison}, [model name] achieved the 
highest accuracy of X.XXX, while [model name] demonstrated superior calibration 
with an ECE of X.XXX.
```

### For Your Discussion Section

The `summary_report.txt` provides:
- Rankings by different criteria (accuracy, AUC, calibration, speed)
- Detailed per-model analysis
- Recommendations based on priorities

Use this to discuss trade-offs:
- Accuracy vs. calibration quality
- Performance vs. inference speed
- Model complexity vs. explainability

## Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size
python src/training/compare_models.py --batch-size 16

# Or use gradient accumulation (modify script)
# Or train one model at a time
python src/training/compare_models.py --architectures resnet50
python src/training/compare_models.py --architectures efficientnet_b3
# etc.
```

### Training Takes Too Long

```bash
# Reduce epochs (early stopping will help)
python src/training/compare_models.py --epochs 10

# Use smaller models only
python src/training/compare_models.py --architectures resnet50 efficientnet_b3
```

### Visualization Errors

If matplotlib/seaborn have issues:

```bash
pip install --upgrade matplotlib seaborn
```

For missing LaTeX fonts in plots:

```python
# Edit visualize_comparison.py
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
```

## Next Steps

After completing model comparison:

1. **Select Best Model**: Based on your priorities (accuracy, calibration, speed)
2. **Update Gradio Interface**: Use best model checkpoint in `serve_gradio.py`
3. **Add to Thesis**: Include generated tables and figures
4. **Grad-CAM Analysis**: Compare XAI quality across models (visual inspection)
5. **Error Analysis**: Deep dive into failure cases for best model

## Citation

If you use this comparison framework in your thesis:

```bibtex
@misc{melanoma_comparison2025,
  title={Comprehensive Deep Learning Model Comparison for Melanoma Detection},
  author={Your Name},
  year={2025},
  note={University Thesis}
}
```

## Questions?

Check the `summary_report.txt` generated after running experiments for detailed analysis and recommendations.
