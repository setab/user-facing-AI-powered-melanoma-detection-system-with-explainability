# 📚 Complete Code Walkthrough: Understanding Every Line

**Purpose**: This guide explains every component of the melanoma detection system in detail, so you can understand, recreate, and adapt it for Azure ML Studio.

**Read this during training time** - it takes 2-4 hours to fully understand all components.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Configuration System](#2-configuration-system)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architectures](#4-model-architectures)
5. [Training Loop](#5-training-loop)
6. [Temperature Calibration](#6-temperature-calibration)
7. [Operating Thresholds](#7-operating-thresholds)
8. [Explainable AI (Grad-CAM)](#8-explainable-ai-grad-cam)
9. [Web Interface](#9-web-interface)
10. [Model Comparison Framework](#10-model-comparison-framework)
11. [Azure ML Adaptation](#11-azure-ml-adaptation)

---

## 1. Project Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  (Gradio Web App: Upload → Predict → Explain)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE ENGINE                           │
│  • Load Model     • Preprocess Image                        │
│  • Forward Pass   • Apply Calibration                       │
│  • Generate XAI   • Format Results                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRAINED MODEL                            │
│  • ResNet-50 Backbone (25M parameters)                      │
│  • Custom Classification Head (7 classes)                   │
│  • Temperature Scaling (T=1.3177)                           │
│  • Operating Thresholds (95% specificity)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                          │
│  • Data Loading      • Augmentation                         │
│  • Optimization      • Validation                           │
│  • Calibration       • Threshold Selection                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      DATASET                                │
│  HAM10000: 10,015 dermoscopic images                        │
│  7 classes (melanoma, nevus, BCC, etc.)                     │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure Explained

```
Melanoma-detection/
├── src/                    # All source code
│   ├── config.py          # Configuration management
│   ├── serve_gradio.py    # Web interface
│   ├── training/          # Training modules
│   │   ├── train.py       # Basic training script
│   │   ├── compare_models.py    # Multi-model comparison
│   │   └── visualize_comparison.py  # Plot generation
│   └── inference/         # Inference modules
│       ├── cli.py         # Command-line interface
│       └── xai.py         # Explainability (Grad-CAM)
│
├── data/                  # Dataset (not in git)
│   ├── HAM10000_metadata.csv     # Labels and patient info
│   ├── splits.json               # Train/val/test indices
│   └── ds/img/                   # 10,015 JPG images
│
├── models/                # Trained models
│   ├── checkpoints/       # Model weights
│   └── label_maps/        # Class name mappings
│
├── experiments/           # Experiment results
│   └── model_comparison/  # Comparison study results
│
└── thesis/                # Thesis writing
    ├── sections/          # LaTeX/Markdown sections
    ├── figures/           # Generated plots
    └── references/        # Bibliography
```

---

## 2. Configuration System

### File: `src/config.py`

This file centralizes all configuration parameters.

```python
from pathlib import Path

class Config:
    # === PROJECT PATHS ===
    ROOT = Path(__file__).parent.parent  # Project root
    DATA_DIR = ROOT / "data"
    IMG_DIR = DATA_DIR / "ds" / "img"
    METADATA_CSV = DATA_DIR / "HAM10000_metadata.csv"
    
    # === MODEL PATHS ===
    MODELS_DIR = ROOT / "models"
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
    LABEL_MAP_PATH = MODELS_DIR / "label_maps" / "label_map_nb.json"
    
    # === TRAINING HYPERPARAMETERS ===
    BATCH_SIZE = 32              # How many images per batch
    LEARNING_RATE = 1e-4         # Step size for gradient descent
    NUM_EPOCHS = 20              # Full passes through dataset
    NUM_WORKERS = 4              # Parallel data loading threads
    
    # === MODEL CONFIGURATION ===
    IMG_SIZE = 224               # ResNet input size
    NUM_CLASSES = 7              # Skin lesion categories
    
    # === DATA SPLIT ===
    TRAIN_RATIO = 0.85           # 85% for training
    VAL_RATIO = 0.15             # 15% for validation
    RANDOM_SEED = 42             # For reproducibility
```

**Key Concepts**:

1. **Pathlib**: Modern Python path handling (cross-platform)
2. **Centralization**: Change batch size once, affects everywhere
3. **Reproducibility**: Fixed random seed ensures same splits

**Why These Values?**:
- `BATCH_SIZE=32`: Fits in most GPUs (11GB VRAM), good convergence
- `LEARNING_RATE=1e-4`: Small for fine-tuning pretrained models
- `NUM_EPOCHS=20`: Balances training time vs performance
- `IMG_SIZE=224`: Standard for ImageNet-pretrained models
- `RANDOM_SEED=42`: Convention (from Hitchhiker's Guide to the Galaxy)

---

## 3. Data Pipeline

### Dataset Class Explained

**File**: `src/training/dataset.py` (conceptual - you'll implement this)

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class HAM10000Dataset(Dataset):
    """
    Custom PyTorch Dataset for HAM10000 skin lesion images.
    
    How PyTorch datasets work:
    1. __init__: Load metadata, create mappings
    2. __len__: Return total number of samples
    3. __getitem__: Load and return one sample
    """
    
    def __init__(self, img_dir, metadata_csv, transform=None, indices=None):
        # Load CSV with pandas
        df = pd.read_csv(metadata_csv)
        
        # Create label mapping (string → integer)
        # Example: {'melanoma': 0, 'nevus': 1, ...}
        unique_labels = sorted(df['dx'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Filter by indices if provided (for train/val split)
        if indices is not None:
            df = df.iloc[indices]
        
        self.img_dir = Path(img_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        """Return number of samples"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Load and return one sample.
        
        Args:
            idx: Index of sample to load (0 to len-1)
        
        Returns:
            image: Tensor of shape (3, 224, 224)
            label: Integer class index
            image_id: String identifier
        """
        # Get metadata row
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label_name = row['dx']
        
        # Load image from disk
        img_path = self.img_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations (resize, augment, normalize)
        if self.transform:
            image = self.transform(image)
        
        # Convert label string to integer
        label = self.label_map[label_name]
        
        return image, label, image_id
```

**Key Concepts**:

1. **Dataset vs DataLoader**:
   - `Dataset`: Defines how to load ONE sample
   - `DataLoader`: Batches multiple samples, shuffles, parallelizes

2. **Transforms**: Image preprocessing pipeline
   ```python
   transform = T.Compose([
       T.Resize((224, 224)),        # Resize to model input size
       T.ToTensor(),                # Convert PIL → Tensor (0-1)
       T.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                   [0.229, 0.224, 0.225])  # ImageNet std
   ])
   ```

3. **Why Normalize?**:
   - Pretrained ResNet expects ImageNet statistics
   - Normalization formula: `(x - mean) / std`
   - Centers data around 0, prevents gradient issues

### Data Augmentation

**Purpose**: Artificially increase dataset size, prevent overfitting

```python
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),      # 50% chance flip left-right
    T.RandomVerticalFlip(p=0.5),        # 50% chance flip up-down
    T.RandomRotation(degrees=20),       # Rotate ±20 degrees
    T.ColorJitter(                      # Random color changes
        brightness=0.2,                 # ±20% brightness
        contrast=0.2,                   # ±20% contrast
        saturation=0.2,                 # ±20% saturation
        hue=0.1                         # ±10% hue
    ),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Why These Augmentations?**:
- **Flips**: Skin lesions have no canonical orientation
- **Rotation**: Dermoscope can be at any angle
- **ColorJitter**: Different camera settings, lighting conditions

**Validation Transform** (no augmentation):
```python
val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## 4. Model Architectures

### ResNet-50 Explained

**File**: Used in `src/training/compare_models.py`

```python
import torchvision.models as models
import torch.nn as nn

def build_resnet50(num_classes=7):
    """
    Build ResNet-50 with custom classification head.
    
    ResNet-50 Architecture:
    - 50 layers deep
    - 25 million parameters
    - Pretrained on ImageNet (1.2M images, 1000 classes)
    - Uses residual connections (skip connections)
    """
    # Load pretrained model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Get size of last layer (2048 features)
    in_features = model.fc.in_features
    
    # Replace final layer for 7 classes
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
```

**What Happens During Forward Pass**:

```python
# Input: Image tensor (batch_size, 3, 224, 224)
x = image

# Stage 1: Initial convolution + pooling
x = model.conv1(x)      # (batch, 64, 112, 112)
x = model.bn1(x)        # Batch normalization
x = model.relu(x)       # Activation
x = model.maxpool(x)    # (batch, 64, 56, 56)

# Stage 2-5: Residual blocks
x = model.layer1(x)     # (batch, 256, 56, 56)
x = model.layer2(x)     # (batch, 512, 28, 28)
x = model.layer3(x)     # (batch, 1024, 14, 14)
x = model.layer4(x)     # (batch, 2048, 7, 7)  ← We extract from here for Grad-CAM

# Final: Global pooling + classification
x = model.avgpool(x)    # (batch, 2048, 1, 1)
x = torch.flatten(x, 1) # (batch, 2048)
x = model.fc(x)         # (batch, 7)  ← Logits

# Output: (batch_size, 7) logits
```

**Residual Connections**:

```python
# Traditional: x → Conv → ReLU → Conv → Output
# Problem: Gradients vanish in deep networks

# ResNet: x → Conv → ReLU → Conv → Add(x) → Output
#         └─────────────────────────────────┘
#                 Skip connection

def residual_block(x):
    identity = x                    # Save input
    out = conv1(x)
    out = relu(out)
    out = conv2(out)
    out = out + identity            # Add skip connection
    out = relu(out)
    return out
```

**Why ResNet Works**:
- Skip connections allow gradients to flow backward easily
- Network can learn identity function (do nothing) if needed
- Enables training very deep networks (50+ layers)

### Other Architectures

**EfficientNet-B3**:
```python
def build_efficientnet_b3(num_classes=7):
    """
    Compound scaling: balance depth, width, resolution.
    More efficient than ResNet (fewer parameters, better accuracy).
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
```

**DenseNet-121**:
```python
def build_densenet121(num_classes=7):
    """
    Dense connections: each layer connects to ALL previous layers.
    Encourages feature reuse, reduces parameters.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
```

**Vision Transformer (ViT)**:
```python
def build_vit_b_16(num_classes=7):
    """
    Transformer architecture (attention-based).
    Divides image into patches, processes with self-attention.
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
```

---

## 5. Training Loop

### File: `src/training/compare_models.py`

**Complete Training Function Explained**:

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch (one pass through training data).
    
    Args:
        model: Neural network to train
        dataloader: Batches of (images, labels)
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimization algorithm (Adam)
        device: 'cuda' or 'cpu'
    
    Returns:
        avg_loss: Average loss over epoch
        accuracy: Classification accuracy
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Iterate through batches
    for batch_idx, (images, labels, _) in enumerate(dataloader):
        # Move data to GPU
        images = images.to(device)  # (batch_size, 3, 224, 224)
        labels = labels.to(device)  # (batch_size,)
        
        # === FORWARD PASS ===
        optimizer.zero_grad()       # Reset gradients from previous batch
        outputs = model(images)     # Get predictions (batch_size, 7)
        loss = criterion(outputs, labels)  # Compute loss
        
        # === BACKWARD PASS ===
        loss.backward()             # Compute gradients (backpropagation)
        optimizer.step()            # Update weights
        
        # === TRACK METRICS ===
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)  # Get predicted class
        correct_predictions += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
    
    # Calculate epoch metrics
    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy
```

**Step-by-Step Breakdown**:

1. **Forward Pass**:
   ```python
   outputs = model(images)  # Shape: (32, 7) for batch_size=32
   # outputs[i, j] = score for image i, class j
   ```

2. **Loss Calculation**:
   ```python
   loss = criterion(outputs, labels)
   # Cross-Entropy Loss:
   # loss = -log(softmax(outputs)[labels])
   # Penalizes wrong predictions
   ```

3. **Backward Pass**:
   ```python
   loss.backward()
   # Computes gradients: ∂loss/∂weights for all parameters
   # Uses chain rule (calculus) automatically
   ```

4. **Weight Update**:
   ```python
   optimizer.step()
   # Adam optimizer update rule:
   # weights = weights - learning_rate * gradients
   # (with momentum and adaptive learning rates)
   ```

### Validation Function

```python
def validate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set (no training).
    """
    model.eval()  # Set to evaluation mode (disables dropout, fixes batch norm)
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient computation (saves memory)
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels
```

### Main Training Loop

```python
def train_model(model, train_loader, val_loader, num_epochs=20):
    """
    Full training procedure.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc
            }, 'best_model.pth')
            print(f"✓ Saved best model (val_acc={val_acc:.4f})")
    
    return history
```

---

## 6. Temperature Calibration

### Why Calibration Matters

**Problem**: Neural networks are often overconfident.

```
Model says: "90% sure it's melanoma"
Reality: Only correct 70% of the time when it says 90%
```

**Solution**: Temperature scaling adjusts probabilities to match true confidence.

### Temperature Scaling Math

**Uncalibrated Probabilities**:
```python
logits = model(image)        # Raw scores
probs = softmax(logits)      # Convert to probabilities
# Problem: probs are too peaked (overconfident)
```

**Calibrated Probabilities**:
```python
logits = model(image)
calibrated_logits = logits / T    # Divide by temperature T
calibrated_probs = softmax(calibrated_logits)
# T > 1: Smooths probabilities (less confident)
# T < 1: Sharpens probabilities (more confident)
```

**Example** (T=1.3):
```
Uncalibrated: [0.05, 0.10, 0.75, 0.05, 0.03, 0.01, 0.01]
Calibrated:   [0.08, 0.12, 0.62, 0.08, 0.05, 0.03, 0.02]
              ↑ Less extreme, more realistic
```

### Finding Optimal Temperature

```python
def find_temperature(model, val_loader, device):
    """
    Find temperature T that minimizes Expected Calibration Error (ECE).
    """
    # Get all validation predictions
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Objective function: ECE
    def calibration_error(T):
        scaled_logits = all_logits / T
        probs = softmax(scaled_logits, axis=1)
        return expected_calibration_error(probs, all_labels)
    
    # Optimize T
    from scipy.optimize import minimize
    result = minimize(calibration_error, x0=1.0, bounds=[(0.1, 10.0)])
    optimal_T = result.x[0]
    
    return optimal_T
```

### Expected Calibration Error (ECE)

```python
def expected_calibration_error(probs, labels, n_bins=15):
    """
    Measure calibration quality.
    
    Idea: Group predictions by confidence, check if accuracy matches.
    """
    confidences = np.max(probs, axis=1)      # Max probability
    predictions = np.argmax(probs, axis=1)   # Predicted class
    accuracies = (predictions == labels)     # Correct or not
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    
    for i in range(n_bins):
        # Find predictions in this confidence bin
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        
        if in_bin.sum() > 0:
            # Average confidence in bin
            avg_confidence = confidences[in_bin].mean()
            # Actual accuracy in bin
            avg_accuracy = accuracies[in_bin].mean()
            # ECE accumulates gaps
            ece += np.abs(avg_confidence - avg_accuracy) * in_bin.mean()
    
    return ece
```

**Interpretation**:
- ECE = 0.05: On average, confidence is off by 5%
- Lower ECE = better calibration
- Perfect calibration: ECE = 0

---

## 7. Operating Thresholds

### Binary Melanoma Detection

**Problem**: 7-class classification, but clinically we care about "melanoma vs not".

**Solution**: Set a threshold on melanoma probability.

```python
def calculate_operating_thresholds(model, val_loader, device):
    """
    Find threshold that achieves desired specificity.
    
    Specificity = True Negative Rate
                = TN / (TN + FP)
                = Correct negatives / All negatives
    
    Goal: 95% specificity (low false alarm rate)
    """
    # Get melanoma probabilities
    model.eval()
    all_probs = []
    all_is_melanoma = []  # Binary: 1 if melanoma, 0 otherwise
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            melanoma_prob = probs[:, melanoma_class_idx]
            all_probs.extend(melanoma_prob.cpu().numpy())
            all_is_melanoma.extend((labels == melanoma_class_idx).numpy())
    
    all_probs = np.array(all_probs)
    all_is_melanoma = np.array(all_is_melanoma)
    
    # Sweep thresholds, compute metrics
    thresholds = np.linspace(0, 1, 1000)
    specificities = []
    sensitivities = []
    
    for thresh in thresholds:
        predictions = (all_probs >= thresh).astype(int)
        
        # True positives: melanoma correctly classified
        TP = ((predictions == 1) & (all_is_melanoma == 1)).sum()
        # True negatives: non-melanoma correctly classified
        TN = ((predictions == 0) & (all_is_melanoma == 0)).sum()
        # False positives: non-melanoma misclassified as melanoma
        FP = ((predictions == 1) & (all_is_melanoma == 0)).sum()
        # False negatives: melanoma misclassified as non-melanoma
        FN = ((predictions == 0) & (all_is_melanoma == 1)).sum()
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        specificities.append(specificity)
        sensitivities.append(sensitivity)
    
    # Find threshold for 95% specificity
    idx_95_spec = np.argmin(np.abs(np.array(specificities) - 0.95))
    threshold_95 = thresholds[idx_95_spec]
    sensitivity_at_95 = sensitivities[idx_95_spec]
    
    return {
        'threshold_spec95': threshold_95,
        'sensitivity_at_spec95': sensitivity_at_95
    }
```

**Clinical Interpretation**:

```
Threshold = 0.25 (for 95% specificity)

If melanoma_prob >= 0.25:
    → Predict "melanoma" → Refer to dermatologist
    → 95% of non-melanoma cases correctly ruled out
    → 62% of melanoma cases correctly detected

If melanoma_prob < 0.25:
    → Predict "non-melanoma" → Routine monitoring
```

**Trade-off**:
- Higher specificity (fewer false alarms) → Lower sensitivity (miss some melanomas)
- Lower specificity (more false alarms) → Higher sensitivity (catch more melanomas)

---

## 8. Explainable AI (Grad-CAM)

### What is Grad-CAM?

**Goal**: Visualize which image regions influenced the prediction.

**Intuition**: 
- CNN final layers contain high-level features
- Gradients show how prediction changes with features
- Combine activation maps + gradients = importance map

### Grad-CAM Algorithm

```python
from torchcam.methods import GradCAM

def generate_gradcam(model, image, target_class):
    """
    Generate Grad-CAM heatmap for a prediction.
    
    Steps:
    1. Forward pass: get activations from target layer
    2. Backward pass: compute gradients w.r.t. target class
    3. Weight activations by gradients
    4. ReLU + normalize to [0, 1]
    """
    # Initialize Grad-CAM extractor
    cam_extractor = GradCAM(model, target_layer='layer4')
    # layer4 = last convolutional layer of ResNet-50
    
    # Forward pass with gradient tracking
    model.eval()
    logits = model(image)
    
    # Extract CAM for target class
    activation_maps = cam_extractor(class_idx=target_class, scores=logits)
    cam = activation_maps[0].squeeze().cpu().numpy()
    # cam shape: (7, 7) - spatial map
    
    # Clean up hooks
    cam_extractor.remove_hooks()
    
    return cam  # Range: [0, 1], shape: (7, 7)
```

**Mathematical Details**:

```
Let A^k = activation map from layer k (7×7×2048 for ResNet layer4)
Let y^c = score for class c (before softmax)

1. Compute gradients:
   α^k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_{ij})
   
   α^k = importance weight for each channel k

2. Weighted combination:
   L^c = ReLU(Σ_k α^k * A^k)
   
   ReLU: Only positive influences (features supporting class c)

3. Normalize:
   CAM = (L^c - min(L^c)) / (max(L^c) - min(L^c))
```

### Overlay on Image

```python
def overlay_gradcam(original_image, cam, alpha=0.5):
    """
    Overlay heatmap on original image.
    
    Args:
        original_image: PIL Image (224, 224, 3)
        cam: Numpy array (7, 7) - Grad-CAM heatmap
        alpha: Blend factor (0.5 = 50% image, 50% heatmap)
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Resize CAM to image size
    cam_resized = Image.fromarray(cam).resize(original_image.size, Image.BILINEAR)
    cam_array = np.array(cam_resized)
    
    # Normalize to [0, 1]
    cam_normalized = (cam_array - cam_array.min()) / (cam_array.max() - cam_array.min() + 1e-8)
    
    # Apply colormap (jet: blue → green → yellow → red)
    colormap = plt.get_cmap('jet')
    heatmap = colormap(cam_normalized)[:, :, :3]  # RGB only, discard alpha
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Blend with original image
    img_array = np.array(original_image)
    overlay = (alpha * img_array + (1 - alpha) * heatmap).astype(np.uint8)
    
    return Image.fromarray(overlay)
```

**Visualization Interpretation**:

```
Red regions:    High activation → Strongly influences "melanoma" prediction
Yellow regions: Medium activation → Moderate influence
Blue regions:   Low activation → Minimal influence
```

---

## 9. Web Interface

### File: `src/serve_gradio.py`

**Gradio Basics**:

```python
import gradio as gr

# Simple example
def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Your Name"),
    outputs=gr.Textbox(label="Greeting")
)

demo.launch()
```

### Our Melanoma Detection Interface

```python
def predict_and_explain(image):
    """
    Main prediction function.
    
    Input: PIL Image
    Output: (gradcam_overlay, predicted_class, probabilities, decision, explanation)
    """
    if image is None:
        return None, "", {}, "", ""
    
    # 1. Preprocess image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 2. Get prediction
    with torch.no_grad():
        logits = model(img_tensor)
        
        # Apply temperature calibration
        if temperature is not None:
            logits = logits / temperature
        
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
    
    # 3. Generate Grad-CAM
    cam = generate_gradcam(model, img_tensor, pred_idx)
    gradcam_overlay = overlay_gradcam(image.resize((224, 224)), cam)
    
    # 4. Format probabilities
    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    # 5. Melanoma decision
    if operating_points is not None:
        mel_prob = float(probs[melanoma_idx])
        threshold = float(operating_points['thresholds']['melanoma_spec95'])
        verdict = "melanoma" if mel_prob >= threshold else "non-melanoma"
        decision = f"p={mel_prob:.3f} | thr={threshold:.3f} → {verdict}"
    else:
        decision = "N/A"
    
    # 6. Generate AI explanation
    explanation = generate_ai_explanation(pred_label, prob_dict, mel_prob, threshold, verdict)
    
    return gradcam_overlay, pred_label, prob_dict, decision, explanation


# Build interface
with gr.Blocks(title="Melanoma Detection") as demo:
    gr.Markdown("# 🔬 Melanoma Detection with XAI")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Lesion Image")
            predict_btn = gr.Button("🔍 Analyze", variant="primary")
        
        with gr.Column():
            gradcam_output = gr.Image(label="Grad-CAM Explanation")
            label_output = gr.Label(label="Predicted Class")
    
    with gr.Row():
        probs_output = gr.JSON(label="Probabilities")
        decision_output = gr.Textbox(label="Melanoma Decision")
    
    with gr.Row():
        explanation_output = gr.Markdown(label="AI Explanation")
    
    # Wire up button click
    predict_btn.click(
        predict_and_explain,
        inputs=[image_input],
        outputs=[gradcam_output, label_output, probs_output, decision_output, explanation_output]
    )

# Launch server
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

**Key Concepts**:

1. **gr.Blocks**: Flexible layout system (vs simple gr.Interface)
2. **gr.Row/Column**: Organize components in grid
3. **Component types**:
   - `gr.Image`: Image upload/display
   - `gr.Label`: Classification results with confidence bars
   - `gr.JSON`: Display dictionary data
   - `gr.Markdown`: Rich text output
4. **Event handling**: `.click()` connects button to function

---

## 10. Model Comparison Framework

### File: `src/training/compare_models.py`

**Purpose**: Train and compare 4 architectures systematically.

```python
def compare_models(architectures, num_epochs=20, batch_size=32):
    """
    Train multiple models and compare performance.
    
    Args:
        architectures: List of model names
        num_epochs: Training epochs per model
        batch_size: Batch size
    
    Returns:
        results: Dictionary with metrics for each model
    """
    results = {}
    
    for arch_name in architectures:
        print(f"\n{'='*60}")
        print(f"Training {arch_name}")
        print(f"{'='*60}\n")
        
        # 1. Build model
        if arch_name == 'resnet50':
            model = build_resnet50(num_classes=7)
        elif arch_name == 'efficientnet_b3':
            model = build_efficientnet_b3(num_classes=7)
        elif arch_name == 'densenet121':
            model = build_densenet121(num_classes=7)
        elif arch_name == 'vit_b_16':
            model = build_vit_b_16(num_classes=7)
        
        # 2. Train model
        history = train_model(model, train_loader, val_loader, num_epochs)
        
        # 3. Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, device)
        
        # 4. Measure inference time
        inference_time = measure_inference_time(model, device)
        
        # 5. Calculate calibration metrics
        cal_metrics = calculate_calibration_metrics(model, val_loader, device)
        
        # 6. Save results
        results[arch_name] = {
            'history': history,
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc_macro'],
            'melanoma_auc': test_metrics['auc_melanoma'],
            'sensitivity_at_95spec': test_metrics['sensitivity_95spec'],
            'inference_time_ms': inference_time,
            'ece_before_calibration': cal_metrics['ece_before'],
            'ece_after_calibration': cal_metrics['ece_after'],
            'temperature': cal_metrics['optimal_T']
        }
        
        # 7. Save checkpoint
        torch.save({
            'architecture': arch_name,
            'state_dict': model.state_dict(),
            'results': results[arch_name]
        }, f'checkpoints/{arch_name}_checkpoint.pth')
    
    # 8. Save comparison results
    with open('experiments/model_comparison/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

### Generating Visualizations

```python
def visualize_comparison(results):
    """
    Create publication-quality comparison plots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    models = list(results.keys())
    
    # 1. Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for model in models:
        epochs = range(1, len(results[model]['history']['train_loss']) + 1)
        ax1.plot(epochs, results[model]['history']['train_loss'], label=model)
        ax2.plot(epochs, results[model]['history']['val_acc'], label=model)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/training_curves.png', dpi=300)
    
    # 2. Metrics comparison
    metrics = ['test_accuracy', 'melanoma_auc', 'sensitivity_at_95spec']
    metric_names = ['Accuracy', 'Melanoma AUC', 'Sensitivity @ 95% Spec']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[m][metric] for m in models]
        axes[idx].bar(models, values, color='steelblue')
        axes[idx].set_ylabel(name)
        axes[idx].set_ylim([0, 1])
        axes[idx].set_title(name)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/metrics_comparison.png', dpi=300)
    
    # 3. Calibration comparison
    # ... (similar plots for ECE, Brier score)
    
    # 4. Generate LaTeX table
    generate_latex_table(results)
```

---

## 11. Azure ML Adaptation

### How to Port This Code to Azure ML Studio

**Azure ML Structure**:

```
Azure ML Workspace
├── Compute (GPU instances)
├── Datasets (HAM10000 uploaded)
├── Experiments (training runs)
└── Models (trained checkpoints)
```

### Step 1: Create Compute Instance

```python
# In Azure ML Studio:
# 1. Go to "Compute" → "Compute Instances"
# 2. Create new: Standard_NC6 (1x NVIDIA K80, 6 cores, 56GB RAM)
# 3. Wait for provisioning (~5 minutes)
```

### Step 2: Upload Dataset

```python
from azureml.core import Workspace, Dataset

# Connect to workspace
ws = Workspace.from_config()

# Upload HAM10000
datastore = ws.get_default_datastore()
datastore.upload(
    src_dir='./data/ds/img',
    target_path='datasets/HAM10000',
    overwrite=False,
    show_progress=True
)

# Register dataset
dataset = Dataset.File.from_files(path=(datastore, 'datasets/HAM10000'))
dataset = dataset.register(
    workspace=ws,
    name='HAM10000',
    description='10,015 dermoscopic images'
)
```

### Step 3: Create Training Script

**Azure ML requires specific structure**:

```python
# train_azure.py

import argparse
from azureml.core import Run

# Get Azure ML run context
run = Run.get_context()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, help='Path to dataset')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

# ... (Your training code here) ...

# Log metrics to Azure ML
for epoch in range(args.epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)
    
    # Log to Azure
    run.log('train_loss', train_loss)
    run.log('train_acc', train_acc)
    run.log('val_loss', val_loss)
    run.log('val_acc', val_acc)

# Upload model
run.upload_file('outputs/model.pth', 'model.pth')
```

### Step 4: Submit Experiment

```python
from azureml.core import Experiment, ScriptRunConfig, Environment

# Define environment
env = Environment.from_conda_specification(
    name='pytorch-env',
    file_path='environment.yml'
)

# Configure run
config = ScriptRunConfig(
    source_directory='./src',
    script='train_azure.py',
    arguments=[
        '--data-folder', dataset.as_named_input('ham10000').as_mount(),
        '--epochs', 20,
        '--batch-size', 32
    ],
    compute_target='gpu-cluster',
    environment=env
)

# Submit experiment
experiment = Experiment(workspace=ws, name='melanoma-detection')
run = experiment.submit(config)

# Monitor progress
run.wait_for_completion(show_output=True)
```

### Key Differences: Local vs Azure

| Aspect | Local | Azure ML |
|--------|-------|----------|
| Data | Direct file access | Mounted datasets |
| Logging | Print statements | `run.log()` |
| Output | Local files | `outputs/` folder (auto-uploaded) |
| Monitoring | Terminal | Azure ML Studio UI |
| Cost | $0 (your GPU) | ~$0.90/hour (NC6) |

---

## Summary: What to Read When

### During Training (8-16 hours):

**Phase 1 (Hours 0-4): Core Concepts**
1. Read Sections 1-3 (Architecture, Config, Data Pipeline)
2. Understand Dataset class and transforms
3. Review augmentation strategies

**Phase 2 (Hours 4-8): Training Deep Dive**
1. Read Sections 4-5 (Models, Training Loop)
2. Understand forward/backward pass
3. Study loss functions and optimizers

**Phase 3 (Hours 8-12): Advanced Topics**
1. Read Sections 6-7 (Calibration, Thresholds)
2. Understand ECE and temperature scaling
3. Study sensitivity/specificity trade-offs

**Phase 4 (Hours 12-16): Deployment**
1. Read Sections 8-9 (Grad-CAM, Web Interface)
2. Understand XAI visualization
3. Review Gradio implementation

**Phase 5 (Hours 16+): Azure Prep**
1. Read Section 11 (Azure ML Adaptation)
2. Compare local vs Azure workflows
3. Plan your Azure implementation

### Quick Reference

**For Understanding Core Training**:
- Section 5: Training Loop (30 min read)

**For Calibration Thesis Section**:
- Section 6: Temperature Calibration (45 min read)

**For XAI Thesis Section**:
- Section 8: Grad-CAM (30 min read)

**For Azure ML Recreation**:
- Section 11 + `docs/THESIS_ROADMAP.md` Section 5 (1 hour)

---

**Total Reading Time**: 3-4 hours for complete understanding  
**Recommended**: Read in chunks during training breaks

**Next**: Start training, open this guide in split-screen view! 🚀
