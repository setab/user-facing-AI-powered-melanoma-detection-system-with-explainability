# Complete Reproduction Guide: AI-Powered Melanoma Detection System

**A Step-by-Step Guide to Recreate This Thesis Project From Scratch**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites & Required Knowledge](#prerequisites--required-knowledge)
3. [System Requirements](#system-requirements)
4. [Step-by-Step Reproduction](#step-by-step-reproduction)
5. [Learning Path to Master This Project](#learning-path-to-master-this-project)
6. [Understanding Each Component](#understanding-each-component)
7. [Troubleshooting Common Issues](#troubleshooting-common-issues)
8. [Extending This Project](#extending-this-project)
9. [Resources for Similar Projects](#resources-for-similar-projects)

---

## Project Overview

### What This Project Does

This is a **medical AI system** that:
- Classifies skin lesion images into 7 categories (melanoma, nevus, basal cell carcinoma, etc.)
- Uses **ResNet-50** deep learning architecture trained on 10,000+ medical images
- Provides **explainable AI** with Grad-CAM visualizations showing where the model looked
- Offers **calibrated probabilities** for reliable uncertainty estimates
- Features a **web interface** (Gradio) for easy interaction
- Includes **AI-generated explanations** of decisions in medical terms
- Has **clinical Q&A chat** for educational context

### Project Statistics

- **Dataset**: HAM10000 (10,015 dermatoscopic images)
- **Model**: ResNet-50 (pretrained on ImageNet, fine-tuned)
- **Performance**: 83% accuracy, 91% melanoma AUC, 95% specificity
- **Code**: ~2,000 lines across training, inference, and serving
- **Technologies**: PyTorch, Gradio, NumPy, Pandas, Matplotlib

---

## Prerequisites & Required Knowledge

### Essential Skills (Must Have)

#### 1. Python Programming (Intermediate Level)
**What you need:**
- Functions, classes, modules
- File I/O and data structures (lists, dicts)
- Error handling (try/except)
- Working with libraries (imports)

**Study resources:**
- Book: "Python Crash Course" by Eric Matthes
- Course: CS50's Introduction to Python (Harvard, free)
- Practice: LeetCode Python problems (Easy → Medium)

#### 2. Deep Learning Fundamentals
**What you need:**
- Neural networks basics (layers, activation functions)
- Convolutional Neural Networks (CNNs)
- Training concepts (loss, optimization, epochs)
- Transfer learning
- Evaluation metrics (accuracy, AUC, confusion matrix)

**Study resources:**
- Course: "Deep Learning Specialization" by Andrew Ng (Coursera)
- Course: "Practical Deep Learning for Coders" (fast.ai)
- Book: "Deep Learning with Python" by François Chollet
- **Estimated time**: 2-3 months of study

#### 3. PyTorch Framework
**What you need:**
- Tensors and operations
- Building models with `nn.Module`
- Training loops
- DataLoaders and transforms
- Saving/loading models

**Study resources:**
- Official PyTorch tutorials: https://pytorch.org/tutorials/
- Course: "PyTorch for Deep Learning" (Udemy)
- Practice: Implement MNIST classifier from scratch
- **Estimated time**: 3-4 weeks

### Recommended Skills (Should Have)

#### 4. Machine Learning Fundamentals
- Train/validation/test splits
- Overfitting and regularization
- Cross-validation
- Hyperparameter tuning

**Study resources:**
- Course: "Machine Learning" by Andrew Ng (Coursera)
- Book: "Hands-On Machine Learning" by Aurélien Géron

#### 5. Computer Vision Basics
- Image preprocessing
- Data augmentation
- Common architectures (ResNet, VGG, EfficientNet)
- Evaluation for imbalanced datasets

**Study resources:**
- Course: "Convolutional Neural Networks" (Coursera)
- Papers: Read ResNet paper (He et al., 2015)

#### 6. Linux/Command Line
- Basic shell commands (cd, ls, mkdir, cp, mv)
- Package management (pip, conda)
- Running scripts
- SSH for remote servers

**Study resources:**
- Tutorial: "Linux Command Line Basics" (Udacity, free)
- Practice: Set up Ubuntu in VirtualBox

### Nice to Have

- Git version control
- Data visualization (Matplotlib, Seaborn)
- Medical image analysis background
- Web development basics (for Gradio customization)

---

## System Requirements

### Hardware

**Minimum (for inference only):**
- CPU: 4 cores (Intel i5 or equivalent)
- RAM: 8GB
- Storage: 5GB free space
- No GPU required for inference

**Recommended (for training):**
- CPU: 8+ cores (Intel i7/Ryzen 7 or better)
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM (GTX 1060, RTX 2060, or better)
- Storage: 20GB free space (for dataset + checkpoints)

**Used in this project:**
- CPU: Modern multi-core processor
- RAM: 16GB+
- GPU: CUDA-capable (optional for inference, recommended for training)

### Software

**Operating System:**
- Linux (Ubuntu 20.04/22.04 recommended)
- macOS (with some path adjustments)
- Windows (with WSL2 recommended)

**Required Software:**
- Python 3.8-3.12
- Conda or pip
- CUDA Toolkit 11.8+ (if using GPU)

---

## Step-by-Step Reproduction

### Phase 1: Environment Setup (30 minutes)

#### Step 1.1: Install Miniconda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, accept license, choose install location
# Restart terminal after installation
```

#### Step 1.2: Create Python Environment

```bash
# Create environment with Python 3.12
conda create -n melanoma-env python=3.12 -y

# Activate environment
conda activate melanoma-env

# You should see (melanoma-env) in your prompt
```

#### Step 1.3: Install PyTorch

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.8.0+cu118
CUDA available: True  # (or False if CPU-only)
```

---

### Phase 2: Get the Dataset (1-2 hours)

#### Step 2.1: Download HAM10000 Dataset

**Option A: From Kaggle (Recommended)**

```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Unzip
unzip skin-cancer-mnist-ham10000.zip -d data/raw/
```

**Option B: From Harvard Dataverse**
- Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- Download all image archives
- Download HAM10000_metadata.csv

#### Step 2.2: Organize Dataset Structure

```bash
# Create project directory
mkdir -p ~/melanoma-detection
cd ~/melanoma-detection

# Create directory structure
mkdir -p data/{raw,ds/{img,ann}}
mkdir -p models/{checkpoints,label_maps}
mkdir -p src/{training,inference}
mkdir -p notebooks
mkdir -p experiments/model_comparison/visualizations
mkdir -p requirements
mkdir -p scripts
mkdir -p tests
mkdir -p docs
```

#### Step 2.3: Prepare Image Data

```bash
# If images are in separate folders (HAM10000_images_part_1, HAM10000_images_part_2)
# Combine them:
cp data/raw/HAM10000_images_part_1/* data/ds/img/
cp data/raw/HAM10000_images_part_2/* data/ds/img/

# Copy metadata
cp data/raw/HAM10000_metadata.csv data/

# Verify dataset
ls data/ds/img/ | wc -l
# Should show: 10015
```

---

### Phase 3: Project Setup (15 minutes)

#### Step 3.1: Install Core Dependencies

Create `requirements/requirements-base.txt`:
```bash
cat > requirements/requirements-base.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
EOF
```

Install:
```bash
pip install -r requirements/requirements-base.txt
```

#### Step 3.2: Install Training Dependencies

Create `requirements/requirements-train.txt`:
```bash
cat > requirements/requirements-train.txt << 'EOF'
torchcam>=0.4.0
albumentations>=1.3.0
EOF
```

Install:
```bash
pip install -r requirements/requirements-train.txt
```

#### Step 3.3: Install Serving Dependencies

Create `requirements/requirements-serve.txt`:
```bash
cat > requirements/requirements-serve.txt << 'EOF'
gradio>=5.0.0
torchcam>=0.4.0
EOF
```

Install:
```bash
pip install -r requirements/requirements-serve.txt
```

---

### Phase 4: Build Core Code (2-3 hours)

#### Step 4.1: Create Configuration Module

Create `src/config.py`:
```python
from pathlib import Path

class Config:
    # Paths
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data"
    IMG_DIR = DATA_DIR / "ds" / "img"
    METADATA_CSV = DATA_DIR / "HAM10000_metadata.csv"
    
    # Model paths
    MODELS_DIR = ROOT / "models"
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
    LABEL_MAP_PATH = MODELS_DIR / "label_maps" / "label_map_nb.json"
    
    # Training config
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    NUM_WORKERS = 4
    
    # Model config
    IMG_SIZE = 224
    NUM_CLASSES = 7
    
    # Data split
    TRAIN_RATIO = 0.85
    VAL_RATIO = 0.15
    RANDOM_SEED = 42
```

#### Step 4.2: Create Data Loading Module

Create `src/training/dataset.py`:
```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path

class HAM10000Dataset(Dataset):
    def __init__(self, img_dir, metadata_csv, transform=None, split='train'):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Load metadata
        df = pd.read_csv(metadata_csv)
        
        # Create label mapping
        self.label_map = {label: idx for idx, label in enumerate(sorted(df['dx'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_map.items()}
        
        # Filter by split if needed
        self.df = df
        self.image_ids = df['image_id'].values
        self.labels = df['dx'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label_name = self.labels[idx]
        
        # Load image
        img_path = self.img_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label index
        label = self.label_map[label_name]
        
        return image, label, image_id
```

#### Step 4.3: Create Training Script

Create `src/training/train.py`:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
import json
from pathlib import Path
from dataset import HAM10000Dataset
from src.config import Config

def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    full_dataset = HAM10000Dataset(
        Config.IMG_DIR,
        Config.METADATA_CSV,
        transform=get_transforms(train=True)
    )
    
    # Save label map
    Config.LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.LABEL_MAP_PATH, 'w') as f:
        json.dump(full_dataset.label_map, f, indent=2)
    
    # Split dataset
    train_size = int(Config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    
    # Build model
    model = build_model(Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_path = Config.CHECKPOINT_DIR / "melanoma_resnet50_nb.pth"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"✓ Saved best model (val_acc={val_acc:.4f})")
    
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
```

#### Step 4.4: Run Training

```bash
# Start training (this will take several hours on GPU, longer on CPU)
cd ~/melanoma-detection
python src/training/train.py

# Monitor training
# You should see progress bars and metrics for each epoch
# Training 20 epochs takes ~2-4 hours on modern GPU
```

Expected output:
```
Using device: cuda
Epoch 1/20
Training: 100%|████████| 266/266 [02:15<00:00]
Validation: 100%|██████| 47/47 [00:22<00:00]
Train Loss: 1.2345, Train Acc: 0.5123
Val Loss: 1.1234, Val Acc: 0.5567
✓ Saved best model (val_acc=0.5567)
...
Epoch 20/20
Train Loss: 0.3456, Train Acc: 0.8912
Val Loss: 0.4123, Val Acc: 0.8323
✓ Saved best model (val_acc=0.8323)

Training complete! Best val accuracy: 0.8323
```

---

### Phase 5: Model Calibration (30 minutes)

#### Step 5.1: Temperature Scaling

Create `src/inference/calibration.py`:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import minimize

def temperature_scale(logits, temperature):
    """Apply temperature scaling to logits"""
    return logits / temperature

def calculate_ece(probs, labels, n_bins=15):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def find_optimal_temperature(model, val_loader, device):
    """Find optimal temperature using validation set"""
    model.eval()
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    
    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    
    def objective(T):
        scaled_logits = logits / T
        probs = torch.softmax(torch.from_numpy(scaled_logits), dim=1).numpy()
        return calculate_ece(probs, labels)
    
    result = minimize(objective, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
    optimal_temp = result.x[0]
    
    return optimal_temp

# Usage:
# temperature = find_optimal_temperature(model, val_loader, device)
# Save to models/checkpoints/temperature.json
```

---

### Phase 6: Explainable AI (30 minutes)

#### Step 6.1: Install Grad-CAM

```bash
pip install torchcam>=0.4.0
```

#### Step 6.2: Create XAI Module

Create `src/inference/xai.py`:
```python
import torch
import numpy as np
from torchcam.methods import GradCAM
from PIL import Image
import matplotlib.pyplot as plt

def generate_gradcam(model, image_tensor, target_class, target_layer='layer4'):
    """Generate Grad-CAM visualization"""
    model.eval()
    
    cam_extractor = GradCAM(model, target_layer=target_layer)
    
    # Forward pass with gradients
    logits = model(image_tensor)
    activation_maps = cam_extractor(class_idx=target_class, scores=logits)
    
    cam = activation_maps[0].squeeze().cpu().numpy()
    
    # Cleanup
    cam_extractor.remove_hooks()
    
    return cam

def overlay_gradcam(image, cam, alpha=0.5, colormap='jet'):
    """Overlay Grad-CAM heatmap on image"""
    # Normalize CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Resize CAM to image size
    cam_resized = np.array(Image.fromarray(cam).resize(image.size))
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam_resized)[:, :, :3] * 255
    heatmap = heatmap.astype(np.uint8)
    
    # Overlay
    img_array = np.array(image)
    overlay = (alpha * img_array + (1 - alpha) * heatmap).astype(np.uint8)
    
    return Image.fromarray(overlay)
```

---

### Phase 7: Web Interface (1 hour)

#### Step 7.1: Create Gradio Server

Create `src/serve_gradio.py`:
```python
import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import json

from inference.xai import generate_gradcam, overlay_gradcam
from config import Config

def load_model():
    """Load trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load label map
    with open(Config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    labels = list(label_map.keys())
    
    # Build model
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    
    # Load weights
    checkpoint = torch.load(
        Config.CHECKPOINT_DIR / "melanoma_resnet50_nb.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model, labels, device

def predict(image):
    """Make prediction on image"""
    if image is None:
        return None, None, None
    
    # Preprocess
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
    
    # Generate Grad-CAM
    cam = generate_gradcam(model, img_tensor, pred_idx)
    gradcam_img = overlay_gradcam(image.resize((224, 224)), cam)
    
    # Format results
    pred_label = labels[pred_idx]
    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    return gradcam_img, pred_label, prob_dict

# Load model globally
model, labels, device = load_model()

# Create Gradio interface
with gr.Blocks(title="Melanoma Detection") as demo:
    gr.Markdown("# 🔬 Melanoma Detection with Explainable AI")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Skin Lesion")
            predict_btn = gr.Button("🔍 Analyze", variant="primary")
        
        with gr.Column():
            gradcam_output = gr.Image(label="Grad-CAM Explanation")
            label_output = gr.Label(label="Predicted Class")
            probs_output = gr.JSON(label="Probabilities")
    
    predict_btn.click(
        predict,
        inputs=[image_input],
        outputs=[gradcam_output, label_output, probs_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

#### Step 7.2: Launch Server

```bash
python src/serve_gradio.py
```

Access at: http://localhost:7860

---

### Phase 8: Testing & Validation (30 minutes)

#### Step 8.1: Create Tests

Create `tests/test_smoke_inference.py`:
```python
import unittest
import torch
from src.serve_gradio import load_model, predict
from PIL import Image
import numpy as np

class SmokeInferenceTest(unittest.TestCase):
    def test_can_load_and_predict(self):
        """Test that model loads and can make predictions"""
        # Create dummy image
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Try prediction
        gradcam, label, probs = predict(dummy_image)
        
        # Assertions
        self.assertIsNotNone(gradcam)
        self.assertIsNotNone(label)
        self.assertIsNotNone(probs)
        self.assertEqual(len(probs), 7)  # 7 classes
        self.assertTrue(sum(probs.values()) > 0.99)  # Probs sum to ~1

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m unittest discover tests/
```

---

## Learning Path to Master This Project

### Timeline: 3-6 Months (Part-Time Study)

#### Month 1-2: Foundations

**Week 1-2: Python Mastery**
- Complete "Python Crash Course" chapters 1-11
- Practice: Build a command-line calculator, file organizer
- Goal: Comfortable with functions, classes, file I/O

**Week 3-4: NumPy & Data Handling**
- NumPy tutorial: https://numpy.org/doc/stable/user/quickstart.html
- Pandas basics: https://pandas.pydata.org/docs/getting_started/
- Practice: Load CSV, manipulate arrays, basic statistics
- Project: Analyze a dataset (e.g., Iris, Titanic)

**Week 5-6: Deep Learning Theory**
- Watch: 3Blue1Brown Neural Networks series (YouTube)
- Read: "Neural Networks and Deep Learning" (Michael Nielsen, free online)
- Understand: Backpropagation, gradient descent, activation functions

**Week 7-8: CNNs & Computer Vision**
- Stanford CS231n lectures (YouTube, first 5 lectures)
- Read: "A guide to convolution arithmetic for deep learning"
- Understand: Convolution, pooling, feature maps

#### Month 3-4: PyTorch & Implementation

**Week 9-10: PyTorch Basics**
- Official PyTorch tutorials (60-minute blitz)
- Build: MNIST classifier from scratch
- Understand: Tensors, autograd, nn.Module, DataLoader

**Week 11-12: Transfer Learning**
- Tutorial: Fine-tuning torchvision models
- Paper: "ImageNet Classification with Deep CNNs" (AlexNet)
- Project: Fine-tune ResNet on Cats vs Dogs dataset

**Week 13-14: Training Best Practices**
- Learn: Data augmentation, learning rate scheduling
- Learn: Regularization (dropout, weight decay)
- Project: Improve your cats/dogs classifier (aim for 95%+ accuracy)

**Week 15-16: Medical Imaging Specifics**
- Read: "Medical Image Analysis with Deep Learning" papers
- Understand: Class imbalance, calibration, evaluation metrics
- Dataset: Download ChestX-ray14 or similar

#### Month 5: Advanced Topics

**Week 17-18: Explainable AI**
- Paper: "Grad-CAM: Visual Explanations from Deep Networks"
- Implement: Grad-CAM from scratch on your classifier
- Read: "Interpretable Machine Learning" book (Christoph Molnar, free online)

**Week 19-20: Model Deployment**
- Gradio tutorial: https://gradio.app/quickstart/
- Learn: Docker basics for containerization
- Project: Deploy your cats/dogs classifier with Gradio

#### Month 6: Reproduce This Project

**Week 21-22: Dataset & Training**
- Download HAM10000 dataset
- Implement training pipeline
- Train ResNet-50 baseline

**Week 23-24: Calibration & XAI**
- Implement temperature scaling
- Add Grad-CAM visualization
- Calculate operating thresholds

**Week 25-26: Polish & Deploy**
- Build Gradio interface
- Add AI explanation system
- Write documentation

---

## Understanding Each Component

### Component Breakdown

#### 1. Data Pipeline (`dataset.py`)

**What it does:**
- Loads images from disk
- Applies augmentations (flips, rotations, color jitter)
- Converts images to tensors
- Batches data for GPU efficiency

**Key concepts:**
- **Dataset**: PyTorch class that defines how to load one sample
- **DataLoader**: Batches samples, shuffles, loads in parallel
- **Transforms**: Image preprocessing (resize, normalize, augment)

**Why it matters:**
- Good augmentation prevents overfitting
- Normalization (ImageNet stats) helps transfer learning
- Efficient loading prevents GPU starvation

#### 2. Model Architecture (`train.py`)

**What it does:**
- Uses ResNet-50 pretrained on ImageNet
- Replaces final layer for 7-class classification
- Fine-tunes all layers

**Key concepts:**
- **Transfer learning**: Start with pretrained weights
- **Fine-tuning**: Update all layers (not just last)
- **Feature extraction**: Early layers detect edges/textures, later layers detect specific patterns

**Why ResNet-50:**
- Proven architecture (152-layer version won ImageNet 2015)
- Residual connections prevent vanishing gradients
- Good balance of accuracy vs speed

#### 3. Training Loop (`train.py`)

**What it does:**
- Forward pass: Image → Model → Logits
- Calculate loss: Cross-entropy between predictions and labels
- Backward pass: Compute gradients
- Update weights: Adam optimizer

**Key concepts:**
- **Epoch**: One pass through entire dataset
- **Batch**: Subset of data processed together
- **Learning rate**: How big a step to take when updating weights
- **Validation**: Evaluate on held-out data to detect overfitting

**Hyperparameters used:**
- Batch size: 32 (fits in GPU memory)
- Learning rate: 1e-4 (small for fine-tuning)
- Optimizer: Adam (adaptive learning rate per parameter)

#### 4. Temperature Calibration (`calibration.py`)

**What it does:**
- Finds a temperature T that makes probabilities well-calibrated
- If model says 70% melanoma, it should be correct 70% of the time

**Key concepts:**
- **Calibration**: Aligning confidence with accuracy
- **Temperature scaling**: Divide logits by T before softmax
- **ECE (Expected Calibration Error)**: Metric for calibration quality

**Why it matters:**
- Neural networks are often overconfident
- Medical AI requires reliable uncertainty estimates
- Calibration doesn't change predictions, just probabilities

#### 5. Grad-CAM (`xai.py`)

**What it does:**
- Shows which image regions influenced the prediction
- Computes gradients of output w.r.t. feature maps
- Highlights important areas in heatmap

**Key concepts:**
- **Activation maps**: Feature maps from convolutional layers
- **Gradients**: Show how output changes with input
- **Weighted combination**: Multiply maps by gradient importance

**Why it matters:**
- Explainability builds trust
- Helps catch spurious correlations (e.g., model using rulers in images)
- Required for medical AI deployment

#### 6. Web Interface (`serve_gradio.py`)

**What it does:**
- Loads trained model
- Provides upload interface
- Displays predictions and explanations

**Key concepts:**
- **Gradio**: Python library for building ML demos
- **Server**: Runs locally or can be deployed to cloud
- **State management**: Tracks conversation in chat

---

## Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or 8

# Use mixed precision training
# Add to train.py:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Issue 2: Low Accuracy

**Possible causes:**
- Insufficient training epochs (try 30-40 instead of 20)
- Learning rate too high or too low (try 1e-3 or 5e-5)
- Not enough augmentation
- Dataset imbalance not handled

**Solutions:**
```python
# Add class weights for imbalanced data
class_counts = df['dx'].value_counts()
class_weights = torch.tensor([1.0/class_counts[label] for label in labels])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
# After each epoch:
scheduler.step(val_loss)
```

### Issue 3: Gradio Server Won't Start

**Error:**
```
Address already in use
```

**Solution:**
```bash
# Kill process on port 7860
lsof -ti:7860 | xargs kill -9

# Or use different port
demo.launch(server_port=7861)
```

### Issue 4: Model File Not Found

**Error:**
```
FileNotFoundError: models/checkpoints/melanoma_resnet50_nb.pth
```

**Solution:**
```bash
# Ensure model was saved during training
# Check path:
ls -lh models/checkpoints/

# If missing, retrain or download pretrained checkpoint
```

---

## Extending This Project

### Ideas for Enhancements

#### 1. Multi-Model Ensemble
```python
# Train multiple architectures
models = {
    'resnet50': train_resnet50(),
    'efficientnet': train_efficientnet(),
    'densenet': train_densenet()
}

# Average predictions
ensemble_probs = sum(model.predict(img) for model in models.values()) / len(models)
```

#### 2. Active Learning
- Deploy model
- Collect predictions where confidence is low
- Send to dermatologist for labeling
- Retrain with new labels
- Improves performance on edge cases

#### 3. Attention Mechanisms
```python
# Add attention module to focus on relevant regions
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attention(x)
```

#### 4. Segmentation Instead of Classification
- Use U-Net to segment lesion boundary
- More interpretable than classification
- Provides size/shape measurements

#### 5. Mobile Deployment
```bash
# Convert to ONNX for mobile
torch.onnx.export(model, dummy_input, "model.onnx")

# Convert to TensorFlow Lite
# Use in Android/iOS app
```

---

## Resources for Similar Projects

### Datasets (Medical Imaging)

**Dermatology:**
- HAM10000 (this project)
- ISIC 2019/2020 (larger, 25k+ images)
- DermNet (organized by disease)

**Radiology:**
- ChestX-ray14 (112k chest X-rays)
- CheXpert (224k chest X-rays from Stanford)
- COVID-19 Image Dataset

**Ophthalmology:**
- Kaggle Diabetic Retinopathy Detection
- DRIVE (retinal vessel segmentation)

**Pathology:**
- Camelyon16/17 (breast cancer metastases)
- PatchCamelyon (histopathology)

### Tools & Libraries

**Deep Learning:**
- PyTorch: https://pytorch.org
- TensorFlow/Keras: https://tensorflow.org
- FastAI: https://fast.ai (higher-level PyTorch)

**Medical Imaging:**
- SimpleITK: Medical image processing
- PyDicom: DICOM file handling
- Torchio: Medical image augmentation

**Explainability:**
- Captum: PyTorch interpretability
- LIME: Local explanations
- SHAP: Shapley values

**Deployment:**
- Gradio: Quick ML demos
- Streamlit: Python web apps
- Flask/FastAPI: Production APIs
- Docker: Containerization

### Papers to Read

**Foundational:**
1. "ImageNet Classification with Deep CNNs" (AlexNet, 2012)
2. "Deep Residual Learning" (ResNet, 2015)
3. "Going Deeper with Convolutions" (Inception, 2014)

**Medical AI:**
1. "Dermatologist-level classification" (Esteva et al., Nature 2017)
2. "CheXNet: Radiologist-Level Pneumonia Detection" (Stanford, 2017)
3. "Deep learning for chest radiograph diagnosis" (Google, 2018)

**Explainability:**
1. "Grad-CAM: Visual Explanations" (2017)
2. "Attention is All You Need" (Transformers, 2017)
3. "A Survey on Explainable AI" (2021)

**Calibration:**
1. "On Calibration of Modern Neural Networks" (2017)
2. "Temperature Scaling" (Guo et al., 2017)

### Online Courses

**Free:**
- Fast.ai Practical Deep Learning for Coders
- Stanford CS231n (YouTube)
- MIT Deep Learning for Self-Driving Cars

**Paid (worth it):**
- Coursera Deep Learning Specialization (Andrew Ng)
- Udacity AI for Healthcare Nanodegree
- Coursera AI for Medicine Specialization

### Communities

**Forums:**
- PyTorch Forums: discuss.pytorch.org
- FastAI Forums: forums.fast.ai
- Reddit: r/MachineLearning, r/computervision

**Competitions:**
- Kaggle: kaggle.com (many medical imaging challenges)
- Grand Challenge: grand-challenge.org (medical imaging focus)

**Research:**
- Papers With Code: paperswithcode.com
- ArXiv: arxiv.org (latest research)

---

## Reproducibility Checklist

### Before You Start

- [ ] Python 3.8+ installed
- [ ] GPU with CUDA (optional but recommended)
- [ ] 20GB+ free disk space
- [ ] Stable internet connection (for downloads)

### Phase 1: Setup (Day 1)

- [ ] Install Miniconda/Anaconda
- [ ] Create virtual environment
- [ ] Install PyTorch + dependencies
- [ ] Verify CUDA (if using GPU)

### Phase 2: Data (Day 1-2)

- [ ] Download HAM10000 dataset (~5GB)
- [ ] Organize directory structure
- [ ] Verify 10,015 images present
- [ ] Check metadata CSV loads correctly

### Phase 3: Training (Day 2-3)

- [ ] Implement dataset class
- [ ] Implement training loop
- [ ] Start training (monitor first epoch)
- [ ] Wait for completion (~2-4 hours on GPU)
- [ ] Verify checkpoint saved (should be ~90MB)

### Phase 4: Calibration (Day 3)

- [ ] Implement temperature scaling
- [ ] Find optimal temperature on validation set
- [ ] Save temperature to JSON
- [ ] Verify calibrated probabilities sum to 1.0

### Phase 5: Explainability (Day 3-4)

- [ ] Install torchcam
- [ ] Implement Grad-CAM
- [ ] Test on sample image
- [ ] Verify heatmap highlights lesion area

### Phase 6: Deployment (Day 4)

- [ ] Install Gradio
- [ ] Implement inference function
- [ ] Build web interface
- [ ] Test locally at localhost:7860
- [ ] Upload test image and verify predictions

### Phase 7: Documentation (Day 5)

- [ ] Write README with usage instructions
- [ ] Document hyperparameters
- [ ] Add example images and outputs
- [ ] Create requirements.txt with exact versions

### Final Verification

- [ ] Run all tests: `python -m unittest discover tests/`
- [ ] Check model accuracy >80% on validation
- [ ] Verify Grad-CAM shows relevant regions
- [ ] Confirm web interface works end-to-end
- [ ] Test on fresh conda environment

---

## Success Metrics

### Minimum Viable Project

- **Accuracy**: >75% on validation set
- **AUC (Melanoma)**: >0.85
- **Grad-CAM**: Highlights lesion regions
- **Web UI**: Functional upload and prediction
- **Calibration**: ECE <0.1

### Strong Project (Thesis-Ready)

- **Accuracy**: >80% on validation set
- **AUC (Melanoma)**: >0.90
- **Specificity**: >90% at high sensitivity
- **Explainability**: Grad-CAM + AI text explanations
- **Interface**: Polished UI with chat Q&A
- **Documentation**: Complete with reproduction guide

### Excellent Project (Publication-Quality)

- **Accuracy**: >85% (approaches human dermatologist)
- **Ensemble**: Multiple models combined
- **External Validation**: Tested on ISIC dataset
- **Clinical Integration**: DICOM support, HL7 FHIR
- **Deployment**: Docker container, cloud-ready
- **Paper**: Written with methods, results, discussion

---

## Final Tips

### For Students

1. **Start simple**: Get a basic model working first, then improve
2. **Version control**: Use Git from day 1
3. **Experiment tracking**: Log all hyperparameters and results
4. **Ask for help**: Join forums, ask questions early
5. **Iterate**: Your first model won't be perfect - that's okay!

### For Practitioners

1. **Clinical validation**: Work with domain experts
2. **Bias & fairness**: Test on diverse populations
3. **Regulatory**: Understand FDA/CE requirements for medical AI
4. **Privacy**: HIPAA compliance if using real patient data
5. **Maintenance**: Plan for model updates as data drifts

### Time Estimates

**With prior ML experience:**
- Setup & training: 1-2 days
- Calibration & XAI: 1 day
- Web interface: 1 day
- **Total**: 3-4 days

**Learning from scratch:**
- Prerequisite learning: 2-3 months (part-time)
- Implementation: 1-2 weeks (part-time)
- **Total**: 3-4 months (part-time)

---

## Conclusion

This guide provides everything needed to reproduce this melanoma detection system from scratch. The project demonstrates:

- **Transfer learning** with ResNet-50
- **Medical image classification** on HAM10000
- **Explainable AI** with Grad-CAM
- **Model calibration** for reliable probabilities
- **Web deployment** with Gradio
- **Clinical integration** with AI explanations

By following this guide, you will:
1. ✅ Understand each component deeply
2. ✅ Build the system step-by-step
3. ✅ Have a working thesis project
4. ✅ Gain skills for similar medical AI projects

**Good luck with your project! 🚀**

---

## Quick Reference Commands

```bash
# Environment setup
conda create -n melanoma-env python=3.12 -y
conda activate melanoma-env
pip install torch torchvision gradio torchcam pandas numpy matplotlib seaborn scikit-learn tqdm

# Download data
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/raw/

# Train model
python src/training/train.py

# Launch web interface
python src/serve_gradio.py

# Run tests
python -m unittest discover tests/ -v
```

---

**Document Version**: 1.0  
**Last Updated**: November 18, 2025  
**Project**: AI-Powered Melanoma Detection with Explainability  
**Author**: Thesis Reproduction Guide
