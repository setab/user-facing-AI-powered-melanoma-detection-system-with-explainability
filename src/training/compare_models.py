"""
Model Comparison Script for Melanoma Detection Thesis
Trains and evaluates multiple architectures on HAM10000 dataset.
Compares: ResNet-50, EfficientNet-B3, DenseNet-121, Vision Transformer (ViT-B/16)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    densenet121, DenseNet121_Weights,
    vit_b_16, ViT_B_16_Weights
)
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.calibration import calibration_curve
import pandas as pd

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score (lower is better)"""
    return np.mean((y_prob - y_true) ** 2)


class HAM10000Dataset(Dataset):
    """HAM10000 dataset loader"""
    
    def __init__(self, metadata_csv: str, img_dir: str, label_map: Dict[str, int], 
                 transform=None, split: str = 'train'):
        """
        Args:
            metadata_csv: Path to HAM10000_metadata.csv
            img_dir: Directory containing images
            label_map: Dictionary mapping diagnosis labels to class indices
            transform: Optional transforms to apply
            split: 'train' or 'val' (for augmentation differences)
        """
        self.df = pd.read_csv(metadata_csv)
        self.img_dir = Path(img_dir)
        self.label_map = label_map
        self.transform = transform
        self.split = split
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        img_path = self.img_dir / f"{img_id}.jpg"
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label
        label = self.label_map[row['dx']]
        
        return img, label


def get_transforms(split: str = 'train', img_size: int = 224):
    """Get data transforms for train or val"""
    if split == 'train':
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Build model architecture"""
    if arch == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'efficientnet_b3':
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def validate(model: nn.Module, loader: DataLoader, criterion, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            
            total_loss += loss.item() * inputs.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    preds = np.argmax(all_probs, axis=1)
    accuracy = accuracy_score(all_labels, preds)
    
    return total_loss / len(all_labels), accuracy, all_labels, all_probs


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int, lr: float, device, patience: int = 5) -> Dict[str, Any]:
    """Train model with early stopping"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device) -> float:
    """Find optimal temperature for calibration using validation set"""
    model.eval()
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Optimize temperature using NLL loss
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    def eval_temp():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_temp)
    
    return temperature.item()


def compute_operating_threshold(y_true: np.ndarray, y_probs: np.ndarray, 
                                melanoma_class_idx: int, target_specificity: float = 0.95) -> float:
    """Compute operating threshold for melanoma detection at target specificity"""
    # Binary problem: melanoma vs. non-melanoma
    y_binary = (y_true == melanoma_class_idx).astype(int)
    y_prob_mel = y_probs[:, melanoma_class_idx]
    
    fpr, tpr, thresholds = roc_curve(y_binary, y_prob_mel)
    specificity = 1 - fpr
    
    # Find threshold closest to target specificity
    idx = np.argmin(np.abs(specificity - target_specificity))
    
    return thresholds[idx]


def evaluate_model_comprehensive(model: nn.Module, loader: DataLoader, device,
                                 temperature: float, melanoma_idx: int) -> Dict[str, Any]:
    """Comprehensive evaluation with all metrics"""
    model.eval()
    
    all_labels = []
    all_probs_raw = []
    all_probs_cal = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_times.append(time.time() - start_time)
            
            # Raw probabilities
            probs_raw = torch.softmax(outputs, dim=1)
            
            # Calibrated probabilities
            outputs_cal = outputs / temperature
            probs_cal = torch.softmax(outputs_cal, dim=1)
            
            all_labels.append(labels.numpy())
            all_probs_raw.append(probs_raw.cpu().numpy())
            all_probs_cal.append(probs_cal.cpu().numpy())
    
    all_labels = np.concatenate(all_labels)
    all_probs_raw = np.concatenate(all_probs_raw)
    all_probs_cal = np.concatenate(all_probs_cal)
    
    preds_raw = np.argmax(all_probs_raw, axis=1)
    preds_cal = np.argmax(all_probs_cal, axis=1)
    
    # Compute metrics
    results = {
        # Accuracy
        'accuracy_raw': accuracy_score(all_labels, preds_raw),
        'accuracy_calibrated': accuracy_score(all_labels, preds_cal),
        
        # AUC (multiclass)
        'auc_macro': roc_auc_score(all_labels, all_probs_cal, multi_class='ovr', average='macro'),
        
        # Calibration metrics
        'ece': compute_ece(all_labels, np.max(all_probs_cal, axis=1)),
        'brier_score': compute_brier_score(
            np.eye(all_probs_cal.shape[1])[all_labels],
            all_probs_cal
        ).mean(),
        
        # Inference speed
        'inference_time_mean': np.mean(inference_times),
        'inference_time_std': np.std(inference_times),
        
        # Confusion matrix
        'confusion_matrix': confusion_matrix(all_labels, preds_cal).tolist(),
        
        # Melanoma-specific metrics
        'melanoma_metrics': {},
    }
    
    # Compute melanoma-specific metrics
    y_binary = (all_labels == melanoma_idx).astype(int)
    y_prob_mel = all_probs_cal[:, melanoma_idx]
    
    # Operating point at 95% specificity
    threshold_95 = compute_operating_threshold(all_labels, all_probs_cal, melanoma_idx, 0.95)
    y_pred_95 = (y_prob_mel >= threshold_95).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_95).ravel()
    
    results['melanoma_metrics'] = {
        'threshold_spec95': float(threshold_95),
        'sensitivity_at_spec95': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity_at_spec95': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'ppv_at_spec95': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'npv_at_spec95': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        'auc_melanoma': float(roc_auc_score(y_binary, y_prob_mel)),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare melanoma detection models')
    parser.add_argument('--metadata', default='data/HAM10000_metadata.csv')
    parser.add_argument('--img-dir', default='data/ds/img')
    parser.add_argument('--label-map', default='models/label_maps/label_map_nb.json')
    parser.add_argument('--output-dir', default='experiments/model_comparison')
    parser.add_argument('--architectures', nargs='+', 
                       default=['resnet50', 'efficientnet_b3', 'densenet121', 'vit_b_16'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load label map
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    
    num_classes = len(label_map)
    melanoma_idx = label_map.get('melanoma', -1)
    
    print(f"📊 Model Comparison Experiment")
    print(f"   Classes: {list(label_map.keys())}")
    print(f"   Melanoma index: {melanoma_idx}")
    print(f"   Architectures: {args.architectures}")
    print()
    
    # Load dataset
    df = pd.read_csv(args.metadata)
    train_df, val_df = train_test_split(df, test_size=args.val_split, 
                                        stratify=df['dx'], random_state=args.seed)
    
    # Save splits
    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'val_split.csv', index=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Results storage
    all_results = {}
    
    # Train and evaluate each architecture
    for arch in args.architectures:
        print(f"\n{'='*80}")
        print(f"🚀 Training {arch.upper()}")
        print('='*80)
        
        # Create datasets
        train_dataset = HAM10000Dataset(
            output_dir / 'train_split.csv', args.img_dir, label_map,
            transform=get_transforms('train', args.img_size), split='train'
        )
        val_dataset = HAM10000Dataset(
            output_dir / 'val_split.csv', args.img_dir, label_map,
            transform=get_transforms('val', args.img_size), split='val'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
        
        # Build model
        model = build_model(arch, num_classes).to(device)
        
        # Train
        print(f"\n📈 Training...")
        history = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
        
        # Calibrate temperature
        print(f"\n🌡️  Calibrating temperature...")
        temperature = calibrate_temperature(model, val_loader, device)
        print(f"   Optimal temperature: {temperature:.4f}")
        
        # Comprehensive evaluation
        print(f"\n📊 Evaluating...")
        results = evaluate_model_comprehensive(model, val_loader, device, temperature, melanoma_idx)
        results['temperature'] = temperature
        results['training_history'] = history
        
        # Save model checkpoint
        checkpoint_path = output_dir / f'{arch}_checkpoint.pth'
        torch.save({
            'model_state': model.state_dict(),
            'temperature': temperature,
            'label_map': label_map,
            'architecture': arch,
            'args': vars(args),
        }, checkpoint_path)
        print(f"💾 Saved checkpoint to {checkpoint_path}")
        
        # Store results
        all_results[arch] = results
        
        # Print summary
        print(f"\n✅ {arch.upper()} Results:")
        print(f"   Accuracy (calibrated): {results['accuracy_calibrated']:.4f}")
        print(f"   AUC (macro): {results['auc_macro']:.4f}")
        print(f"   ECE: {results['ece']:.4f}")
        print(f"   Brier Score: {results['brier_score']:.4f}")
        print(f"   Inference time: {results['inference_time_mean']*1000:.2f} ± {results['inference_time_std']*1000:.2f} ms")
        print(f"   Melanoma AUC: {results['melanoma_metrics']['auc_melanoma']:.4f}")
        print(f"   Sensitivity @ Spec=95%: {results['melanoma_metrics']['sensitivity_at_spec95']:.4f}")
    
    # Save comparison results
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Comparison complete! Results saved to {results_path}")
    print('='*80)
    
    # Print comparison table
    print("\n📊 COMPARISON TABLE:")
    print(f"{'Architecture':<20} {'Accuracy':<12} {'AUC':<10} {'ECE':<10} {'Inference (ms)':<15} {'Mel AUC':<10}")
    print('-' * 80)
    for arch, res in all_results.items():
        print(f"{arch:<20} "
              f"{res['accuracy_calibrated']:<12.4f} "
              f"{res['auc_macro']:<10.4f} "
              f"{res['ece']:<10.4f} "
              f"{res['inference_time_mean']*1000:<15.2f} "
              f"{res['melanoma_metrics']['auc_melanoma']:<10.4f}")


if __name__ == '__main__':
    main()
