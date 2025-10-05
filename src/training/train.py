import os
import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from PIL import Image


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(img_size=224):
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, test_tf


def compute_class_weights(labels: np.ndarray, num_classes: int):
    counts = np.bincount(labels, minlength=num_classes)
    weights = labels.shape[0] / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def build_model(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    all_probs = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # macro ROC-AUC (one-vs-rest)
    try:
        y_true = np.eye(num_classes)[all_labels]
        auc = roc_auc_score(y_true, all_probs, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    return auc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', default='HAM10000_metadata.csv')
    ap.add_argument('--img-dir', default='./ds/img')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out-weights', default='melanoma_resnet50.pth')
    ap.add_argument('--label-map', default='label_map.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(args.metadata)
    labels = df['dx'].astype('category')
    label_map = {cat: i for i, cat in enumerate(labels.cat.categories)}
    df['label'] = df['dx'].map(label_map)

    # save label_map for inference
    with open(args.label_map, 'w') as f:
        json.dump(label_map, f)

    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df['label'], random_state=42
    )

    train_tf, test_tf = build_transforms(args.img_size)
    train_ds = SkinDataset(train_df, args.img_dir, transform=train_tf)
    val_ds = SkinDataset(val_df, args.img_dir, transform=test_tf)

    class_weights = compute_class_weights(train_df['label'].values, num_classes=len(label_map))
    class_weights = class_weights.to(device)

    # Optional: Weighted sampling for imbalance
    sample_weights = class_weights.cpu().numpy()[train_df['label'].values]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    best_auc = -1.0
    patience = 8
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_auc = evaluate(model, val_loader, num_classes=len(label_map), device=device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(model.state_dict(), args.out_weights)
            print(f"Saved best weights to {args.out_weights}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break


if __name__ == '__main__':
    main()
