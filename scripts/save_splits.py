#!/usr/bin/env python3
"""
Save train/validation split indices for reproducibility.
Run this after creating splits to ensure experiments can be reproduced.
"""
import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Find project root
CWD = Path.cwd()
candidates = [CWD, CWD.parent, CWD.parent.parent]
ROOT = CWD
for root in candidates:
    if (root / 'data' / 'HAM10000_metadata.csv').exists():
        ROOT = root
        break

DATA_DIR = ROOT / 'data'
META_CSV = DATA_DIR / 'HAM10000_metadata.csv'
SPLIT_OUTPUT = DATA_DIR / 'splits.json'

# Match notebook settings
SEED = 42
TEST_SIZE = 0.15

def save_splits():
    """Generate and save train/val split indices."""
    # Load metadata
    meta = pd.read_csv(META_CSV)
    labels = meta['dx'].astype('category')
    label_map = {cat: i for i, cat in enumerate(labels.cat.categories)}
    meta['label'] = meta['dx'].map(label_map)
    
    # Create splits (same as notebook)
    train_df, val_df = train_test_split(
        meta, 
        test_size=TEST_SIZE, 
        stratify=meta['label'], 
        random_state=SEED
    )
    
    # Save indices and image_ids for reproducibility
    splits_data = {
        'seed': SEED,
        'test_size': TEST_SIZE,
        'train_indices': train_df.index.tolist(),
        'val_indices': val_df.index.tolist(),
        'train_image_ids': train_df['image_id'].tolist(),
        'val_image_ids': val_df['image_id'].tolist(),
        'label_map': label_map,
        'train_size': len(train_df),
        'val_size': len(val_df)
    }
    
    # Save to JSON
    with open(SPLIT_OUTPUT, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"✓ Saved splits to {SPLIT_OUTPUT}")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Classes: {len(label_map)}")
    
    # Verify distribution
    print("\nTrain distribution:")
    print(train_df['dx'].value_counts().to_dict())
    print("\nVal distribution:")
    print(val_df['dx'].value_counts().to_dict())

if __name__ == '__main__':
    save_splits()
