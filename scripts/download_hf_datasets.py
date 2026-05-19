#!/usr/bin/env python3
"""
Download and prepare public HuggingFace datasets for evaluation.
Extracts up to 400 balanced rows (200 benign, 200 attack) from each dataset.
"""

import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

def _balanced_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df['Answer'] = df['Answer'].astype(int)
    benign_pool = df[df['Answer'] == 0]
    attack_pool = df[df['Answer'] == 1]
    benign = benign_pool.sample(n=min(per_class, len(benign_pool)), random_state=seed)
    attacks = attack_pool.sample(n=min(per_class, len(attack_pool)), random_state=seed + 1)
    return pd.concat([benign, attacks], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_deepset_dataset(per_class: int = 200, seed: int = 42):
    """Download and prepare deepset/prompt-injections dataset."""
    print("\n=== Preparing deepset/prompt-injections dataset ===")
    try:
        ds = load_dataset("deepset/prompt-injections")
        print(f"Dataset splits: {ds.keys()}")
        
        # Usually has 'train' split
        if 'train' in ds:
            df = pd.DataFrame(ds['train'])
        else:
            df = pd.DataFrame(ds[list(ds.keys())[0]])
        
        print(f"Dataset columns: {df.columns.tolist()}")
        print(f"Total rows: {len(df)}")
        print(f"First row: {df.iloc[0]}")
        
        # Normalize columns - expect 'text' and 'label' or similar
        if 'text' in df.columns and 'label' in df.columns:
            df_normalized = df[['text', 'label']].copy()
            df_normalized.columns = ['Question', 'Answer']
        elif 'prompt' in df.columns and 'label' in df.columns:
            df_normalized = df[['prompt', 'label']].copy()
            df_normalized.columns = ['Question', 'Answer']
        else:
            print("Unexpected column names, attempting auto-detection...")
            text_cols = [c for c in df.columns if 'text' in c.lower() or 'prompt' in c.lower() or 'question' in c.lower()]
            label_cols = [c for c in df.columns if 'label' in c.lower() or 'is_attack' in c.lower() or 'attack' in c.lower()]
            if text_cols and label_cols:
                df_normalized = df[[text_cols[0], label_cols[0]]].copy()
                df_normalized.columns = ['Question', 'Answer']
            else:
                print(f"Cannot auto-detect columns. Available: {df.columns.tolist()}")
                return None
        
        # Balance to up to 200 benign + 200 attack (400 total by default)
        df_balanced = _balanced_sample(df_normalized, per_class=per_class, seed=seed)
        benign = df_balanced[df_balanced['Answer'] == 0]
        attacks = df_balanced[df_balanced['Answer'] == 1]
        
        # Add Scenario column (used as category)
        df_balanced['Scenario'] = 'HF_Deepset'
        
        output_path = Path(f"data/HF_deepset_{len(df_balanced)}_balanced.csv")
        df_balanced.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df_balanced)} rows to {output_path}")
        print(f"  - Benign: {len(benign)}, Attack: {len(attacks)}")
        return output_path
        
    except Exception as e:
        print(f"✗ Error preparing deepset dataset: {e}")
        return None

def prepare_multilingual_dataset(per_class: int = 200, seed: int = 42):
    """Download and prepare Octavio-Santana/prompt-injection-attack-detection-multilingual dataset."""
    print("\n=== Preparing Octavio-Santana multilingual dataset ===")
    try:
        ds = load_dataset("Octavio-Santana/prompt-injection-attack-detection-multilingual")
        print(f"Dataset splits: {ds.keys()}")
        
        # Get first available split
        split_name = list(ds.keys())[0]
        df = pd.DataFrame(ds[split_name])
        
        print(f"Dataset columns: {df.columns.tolist()}")
        print(f"Total rows: {len(df)}")
        print(f"First row: {df.iloc[0]}")
        
        # Normalize columns
        if 'text' in df.columns and 'label' in df.columns:
            df_normalized = df[['text', 'label']].copy()
            df_normalized.columns = ['Question', 'Answer']
        elif 'prompt' in df.columns and 'label' in df.columns:
            df_normalized = df[['prompt', 'label']].copy()
            df_normalized.columns = ['Question', 'Answer']
        else:
            text_cols = [c for c in df.columns if 'text' in c.lower() or 'prompt' in c.lower() or 'question' in c.lower()]
            label_cols = [c for c in df.columns if 'label' in c.lower() or 'is_attack' in c.lower() or 'attack' in c.lower()]
            if text_cols and label_cols:
                df_normalized = df[[text_cols[0], label_cols[0]]].copy()
                df_normalized.columns = ['Question', 'Answer']
            else:
                print(f"Cannot auto-detect columns. Available: {df.columns.tolist()}")
                return None
        
        # Balance to up to 200 benign + 200 attack (400 total by default)
        df_balanced = _balanced_sample(df_normalized, per_class=per_class, seed=seed)
        benign = df_balanced[df_balanced['Answer'] == 0]
        attacks = df_balanced[df_balanced['Answer'] == 1]
        
        # Add category column
        df_balanced['Scenario'] = 'HF_Multilingual'
        
        output_path = Path(f"data/HF_multilingual_{len(df_balanced)}_balanced.csv")
        df_balanced.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df_balanced)} rows to {output_path}")
        print(f"  - Benign: {len(benign)}, Attack: {len(attacks)}")
        return output_path
        
    except Exception as e:
        print(f"✗ Error preparing multilingual dataset: {e}")
        return None

if __name__ == "__main__":
    print("Downloading public HuggingFace datasets for evaluation...")
    print("Target: up to 400 balanced rows (200 benign, 200 attack) per dataset")
    
    path1 = prepare_deepset_dataset()
    path2 = prepare_multilingual_dataset()
    
    print("\n" + "="*60)
    print("Summary:")
    if path1:
        print(f"✓ deepset/prompt-injections → {path1}")
    if path2:
        print(f"✓ Octavio-Santana multilingual → {path2}")
    print("="*60)
