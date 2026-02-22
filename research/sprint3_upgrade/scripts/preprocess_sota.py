
import os
import glob
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from tqdm import tqdm
import sys

import re
from typing import List, Tuple, Optional, Iterable

def canonical_key(name: str) -> str:
    """Standardize feature names for cross-dataset mapping."""
    # Convert to string and strip
    x = str(name).strip()
    
    # Remove trailing .1 (found in Fwd Header Length.1)
    x = re.sub(r"\.\d+$", "", x)
    
    # Split camelCase or joined words (like TotLen or InitFwd)
    x = re.sub(r"([a-z])([A-Z])", r"\1 \2", x)
    
    x = x.lower()
    x = x.replace("/", " ")
    x = re.sub(r"[_\-]+", " ", x)
    x = re.sub(r"\s+", " ", x)

    replacements = {
        "dst": "destination", "src": "source", "byts": "bytes", "byt": "byte",
        "pkts": "packets", "pkt": "packet", "cnt": "count",
        "len": "length", "avg": "average", "tot": "total", 
        "seg": "segment", "fwd": "forward", "bwd": "backward",
        "init": "initialization", "win": "window", "act": "active"
    }

    tokens = []
    for tok in x.split():
        tok = replacements.get(tok, tok)
        # Handle pluralization simply for matching
        if tok.endswith("s") and len(tok) > 3:
            tok = tok[:-1]
        tokens.append(tok)
    
    # Final cleanup of common stop words in these datasets
    stop_words = {"of", "size"} # "size" can be inconsistent but usually paired
    tokens = [t for t in tokens if t not in stop_words]
    
    tokens.sort()
    return " ".join(tokens)

def build_column_mapper(source_cols: List[str], target_cols: List[str]) -> dict[str, str]:
    """Map target columns to source/reference columns using canonical keys."""
    key_to_ref: dict[str, str] = {}
    for col in source_cols:
        key = canonical_key(col)
        key_to_ref[key] = col

    mapper: dict[str, str] = {}
    for col in target_cols:
        key = canonical_key(col)
        if key in key_to_ref:
            mapper[col] = key_to_ref[key]
    return mapper

def select_features_by_statistics(sample_df, nzv_threshold=0.01, corr_threshold=0.9):
    """
    Perform NZV and Correlation filtering on a sample dataframe.
    """
    # 1. NZV
    variances = sample_df.var()
    dropped_nzv = variances[variances < nzv_threshold].index.tolist()
    selected = sample_df.columns.drop(dropped_nzv)
    
    # 2. Correlation
    if len(selected) > 1:
        corr_matrix = sample_df[selected].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        dropped_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        selected = selected.drop(dropped_corr)
        return selected.tolist(), dropped_nzv, dropped_corr
    
    return selected.tolist(), dropped_nzv, []

def clean_dataframe(df, extra_drop_cols=None, mapper=None):
    """
    Clean the dataframe: drop non-numeric, handle Inf/NaN, and optionally map columns.
    """
    # 0. Deduplicate columns first (common in CIC-IDS2017)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 1. Mapping
    if mapper:
        df = df.rename(columns=mapper)

    # Standardize column names (strip)
    df.columns = df.columns.str.strip()

    # Columns to drop (identifiers, timestamps) - Canonical drop list
    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
        'Protocol', 'Timestamp', 'SimillarHTTP', 'Inbound', 'Unnamed: 0',
        # CSE-IDS specific variations
        'Dst Port', 'Src IP', 'Src Port', 'Dst IP'
    ]
    
    actual_to_drop = [c for c in drop_cols if c in df.columns]
    if extra_drop_cols:
        for c in extra_drop_cols:
            if c in df.columns:
                actual_to_drop.append(c)
            elif re.match(r'^f\d+$', str(c)):
                idx = int(c[1:])
                if idx < len(df.columns):
                    actual_to_drop.append(df.columns[idx])
    
    # Drop columns if they exist (before numeric conversion)
    df = df.drop(columns=list(set(actual_to_drop)), errors='ignore')
    
    # Force numeric conversion for remaining columns (except Label) 
    # This handles mixed types loaded as objects
    # Use loop with iloc to avoid issues with potential duplicates/series mismatches
    for i, col in enumerate(df.columns):
        if col != 'Label':
            df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')
    
    # Replace Inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN (including those created by coerce)
    df = df.dropna()
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip().str.upper()
        # Remove header rows that might have leaked into the data
        df = df[df['Label'] != 'LABEL']
    
    # Replace Inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def get_benign_data(df):
    """
    Filter for Benign traffic only.
    Assumes 'Label' column exists.
    """
    if 'Label' in df.columns:
        # Label is already uppercase from clean_dataframe
        return df[df['Label'] == 'BENIGN'].drop(columns=['Label'], errors='ignore')
    return df

def fit_scaler_incrementally(files, scaler, extra_drop_cols=None):
    """
    Compute global Min/Max by iterating through files.
    """
    print("Computing global statistics for scaling...")
    for f in tqdm(files, desc="Fitting Scaler"):
        try:
            df = pd.read_csv(f)
            df = clean_dataframe(df, extra_drop_cols)
            df = get_benign_data(df) # Only fit on Benign data
            
            # Ensure only numeric columns remain
            df = df.select_dtypes(include=[np.number])
            
            if not df.empty:
                scaler.partial_fit(df.values)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return scaler

def create_sequences(data, seq_len, stride):
    """
    Create sliding window sequences.
    """
    sequences = []
    # If data is smaller than seq_len, skip
    if len(data) < seq_len:
        return np.array([])
        
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i : i + seq_len])
    return np.array(sequences)

def create_label_sequences(labels, seq_len, stride):
    """
    Create label sequences (taking the max/attack label in the window).
    """
    sequences = []
    if len(labels) < seq_len:
        return np.array([])
        
    for i in range(0, len(labels) - seq_len + 1, stride):
        window = labels[i : i + seq_len]
        # If any attack in window -> Attack (1)
        if np.any(window == 1):
            sequences.append(1)
        else:
            sequences.append(0)
    return np.array(sequences)

def process_and_save_shard(df, output_dir, scaler, seq_len, stride, mode, shard_id, extra_drop_cols=None, selected_features=None, master_columns=None):
    """
    Process a single dataframe and save as npz shard.
    """
    # 0. Mapping for consistency across datasets (especially for Test)
    if master_columns is not None:
        mapper = build_column_mapper(master_columns, df.columns.tolist())
        df = clean_dataframe(df, extra_drop_cols, mapper)
    else:
        df = clean_dataframe(df, extra_drop_cols)
    
    # 2. Handle Label
    labels = None
    if 'Label' in df.columns:
        if mode == 'train':
            # For training, we only want Benign
            df = df[df['Label'] == 'BENIGN']
        else:
            # For testing: 0 = Benign, 1 = Attack
            labels = (df['Label'] != 'BENIGN').astype(int).values
        
        # Drop Label column only after filtering/preserving labels
        df = df.drop(columns=['Label'], errors='ignore')
    
    # 3. Filter Features (Drop drift + NZV/Corr)
    if selected_features:
        # ENSURE all selected features are present and in the RIGHT ORDER
        # Fill missing with 0 to maintain shape consistency
        final_cols = []
        for feat in selected_features:
            if feat in df.columns:
                final_cols.append(feat)
            else:
                # Log or handle missing feature
                df[feat] = 0.0
                final_cols.append(feat)
        df = df[selected_features] # Use exact list for alignment
    else:
        # Fallback: Ensure only numeric columns
        df = df.select_dtypes(include=[np.number])

    if df.empty:
        return False
        
    # 4. Scale
    try:
        data_scaled = scaler.transform(df.values)
        # CRITICAL: Clip to [0, 1] to handle outliers in Test data
        data_scaled = np.clip(data_scaled, 0.0, 1.0)
    except ValueError as e:
        print(f"Skipping shard due to shape mismatch: {e}")
        return False
    
    # 4. Sequence
    X_seq = create_sequences(data_scaled, seq_len, stride)
    if len(X_seq) == 0:
        return False
        
    y_seq = None
    if labels is not None:
        y_seq = create_label_sequences(labels, seq_len, stride)
    else:
        # Default to 0 (Benign)
        y_seq = np.zeros(len(X_seq))
    
    # 5. Save Shard
    # Ensure X and y have same length
    min_len = min(len(X_seq), len(y_seq))
    X_seq = X_seq[:min_len]
    y_seq = y_seq[:min_len]
    
    if len(X_seq) > 0:
        shard_name = f"{mode}_shard_{shard_id}.npz"
        save_path = os.path.join(output_dir, shard_name)
        # Use 'X' (uppercase) to match typical loader expectation
        np.savez_compressed(save_path, X=X_seq, y=y_seq)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Generate SOTA Preprocessed Data (Sprint 3)")
    parser.add_argument("--raw_train", type=str, required=True, help="Path to raw training CSVs or folder")
    parser.add_argument("--raw_test", type=str, required=True, help="Path to raw test CSVs or folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--scaler_type", type=str, default="minmax", help="Scaler type (minmax or quantile)")
    parser.add_argument("--drop_features", type=str, default=None, help="Comma-separated list of features to drop or path to file")
    
    args = parser.parse_args()
    
    # Load drop features
    extra_drop_cols = []
    if args.drop_features:
        if os.path.exists(args.drop_features):
            print(f"Loading drop list from: {args.drop_features}")
            try:
                with open(args.drop_features, 'r', encoding='utf-8') as f:
                    extra_drop_cols = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading drop list file: {e}")
        else:
            print(f"Drop list file NOT FOUND at: {args.drop_features}. Treating as comma-separated list.")
            extra_drop_cols = [x.strip() for x in args.drop_features.split(',') if x.strip()]
        print(f"Dropping {len(extra_drop_cols)} features: {extra_drop_cols}")
    
    # Expand wildcards
    def get_files(path_pattern):
        if os.path.isdir(path_pattern):
            return sorted(glob.glob(os.path.join(path_pattern, "*.csv")))
        else:
            return sorted(glob.glob(path_pattern))

    train_files = get_files(args.raw_train)
    test_files = get_files(args.raw_test)
    
    print(f"Found {len(train_files)} training files.")
    print(f"Found {len(test_files)} test files.")
    
    if not train_files:
        print("No training files found! Check path.")
        return

    # 2. Fit Scaler & Select Features
    print("Collecting sample for feature selection...")
    sample_dfs = []
    total_samples = 0
    # Capture original columns for mapping references later
    master_columns = None
    
    for f in train_files[:10]: # Use first 10 files as sample
        df = pd.read_csv(f, low_memory=False)
        df = clean_dataframe(df, extra_drop_cols)
        if master_columns is None:
            master_columns = df.columns.tolist()
        
        df = get_benign_data(df)
        df = df.select_dtypes(include=[np.number])
        if not df.empty:
            sample_dfs.append(df.sample(min(len(df), 10000)))
            total_samples += len(sample_dfs[-1])
            if total_samples > 100000: break
    
    if not sample_dfs:
        print("Error: No data found for feature selection")
        return

    sample_comb = pd.concat(sample_dfs)
    selected_features, dr_nzv, dr_corr = select_features_by_statistics(sample_comb)
    print(f"Feature Selection: {len(selected_features)} selected. Dropped {len(dr_nzv)} NZV, {len(dr_corr)} Correlation.")
    
    if args.scaler_type == "quantile":
        scaler = QuantileTransformer(output_distribution="uniform", n_quantiles=1000, random_state=42)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        
    print("Fitting Scaler on selected features...")
    for f in tqdm(train_files, desc="Fitting Scaler"):
        try:
            df = pd.read_csv(f, low_memory=False)
            df = clean_dataframe(df, extra_drop_cols)
            df = get_benign_data(df)
            df = df[selected_features] # Use only selected
            if not df.empty:
                scaler.partial_fit(df.values)
        except Exception as e:
            print(f"Error fitting {f}: {e}")
            
    # Save Scaler and Feature List
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    with open(os.path.join(args.output_dir, "selected_features.txt"), 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"Selected features saved to selected_features.txt")
    
    # 3. Process Train Data (Benign Only)
    train_out = os.path.join(args.output_dir, "train")
    os.makedirs(train_out, exist_ok=True)
    
    shard_count = 0
    for f in tqdm(train_files, desc="Processing Train"):
        try:
            df = pd.read_csv(f, low_memory=False)
            if process_and_save_shard(df, train_out, scaler, args.seq_len, args.stride, 'train', shard_count, extra_drop_cols, selected_features, master_columns):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # 4. Process Test Data (Mixed)
    test_out = os.path.join(args.output_dir, "test")
    os.makedirs(test_out, exist_ok=True)
    
    shard_count = 0
    for f in tqdm(test_files, desc="Processing Test (CSE)"):
        try:
            df = pd.read_csv(f, low_memory=False)
            if process_and_save_shard(df, test_out, scaler, args.seq_len, args.stride, 'test', shard_count, extra_drop_cols, selected_features, master_columns):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # 5. Process CIC-IDS2017 Test Data (Mixed)
    test_cic_out = os.path.join(args.output_dir, "test_cic")
    os.makedirs(test_cic_out, exist_ok=True)
            
    shard_count = 0
    for f in tqdm(train_files, desc="Processing Test (CIC)"):
        try:
            df = pd.read_csv(f, low_memory=False)
            if process_and_save_shard(df, test_cic_out, scaler, args.seq_len, args.stride, 'test', shard_count, extra_drop_cols, selected_features, master_columns):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    print("Preprocessing Complete! Output saved to:", args.output_dir)

if __name__ == "__main__":
    main()
