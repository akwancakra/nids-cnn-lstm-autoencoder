
import os
import glob
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from tqdm import tqdm
import sys

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

def clean_dataframe(df, extra_drop_cols=None):
    """
    Clean the dataframe: drop non-numeric, handle Inf/NaN.
    """
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Columns to drop (identifiers, timestamps)
    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
        'Protocol', 'Timestamp', 'SimillarHTTP', 'Inbound', 'Unnamed: 0'
    ]
    
    if extra_drop_cols:
        drop_cols.extend(extra_drop_cols)
    
    # Drop existing columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Standardize Label column if it exists
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip().str.upper()
        # Remove header rows that might have leaked into the data
        df = df[df['Label'] != 'LABEL']
    
    # Replace Inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN
    df = df.dropna()
    
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

def process_and_save_shard(df, output_dir, scaler, seq_len, stride, mode, shard_id, extra_drop_cols=None, selected_features=None):
    """
    Process a single dataframe and save as npz shard.
    """
    # 1. Clean
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
        # Ensure selected_features exist in df
        avail = [f for f in selected_features if f in df.columns]
        df = df[avail]
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
            with open(args.drop_features, 'r') as f:
                extra_drop_cols = [line.strip() for line in f if line.strip()]
        else:
            extra_drop_cols = [x.strip() for x in args.drop_features.split(',')]
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
    for f in train_files[:10]: # Use first 10 files as sample
        df = pd.read_csv(f)
        df = clean_dataframe(df, extra_drop_cols)
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
        
    # Fit scaler on selected features only
    print("Fitting Scaler on selected features...")
    for f in tqdm(train_files, desc="Fitting Scaler"):
        try:
            df = pd.read_csv(f)
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
        df = pd.read_csv(f)
        # process_and_save_shard needs to know about selected_features
        if process_and_save_shard(df, train_out, scaler, args.seq_len, args.stride, 'train', shard_count, extra_drop_cols, selected_features):
            shard_count += 1
            
    # 4. Process Test Data (Mixed)
    test_out = os.path.join(args.output_dir, "test")
    os.makedirs(test_out, exist_ok=True)
    
    shard_count = 0
    for f in tqdm(test_files, desc="Processing Test (CSE)"):
        try:
            df = pd.read_csv(f)
            if process_and_save_shard(df, test_out, scaler, args.seq_len, args.stride, 'test', shard_count, extra_drop_cols, selected_features):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # 5. Process CIC-IDS2017 Test Data (Mixed)
    test_cic_out = os.path.join(args.output_dir, "test_cic")
    os.makedirs(test_cic_out, exist_ok=True)
    
    shard_count = 0
    for f in tqdm(train_files, desc="Processing Test (CIC)"):
        try:
            df = pd.read_csv(f)
            if process_and_save_shard(df, test_cic_out, scaler, args.seq_len, args.stride, 'test', shard_count, extra_drop_cols, selected_features):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    print("Preprocessing Complete! Output saved to:", args.output_dir)

if __name__ == "__main__":
    main()
