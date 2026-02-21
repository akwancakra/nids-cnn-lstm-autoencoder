
import os
import glob
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import sys

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

def fit_scaler_incrementally(files, scaler):
    """
    Compute global Min/Max by iterating through files.
    """
    print("Computing global statistics for scaling...")
    for f in tqdm(files, desc="Fitting Scaler"):
        try:
            df = pd.read_csv(f)
            df = clean_dataframe(df)
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

def process_and_save_shard(df, output_dir, scaler, seq_len, stride, mode, shard_id, extra_drop_cols=None):
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
            # Drop Label
            df = df.drop(columns=['Label'], errors='ignore')
        else:
            # For testing: 0 = Benign, 1 = Attack
            labels = (df['Label'] != 'BENIGN').astype(int).values
            df = df.drop(columns=['Label'], errors='ignore')
    
    # Ensure only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    if df.empty:
        return False
        
    # 3. Scale
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
    
    args = parser.parse_args()
    
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

    # 2. Fit Scaler (MinMax 0-1)
    # Use GlobalScaler logic: Fit on Benign Training Data ONLY
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = fit_scaler_incrementally(train_files, scaler, extra_drop_cols)
    
    # Save Scaler
    os.makedirs(args.output_dir, exist_ok=True)
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 3. Process Train Data (Benign Only)
    train_out = os.path.join(args.output_dir, "train")
    os.makedirs(train_out, exist_ok=True)
    
    shard_count = 0
    for f in tqdm(train_files, desc="Processing Train Data"):
        try:
            df = pd.read_csv(f)
            if process_and_save_shard(df, train_out, scaler, args.seq_len, args.stride, 'train', shard_count, extra_drop_cols):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # 4. Process Test Data (Mixed)
    test_out = os.path.join(args.output_dir, "test")
    os.makedirs(test_out, exist_ok=True)
    
    shard_count = 0
    for f in tqdm(test_files, desc="Processing Test Data (CSE-CIC-IDS2018)"):
        try:
            df = pd.read_csv(f)
            # Use 'test' mode to preserve Attack labels
            if process_and_save_shard(df, test_out, scaler, args.seq_len, args.stride, 'test', shard_count, extra_drop_cols):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # 5. Process CIC-IDS2017 Test Data (Mixed)
    test_cic_out = os.path.join(args.output_dir, "test_cic")
    os.makedirs(test_cic_out, exist_ok=True)
    
    shard_count = 0
    for f in tqdm(train_files, desc="Processing Test Data (CIC-IDS2017)"):
        try:
            df = pd.read_csv(f)
            # Use 'test' mode to preserve Attack labels from CIC-IDS2017
            if process_and_save_shard(df, test_cic_out, scaler, args.seq_len, args.stride, 'test', shard_count, extra_drop_cols):
                shard_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    print("Preprocessing Complete! Output saved to:", args.output_dir)

if __name__ == "__main__":
    main()
