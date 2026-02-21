
import os
import glob
import argparse
import pandas as pd
import numpy as np
import json
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

def clean_dataframe(df):
    """
    Same cleaning as preprocess_sota.py to ensure consistent columns.
    """
    df.columns = df.columns.str.strip()
    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
        'Protocol', 'Timestamp', 'SimillarHTTP', 'Inbound', 'Unnamed: 0'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip().str.upper()
        df = df[df['Label'] != 'LABEL'] # Remove header rows
        
    # Force numeric on all columns except Label
    for col in df.columns:
        if col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def get_benign_data(df):
    if 'Label' in df.columns:
        return df[df['Label'] == 'BENIGN'].drop(columns=['Label'], errors='ignore')
    return df.drop(columns=['Label'], errors='ignore')

def load_sample_data(path_pattern, limit_files=2, benign_only=True):
    files = sorted(glob.glob(os.path.join(path_pattern, "*.csv"))) if os.path.isdir(path_pattern) else sorted(glob.glob(path_pattern))
    
    dfs = []
    for f in files[:limit_files]:
        try:
            print(f"Loading {f}...")
            # low_memory=False to avoid DtypeWarning, but might use more RAM
            df = pd.read_csv(f, low_memory=False)
            df = clean_dataframe(df)
            if benign_only:
                df = get_benign_data(df)
            
            # Ensure we actually have data
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Analyze Feature Drift (Covariate Shift)")
    parser.add_argument("--raw_train", type=str, required=True, help="Path to 2017 (Source) Data")
    parser.add_argument("--raw_test", type=str, required=True, help="Path to 2018 (Target) Data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for report")
    parser.add_argument("--ks_threshold", type=float, default=0.5, help="KS Statistic threshold (0.0 - 1.0). Higher means more drift.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading Source Data (2017)...")
    df_source = load_sample_data(args.raw_train, limit_files=3, benign_only=True)
    
    print("Loading Target Data (2018)...")
    # For target, we ideally want to compare Benign vs Benign to isolate covariate shift
    df_target = load_sample_data(args.raw_test, limit_files=3, benign_only=True)
    
    if df_source.empty or df_target.empty:
        print("Error: Could not load data. Check paths or file contents.")
        return

    # Align columns
    common_cols = [c for c in df_source.columns if c in df_target.columns]
    
    # Filter only numeric columns that are actually numeric in both
    valid_cols = []
    for c in common_cols:
        if pd.api.types.is_numeric_dtype(df_source[c]) and pd.api.types.is_numeric_dtype(df_target[c]):
            valid_cols.append(c)
            
    df_source = df_source[valid_cols]
    df_target = df_target[valid_cols]
    
    print(f"Analyzing {len(valid_cols)} common numeric features...")
    
    # 2. Compute Drift
    drift_results = []
    high_drift_features = []
    
    for col in valid_cols:
        src_vals = df_source[col].values
        tgt_vals = df_target[col].values
        
        try:
            # KS Test
            ks_stat, p_value = ks_2samp(src_vals, tgt_vals)
            
            # Simple Stats
            src_mean = np.mean(src_vals)
            tgt_mean = np.mean(tgt_vals)
            mean_diff = abs(src_mean - tgt_mean)
            
            drift_results.append({
                "feature": col,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "src_mean": src_mean,
                "tgt_mean": tgt_mean,
                "mean_diff": mean_diff
            })
            
            if ks_stat > args.ks_threshold:
                high_drift_features.append(col)
        except Exception as e:
            print(f"Error calculating drift for {col}: {e}")
            
    # 3. Save Report
    if not drift_results:
        print("No drift results generated.")
        return
        
    drift_df = pd.DataFrame(drift_results).sort_values(by="ks_stat", ascending=False)
    report_path = os.path.join(args.output_dir, "drift_report.csv")
    drift_df.to_csv(report_path, index=False)
    print(f"Drift report saved to {report_path}")
    
    # Save High Drift Features List
    drop_list_path = os.path.join(args.output_dir, "high_drift_features.txt")
    with open(drop_list_path, "w") as f:
        for feat in high_drift_features:
            f.write(f"{feat}\n")
            
    print(f"Identified {len(high_drift_features)} features with KS > {args.ks_threshold}")
    print(f"Drop list saved to {drop_list_path}")
    
    # 4. Plot Top 5 Drifting Features
    top_5 = drift_df.head(5)['feature'].tolist()
    if top_5:
        try:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(top_5):
                plt.subplot(2, 3, i+1)
                sns.kdeplot(df_source[col], label='Source (2017)', fill=True, alpha=0.3)
                sns.kdeplot(df_target[col], label='Target (2018)', fill=True, alpha=0.3)
                plt.title(f"{col} (KS={drift_df[drift_df['feature']==col]['ks_stat'].values[0]:.2f})")
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "top_drift_features.png"))
        except Exception as e:
            print(f"Error plotting: {e}")

if __name__ == "__main__":
    main()
