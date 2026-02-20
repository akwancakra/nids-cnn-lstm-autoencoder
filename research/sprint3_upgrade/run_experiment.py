
import os
import yaml
import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

def run_command(cmd):
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run Sprint 3 Upgrade Experiment")
    parser.add_argument("--config", type=str, default="research/sprint3_upgrade/config/experiment_v1.yaml", help="Path to config file")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing step")
    args = parser.parse_args()
    
    # Load Config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"--- Running Experiment: {config['experiment_name']} ---")
    
    # 1. Preprocessing
    if not args.skip_preprocessing:
        print("\n[Phase 1] Preprocessing...")
        raw_train = PROJECT_ROOT / config['paths']['raw_train']
        raw_test = PROJECT_ROOT / config['paths']['raw_test']
        output_dir = PROJECT_ROOT / config['paths']['processed_data']
        
        # Check if raw data exists (mock check for now, user might need to adjust path)
        if not raw_train.exists():
            print(f"Warning: Raw train path {raw_train} does not exist. Please adjust config.")
            # return # Commented out to allow testing logic
            
        script_path = PROJECT_ROOT / "research/sprint3_upgrade/scripts/preprocess_sota.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--raw_train", str(raw_train),
            "--raw_test", str(raw_test),
            "--output_dir", str(output_dir),
            "--seq_len", str(config['preprocessing']['sequence_length']),
            "--stride", str(config['preprocessing']['stride'])
        ]
        
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Preprocessing failed: {e}")
            return

    # 2. Training (Using SOTA script logic)
    # We need to adapt the training script too. For now, let's create a placeholder
    # that points to the new training script we are about to create.
    print("\n[Phase 2] Training & Evaluation...")
    train_script = PROJECT_ROOT / "research/sprint3_upgrade/scripts/train_sota.py"
    
    cmd_train = [
        sys.executable, str(train_script),
        "--config", str(config_path)
    ]
    
    try:
        # Check if train script exists (we will create it next)
        if train_script.exists():
            run_command(cmd_train)
        else:
            print(f"Training script {train_script} not created yet.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
