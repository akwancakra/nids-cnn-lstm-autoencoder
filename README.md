# NIDS CNN-LSTM Autoencoder

> **Hybrid CNN-LSTM Autoencoder for Zero-Day Anomaly Detection in Network-based Intrusion Detection Systems**

A thesis project implementing an unsupervised deep learning approach for detecting unknown network attacks using hybrid convolutional and recurrent autoencoder architecture with cross-dataset validation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Background](#research-background)
- [System Requirements](#system-requirements)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results & Outputs](#results--outputs)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **Hybrid CNN-LSTM Autoencoder** for network intrusion detection with zero-day attack detection capability. The system is trained on normal network traffic only (unsupervised learning) and detects anomalies through reconstruction error analysis.

### Key Features

- ✅ **Unsupervised Anomaly Detection**: Trained only on benign traffic
- ✅ **Hybrid Architecture**: Combines CNN (spatial patterns) + LSTM (temporal dependencies)
- ✅ **Cross-Dataset Validation**: Trained on CIC-IDS2017, validated on CSE-CIC-IDS2018
- ✅ **Zero-Shot & Few-Shot Evaluation**: Configurable cross-dataset modes in one pipeline
- ✅ **Adaptive Thresholding**: Percentile/source-target recalibration/gaussian modes
- ✅ **Zero-Day Capability**: Detects previously unseen attack patterns
- ✅ **Scalable Processing**: Sharded preprocessing for large datasets (>10M rows)
- ✅ **Configurable Robust Preprocessing**: RobustScaler + optional statistical feature filtering
- ✅ **Baseline Comparison**: LSTM Autoencoder + Traditional ML methods
- ✅ **Reproducible Research**: Fixed seeds, versioned dependencies, documented pipeline

### Research Objectives

1. Implement hybrid CNN-LSTM Autoencoder for unsupervised anomaly detection
2. Achieve **F1-score ≥ 90%** on CIC-IDS2017 (in-distribution)
3. Achieve **Accuracy ≥ 90%** on CSE-CIC-IDS2018 (out-of-distribution)
4. Maintain **generalization gap ≤ 15%** between datasets
5. Compare performance against baseline methods with statistical significance tests

---

## 🔬 Research Background

### Problem Statement

Traditional signature-based IDS cannot detect **zero-day attacks** (previously unknown threats). Machine learning approaches often struggle with **generalization** to new attack patterns not seen during training.

### Proposed Solution

An **unsupervised hybrid deep learning approach** that:

- Learns patterns from normal traffic only (no attack signatures needed)
- Combines CNN for feature extraction and LSTM for temporal modeling
- Uses reconstruction error as anomaly indicator
- Validates generalization across different datasets simulating real-world deployment

### Datasets Used

| Dataset             | Records | Features | Days | Attack Types                                          | Usage                           |
| ------------------- | ------- | -------- | ---- | ----------------------------------------------------- | ------------------------------- |
| **CIC-IDS2017**     | ~2.7M   | 79 columns total (78 predictor features + Label) | 5    | DoS/DDoS, Web Attacks, Infiltration, Botnet, PortScan | Training & In-distribution Test |
| **CSE-CIC-IDS2018** | ~16.2M  | 80 columns total (79 predictor features + Label) | 10   | Heartbleed, Infiltration, DoS/DDoS-LOIC-HTTP          | Cross-dataset Validation        |

---

## 💻 System Requirements

### Hardware

- **CPU**: Multi-core processor (minimum 4 cores, 8+ cores recommended)
- **RAM**:
  - Minimum: 8 GB
  - Recommended: 16 GB
  - For full datasets: 32 GB
- **Disk Space**: Minimum 15 GB free space
  - CIC-IDS2017: ~2-3 GB (raw CSV)
  - CSE-CIC-IDS2018: ~5-7 GB (raw CSV)
  - Processed data: ~3-5 GB
  - Models & results: ~2-3 GB
- **GPU** (Optional but Recommended):
  - NVIDIA GPU with CUDA support
  - Minimum 4 GB VRAM (8 GB+ for faster training)
  - Reduces training time from hours to minutes

### Software Requirements

| Component            | Version                                            | Notes                         |
| -------------------- | -------------------------------------------------- | ----------------------------- |
| **Operating System** | Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.14+ | Any modern OS                 |
| **Python**           | **3.10** (recommended)                             | Also supports 3.8, 3.9, 3.11  |
| **pip**              | Latest                                             | Comes with Python             |
| **UV**               | Latest (optional)                                  | 10-100x faster than pip       |
| **CUDA Toolkit**     | 11.2+ (if using GPU)                               | Match with TensorFlow version |
| **cuDNN**            | 8.1+ (if using GPU)                                | Match with CUDA version       |

> 💡 **Recommendation**: Use **Python 3.10** for best compatibility with TensorFlow and all dependencies.

---

## 🔧 Prerequisites

### 1. Python Installation

Verify Python is installed:

```bash
python --version  # Windows
python3 --version  # Linux/macOS
```

If not installed, download from [python.org](https://www.python.org/downloads/) or use a package manager:

```bash
# Windows (using Chocolatey)
choco install python310

# Linux (Ubuntu/Debian)
sudo apt install python3.10 python3.10-venv

# macOS (using Homebrew)
brew install python@3.10
```

### 2. Dataset Files

Download the required datasets:

#### CIC-IDS2017

- **Source**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Direct CSV Download**: [CIC-IDS-2017 CSVs](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/)
- **Format**: CSV files (preprocessed from PCAP)
- **Size**: ~2.5 GB compressed
- **Files needed**:
  - `Monday-WorkingHours.csv`
  - `Tuesday-WorkingHours.pcap_ISCX.csv`
  - `Wednesday-workingHours.pcap_ISCX.csv`
  - `Thursday-WorkingHours-Morning-WebAttacks.csv`
  - `Thursday-WorkingHours-Afternoon-Infilteration.csv`
  - `Friday-WorkingHours-Morning.csv`
  - `Friday-WorkingHours-Afternoon-PortScan.csv`
  - `Friday-WorkingHours-Afternoon-DDos.csv`

#### CSE-CIC-IDS2018

- **Source**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html)
- **Format**: CSV files (preprocessed from PCAP)
- **Size**: ~6 GB compressed
- **Files needed**: All CSV files from February 14-16, 2018

### 3. GPU Setup (Optional)

For GPU acceleration, install CUDA and cuDNN:

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Install CUDA Toolkit and cuDNN
# Follow instructions at: https://www.tensorflow.org/install/gpu
```

### 4. Optional Tools

- **Git**: For version control
- **Jupyter**: For exploratory notebooks (`pip install jupyter`)
- **VS Code**: Recommended IDE with Python extension

---

## 📦 Installation

### Option 1: Using UV (Recommended - 10-100x Faster)

[UV](https://github.com/astral-sh/uv) is a blazing-fast Python package installer that can also manage Python versions.

```bash
# Install UV (one-time setup)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.10 (auto-installs Python if needed)
uv venv --python 3.10

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies (super fast!)
uv pip install -r requirements.txt
```

### Option 2: Using Standard venv + pip

```bash
# Create virtual environment with Python 3.10
python3.10 -m venv .venv

# OR use default Python version
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check Python version
python --version  # Should show Python 3.10.x

# Check TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Check GPU availability (if using GPU)
python -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"
```

---

## 📂 Dataset Setup

### Step 1: Download Datasets

Download both datasets from the Canadian Institute for Cybersecurity:

- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)

### Step 2: Extract Files

Extract all CSV files to the appropriate directories:

```
data/raw/
├── CIC-IDS2017/
│   ├── Monday-WorkingHours.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.csv
│   ├── Friday-WorkingHours-Morning.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.csv
│   └── Friday-WorkingHours-Afternoon-DDos.csv
└── CSE-CIC-IDS2018/
    ├── 02-14-2018.csv
    ├── 02-15-2018.csv
    └── 02-16-2018.csv
```

### Step 3: Verify Files

```bash
# Check if files exist
python -c "from scripts.utils import list_csv_files; print(f'CIC-IDS2017: {len(list_csv_files(\"data/raw/CIC-IDS2017\"))} files'); print(f'CSE-CIC-IDS2018: {len(list_csv_files(\"data/raw/CSE-CIC-IDS2018\"))} files')"
```

Expected output:

```
CIC-IDS2017: 8 files
CSE-CIC-IDS2018: 3 files
```

---

## 📁 Project Structure

```
nids-cnn-lstm-autoencoder/
├── config.yaml                    # Legacy compatibility config entrypoint
├── config/
│   ├── base.yaml                  # Shared defaults
│   └── profiles/                  # Scenario-specific overrides
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── TASKS.md                       # Implementation checklist
├── KNOWLEDGE.md                   # Project knowledge base
├── SKILL.md                       # Academic research agent skill
│
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets (not in git)
│   │   ├── CIC-IDS2017/          # CIC-IDS2017 CSV files
│   │   └── CSE-CIC-IDS2018/      # CSE-CIC-IDS2018 CSV files
│   └── processed/                 # Processed data (not in git)
│       ├── shards/               # Sharded data for streaming
│       │   ├── cic/train/*.npz + manifest.json
│       │   ├── cic/val/*.npz + manifest.json
│       │   ├── cic/test/*.npz + manifest.json
│       │   └── cse/test/*.npz + manifest.json
│       ├── scaler.pkl            # Fitted scaler (standard/minmax/robust)
│       └── feature_columns.json  # Final aligned feature list
│       └── feature_filter_report.json  # Optional feature filtering report
│
├── scripts/                       # Python scripts
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── train_cnn_lstm_ae.py      # Train hybrid CNN-LSTM Autoencoder
│   ├── train_lstm_ae.py          # Train baseline LSTM Autoencoder
│   ├── thresholding.py           # Compute optimal threshold
│   ├── eval_metrics.py           # Evaluation & metrics
│   ├── cross_dataset_eval.py     # Cross-dataset validation
│   └── utils.py                  # Utility functions
│
├── models/                        # Saved models (not in git)
│   ├── cnn_lstm_ae/              # Hybrid CNN-LSTM Autoencoder
│   │   ├── best_model.keras      # Best model weights
│   │   └── final_model.keras     # Final model after training
│   └── lstm_ae/                  # Baseline LSTM Autoencoder
│       ├── best_model.keras
│       └── final_model.keras
│
├── notebooks/                     # Jupyter notebooks
│   ├── colab_ready.ipynb         # Colab/local-ready experiment notebook
│   └── colab_ready_online.ipynb  # Colab online-focused notebook variant
│
├── tests/                         # Unit tests
│   ├── test_preprocess_features.py
│   └── test_threshold_modes.py
│
├── docs/                          # Supporting technical docs
│   ├── CHANGELOG_EXPERIMENTS.md  # Experiment changelog
│   └── RELEASE_NOTES_v1.0.0.md   # Previous release notes
│
├── results/                       # Experiment results (not in git)
│   ├── metrics/                  # Evaluation metrics (JSON)
│   ├── plots/                    # Figures (ROC, confusion matrix, etc.)
│   └── logs/                     # Training logs
│
├── reports/                       # Reports and documentation
│   ├── figures/                  # Publication-ready figures
│   ├── tables/                   # Result tables
│   └── drafts/                   # Draft writings
│
└── my-papers/                     # Thesis writing materials
    ├── isi-bab-1.md              # Chapter 1 (Introduction)
    ├── isi-bab-2.md              # Chapter 2 (Literature Review)
    ├── isi-bab-3.md              # Chapter 3 (Methodology)
    ├── isi-daftar-pustaka.md     # References
    └── ...                       # Other thesis materials
```

---

## ⚙️ Configuration

Main setup now uses:

- `config/base.yaml` for shared defaults
- `config/profiles/*.yaml` for scenario-specific deltas
- `research/sprint2/run_registry.yaml` for run selection and isolation policy

Legacy `config.yaml` is still supported by scripts for backward compatibility.

Key sections:

### Data Paths

```yaml
paths:
  data_raw_cic: data/raw/CIC-IDS2017
  data_raw_cse: data/raw/CSE-CIC-IDS2018
  data_processed: data/processed
  models_dir: models
  results_dir: results
```

### Preprocessing Settings

```yaml
preprocess:
  window_size: 10 # Time window length
  stride: 1 # Window stride
  scaler: standard # standard|minmax|robust
  scaler_fit_mode: auto # auto|stream_partial|full_benign
  benign_label: BENIGN # Normal traffic label
  test_size: 0.15 # Test split ratio
  val_size: 0.15 # Validation split ratio
  random_seed: 42 # Reproducibility
  shard_enable: true # Enable sharded processing
  shard_size: 50000 # Rows per shard
  feature_filter:
    enable_nzv: false
    nzv_threshold: 1e-6
    enable_corr: false
    corr_threshold: 0.95
    sample_rows_limit: 100000
```

### Training Hyperparameters

```yaml
training:
  batch_size: 128
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
  restore_best_weights: true
  lr_scheduler: reduce_on_plateau # reduce_on_plateau|cosine|none
  clipnorm: null # set numeric value (e.g., 1.0) to enable
  dropout: 0.2
  cnn_filters: [64, 128] # CNN filter sizes
  cnn_kernel_size: 3
  lstm_units: [128, 64] # LSTM hidden units
  latent_dim: 32 # Bottleneck dimension
```

### Threshold & Evaluation

```yaml
threshold:
  percentile: 99 # 99th percentile for anomaly threshold

evaluation:
  batch_size: 256
  sample_size: 200000 # Max samples for evaluation
  mode: zero_shot # zero_shot|few_shot
  threshold_method: percentile # percentile|target_percentile|target_gaussian
  threshold_k_sigma: 2.5
  few_shot_benign_frac: 0.01
  few_shot_max_samples: 50000
  few_shot_finetune_epochs: 0
  few_shot_finetune_lr: 0.0005
  few_shot_finetune_batch_size: 256
```

To modify settings, edit `config.yaml` directly.

---

## 🚀 Usage

### Complete Workflow

### Sprint2 Isolated Runner (Recommended for research experiments)

```bash
# Dry-run all active Sprint2 runs (no execution, generate per-run configs)
python scripts/research_sprint2.py --registry research/sprint2/run_registry.yaml --base-config config/base.yaml --dry-run

# Execute active runs with isolated outputs per run_id
python scripts/research_sprint2.py --registry research/sprint2/run_registry.yaml --base-config config/base.yaml
```

#### 1. Data Preprocessing

Process raw CSV files into windowed sequences:

```bash
python scripts/preprocess.py --config config.yaml
```

**What it does**:

- Loads CIC-IDS2017 CSV files
- Drops non-numeric columns (IP addresses, timestamps)
- Handles NaN/Inf values
- Aligns features between datasets
- Harmonizes CSE-CIC-IDS2018 column names to CIC-IDS2017 schema before feature intersection
- (Optional) Applies statistical feature filtering (near-zero variance + high correlation)
- Splits into train/val/test (only BENIGN for train/val)
- Applies configured scaler (`standard|minmax|robust`) fit on source benign train only
- Scaler fit mode:
  - `auto`: use `partial_fit` when available, otherwise fallback to full BENIGN fit
  - `stream_partial`: force streaming partial fit (scalers that support it)
  - `full_benign`: fit once on all BENIGN rows from CIC-IDS2017 source data
- Creates windowed sequences (default: 10 timesteps)
- Saves sharded NPZ files for memory-efficient loading

**Outputs**:

- `data/processed/shards/` - Sharded data files
- `data/processed/scaler.pkl` - Fitted scaler
- `data/processed/feature_columns.json` - Final aligned feature list + label column
- `data/processed/feature_filter_report.json` - Feature filtering report (if enabled)

**Time**: ~10-30 minutes depending on hardware

#### 2. Train Hybrid CNN-LSTM Autoencoder

Train the main model:

```bash
python scripts/train_cnn_lstm_ae.py --config config.yaml
```

**What it does**:

- Loads preprocessed training data (BENIGN only)
- Builds hybrid CNN-LSTM Autoencoder
- Trains with early stopping and checkpointing
- Saves best model and training history

**Outputs**:

- `models/cnn_lstm_ae/best_model.keras` - Best model weights
- `models/cnn_lstm_ae/final_model.keras` - Final model
- `results/logs/cnn_lstm_history.json` - Loss curves
- `results/logs/config_snapshot.json` - Config snapshot used in run

**Time**: ~1-3 hours (CPU) / ~15-30 minutes (GPU)

#### 3. Compute Threshold (Optional Utility Script)

Determine anomaly detection threshold:

```bash
python scripts/thresholding.py \
  --config config.yaml \
  --model models/cnn_lstm_ae/best_model.keras
```

**What it does**:

- Computes reconstruction errors on validation set (BENIGN)
- Computes percentile-based threshold from `config.yaml` (`threshold.percentile`)
- Prints threshold value to stdout

> Note: Main evaluation flow (`scripts/eval_metrics.py`) computes threshold internally using the configured evaluation mode and threshold method.

**Outputs**:

- Console output only (threshold is not saved by this script)

**Time**: ~5-10 minutes

#### 4. Evaluate (CIC + CSE) - Zero-Shot or Few-Shot

Evaluate on test set from same dataset:

```bash
python scripts/eval_metrics.py \
  --config config.yaml \
  --model models/cnn_lstm_ae/best_model.keras \
  --tag cnn_lstm
```

**What it does**:

- Reads evaluation strategy from `config.yaml`:
  - `evaluation.mode=zero_shot` or `few_shot`
  - `evaluation.threshold_method=percentile|target_percentile|target_gaussian`
- Computes threshold according to selected method
- For `few_shot`: samples benign target windows and can run lightweight unsupervised fine-tuning (optional)
- Evaluates both CIC test and CSE test in one run
- Calculates metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Generates plots (ROC curve, confusion matrix)

**Outputs**:

- `results/metrics/cnn_lstm_cic_metrics.json` - CIC metrics
- `results/metrics/cnn_lstm_cse_metrics.json` - CSE metrics
- `results/metrics/cnn_lstm_generalization_gap.json` - Gap analysis
- `results/plots/cnn_lstm/roc_cic.png` - ROC curve (CIC)
- `results/plots/cnn_lstm/roc_cse.png` - ROC curve (CSE)
- `results/plots/cnn_lstm/cm_cic.png` - Confusion matrix (CIC)
- `results/plots/cnn_lstm/cm_cse.png` - Confusion matrix (CSE)

##### Example: Run Both Modes with Separate Tags

```bash
# Zero-shot
python scripts/eval_metrics.py --config config.yaml --model models/cnn_lstm_ae/best_model.keras --tag cnn_lstm_zero_shot

# Few-shot (set evaluation.mode=few_shot in config first)
python scripts/eval_metrics.py --config config.yaml --model models/cnn_lstm_ae/best_model.keras --tag cnn_lstm_few_shot
```

#### 5. Cross-Dataset Evaluation Wrapper

Convenience wrapper around the same evaluator:

```bash
python scripts/cross_dataset_eval.py \
  --config config.yaml \
  --model models/cnn_lstm_ae/best_model.keras \
  --tag cnn_lstm_cross
```

**What it does**:

- Wrapper for `eval_metrics.py` (same behavior and arguments)
- Evaluates CIC + CSE together using the given `--tag`

**Outputs**:

- `results/metrics/cnn_lstm_cross_cic_metrics.json`
- `results/metrics/cnn_lstm_cross_cse_metrics.json`
- `results/metrics/cnn_lstm_cross_generalization_gap.json`

#### 6. Train & Evaluate Baselines

For comparison with baseline methods:

```bash
# LSTM Autoencoder (without CNN)
python scripts/train_lstm_ae.py --config config.yaml
python scripts/eval_metrics.py --config config.yaml --model models/lstm_ae/best_model.keras --tag lstm_ae

# Additional baselines can be added (Isolation Forest, Random Forest, etc.)
```

---

## 🏗️ Model Architecture

### Hybrid CNN-LSTM Autoencoder

```
Input: (batch, 10, n_features)
│
├─ ENCODER ────────────────────────────
│  ├─ Conv1D(64, k=3) + BN + ReLU + MaxPool(2)
│  ├─ Conv1D(128, k=3) + BN + ReLU + MaxPool(2)
│  ├─ LSTM(128, return_sequences=True) + Dropout(0.2)
│  ├─ LSTM(64, return_sequences=False) + Dropout(0.2)
│  └─ Dense(32, relu)  [Latent/Bottleneck]
│
├─ DECODER ────────────────────────────
│  ├─ RepeatVector(10)
│  ├─ LSTM(64, return_sequences=True) + Dropout(0.2)
│  ├─ LSTM(128, return_sequences=True) + Dropout(0.2)
│  └─ TimeDistributed(Dense(n_features))
│
Output: (batch, 10, n_features)
```

### Architecture Components

| Layer Type             | Purpose                                             | Parameters                 |
| ---------------------- | --------------------------------------------------- | -------------------------- |
| **Conv1D**             | Extract spatial features from time windows          | filters=[64,128], kernel=3 |
| **BatchNorm**          | Stabilize training, reduce internal covariate shift | -                          |
| **MaxPool1D**          | Downsample sequences, reduce dimensionality         | pool_size=2                |
| **LSTM**               | Capture temporal dependencies in sequences          | units=[128,64]             |
| **Dropout**            | Prevent overfitting, improve generalization         | rate=0.2                   |
| **Dense (Bottleneck)** | Compressed representation (latent space)            | units=32                   |
| **RepeatVector**       | Expand latent code for decoding                     | timesteps=10               |
| **TimeDistributed**    | Apply same dense layer to each timestep             | -                          |

### Loss Function

**Mean Squared Error (MSE)**:

```
Loss = (1/N) * Σ(X - X̂)²
```

Where:

- `X`: Original input sequence
- `X̂`: Reconstructed sequence
- `N`: Number of elements

**Anomaly Detection**:

- Normal traffic → Low reconstruction error
- Attack traffic → High reconstruction error
- Threshold (99th percentile) separates normal from anomalous

---

## 📊 Evaluation Metrics

### Classification Metrics

| Metric        | Formula                                         | Target   |
| ------------- | ----------------------------------------------- | -------- |
| **Accuracy**  | (TP + TN) / Total                               | ≥ 90%    |
| **Precision** | TP / (TP + FP)                                  | Maximize |
| **Recall**    | TP / (TP + FN)                                  | Maximize |
| **F1-Score**  | 2 × (Precision × Recall) / (Precision + Recall) | ≥ 90%    |
| **AUC-ROC**   | Area under ROC curve                            | ≥ 0.90   |
| **FPR**       | FP / (FP + TN)                                  | Minimize |

### Generalization Metrics

**Generalization Gap**:

```
Gap = |Metric_CIC - Metric_CSE|
```

Target: ≤ 15% for all metrics

### Statistical Significance

**McNemar's Test**:

- Compares model predictions with baseline
- Null hypothesis: No difference in performance
- p-value < 0.05 indicates significant difference

---

## 📈 Results & Outputs

### Directory Structure

```
results/
├── metrics/
│   ├── cnn_lstm_cic_metrics.json                # In-distribution results
│   ├── cnn_lstm_cse_metrics.json                # Cross-dataset results
│   ├── cnn_lstm_generalization_gap.json         # Gap analysis
│   ├── lstm_ae_cic_metrics.json                 # Baseline comparison (CIC)
│   ├── lstm_ae_cse_metrics.json                 # Baseline comparison (CSE)
│   └── lstm_ae_generalization_gap.json          # Baseline gap analysis
│
├── plots/
│   ├── cnn_lstm/                       # Figures for tag=cnn_lstm
│   │   ├── roc_cic.png
│   │   ├── roc_cse.png
│   │   ├── cm_cic.png
│   │   ├── cm_cse.png
│   │   ├── err_dist_cic.png
│   │   └── err_dist_cse.png
│   └── lstm_ae/                        # Figures for tag=lstm_ae
│
└── logs/
    ├── cnn_lstm_history.json
    ├── lstm_history.json
    └── config_snapshot.json
```

### Metrics JSON Format

```json
{
  "accuracy": 0.9234,
  "precision": 0.9156,
  "recall": 0.9312,
  "f1": 0.9234,
  "roc_auc": 0.9501,
  "fpr": 0.0823,
  "tp": 1000,
  "fp": 89,
  "tn": 9911,
  "fn": 74
}
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Problem**: Training crashes with OOM error

**Solutions**:

```yaml
# Edit config.yaml
training:
  batch_size: 64 # Reduce from 128

preprocess:
  shard_size: 25000 # Reduce from 50000
  sample_frac: 0.5 # Use 50% of data for testing
```

#### 2. CUDA Out of Memory (GPU)

**Problem**: GPU memory exhausted

**Solutions**:

```python
# Add to training script
import tensorflow as tf
tf.config.experimental.set_memory_growth(gpu, True)
```

Or force CPU:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/train_cnn_lstm_ae.py
```

#### 3. Dataset Files Not Found

**Problem**: `FileNotFoundError` for CSV files

**Solution**:

```bash
# Verify files exist
ls data/raw/CIC-IDS2017/
ls data/raw/CSE-CIC-IDS2018/

# Check config.yaml paths match actual structure
```

#### 4. Feature Mismatch Between Datasets

**Problem**: Different number of features in CIC vs CSE

**Solution**: Preprocessing harmonizes CSE column names to CIC schema first, then takes intersection. Check logs:

```bash
# Look for this in preprocessing output:
# "Feature intersection count: <number>"
```

#### 5. Slow Preprocessing

**Problem**: Preprocessing takes too long

**Solutions**:

```yaml
# Use sampling for quick testing
preprocess:
  sample_frac: 0.1 # Use 10% of data
  max_rows_per_file: 100000 # Limit rows per file
```

#### 6. scikit-learn Version Conflict

**Problem**: `ValueError` when loading scaler

**Solution**:

```bash
# Check scikit-learn version
pip show scikit-learn

# If mismatch, reinstall
pip install --force-reinstall scikit-learn==1.3.0
```

#### 7. Few-Shot Evaluation Requires Sharded Data

**Problem**: `ValueError: few_shot mode currently requires shard manifests`

**Solution**:

- Ensure preprocessing was run with shard mode enabled (`preprocess.shard_enable: true`)
- Re-run preprocessing if shard manifests are missing
- Verify files exist:
  - `data/processed/shards/cic/val/manifest.json`
  - `data/processed/shards/cse/test/manifest.json`

---

## 🤝 Contributing

This is a thesis project. For questions or collaboration:

1. Check [TASKS.md](TASKS.md) for implementation progress
2. Review [KNOWLEDGE.md](KNOWLEDGE.md) for project context
3. See [SKILL.md](SKILL.md) for academic research guidelines

---

## 📄 License

This project is for academic research purposes. Please cite appropriately if you use this work.

---

## 📚 Key Documentation Files

- **[TASKS.md](TASKS.md)**: Complete implementation checklist with 14 stages
- **[KNOWLEDGE.md](KNOWLEDGE.md)**: Project knowledge base, concepts, and targets
- **[SKILL.md](SKILL.md)**: Academic research agent skill for literature review
- **[config/base.yaml](config/base.yaml)**: Shared configuration defaults
- **[research/sprint2/run_registry.yaml](research/sprint2/run_registry.yaml)**: Isolated Sprint2 run definitions
- **[docs/CHANGELOG_EXPERIMENTS.md](docs/CHANGELOG_EXPERIMENTS.md)**: Experiment evolution and reproducibility notes

---

## 🎓 Academic Context

### Research Questions

1. Can a hybrid CNN-LSTM Autoencoder effectively detect zero-day attacks in enterprise NIDS?
2. How well does the model generalize from CIC-IDS2017 to CSE-CIC-IDS2018?
3. Does the hybrid architecture outperform baseline LSTM Autoencoder and traditional ML?
4. What is the trade-off between false positive rate and detection accuracy?

### Success Criteria

- [ ] F1-score ≥ 90% on CIC-IDS2017 (in-distribution)
- [ ] Accuracy ≥ 90% on CSE-CIC-IDS2018 (cross-dataset)
- [ ] Generalization gap ≤ 15%
- [ ] Statistical significance (McNemar's test, p < 0.05)

### Expected Contributions

1. Implementation of hybrid CNN-LSTM architecture for NIDS
2. Cross-dataset validation methodology for zero-day detection
3. Comparative analysis with established baselines
4. Open-source reproducible research codebase

---

## 📞 Contact & Support

For issues or questions related to this research:

1. Check [Troubleshooting](#troubleshooting) section
2. Review [KNOWLEDGE.md](KNOWLEDGE.md) for technical details
3. See [TASKS.md](TASKS.md) for implementation status

---

**Last Updated**: February 2026 (v1.1.0 experiment sync)  
**Project Status**: Active Development  
**Python Version**: 3.10 (Recommended)
