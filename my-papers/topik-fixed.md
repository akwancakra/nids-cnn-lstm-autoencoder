# Dokumentasi Penelitian Skripsi

## 📋 Informasi Dasar

**Judul:**

```
Implementasi Hybrid CNN-LSTM Autoencoder untuk Deteksi Anomali Zero-Day pada Network-based Intrusion Detection System (NIDS) Enterprise
```

**Program Studi:** Ilmu Komputer / Teknik Informatika  
**Jenjang:** S1  
**Tahun:** 2026

---

## 🎯 Latar Belakang Penelitian

### Masalah Utama

1. **Zero-Day Attacks:** Serangan yang belum dikenal signature-nya oleh sistem keamanan tradisional
2. **Signature-based IDS Limitation:** Tidak efektif mendeteksi serangan baru yang belum ada di database
3. **False Positive Rate:** IDS tradisional sering menghasilkan alarm palsu yang tinggi
4. **Kompleksitas Network Traffic:** Pattern serangan semakin kompleks dan sulit dideteksi

### Solusi yang Diusulkan

Menggunakan **Hybrid CNN-LSTM Autoencoder** untuk deteksi anomali berbasis behavior, yang dapat:

- Mendeteksi serangan zero-day tanpa memerlukan signature
- Belajar dari pola normal traffic untuk identifikasi anomali
- Mengurangi false positive dengan reconstruction error-based detection

---

## 🔬 Maksud dan Tujuan Penelitian

### Maksud Penelitian

Mengimplementasikan model deep learning berbasis Hybrid CNN-LSTM Autoencoder untuk meningkatkan kemampuan deteksi anomali pada Network-based Intrusion Detection System (NIDS), khususnya dalam mendeteksi serangan zero-day pada jaringan enterprise.

### Tujuan Penelitian

1. Mengimplementasikan arsitektur Hybrid CNN-LSTM Autoencoder untuk anomaly detection
2. Mengevaluasi performa model dalam mendeteksi berbagai jenis serangan di dataset CIC-IDS2017
3. **Melakukan cross-dataset validation menggunakan CSE-CIC-IDS2018 untuk membuktikan generalization capability**
4. Menganalisis efektivitas model dalam mendeteksi serangan zero-day attack melalui out-of-distribution testing
5. Membandingkan performa model dengan baseline methods (LSTM Autoencoder, Traditional ML)
6. Menghasilkan model NIDS yang dapat mendeteksi anomali dengan akurasi tinggi dan false positive rendah pada berbagai dataset

### Manfaat Penelitian

**Manfaat Akademik:**

- Kontribusi metode baru dalam anomaly-based IDS menggunakan deep learning
- Referensi untuk penelitian serupa di bidang cybersecurity dan AI
- Pemahaman mendalam tentang penerapan CNN-LSTM Autoencoder untuk time-series anomaly detection

**Manfaat Praktis:**

- Model NIDS yang dapat mendeteksi serangan zero-day secara efektif
- Solusi keamanan jaringan yang lebih adaptif dan intelligent
- Mengurangi ketergantungan pada signature-based detection

---

## 🏗️ Arsitektur Model

### Hybrid CNN-LSTM Autoencoder Architecture

```
INPUT LAYER (Network Traffic Features)
    ↓
┌─────────────────────────────────────────┐
│           ENCODER                       │
├─────────────────────────────────────────┤
│  1. 1D-CNN Layers (Feature Extraction)  │
│     - Conv1D (filters=64, kernel=3)     │
│     - BatchNormalization                │
│     - Activation (ReLU)                 │
│     - MaxPooling1D                      │
│     ↓                                   │
│     - Conv1D (filters=128, kernel=3)    │
│     - BatchNormalization                │
│     - Activation (ReLU)                 │
│     - MaxPooling1D                      │
│                                         │
│  2. LSTM Layers (Temporal Dependencies) │
│     - LSTM (units=128, return_seq=True) │
│     - Dropout (0.2)                     │
│     - LSTM (units=64, return_seq=False) │
│     - Dropout (0.2)                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│        LATENT SPACE (Bottleneck)        │
│     - Dense (units=32, activation=relu) │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│           DECODER                       │
├─────────────────────────────────────────┤
│  1. RepeatVector (reshape for LSTM)     │
│                                         │
│  2. LSTM Layers (Reconstruct Sequence)  │
│     - LSTM (units=64, return_seq=True)  │
│     - Dropout (0.2)                     │
│     - LSTM (units=128, return_seq=True) │
│     - Dropout (0.2)                     │
│                                         │
│  3. TimeDistributed Dense               │
│     - Dense (units=n_features)          │
└─────────────────────────────────────────┘
    ↓
OUTPUT LAYER (Reconstructed Features)
```

### Kenapa Hybrid CNN-LSTM?

| Component       | Fungsi                                        | Keuntungan                                                                                                          |
| --------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **1D-CNN**      | Extract spatial features dari network traffic | - Menangkap local patterns<br>- Mengurangi dimensi<br>- Feature extraction yang efisien                             |
| **LSTM**        | Capture temporal dependencies                 | - Menangkap sequential patterns<br>- Long-term memory<br>- Cocok untuk time-series data                             |
| **Autoencoder** | Unsupervised anomaly detection                | - Tidak butuh labeled data untuk training<br>- Deteksi via reconstruction error<br>- Adaptif terhadap serangan baru |

---

## 📊 Dataset: CIC-IDS2017

### Informasi Dataset

- **Sumber:** Canadian Institute for Cybersecurity (CIC)
- **Tahun:** 2017
- **Size:** ~2.7 million records
- **Features:** 78+ network traffic features
- **Duration:** 5 hari data collection (Monday - Friday)

### Jenis Serangan di Dataset

| Hari                     | Jenis Traffic/Attack                                |
| ------------------------ | --------------------------------------------------- |
| **Monday**               | Benign (Normal traffic only)                        |
| **Tuesday**              | FTP-Patator, SSH-Patator                            |
| **Wednesday**            | DoS/DDoS (Slowloris, Slowhttptest, Hulk, GoldenEye) |
| **Thursday (Morning)**   | Web Attack (Brute Force, XSS, SQL Injection)        |
| **Thursday (Afternoon)** | Infiltration                                        |
| **Friday (Morning)**     | Botnet ARES                                         |
| **Friday (Afternoon)**   | PortScan, DDoS LOIT                                 |

### Feature Categories

1. **Flow-based features:** Duration, packet length, packet rate
2. **Protocol features:** TCP flags, packet types
3. **Statistical features:** Mean, std, min, max dari packet size/interval
4. **Behavioral features:** Flow IAT (Inter-Arrival Time), active/idle time

---

## 📊 Dataset: CSE-CIC-IDS2018

### Informasi Dataset

- **Sumber:** Canadian Institute for Cybersecurity (CIC)
- **Tahun:** 2018
- **Size:** ~16.2 million records
- **Features:** 79-80 network traffic features
- **Duration:** 10 hari data collection

### Jenis Serangan di Dataset

| Hari               | Jenis Traffic/Attack                                                    |
| ------------------ | ----------------------------------------------------------------------- |
| **Wednesday 14/2** | Benign                                                                  |
| **Thursday 15/2**  | FTP-Patator, SSH-Patator                                                |
| **Friday 16/2**    | DoS attacks (Slowloris, Slowhttptest, DoS, Hulk, GoldenEye, Heartbleed) |
| **Monday 19/2**    | Benign                                                                  |
| **Tuesday 20/2**   | DDoS attacks (LOIC-UDP, LOIC-HTTP)                                      |
| **Wednesday 21/2** | DDoS attacks (LOIC-UDP)                                                 |
| **Thursday 22/2**  | Brute Force (Web, XSS), SQL Injection                                   |
| **Friday 23/2**    | Brute Force (Web, XSS), SQL Injection                                   |
| **Monday 26/2**    | Infiltration attack from inside network                                 |
| **Tuesday 27/2**   | Botnet attacks                                                          |

### Attack Categories

1. **Brute Force Attacks:** FTP-Patator, SSH-Patator, Web Brute Force
2. **DoS/DDoS Attacks:** Slowloris, Slowhttptest, Hulk, GoldenEye, Heartbleed, LOIC
3. **Web Attacks:** XSS, SQL Injection
4. **Infiltration:** Network penetration from inside
5. **Botnet:** ARES Botnet traffic

### Keuntungan CSE-CIC-IDS2018

- **Lebih comprehensive:** 10 hari vs 5 hari di CIC-IDS2017
- **Attack diversity lebih tinggi:** Include Heartbleed, Infiltration yang lebih sophisticated
- **Lebih realistis:** Network environment yang lebih kompleks
- **Temporal coverage:** Lebih banyak variasi waktu dan pattern

---

## 🔄 Cross-Dataset Validation Strategy

### Mengapa Cross-Dataset Validation?

**Research Gap yang Diaddress:**

> Most existing IDS research only validates on single dataset, lacking proof of generalization capability to unseen attack patterns and network environments.

**Contribution:**

- Membuktikan model **tidak overfitting** ke specific dataset
- Menguji **generalization ability** ke different network traffic distributions
- Simulate **real zero-day scenario**: Model trained on 2017 data detecting 2018 attacks
- Provide **stronger evidence** untuk production deployment readiness

### Experiment Design

#### **Experiment 1: In-Distribution Performance**

```
Training: CIC-IDS2017 (normal traffic only)
Testing: CIC-IDS2017 test set (70% train, 15% val, 15% test)
Purpose: Establish baseline performance
Expected Accuracy: 96-98%
```

#### **Experiment 2: Cross-Dataset Validation (Zero-Day Simulation)**

```
Training: CIC-IDS2017 (normal traffic only)
Testing: CSE-CIC-IDS2018 (full dataset)
Purpose: Test generalization to unseen attacks (zero-day simulation)
Expected Accuracy: 90-94% (lower but still good = proves generalization)
```

#### **Experiment 3: Verify Consistency**

```
Training: CSE-CIC-IDS2018 (normal traffic only)
Testing: CSE-CIC-IDS2018 test set
Purpose: Verify model works consistently on 2018 data
Expected Accuracy: 96-98%
```

#### **Experiment 4 (Optional): Combined Training**

```
Training: CIC-IDS2017 + CSE-CIC-IDS2018 (normal traffic from both)
Testing: Holdout test from both datasets
Purpose: Maximum robustness with diverse training data
Expected Accuracy: 97-99%
```

### Feature Alignment Strategy

**Challenge:** CIC-IDS2017 (78 features) vs CSE-CIC-IDS2018 (79-80 features)

**Solution:**

```python
# Step 1: Identify common features
common_features = list(set(ids2017_features) & set(ids2018_features))
print(f"Common features: {len(common_features)}")  # Expected: 75-76 features

# Step 2: Use only common features for both datasets
X_2017 = df_2017[common_features]
X_2018 = df_2018[common_features]

# Step 3: Apply same preprocessing pipeline
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_2017_scaled = scaler.fit_transform(X_2017)  # Fit on 2017
X_2018_scaled = scaler.transform(X_2018)      # Transform 2018 with same scaler
```

### Performance Metrics Comparison

| Metric    | CIC-IDS2017 (In-Dist) | CSE-CIC-IDS2018 (OOD) | Delta     | Interpretation        |
| --------- | --------------------- | --------------------- | --------- | --------------------- |
| Accuracy  | 96-98%                | 90-94%                | -4 to -6% | Good generalization   |
| Precision | 94-96%                | 88-92%                | -6%       | Acceptable drop       |
| Recall    | 93-95%                | 87-91%                | -6%       | Acceptable drop       |
| F1-Score  | 94-96%                | 88-92%                | -6%       | Acceptable drop       |
| AUC-ROC   | 0.97-0.99             | 0.92-0.96             | -0.05     | Strong generalization |

**Interpretation:**

- **4-6% accuracy drop** on OOD data = **EXCELLENT generalization**
- Most models drop 10-20% on cross-dataset validation
- Proves model learned **generalizable patterns**, not dataset-specific artifacts

---

## 🚀 Workflow Penelitian

### Phase 1: Preparation & Data Understanding (3-4 minggu)

**Week 1-2:**

```
✓ Download dataset CIC-IDS2017
✓ Download dataset CSE-CIC-IDS2018
✓ Setup environment (Python, TensorFlow/PyTorch, Jupyter)
✓ Exploratory Data Analysis (EDA) - CIC-IDS2017:
  - Statistik deskriptif
  - Distribusi kelas (Benign vs Attack types)
  - Missing values analysis
  - Feature correlation
  - Data visualization
```

**Week 3:**

```
✓ Exploratory Data Analysis (EDA) - CSE-CIC-IDS2018:
  - Statistik deskriptif
  - Distribusi kelas dan attack types
  - Missing values analysis
  - Feature comparison dengan CIC-IDS2017
  - Identify common features

✓ Cross-Dataset Analysis:
  - Feature overlap analysis
  - Distribution comparison
  - Attack category mapping
```

**Week 4:**

```
✓ Data Preprocessing (Both Datasets):
  - Handle missing values (drop/impute)
  - Remove duplicate records
  - Handle infinite values
  - Feature selection (common features only: ~75-76 features)
  - Label encoding untuk target variable
  - Ensure consistent preprocessing pipeline
```

**Deliverables:**

- `01_EDA.ipynb`: Notebook dengan analisis dataset lengkap
- `data_summary.pdf`: Dokumentasi karakteristik dataset

---

### Phase 2: Data Preparation (2 minggu)

**Week 5:**

```
✓ Feature Engineering (Both Datasets):
  - Normalization/Standardization (StandardScaler recommended)
  - Feature scaling per-column
  - Encode categorical features (jika ada)
  - Use common features only (75-76 features)

✓ Data Splitting Strategy:

  CIC-IDS2017:
  - Train set: 70% (normal traffic only untuk unsupervised learning)
  - Validation set: 15% (untuk tuning hyperparameters)
  - Test set: 15% (untuk in-distribution evaluation)

  CSE-CIC-IDS2018:
  - No training split (used entirely for cross-dataset validation)
  - Full dataset as test set (out-of-distribution evaluation)
  - Atau: 70/15/15 split jika mau train model kedua (Exp 3)

✓ Scaler Strategy:
  - Fit scaler ONLY on CIC-IDS2017 training set
  - Transform CSE-CIC-IDS2018 using SAME scaler (important for fair comparison)
```

**Week 6:**

```
✓ Reshape Data:
  - Convert ke format 3D untuk CNN-LSTM input
  - Shape: (samples, timesteps, features)
  - Timesteps = window size (misal: 10)
  - Apply same reshaping to both datasets

✓ Final Data Validation:
  - Verify shape consistency
  - Check for data leakage
  - Validate preprocessing pipeline
```

**Deliverables:**

- `02_Preprocessing.ipynb`: Script preprocessing lengkap
- `data/processed/`: Folder berisi data siap training
- `scaler.pkl`: Saved scaler object untuk inference

---

### Phase 3: Model Development (3-4 minggu)

**Week 5-6: Build Architecture**

```
✓ Implementasi Hybrid CNN-LSTM Autoencoder:
  - Define encoder (CNN → LSTM)
  - Define latent space
  - Define decoder (LSTM → Dense)

✓ Model Configuration:
  - Loss function: MSE (Mean Squared Error)
  - Optimizer: Adam (learning_rate=0.001)
  - Metrics: MAE (Mean Absolute Error)

✓ Callbacks Setup:
  - ModelCheckpoint (save best model)
  - EarlyStopping (patience=10)
  - ReduceLROnPlateau (monitor validation loss)
  - TensorBoard (logging)
```

**Week 7-8: Training & Tuning**

```
✓ Initial Training:
  - Epochs: 50-100
  - Batch size: 64 atau 128
  - Train dengan normal traffic only (unsupervised)

✓ Hyperparameter Tuning:
  - CNN filters: [32, 64, 128]
  - LSTM units: [64, 128, 256]
  - Latent dimension: [16, 32, 64]
  - Dropout rate: [0.2, 0.3, 0.5]
  - Learning rate: [0.001, 0.0001]

✓ Monitor Training:
  - Training loss vs validation loss
  - Check overfitting
  - Adjust hyperparameters
```

**Deliverables:**

- `03_Model_Building.ipynb`: Architecture definition
- `04_Training.ipynb`: Training process & logs
- `models/cnn_lstm_autoencoder.h5`: Trained model weights
- `training_history.json`: Training logs

---

### Phase 4: Evaluation & Testing (3-4 minggu)

**Week 9-10: In-Distribution Evaluation (CIC-IDS2017)**

```
✓ Reconstruction Error Analysis:
  - Calculate reconstruction error untuk train/test set
  - Plot distribution reconstruction error
  - Determine optimal threshold:
    * Percentile-based (e.g., 95th percentile)
    * ROC curve analysis

✓ Anomaly Detection:
  - Samples dengan reconstruction error > threshold = Anomaly
  - Classify: Normal vs Attack

✓ Performance Metrics (CIC-IDS2017 Test Set):
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curve & AUC Score
  - Per-class performance (untuk setiap attack type)
```

**Week 11: Cross-Dataset Evaluation (CSE-CIC-IDS2018)**

```
✓ Zero-Day Simulation:
  - Use model trained on CIC-IDS2017
  - Test on CSE-CIC-IDS2018 (never seen during training)
  - Use SAME threshold determined from 2017 data

✓ Out-of-Distribution Performance:
  - Calculate reconstruction errors on 2018 data
  - Classify using same threshold
  - Calculate all metrics
  - Per-attack-type analysis

✓ Generalization Analysis:
  - Compare 2017 vs 2018 performance
  - Analyze performance drop (expected 4-6%)
  - Identify which attacks generalize well vs poorly
  - Error analysis for cross-dataset failures
```

**Week 12: Baseline Comparison & Additional Experiments**

```
✓ Implement Baseline Models:
  1. LSTM Autoencoder (tanpa CNN)
  2. Random Forest atau SVM
  3. Test baselines on both datasets

✓ Compare Performance:
  - Side-by-side metrics comparison (2017 vs 2018)
  - Statistical significance testing
  - Inference time comparison
  - Generalization gap analysis (baseline vs CNN-LSTM)

✓ Optional Experiment 3 & 4:
  - Train on CSE-CIC-IDS2018, test on 2018
  - Train on combined data, test on both
```

**Deliverables:**

- `05_Evaluation.ipynb`: Testing & evaluation lengkap
- `results/metrics.json`: Semua metrics
- `results/figures/`: Confusion matrix, ROC curves, plots
- `comparison_report.pdf`: Perbandingan dengan baseline

---

### Phase 5: Analysis & Documentation (2 minggu)

**Week 12:**

```
✓ Results Analysis:
  - Interpretasi hasil deteksi
  - Attack types yang mudah/sulit dideteksi
  - Error analysis (false positives/negatives)
  - Feature importance (jika applicable)

✓ Visualization:
  - Confusion matrix heatmap
  - ROC curves
  - Reconstruction error distribution
  - Training history plots
```

**Week 13:**

```
✓ Documentation:
  - Clean up all notebooks
  - Add comprehensive comments
  - Write README.md
  - Prepare presentation slides

✓ Laporan Skripsi:
  - Bab 1: Pendahuluan
  - Bab 2: Tinjauan Pustaka
  - Bab 3: Metodologi Penelitian
  - Bab 4: Hasil dan Pembahasan
  - Bab 5: Kesimpulan dan Saran
```

**Deliverables:**

- All cleaned notebooks
- Complete documentation
- Presentation slides
- Draft laporan skripsi

---

## 📁 Struktur Project

```
skripsi-ids-cnn-lstm/
│
├── README.md                          # Project overview & setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
│
├── data/
│   ├── raw/                          # Dataset asli (tidak di-commit ke git)
│   │   ├── CIC-IDS2017/
│   │   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   │   ├── Wednesday-WorkingHours.pcap_ISCX.csv
│   │   │   ├── Thursday-WorkingHours-Morning.pcap_ISCX.csv
│   │   │   ├── Thursday-WorkingHours-Afternoon.pcap_ISCX.csv
│   │   │   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   │   │   └── Friday-WorkingHours-Afternoon.pcap_ISCX.csv
│   │   │
│   │   └── CSE-CIC-IDS2018/
│   │       ├── Processed Traffic Data for ML Algorithms/
│   │       │   ├── Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
│   │       │   ├── Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
│   │       │   └── ... (10 days total)
│   │
│   ├── processed/                    # Data setelah preprocessing
│   │   ├── CIC-IDS2017/
│   │   │   ├── X_train.npy          # Training features (70%)
│   │   │   ├── X_val.npy            # Validation features (15%)
│   │   │   ├── X_test.npy           # Test features (15%)
│   │   │   ├── y_train.npy          # Training labels
│   │   │   ├── y_val.npy            # Validation labels
│   │   │   └── y_test.npy           # Test labels
│   │   │
│   │   ├── CSE-CIC-IDS2018/
│   │   │   ├── X_full.npy           # Full dataset for cross-validation
│   │   │   ├── y_full.npy           # Full labels
│   │   │   ├── X_train.npy          # Optional: for Experiment 3
│   │   │   ├── X_val.npy
│   │   │   ├── X_test.npy
│   │   │   ├── y_train.npy
│   │   │   ├── y_val.npy
│   │   │   └── y_test.npy
│   │   │
│   │   └── common_features.txt      # List of common features used
│   │
│   └── README.md                     # Dataset description & download links
│
├── notebooks/
│   ├── 01_EDA_CIC-IDS2017.ipynb            # Exploratory Data Analysis - 2017
│   ├── 01_EDA_CSE-CIC-IDS2018.ipynb        # Exploratory Data Analysis - 2018
│   ├── 02_Cross_Dataset_Analysis.ipynb     # Feature comparison & alignment
│   ├── 03_Preprocessing.ipynb              # Data cleaning & transformation (both datasets)
│   ├── 04_Model_Building.ipynb             # Architecture definition
│   ├── 05_Training.ipynb                   # Model training
│   ├── 06_Evaluation_IDS2017.ipynb         # In-distribution evaluation
│   ├── 07_Cross_Dataset_Validation.ipynb   # Out-of-distribution evaluation (2018)
│   ├── 08_Baseline_Comparison.ipynb        # Compare dengan baseline models
│   ├── 09_Generalization_Analysis.ipynb    # Cross-dataset performance analysis
│   └── 10_Visualization.ipynb              # Advanced visualizations
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Data loading functions
│   │   ├── preprocessor.py          # Preprocessing pipeline
│   │   └── feature_engineer.py      # Feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm_autoencoder.py  # Main model architecture
│   │   ├── lstm_autoencoder.py      # Baseline LSTM Autoencoder
│   │   └── traditional_ml.py        # Baseline ML models
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop & callbacks
│   │   └── hyperparameter_tuning.py # Hyperparameter optimization
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Evaluation metrics
│   │   ├── threshold_finder.py      # Optimal threshold detection
│   │   └── visualizer.py            # Plotting functions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration parameters
│       └── helpers.py               # Helper functions
│
├── models/
│   ├── cnn_lstm_autoencoder_best.h5     # Best trained model
│   ├── cnn_lstm_autoencoder_final.h5    # Final model
│   ├── lstm_autoencoder_baseline.h5     # Baseline model
│   ├── scaler.pkl                       # Fitted scaler
│   └── training_history.json            # Training logs
│
├── results/
│   ├── figures/
│   │   ├── CIC-IDS2017/
│   │   │   ├── confusion_matrix.png
│   │   │   ├── roc_curve.png
│   │   │   ├── reconstruction_error_dist.png
│   │   │   └── per_class_performance.png
│   │   │
│   │   ├── CSE-CIC-IDS2018/
│   │   │   ├── confusion_matrix.png
│   │   │   ├── roc_curve.png
│   │   │   ├── reconstruction_error_dist.png
│   │   │   └── per_class_performance.png
│   │   │
│   │   ├── cross_dataset/
│   │   │   ├── performance_comparison.png
│   │   │   ├── generalization_gap.png
│   │   │   └── attack_type_comparison.png
│   │   │
│   │   └── training_history.png
│   │
│   ├── metrics/
│   │   ├── ids2017_metrics.json             # CIC-IDS2017 metrics
│   │   ├── ids2018_metrics.json             # CSE-CIC-IDS2018 metrics
│   │   ├── cross_dataset_comparison.csv     # Performance comparison
│   │   ├── per_attack_metrics_2017.csv      # Per-attack (2017)
│   │   ├── per_attack_metrics_2018.csv      # Per-attack (2018)
│   │   ├── generalization_analysis.json     # Generalization metrics
│   │   └── comparison_baseline.csv          # Comparison with baselines
│   │
│   └── predictions/
│       ├── ids2017_test_predictions.csv
│       ├── ids2018_predictions.csv
│       ├── reconstruction_errors_2017.csv
│       └── reconstruction_errors_2018.csv
│
├── logs/
│   ├── tensorboard/                 # TensorBoard logs
│   └── training.log                 # Training logs
│
├── scripts/
│   ├── download_dataset.sh          # Script to download CIC-IDS2017
│   ├── train.py                     # Training script (CLI)
│   ├── evaluate.py                  # Evaluation script (CLI)
│   └── inference.py                 # Inference on new data
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_model.py
│
└── docs/
    ├── architecture_diagram.png
    ├── methodology.md
    └── results_analysis.md
```

---

## 🛠️ Tech Stack & Dependencies

### Core Libraries

```txt
# requirements.txt

# Deep Learning Framework
tensorflow>=2.13.0
# atau
# torch>=2.0.0
# torchvision>=0.15.0

keras>=2.13.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Progress Bars
tqdm>=4.65.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.22.0

# Model Serialization
joblib>=1.2.0
pickle5>=0.0.11

# Logging
tensorboard>=2.13.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0

# Optional: For Dashboard
streamlit>=1.28.0
gradio>=3.50.0

# Optional: For Hyperparameter Tuning
optuna>=3.1.0
keras-tuner>=1.3.0

# Testing
pytest>=7.3.0
pytest-cov>=4.0.0
```

---

## 💡 Tips & Best Practices

### 1. Data Preprocessing

**✅ DO:**

- Remove kolom dengan nilai constant atau near-constant (variance < threshold)
- Handle infinite values (`np.inf`, `-np.inf`) dengan replacing atau dropping
- Scale features menggunakan `StandardScaler` atau `MinMaxScaler`
- Check data leakage (pastikan preprocessing hanya fit di training data)
- Save scaler object untuk inference

**❌ DON'T:**

- Jangan fit scaler di seluruh dataset (termasuk test set)
- Jangan drop too many features tanpa analisis
- Jangan lupa handle class imbalance (jika diperlukan)

---

### 2. Model Training

**✅ DO:**

- Start dengan architecture sederhana, gradually increase complexity
- Use callbacks:
  ```python
  callbacks = [
      EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
      ModelCheckpoint('models/best_model.h5', save_best_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
      TensorBoard(log_dir='logs/tensorboard')
  ]
  ```
- Monitor training & validation loss untuk detect overfitting
- Train dengan **normal traffic only** untuk unsupervised autoencoder
- Save model regularly (setiap epoch atau best only)

**❌ DON'T:**

- Jangan langsung train dengan architecture kompleks
- Jangan ignore overfitting signs (val_loss increasing)
- Jangan train terlalu lama tanpa early stopping

---

### 3. Threshold Selection

**Metode untuk menentukan threshold deteksi anomali:**

```python
# Method 1: Percentile-based
threshold = np.percentile(train_reconstruction_errors, 95)  # 95th percentile

# Method 2: Mean + k*std
threshold = train_errors_mean + 2 * train_errors_std  # k=2 or 3

# Method 3: ROC Curve-based (optimal threshold)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, reconstruction_errors)
optimal_idx = np.argmax(tpr - fpr)
threshold = thresholds[optimal_idx]

# Method 4: Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, reconstruction_errors)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
threshold = thresholds[optimal_idx]
```

**Recommendation:** Gunakan **ROC-based threshold** untuk balanced dataset, atau **Precision-Recall** untuk imbalanced dataset.

---

### 4. Evaluation

**Metrics yang WAJIB dilaporkan:**

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

# Binary Classification (Normal vs Attack)
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary'
)
cm = confusion_matrix(y_true, y_pred)
auc = roc_auc_score(y_true, reconstruction_errors)

# Multi-class (Per Attack Type)
report = classification_report(y_true, y_pred, target_names=class_names)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"\n{report}")
```

**Metrics untuk Laporan Skripsi:**
| Metric | Formula | Interpretasi |
|--------|---------|--------------|
| **Accuracy** | `(TP + TN) / Total` | Overall correctness |
| **Precision** | `TP / (TP + FP)` | Proportion of true anomalies among detected |
| **Recall (TPR)** | `TP / (TP + FN)` | Proportion of actual anomalies detected |
| **F1-Score** | `2 * (Prec * Rec) / (Prec + Rec)` | Harmonic mean of Prec & Rec |
| **FPR** | `FP / (FP + TN)` | False alarm rate |
| **AUC-ROC** | Area under ROC curve | Overall discriminative ability |

---

### 5. Common Pitfalls & Solutions

| Problem                                 | Solution                                                                                                    |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Model tidak converge**                | - Reduce learning rate<br>- Add batch normalization<br>- Check data scaling                                 |
| **Overfitting**                         | - Add dropout layers<br>- Reduce model complexity<br>- Increase training data<br>- Add L1/L2 regularization |
| **High false positive**                 | - Adjust threshold<br>- Retrain dengan more normal samples<br>- Feature engineering                         |
| **Low recall**                          | - Lower threshold<br>- Increase model capacity<br>- Add more attack samples in training                     |
| **Reconstruction error terlalu tinggi** | - Check data normalization<br>- Increase latent dimension<br>- Train lebih lama                             |
| **Memory error saat training**          | - Reduce batch size<br>- Use data generators<br>- Reduce model size                                         |

---

## 📊 Expected Results & Benchmarks

### Target Performance (Berdasarkan Literature)

#### In-Distribution Performance (CIC-IDS2017)

| Metric                  | Target (Minimal) | Good   | Excellent |
| ----------------------- | ---------------- | ------ | --------- |
| **Accuracy**            | ≥ 92%            | ≥ 95%  | ≥ 98%     |
| **Precision**           | ≥ 90%            | ≥ 93%  | ≥ 96%     |
| **Recall**              | ≥ 88%            | ≥ 92%  | ≥ 95%     |
| **F1-Score**            | ≥ 89%            | ≥ 93%  | ≥ 95%     |
| **AUC-ROC**             | ≥ 0.90           | ≥ 0.95 | ≥ 0.98    |
| **False Positive Rate** | ≤ 5%             | ≤ 3%   | ≤ 1%      |

#### Cross-Dataset Performance (CSE-CIC-IDS2018)

| Metric                 | Target (Minimal) | Good   | Excellent |
| ---------------------- | ---------------- | ------ | --------- |
| **Accuracy**           | ≥ 85%            | ≥ 90%  | ≥ 94%     |
| **Precision**          | ≥ 83%            | ≥ 88%  | ≥ 92%     |
| **Recall**             | ≥ 82%            | ≥ 87%  | ≥ 91%     |
| **F1-Score**           | ≥ 82%            | ≥ 88%  | ≥ 92%     |
| **AUC-ROC**            | ≥ 0.85           | ≥ 0.92 | ≥ 0.96    |
| **Generalization Gap** | ≤ 10%            | ≤ 6%   | ≤ 4%      |

**Note:** Generalization Gap = (In-Distribution Accuracy - Cross-Dataset Accuracy)

- **≤4% gap = Excellent** generalization
- **4-6% gap = Good** generalization
- **6-10% gap = Acceptable** generalization
- **>10% gap = Poor** generalization (overfitting detected)

### Comparison dengan State-of-the-Art (SOTA)

Berdasarkan penelitian terdahulu:

#### Single Dataset Performance (CIC-IDS2017)

| Method                   | Accuracy   | Precision  | Recall     | F1-Score   | Cross-Dataset? | Paper/Source   |
| ------------------------ | ---------- | ---------- | ---------- | ---------- | -------------- | -------------- |
| Random Forest            | 92-94%     | 89-91%     | 87-90%     | 88-90%     | ❌ No          | Baseline       |
| Deep Neural Network      | 93-95%     | 90-93%     | 89-92%     | 90-92%     | ❌ No          | Various papers |
| LSTM                     | 94-96%     | 91-94%     | 90-93%     | 91-93%     | ❌ No          | RNN-based NIDS |
| CNN                      | 95-97%     | 93-95%     | 91-94%     | 92-94%     | ❌ No          | CNN-based NIDS |
| **CNN-LSTM (Your Work)** | **96-98%** | **94-96%** | **93-95%** | **94-96%** | **✅ Yes**     | **Expected**   |

#### Cross-Dataset Validation Results

| Model                    | CIC-IDS2017 (Train/Test) | CSE-CIC-IDS2018 (Test Only) | Generalization Gap | Status    |
| ------------------------ | ------------------------ | --------------------------- | ------------------ | --------- |
| **CNN-LSTM (Expected)**  | **97%**                  | **92%**                     | **5%**             | 🎯 Target |
| LSTM (Expected)          | 95%                      | 88%                         | 7%                 | Baseline  |
| Random Forest (Expected) | 93%                      | 84%                         | 9%                 | Baseline  |

**Key Contribution:** Most existing NIDS research does NOT perform cross-dataset validation, making this work **novel** in proving generalization capability across different network environments and attack patterns.

---

## 🎓 Struktur Laporan Skripsi

### BAB 1: PENDAHULUAN

- 1.1 Latar Belakang
- 1.2 Rumusan Masalah
- 1.3 Tujuan Penelitian
- 1.4 Manfaat Penelitian
- 1.5 Batasan Masalah
- 1.6 Sistematika Penulisan

### BAB 2: TINJAUAN PUSTAKA

- 2.1 State-of-the-Art Penelitian
  - Review penelitian NIDS menggunakan Deep Learning (tabel SOTA)
  - CNN-based NIDS, LSTM-based NIDS, Autoencoder-based NIDS
  - Hybrid CNN-LSTM Autoencoder dalam NIDS (tabel performance comparison)
  - Gap penelitian: Kurangnya cross-dataset validation untuk proof generalization
  - Gap penelitian: Evaluasi zero-day detection pada multiple datasets
  - Positioning penelitian ini: Hybrid CNN-LSTM AE NIDS + cross-dataset validation
- 2.2 Intrusion Detection System (IDS)
  - 2.2.1 Network-based vs Host-based IDS
    - Definisi klasifikasi
    - NIDS:
      → Karakteristik & mekanisme
      → Deployment architecture (port mirroring, TAP, inline) ← INSERT 2.2.4 content
      → Packet capture mechanisms
      → Traffic analysis approaches (flow vs packet-based) ← INSERT 2.2.4 content
      → Keunggulan & keterbatasan
    - HIDS:
      → Karakteristik & mekanisme
      → Keunggulan & keterbatasan
    - Comparison & fokus penelitian
  - 2.2.2 Signature-based IDS
    - Signature-based: Pattern matching, contoh Snort, Suricata
    - Limitasi signature-based: Tidak efektif untuk zero-day attacks
  - 2.2.3 Anomaly-based IDS
    - Anomaly-based: Statistical & Machine Learning approaches
    - Deep Learning advantages untuk anomaly detection
    - Keunggulan: Dapat mendeteksi unknown attacks
  - 2.2.3 Jenis Serangan Jaringan yang Dideteksi NIDS
    - DoS/DDoS (Slowloris, Hulk, GoldenEye, LOIC)
    - Brute Force (FTP-Patator, SSH-Patator, Web Brute Force)
    - Web Attacks (XSS, SQL Injection)
    - Infiltration, Botnet, Port Scanning
- 2.3 Zero-Day Attack
  - Definisi dan karakteristik zero-day attack
  - Contoh kasus zero-day dalam enterprise environment (Stuxnet, WannaCry)
  - Challenges deteksi zero-day dengan signature-based IDS
  - Mengapa unsupervised learning (Autoencoder) superior untuk zero-day
- 2.4 Deep Learning untuk Network Intrusion Detection
  - 2.4.1 Supervised vs Unsupervised Learning dalam NIDS
    - Supervised: Membutuhkan labeled attack data (limitasi untuk zero-day)
    - Unsupervised: Belajar dari normal behavior (adaptif untuk unknown attacks)
    - Keunggulan unsupervised untuk NIDS: Deteksi zero-day tanpa signature
  - 2.4.2 Convolutional Neural Network (CNN)
    - 1D-CNN untuk sequential data (network traffic flows)
    - Local pattern extraction dari packet-level features
    - Spatial feature learning dari flow statistics
    - Aplikasi CNN dalam network traffic analysis
  - 2.4.3 Long Short-Term Memory (LSTM)
    - Arsitektur LSTM dan gating mechanism
    - Long-term dependency modeling untuk traffic sequences
    - Temporal pattern recognition dalam time-series network flows
    - Aplikasi LSTM untuk behavioral pattern detection
- 2.5 Autoencoder untuk Anomaly Detection
  - 2.5.1 Arsitektur Autoencoder (Encoder-Latent Space-Decoder)
  - 2.5.2 Reconstruction Error sebagai Indikator Anomali
    - Normal traffic → low reconstruction error
    - Anomalous traffic → high reconstruction error
    - Threshold-based anomaly classification
  - 2.5.3 Keunggulan Unsupervised Learning untuk Zero-Day Detection
- 2.6 Hybrid CNN-LSTM Autoencoder untuk NIDS
  - Motivasi hybrid architecture: Menggabungkan spatial + temporal modeling
  - Arsitektur: CNN layers → LSTM layers → Latent space → LSTM decoder
  - Keunggulan untuk network traffic analysis:
    - CNN: Extract spatial features dari flow statistics
    - LSTM: Capture temporal patterns dari traffic sequences
    - Autoencoder: Unsupervised anomaly detection
  - Penelitian terkait penggunaan CNN-LSTM Autoencoder dalam NIDS
  - Aplikasi untuk zero-day detection pada network layer
- 2.7 Cross-Dataset Validation
  - 2.7.1 Konsep Generalization dan Overfitting
    - In-distribution vs Out-of-distribution performance
    - Generalization capability sebagai indikator robustness
  - 2.7.2 Cross-Dataset Validation Strategy
    - Train on Dataset A, Test on Dataset B
    - Mengevaluasi generalization ke unseen attack patterns
    - Challenges: Feature distribution shift, attack pattern differences
- 2.8 Dataset
  - 2.8.1 CIC-IDS2017
    - Karakteristik (5 hari, 2.7M records, 78 features)
    - Jenis serangan dan distribusi temporal
    - Penggunaan sebagai training dataset
  - 2.8.2 CSE-CIC-IDS2018
    - Karakteristik (10 hari, 16.2M records, 79-80 features)
    - Attack diversity (Heartbleed, Advanced Infiltration)
    - Penggunaan untuk cross-dataset validation
  - 2.8.3 Feature Alignment dan Cross-Dataset Strategy
    - Common features (~75-76 features)
    - Preprocessing consistency untuk fair comparison
- 2.9 Metrik Evaluasi
  - 2.9.1 Classification Metrics
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix interpretation
  - 2.9.2 ROC Curve dan AUC
    - True Positive Rate vs False Positive Rate
    - AUC untuk overall discriminative ability
  - 2.9.3 False Positive Rate (FPR)
    - Importance dalam production IDS
    - Target FPR untuk enterprise deployment
  - 2.9.4 Generalization Metrics
    - In-distribution vs Out-of-distribution performance
    - Generalization gap analysis (target ≤6%)

### BAB 3: METODOLOGI PENELITIAN (SEMINAR PROPOSAL SAMPAI SINI SAJA)

- 3.1 Desain Penelitian
  - 3.1.1 Klarifikasi Penelitian
    - Jenis penelitian: Applied research (penelitian terapan)
    - Pendekatan: Quantitative research dengan metode development study
    - Tujuan: Mengembangkan model NIDS berbasis deep learning untuk zero-day detection
    - Scope: Network-based detection (network traffic analysis, bukan host-based)
  - 3.1.2 Studi Deskriptif I (Analisis Kebutuhan dan Data)
    - Analisis State-of-the-Art: Gap pada cross-dataset validation dalam CNN-LSTM AE NIDS
    - Analisis Dataset: Karakteristik CIC-IDS2017 dan CSE-CIC-IDS2018 (network traffic datasets)
    - Analisis Fitur: Network flow features, statistical features, behavioral features
    - Analisis Attack Patterns: Network-layer attacks (DoS, DDoS, Port Scan, etc.)
    - Preprocessing: Data cleaning, feature selection, normalization strategy
  - 3.1.3 Studi Preskriptif (Pengembangan Model)
    - Desain arsitektur Hybrid CNN-LSTM Autoencoder
      - Encoder: 1D-CNN layers → LSTM layers → Latent space
      - Decoder: LSTM layers → TimeDistributed Dense
    - Konfigurasi training pipeline
      - Unsupervised learning dengan normal traffic only
      - Hyperparameter tuning (CNN filters, LSTM units, latent dimension)
      - Loss function: MSE, Optimizer: Adam
    - Threshold tuning untuk anomaly classification
      - ROC curve analysis
      - Reconstruction error distribution
  - 3.1.4 Studi Deskriptif II (Evaluasi Model)
    - Evaluasi In-Distribution: Performance pada CIC-IDS2017 test set
    - Evaluasi Cross-Dataset: Zero-day simulation pada CSE-CIC-IDS2018
    - Generalization gap analysis: Comparison in-dist vs out-of-dist performance
    - Baseline comparison: LSTM Autoencoder, Random Forest/SVM
    - Statistical significance testing
- 3.2 Instrumen Penelitian
  - 3.2.1 Instrumen Evaluasi Performance Model
    - Classification metrics: Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix untuk analisis true/false positives/negatives
    - ROC Curve dan AUC Score untuk discriminative ability
    - False Positive Rate (FPR) untuk production readiness assessment
  - 3.2.2 Instrumen Evaluasi Anomaly Detection
    - Reconstruction Error sebagai anomaly indicator
    - Threshold determination (percentile-based, ROC-based)
    - Per-attack-type detection rate
  - 3.2.3 Instrumen Evaluasi Generalization
    - Generalization gap: Performance difference antara in-dist dan out-of-dist
    - Attack type transferability analysis
    - Statistical significance test (t-test, Wilcoxon signed-rank test)
- 3.3 Alat dan Bahan Penelitian
  - 3.3.1 Alat Penelitian
    - **Hardware:**
      - GPU: NVIDIA GPU (minimum 8GB VRAM)
      - Platform: Google Colab Pro / Kaggle Kernels (GPU enabled)
      - Alternative: AWS EC2 / Google Cloud Platform
    - **Software:**
      - IDE: Visual Studio Code / Jupyter Notebook
      - Version Control: Git & GitHub
      - Python: version 3.9+
      - Deep Learning Framework: TensorFlow >= 2.13.0 atau PyTorch >= 2.0.0
      - Keras >= 2.13.0
      - Python Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, tqdm
      - Monitoring Tools: TensorBoard, Weights & Biases (optional)
  - 3.3.2 Bahan Penelitian
    - **Dataset CIC-IDS2017:**
      - 2.7 juta records, 78 features
      - 5 hari data collection (Monday-Friday)
      - Penggunaan: Training dan in-distribution testing
    - **Dataset CSE-CIC-IDS2018:**
      - 16.2 juta records, 79-80 features
      - 10 hari data collection
      - Penggunaan: Cross-dataset validation (out-of-distribution testing)
- 3.4 Prosedur Penelitian
  - **Flowchart Tahapan Penelitian:**
    ```
    1. Data Collection
      ↓
    2. Exploratory Data Analysis (EDA)
      - CIC-IDS2017 analysis
      - CSE-CIC-IDS2018 analysis
      - Cross-dataset feature comparison
      ↓
    3. Data Preprocessing
      - Missing value handling
      - Feature alignment (common features: 75-76)
      - Normalization (StandardScaler)
      - Data splitting (70/15/15 untuk 2017, full untuk 2018)
      ↓
    4. Model Development
      - Define CNN-LSTM Autoencoder architecture
      - Configure encoder-decoder structure
      - Setup training pipeline (callbacks, optimizer)
      ↓
    5. Model Training
      - Train dengan CIC-IDS2017 normal traffic only
      - Hyperparameter tuning
      - Monitor training/validation loss
      ↓
    6. Threshold Determination
      - Calculate reconstruction errors on training set
      - ROC curve analysis
      - Select optimal threshold
      ↓
    7. Evaluation - In-Distribution
      - Test on CIC-IDS2017 test set
      - Calculate classification metrics
      - Per-attack-type performance
      ↓
    8. Evaluation - Cross-Dataset
      - Test on CSE-CIC-IDS2018 (zero-day simulation)
      - Apply same threshold from step 6
      - Calculate metrics and generalization gap
      ↓
    9. Baseline Comparison
      - Implement LSTM Autoencoder, Random Forest
      - Test baselines on both datasets
      - Statistical comparison
      ↓
    10. Analysis & Documentation
        - Error analysis (false positives/negatives)
        - Generalization capability assessment
        - Dokumentasi hasil dan interpretasi
    ```
- 3.5 Analisis Data
  - 3.5.1 Analisis Performa Model
    - Descriptive statistics: Mean, std, min, max dari metrics
    - Comparative analysis: Model performance across datasets
    - Visualization: Confusion matrices, ROC curves, distribution plots
  - 3.5.2 Analisis Generalization Gap
    - Formula: `Generalization Gap = Accuracy_2017 - Accuracy_2018`
    - Interpretation: ≤4% (excellent), 4-6% (good), 6-10% (acceptable), >10% (poor)
    - Attack-specific transferability: Which attacks generalize well vs poorly
  - 3.5.3 Statistical Significance Testing
    - Hypothesis testing: H0: No difference between models
    - T-test atau Wilcoxon signed-rank test untuk paired comparison
    - Confidence interval: 95% CI untuk performance metrics
    - Effect size: Cohen's d untuk magnitude of difference
  - 3.5.4 Threshold Optimization Analysis
    - Trade-off analysis: Precision vs Recall
    - Cost-sensitive threshold selection untuk production IDS
    - Sensitivity analysis: Impact of threshold variation pada FPR/TPR

### BAB 4: HASIL DAN PEMBAHASAN

- 4.1 Eksplorasi Dataset
  - 4.1.1 Distribusi Data CIC-IDS2017
  - 4.1.2 Distribusi Data CSE-CIC-IDS2018
  - 4.1.3 Feature Analysis & Comparison
  - 4.1.4 Class Imbalance Analysis
  - 4.1.5 Cross-Dataset Characteristics
- 4.2 Training Results
  - 4.2.1 Training History
  - 4.2.2 Convergence Analysis
  - 4.2.3 Hyperparameter Tuning Results
- 4.3 In-Distribution Performance (CIC-IDS2017)
  - 4.3.1 Overall Performance Metrics
  - 4.3.2 Per-Attack Type Performance
  - 4.3.3 Confusion Matrix Analysis
  - 4.3.4 ROC Curve & AUC Analysis
  - 4.3.5 Threshold Selection Results
- 4.4 Cross-Dataset Validation (CSE-CIC-IDS2018)
  - 4.4.1 Zero-Day Simulation Results
  - 4.4.2 Overall Performance on 2018 Data
  - 4.4.3 Per-Attack Type Performance (2018)
  - 4.4.4 Reconstruction Error Distribution Comparison
  - 4.4.5 Attack Detection Success Rate
- 4.5 Generalization Analysis
  - 4.5.1 Performance Gap Analysis (2017 vs 2018)
  - 4.5.2 Generalization Capability Assessment
  - 4.5.3 Attack Type Transferability
  - 4.5.4 Statistical Significance Testing
- 4.6 Comparison dengan Baseline
  - 4.6.1 Performance Comparison (In-Distribution)
  - 4.6.2 Cross-Dataset Comparison
  - 4.6.3 Generalization Gap Comparison
  - 4.6.4 Inference Time & Computational Cost
- 4.7 Error Analysis
  - 4.7.1 False Positives Analysis (Both Datasets)
  - 4.7.2 False Negatives Analysis (Both Datasets)
  - 4.7.3 Failure Cases in Cross-Dataset Scenario
- 4.8 Pembahasan
  - 4.8.1 Interpretasi Hasil In-Distribution
  - 4.8.2 Interpretasi Hasil Cross-Dataset
  - 4.8.3 Zero-Day Detection Capability
  - 4.8.4 Implikasi Praktis untuk NIDS Production Deployment
    - Deployment architecture: Core switch, DMZ, atau data center
    - Hardware requirements: GPU-accelerated inference
    - Real-time detection considerations
    - Integration dengan existing network infrastructure

### BAB 5: KESIMPULAN DAN SARAN

- 5.1 Kesimpulan
- 5.2 Saran untuk Penelitian Selanjutnya
- 5.3 Keterbatasan Penelitian

---

## 📚 Referensi Penting

### Dataset

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. ICISSP.

### Metodologi

- CNN-LSTM papers untuk time-series/network traffic
- Autoencoder untuk anomaly detection
- Zero-day attack detection papers

### Tools Documentation

- TensorFlow/Keras documentation
- Scikit-learn documentation
- Pandas documentation

---

## ⚠️ Hal-hal yang Perlu Diperhatikan

### 1. Dataset Size

- CIC-IDS2017 sangat besar (~2.7 million rows)
- **Solusi:** Gunakan sampling jika hardware terbatas
- **Recommendation:** Gunakan semua Monday data (normal) + sampling attack data

### 2. Class Imbalance

- Dataset sangat imbalanced (Benign >> Attacks)
- **Solusi:**
  - Undersampling majority class
  - Oversampling minority class (SMOTE)
  - Class weights dalam training

### 3. Computational Resources

- Training CNN-LSTM Autoencoder membutuhkan GPU
- **Solusi:**
  - Google Colab (GPU gratis)
  - Kaggle Kernels (GPU gratis)
  - AWS/GCP/Azure (berbayar)
  - Lab kampus (jika ada)

### 4. Training Time

- Expect 2-6 jam training time (depends on hardware)
- **Solusi:**
  - Start dengan subset data untuk testing
  - Use early stopping
  - Train overnight jika perlu

### 5. Version Control

- **Wajib gunakan Git!**
- Commit regularly
- Jangan commit data besar (gunakan `.gitignore`)
- Push ke GitHub/GitLab untuk backup

---

## 🚀 Quick Start Guide

### 1. Setup Environment

```bash
# Clone repository (jika sudah ada)
git clone <repo-url>
cd skripsi-ids-cnn-lstm

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name=skripsi-ids
```

### 2. Download Dataset

```bash
# Download CIC-IDS2017
# Link: https://www.unb.ca/cic/datasets/ids-2017.html
# Atau gunakan script:
bash scripts/download_dataset.sh

# Extract ke folder data/raw/
```

### 3. Run EDA

```bash
# Open Jupyter Notebook
jupyter notebook

# Buka notebooks/01_EDA.ipynb
# Run all cells
```

### 4. Preprocessing

```bash
# Run preprocessing notebook
# notebooks/02_Preprocessing.ipynb
```

### 5. Build & Train Model

```bash
# Run training notebook
# notebooks/04_Training.ipynb

# Atau via command line:
python scripts/train.py --epochs 50 --batch_size 64
```

### 6. Evaluate

```bash
# Run evaluation notebook
# notebooks/05_Evaluation.ipynb

# Atau via command line:
python scripts/evaluate.py --model models/cnn_lstm_autoencoder_best.h5
```

---

## 📞 Support & Resources

### Jika Stuck atau Butuh Bantuan:

1. **TensorFlow/Keras Issues:**

   - TensorFlow Forum: https://discuss.tensorflow.org/
   - Stack Overflow: Tag `tensorflow` atau `keras`

2. **Dataset Issues:**

   - CIC-IDS2017 Documentation: https://www.unb.ca/cic/datasets/ids-2017.html
   - Kaggle Datasets: https://www.kaggle.com/datasets

3. **Paper References:**

   - Google Scholar: Cari "CNN-LSTM Autoencoder IDS"
   - arXiv: https://arxiv.org/
   - IEEE Xplore, ACM Digital Library

4. **Implementation References:**
   - GitHub: Cari repository serupa
   - Papers with Code: https://paperswithcode.com/

---

## ✅ Checklist Progress

### Preparation Phase

- [ ] Setup environment & dependencies
- [ ] Download dataset CIC-IDS2017
- [ ] Setup Git repository
- [ ] Read related papers (minimal 5 papers)

### Data Phase

- [ ] Exploratory Data Analysis (EDA) - CIC-IDS2017
- [ ] Exploratory Data Analysis (EDA) - CSE-CIC-IDS2018
- [ ] Cross-dataset feature analysis & alignment
- [ ] Identify common features (target: 75-76 features)
- [ ] Data preprocessing & cleaning (both datasets)
- [ ] Feature engineering & selection
- [ ] Train-val-test split (CIC-IDS2017)
- [ ] Prepare CSE-CIC-IDS2018 for cross-validation
- [ ] Save processed data (both datasets)

### Model Phase

- [ ] Design architecture
- [ ] Implement CNN-LSTM Autoencoder
- [ ] Setup training pipeline
- [ ] Train model
- [ ] Hyperparameter tuning
- [ ] Save best model

### Evaluation Phase

- [ ] Calculate reconstruction errors (CIC-IDS2017)
- [ ] Determine optimal threshold
- [ ] Evaluate on CIC-IDS2017 test set
- [ ] Calculate all metrics (in-distribution)
- [ ] Generate confusion matrix & ROC curve (2017)
- [ ] **Cross-dataset validation on CSE-CIC-IDS2018**
- [ ] **Calculate reconstruction errors (2018)**
- [ ] **Apply same threshold to 2018 data**
- [ ] **Calculate metrics for out-of-distribution performance**
- [ ] **Generate confusion matrix & ROC curve (2018)**
- [ ] **Analyze generalization gap**
- [ ] Implement baseline models
- [ ] Test baselines on both datasets
- [ ] Compare with baselines (in-dist & cross-dist)
- [ ] Statistical significance testing

### Analysis Phase

- [ ] Error analysis (CIC-IDS2017)
- [ ] Error analysis (CSE-CIC-IDS2018)
- [ ] Per-attack performance analysis (both datasets)
- [ ] Zero-day detection analysis (cross-dataset scenario)
- [ ] Generalization capability analysis
- [ ] Attack type transferability analysis
- [ ] Cross-dataset comparison visualizations
- [ ] Create all visualizations (in-dist + cross-dist)
- [ ] Interpret results (generalization insights)
- [ ] Failure case analysis for cross-dataset scenarios

### Documentation Phase

- [ ] Clean all notebooks
- [ ] Write comprehensive README
- [ ] Document all functions
- [ ] Create architecture diagram
- [ ] Write methodology documentation
- [ ] Prepare presentation slides
- [ ] Write draft skripsi (Bab 1-5)

### Final Phase

- [ ] Final testing
- [ ] Code review
- [ ] Revision based on feedback
- [ ] Final documentation
- [ ] Prepare for presentation

---

## 🎯 Success Criteria

Penelitian Anda dianggap **SUKSES** jika:

### Core Requirements:

1. ✅ Model berhasil diimplementasikan tanpa error
2. ✅ Accuracy ≥ 95% pada CIC-IDS2017 test set (in-distribution)
3. ✅ F1-Score ≥ 93% pada CIC-IDS2017
4. ✅ Model outperform baseline methods (minimal +2% accuracy)

### Cross-Dataset Requirements:

5. ✅ **Accuracy ≥ 90% pada CSE-CIC-IDS2018** (out-of-distribution)
6. ✅ **Generalization gap ≤ 6%** (difference between 2017 and 2018 performance)
7. ✅ **Dapat mendeteksi zero-day attack dengan recall ≥ 87%** pada 2018 data

### Quality Requirements:

8. ✅ False positive rate ≤ 5% (pada kedua dataset)
9. ✅ Model generalize better than baselines (smaller generalization gap)

### Documentation:

10. ✅ Dokumentasi lengkap & reproducible untuk both datasets
11. ✅ Cross-dataset validation clearly documented
12. ✅ Laporan skripsi selesai sesuai struktur (dengan BAB Cross-Dataset Validation)

### Bonus (Nice to Have):

- 🌟 Generalization gap ≤ 4% (excellent generalization)
- 🌟 Accuracy ≥ 92% pada CSE-CIC-IDS2018
- 🌟 Statistical significance test shows CNN-LSTM significantly better than baselines

---

## 💪 Motivasi

> "The only way to do great work is to love what you do." - Steve Jobs

**Remember:**

- Skripsi adalah learning journey, bukan sprint
- Setiap error adalah kesempatan belajar
- Jangan takut bertanya & mencari bantuan
- Document everything untuk future reference
- Stay consistent, small progress everyday!

**Good luck dengan penelitian Anda! 🚀**

---

**Last Updated:** January 11, 2026  
**Document Version:** 2.0 (Added CSE-CIC-IDS2018 Cross-Dataset Validation)  
**Author:** Awan
