# TASKS.md

## 0) Scope & Target
- [ ] Konfirmasi scope: Hybrid CNN-LSTM Autoencoder (unsupervised), train normal-only CIC-IDS2017, test CIC-IDS2017 + CSE-CIC-IDS2018
- [ ] Baseline yang dipakai: LSTM Autoencoder + Traditional ML (Isolation Forest / Random Forest)
- [ ] Target metrik: F1 ≥ 90% (CIC-IDS2017) + generalization gap ≤ 15% + Accuracy ≥ 90% (CSE-CIC-IDS2018)

## 1) Environment & Repo Setup
- [ ] Setup Python 3.8+ + venv/conda
- [ ] Install deps: tensorflow/keras, numpy, pandas, scikit-learn, matplotlib, seaborn
- [ ] Siapkan GPU (CUDA/cuDNN) atau fallback CPU
- [ ] Buat struktur folder: `data/raw`, `data/processed`, `models/`, `reports/`, `notebooks/`, `scripts/`, `results/`

## 2) Dataset Acquisition
- [ ] Download CIC-IDS2017 (CSV/PCAP processed)
- [ ] Download CSE-CIC-IDS2018 (CSV/PCAP processed)
- [ ] Verifikasi integritas file (size/row count) + dokumentasi sumber data

## 3) Exploratory Data Analysis (EDA)
- [ ] EDA CIC-IDS2017: missing/inf values, distribusi fitur, class imbalance
- [ ] EDA CSE-CIC-IDS2018: missing/inf values, distribusi fitur, perbandingan dengan 2017
- [ ] Cek anomali kualitas data (outlier ekstrem, feature errors)
- [ ] Simpan laporan EDA: grafik distribusi, korelasi, ringkasan statistik

## 4) Preprocessing & Feature Alignment
- [ ] Drop kolom non-numerik/ID (Flow ID, Timestamp, IP, Port jika perlu)
- [ ] Handle NaN/Inf: impute/remove sesuai kebijakan
- [ ] Dedup/trim outlier ekstrem (jika diperlukan)
- [ ] Feature alignment: ambil intersection fitur CIC-IDS2017 & CSE-CIC-IDS2018
- [ ] Split CIC-IDS2017: train/val/test (70/15/15)
- [ ] Train hanya data BENIGN untuk model unsupervised
- [ ] Fit scaler hanya di train CIC-IDS2017, apply ke val/test CIC-IDS2017 & seluruh CSE-CIC-IDS2018
- [ ] Reshape ke format time-series 3D: `(samples, timesteps, features)`

## 5) Model Development (Hybrid CNN-LSTM Autoencoder)
- [ ] Implement encoder: Conv1D -> BN -> ReLU -> MaxPool (2 blok)
- [ ] Implement LSTM layers: LSTM(128, return_seq=True) -> Dropout -> LSTM(64)
- [ ] Bottleneck: Dense(32, relu)
- [ ] Decoder: RepeatVector -> LSTM(64, return_seq=True) -> LSTM(128, return_seq=True)
- [ ] Output: TimeDistributed(Dense(n_features))
- [ ] Loss: MSE; Optimizer: Adam (lr=1e-3)

## 6) Training & Hyperparameter Tuning
- [ ] Early stopping (patience=10) + model checkpoint
- [ ] Grid/Random search sederhana: batch size (128/256), lr (1e-3/1e-4), dropout (0.2/0.3), bottleneck size
- [ ] Monitor train/val loss + reconstruct error distribution

## 7) Thresholding (Detection Rule)
- [ ] Hitung reconstruction error di validation set (BENIGN)
- [ ] Uji threshold: 95th percentile vs 99th percentile
- [ ] Pilih threshold optimal (maximize F1 atau Youden Index)
- [ ] Simpan threshold final + justifikasi trade-off TPR/FPR

## 8) In-Distribution Evaluation (CIC-IDS2017)
- [ ] Test di CIC-IDS2017 test set (normal + attack)
- [ ] Hitung metrics: Accuracy, Precision, Recall, F1, AUC-ROC, FPR
- [ ] Plot: ROC curve, confusion matrix, error distribution
- [ ] Per-attack-type analysis (jika label tersedia)

## 9) Cross-Dataset Validation (CSE-CIC-IDS2018)
- [ ] Zero-shot: model CIC-IDS2017 -> test full CSE-CIC-IDS2018
- [ ] Evaluate metrics + generalization gap
- [ ] Plot ROC + confusion matrix + error distribution
- [ ] Analisis performa per attack category (opsional)

## 10) Few-Shot Adaptation (Opsional sesuai rumusan masalah)
- [ ] Fine-tune unsupervised dengan <1% data BENIGN dari CSE-CIC-IDS2018
- [ ] Re-evaluate metrics + compare dengan zero-shot
- [ ] Laporkan perubahan generalization gap

## 10A) Optional Improvement Plan (Jika hasil kurang baik)
- [ ] Multi-scale Conv + Attention sebelum LSTM (uji dampak F1/AUC)
- [ ] Denoising AE (noise injection) untuk robustness
- [ ] Threshold EVT/POT (bandingkan vs percentile 95/99)
- [ ] Quantile normalization + drop constant features (uji generalization gap)
- [ ] Ablation: baseline vs tiap improvement (1 perubahan per run)

## 11) Baseline Methods
- [ ] Implement LSTM Autoencoder (tanpa CNN)
- [ ] Implement Traditional ML:
- [ ] Isolation Forest (unsupervised)
- [ ] Random Forest (supervised, optional jika label dipakai untuk baseline)
- [ ] Evaluasi baseline pada CIC-IDS2017 & CSE-CIC-IDS2018

## 12) Statistical Significance
- [ ] McNemar’s Test: bandingkan model vs baseline di CIC-IDS2017
- [ ] Laporkan p-value (alpha=0.05) + interpretasi

## 13) Reporting & Reproducibility
- [ ] Simpan artefak: model weights, scaler, threshold, metrics JSON
- [ ] Buat tabel hasil: in-distribution vs cross-dataset + generalization gap
- [ ] Dokumentasi pipeline end-to-end + struktur folder
- [ ] Reproducibility: seed random, versi library, hardware spec

## 14) Final Deliverables
- [ ] Notebook EDA + preprocessing
- [ ] Script training & evaluation
- [ ] Laporan hasil eksperimen (grafik + tabel)
- [ ] Kesimpulan & rekomendasi untuk riset lanjutan
