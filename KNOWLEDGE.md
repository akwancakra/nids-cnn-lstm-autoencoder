# KNOWLEDGE.md

## Ringkasan Proyek
Judul: Implementasi Hybrid CNN-LSTM Autoencoder untuk Deteksi Anomali Zero-Day pada Network-based Intrusion Detection System (NIDS) Enterprise.

Tujuan utama:
1) Mengimplementasikan Hybrid CNN-LSTM Autoencoder (unsupervised) untuk deteksi anomali.
2) Evaluasi in-distribution di CIC-IDS2017.
3) Validasi lintas dataset (cross-dataset) ke CSE-CIC-IDS2018 (zero-shot + opsional few-shot).
4) Bandingkan dengan baseline (LSTM Autoencoder + Traditional ML).
5) Ukur generalization gap + uji statistik (McNemar).

## Konsep Kunci
- Unsupervised learning: model dilatih hanya pada trafik normal (BENIGN).
- Anomaly detection: deteksi berdasarkan reconstruction error.
- Cross-dataset validation: model training pada CIC-IDS2017, testing pada CSE-CIC-IDS2018 untuk simulasi zero-day.
- Generalization gap: selisih performa in-distribution vs out-of-distribution.

## Dataset
### CIC-IDS2017
- ~2.7 juta records, 78+ fitur
- 5 hari trafik, variasi serangan: FTP/SSH-Patator, DoS/DDoS, Web Attacks, Infiltration, Botnet, PortScan
- Dipakai untuk training (normal only) + in-distribution test

### CSE-CIC-IDS2018
- ~16.2 juta records, 79–80 fitur
- 10 hari trafik, serangan lebih kompleks (Heartbleed, LOIC, Botnet, Infiltration)
- Dipakai untuk cross-dataset validation (zero-day simulation)

## Arsitektur Model (Hybrid CNN-LSTM Autoencoder)
Encoder:
- Conv1D(64, k=3) + BN + ReLU + MaxPool
- Conv1D(128, k=3) + BN + ReLU + MaxPool
- LSTM(128, return_seq=True) + Dropout(0.2)
- LSTM(64, return_seq=False) + Dropout(0.2)

Bottleneck:
- Dense(32, relu)

Decoder:
- RepeatVector
- LSTM(64, return_seq=True) + Dropout(0.2)
- LSTM(128, return_seq=True) + Dropout(0.2)
- TimeDistributed(Dense(n_features))

## Preprocessing & Data Pipeline
- Handling NaN/Inf + outliers
- Drop ID-like fields (Flow ID, Timestamp, IP/Port jika perlu)
- Feature alignment: intersection CIC-IDS2017 vs CSE-CIC-IDS2018
- Scaling: fit scaler di train CIC-IDS2017, apply ke semua set
- Reshape ke 3D (samples, timesteps, features)

## Training & Threshold
- Loss: MSE
- Optimizer: Adam (lr 1e-3)
- Early stopping (patience 10)
- Threshold detection: 95th/99th percentile reconstruction error (validation BENIGN)
- Pilih threshold optimal via F1 atau Youden Index

## Evaluasi
Metrics:
- Accuracy, Precision, Recall, F1
- AUC-ROC, FPR
- Generalization gap = M_CIC - M_CSE

Eksperimen:
1) In-distribution: CIC-IDS2017 test
2) Cross-dataset zero-shot: CSE-CIC-IDS2018 full
3) Few-shot adaptation (opsional): fine-tune unsupervised <1% data target

Statistik:
- McNemar’s Test untuk signifikansi perbandingan model vs baseline

## Baseline
- LSTM Autoencoder (tanpa CNN)
- Traditional ML: Isolation Forest / Random Forest

## Target Keberhasilan
- F1-score ≥ 90% (CIC-IDS2017)
- Accuracy ≥ 90% (CSE-CIC-IDS2018)
- Generalization gap ≤ 15%