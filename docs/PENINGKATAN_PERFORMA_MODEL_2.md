<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# coba buat dokumen lengkap yang mungkin dapat meningkatkan hasil dari model yang gw usulkan gitu, berikan beberapa pendekatan its oke ya, nanti biar gw yang decide, pastikan jelas, detil dan komprehensif.

Berikut dokumen komprehensif strategi peningkatan model CNN-LSTM-AE kamu:

***

# 📄 Dokumen Strategi Peningkatan Model CNN-LSTM-AE untuk NIDS Unsupervised

**Target masalah:** F1-score 0.7 di CIC-IDS2017, FPR 0.2 → 0.4 saat cross-dataset ke CIC-IDS2018

***

## A. Perbaikan Arsitektur Model

### A1. Upgrade LSTM → BiLSTM

**Masalah yang diselesaikan:** LSTM unidirectional hanya membaca sekuens dari kiri ke kanan, sehingga kehilangan konteks "masa depan" dalam flow traffic.

**Cara implementasi:**

- Ganti layer `LSTM(units)` dengan `Bidirectional(LSTM(units))` di encoder dan decoder
- Output BiLSTM = 2× dimensi LSTM biasa, jadi sesuaikan ukuran decoder

**Estimasi dampak:** Park (2025) menggunakan BiLSTM dan mencapai F1 0.983. Upgrade ini kemungkinan meningkatkan F1 sebesar 5–15% berdasarkan literatur[^1][^2]

```
Encoder: Input → Conv1D → BiLSTM → Bottleneck
Decoder: Bottleneck → RepeatVector → BiLSTM → TimeDistributed(Dense)
```


***

### A2. Tambahkan Attention Mechanism

**Masalah yang diselesaikan:** Autoencoder standar memperlakukan semua timestep secara sama; padahal pola anomali biasanya terlokalisasi di beberapa timestep tertentu.[^3]

**Cara implementasi:**

- Tambahkan **Self-Attention** atau **Bahdanau Attention** di antara encoder LSTM dan bottleneck
- Attention akan memberikan bobot lebih besar pada timestep yang paling "mencurigakan"
- Bisa pakai `tf.keras.layers.Attention` atau implementasi manual

**Estimasi dampak:** Attention secara konsisten meningkatkan kemampuan deteksi anomali lokal dan mereduksi FPR[^3]

```
Encoder: Input → Conv1D → BiLSTM → Attention Layer → Bottleneck
```


***

### A3. Multi-Scale CNN (adopsi dari Singh 2022)

**Masalah yang diselesaikan:** Conv1D single-scale hanya menangkap pola lokal pada satu granularitas; serangan berbeda memiliki "signature" pada skala kernel yang berbeda[^4]

**Cara implementasi:**

- Buat 3 branch Conv1D paralel dengan kernel size berbeda (misal: 3, 5, 7)
- Concatenate output ketiganya sebelum masuk ke LSTM
- Tambahkan BatchNormalization di setiap branch

**Estimasi dampak:** Singh (2022) menunjukkan MSCNN meningkatkan kemampuan ekstraksi fitur spasial secara signifikan dibanding single-scale CNN[^4]

```
Input → [Conv1D(k=3) | Conv1D(k=5) | Conv1D(k=7)] → Concat → BiLSTM → Attention → Bottleneck
```


***

### A4. Variational Autoencoder (VAE) sebagai Alternatif AE

**Masalah yang diselesaikan:** Standard AE cenderung "menghafal" noise, sehingga reconstruction error tidak selalu informatif untuk anomali yang sangat berbeda dari normal[^5]

**Cara implementasi:**

- Ganti bottleneck deterministic dengan **distribusi probabilistik** (mean + log-variance)
- Loss = Reconstruction Loss + KL-Divergence
- Sampling via reparameterization trick: `z = μ + ε × σ`

**Estimasi dampak:** VAE menghasilkan latent space yang lebih smooth dan generalized, cocok untuk zero-day detection[^5]

**Trade-off:** Training lebih kompleks, perlu tuning bobot antara reconstruction loss dan KL term (β-VAE)

***

## B. Perbaikan Threshold Detection

### B1. Adaptive Threshold (Pengganti Static Percentile)

**Masalah yang diselesaikan:** Threshold persentil 95/99 dari source domain (CIC-IDS2017) tidak optimal ketika diaplikasikan ke target domain (CIC-IDS2018) karena distribusi error bergeser — ini penyebab utama FPR naik dari 0.2 ke 0.4[^5]

**Pendekatan 1 — Statistical Adaptive Threshold:**

- Pada skenario few-shot: fit distribusi Gaussian atau Gamma pada reconstruction error dari 1% benign CIC-IDS2018
- Threshold = μ + k×σ, di mana k di-tune via grid search (misal k = 2, 2.5, 3)

**Pendekatan 2 — Peak Over Threshold (POT/SPOT):**

- Metode berbasis Extreme Value Theory (EVT)
- Secara otomatis mempelajari tail distribution dari error untuk set threshold optimal
- Tidak memerlukan labeled data[^5]

**Pendekatan 3 — Percentile Recalibration:**

- Pada few-shot: hitung ulang persentil threshold dari 1% benign target domain
- Lebih sederhana dari POT, cocok sebagai baseline improvement

***

### B2. Ensemble Threshold (Multiple AE)

**Masalah yang diselesaikan:** Satu threshold dari satu model rentan terhadap noise dan distribusi shift[^6]

**Cara implementasi:**

- Latih 3–5 AE dengan inisialisasi random seed berbeda
- Aggregasi reconstruction error: gunakan **median** (lebih robust dari mean)
- Threshold diterapkan pada median error

**Estimasi dampak:** Ensemble AE secara konsisten mereduksi variance deteksi dan menurunkan FPR[^7][^6]

***

## C. Perbaikan Cross-Dataset Generalization

### C1. Feature Selection yang Lebih Ketat

**Masalah yang diselesaikan:** Fitur-fitur tertentu di CIC-IDS2017 memiliki distribusi berbeda di CIC-IDS2018 (karena perbedaan lingkungan jaringan, tool capture, dsb.) — ini penyebab distribution shift[^8][^9]

**Cara implementasi:**

- Gunakan **Mutual Information** atau **SHAP values** untuk ranking fitur berdasarkan kontribusi deteksi anomali di source domain
- Seleksi top-N fitur (misal: top 20–30 dari 78 fitur CIC) yang paling stabil lintas dataset
- Exclude fitur yang highly correlated dengan environment-specific metadata (misal: IP address derived features)

**Tools:** `sklearn.feature_selection.mutual_info_classif`, SHAP library

***

### C2. Domain Adaptation dengan Adversarial Training (DANN)

**Masalah yang diselesaikan:** Model belajar feature representation yang terlalu spesifik pada source domain, tidak generalizable ke target domain[^10][^11]

**Cara implementasi:**

- Tambahkan **Domain Classifier** kecil (MLP 2 layer) di atas bottleneck representation
- Latih dengan 3 loss secara bersamaan:

1. Reconstruction loss (AE normal)
2. Domain classification loss (source vs target)
3. Gradient Reversal Layer (GRL) — memaksa encoder menghasilkan domain-invariant features
- Pada saat training, gunakan campuran: benign CIC-IDS2017 (labeled source) + unlabeled CIC-IDS2018 benign

**Estimasi dampak:** DANN untuk NIDS dilaporkan meningkatkan cross-domain F1 secara signifikan[^12][^10]

**Catatan:** Ini pendekatan paling kompleks di kategori ini, cocok jika kamu punya waktu implementasi lebih

***

### C3. Data Augmentation pada Source Domain

**Masalah yang diselesaikan:** Model terlalu overfit pada pola normal yang spesifik di CIC-IDS2017[^13]

**Cara implementasi:**

- **Gaussian Noise Injection:** tambahkan noise kecil (σ = 0.01–0.05) pada fitur numerik saat training
- **Temporal Jittering:** sedikit geser urutan timestep dalam window LSTM
- **Feature Dropout:** random zero-out beberapa fitur per sample (mirip Dropout, tapi di level input)

Augmentasi ini memaksa model belajar representasi yang lebih robust, bukan hanya menghafal pola exact dari training data

***

## D. Perbaikan Preprocessing \& Feature Engineering

### D1. Normalisasi yang Lebih Robust

**Masalah yang diselesaikan:** StandardScaler (z-score) sensitif terhadap outlier ekstrem yang umum ada di traffic data (burst packets, jumbo frames)[^8]

**Rekomendasi:**

- Ganti dengan **RobustScaler** (menggunakan median dan IQR, bukan mean dan std)
- Atau **Quantile Transformer** (mengubah distribusi menjadi uniform/normal)
- Scaler tetap di-fit **hanya di source train** dan di-apply ke semua data (hindari leakage)

***

### D2. Time-Window Optimization

**Masalah yang diselesaikan:** Ukuran timestep window untuk LSTM sangat menentukan pola temporal yang bisa ditangkap, namun jarang dioptimasi secara sistematis

**Cara implementasi:**

- Lakukan eksperimen grid: window size = {5, 10, 20, 30, 50}
- Stride = {1, 5, 10}
- Evaluasi trade-off antara granularitas temporal vs. memory dan waktu training

***

### D3. Hapus Fitur Constant \& Near-Zero Variance

**Masalah yang diselesaikan:** CIC-IDS2017 dan CIC-IDS2018 memiliki beberapa fitur dengan variance sangat rendah atau konstan yang menambah noise tanpa informasi[^9]

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_clean = selector.fit_transform(X_train)
```


***

## E. Perbaikan Training Strategy

### E1. Learning Rate Scheduling

**Masalah yang diselesaikan:** Learning rate tetap (0.001) bisa menyebabkan model terjebak di local minima atau overshoot saat konvergensi[^14]

**Rekomendasi:**

- **ReduceLROnPlateau:** turunkan lr 0.5× jika val_loss tidak membaik dalam 5 epoch
- **Cosine Annealing:** lr mengikuti kurva cosine dari lr_max ke lr_min
- Kombinasikan dengan warmup 3–5 epoch di awal

***

### E2. Perbaikan Early Stopping

**Masalah yang diselesaikan:** Patience=10 mungkin terlalu agresif atau terlalu permisif tergantung karakteristik dataset

**Rekomendasi:**

- Naikkan patience ke 15–20 epoch
- Monitor `val_loss` bukan `train_loss`
- Tambahkan `restore_best_weights=True` untuk pastikan model yang disimpan adalah yang terbaik

***

### E3. Gradient Clipping

**Mencegah exploding gradient** yang umum terjadi pada LSTM/BiLSTM dengan sekuens panjang:

```python
optimizer = Adam(lr=0.001, clipnorm=1.0)
```


***

## F. Ringkasan Prioritas Implementasi

Berikut urutan yang disarankan dari yang paling mudah dan berdampak besar, ke yang paling kompleks:


| Prioritas | Strategi | Kompleksitas | Estimasi Dampak pada F1/FPR |
| :-- | :-- | :-- | :-- |
| ⭐⭐⭐ | **A1 — BiLSTM** | Rendah | F1 +5–15% |
| ⭐⭐⭐ | **B1 — Adaptive Threshold** | Rendah | FPR −0.1–0.15 |
| ⭐⭐⭐ | **D1 — RobustScaler** | Rendah | FPR −0.05–0.1 |
| ⭐⭐ | **A2 — Attention** | Sedang | F1 +3–8% |
| ⭐⭐ | **C1 — Feature Selection** | Sedang | F1 +3–7%, FPR turun |
| ⭐⭐ | **E1 — LR Scheduling** | Rendah | F1 +2–5% |
| ⭐⭐ | **C3 — Data Augmentation** | Sedang | Generalisasi +5–10% |
| ⭐⭐ | **B2 — Ensemble AE** | Sedang | FPR −0.05–0.1 |
| ⭐ | **A3 — Multi-Scale CNN** | Sedang | F1 +3–6% |
| ⭐ | **A4 — VAE** | Tinggi | Generalisasi signifikan |
| ⭐ | **C2 — DANN** | Tinggi | Cross-domain F1 +10–20% |


***

## G. Rekomendasi Kombinasi Terbaik

**Jika target utama: naikkan F1 di CIC-IDS2017**
→ A1 (BiLSTM) + A2 (Attention) + E1 (LR Scheduling) + D1 (RobustScaler)

**Jika target utama: turunkan FPR di cross-dataset (CIC-IDS2018)**
→ B1 (Adaptive Threshold) + C1 (Feature Selection) + D1 (RobustScaler) + C3 (Augmentation)

**Jika target: novelty tinggi untuk skripsi/publikasi**
→ A1 (BiLSTM) + A2 (Attention) + C2 (DANN) + B1 (Adaptive Threshold)

<div align="center">⁂</div>

[^1]: https://sejong.elsevierpure.com/en/publications/unsupervised-machine-learning-methods-for-anomaly-detection-in-ne-2/

[^2]: https://www.nature.com/articles/s41598-025-29079-y

[^3]: https://www.sciencedirect.com/science/article/abs/pii/S0957417423034681

[^4]: https://arxiv.org/pdf/2204.03779.pdf

[^5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12788306/

[^6]: https://arxiv.org/html/2405.13031v2

[^7]: https://arxiv.org/abs/2405.13031

[^8]: https://arxiv.org/html/2402.10974

[^9]: https://arxiv.org/abs/2402.10974

[^10]: https://www.sciencedirect.com/science/article/abs/pii/S1084804523001790

[^11]: https://arxiv.org/pdf/2304.10550.pdf

[^12]: https://sciety-labs.elifesciences.org/articles/by?article_doi=10.21203%2Frs.3.rs-6852869%2Fv1

[^13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11794860/

[^14]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425004762

