# Changelog Experiments

Dokumen ini mencatat perubahan eksperimen yang memengaruhi pipeline penelitian agar reproducible dan mudah ditelusuri saat penulisan skripsi.

## v1.1.0 - 2026-02-18

### Ringkasan
- Fokus perubahan: peningkatan performa tanpa mengubah model inti/judul (`CNN-LSTM Autoencoder`).
- Ruang lingkup: preprocess, training strategy, evaluasi zero-shot/few-shot, dan notebook workflow.

### Baseline Referensi (sebelum perubahan ini)
- Sumber: rilis eksperimen `v1.0.0` (2026-02-15).
- CIC-IDS2017: `f1=0.7561`, `fpr=0.0252`, `roc_auc=0.8158`.
- CSE-CIC-IDS2018: `f1=0.6790`, `fpr=0.4005`, `roc_auc=0.6574`.
- Gap: `f1_gap=0.0771`, `auc_gap=0.1583`, `accuracy_gap=0.1618`.

### Perubahan Teknis

#### 1) Preprocessing
- Tambah dukungan scaler: `robust` (selain `standard` dan `minmax`).
- Tambah filter fitur statistik (opsional):
  - near-zero variance (NZV),
  - high correlation threshold.
- Tambah keluaran laporan filter:
  - `data/processed/feature_filter_report.json`.
- File terkait:
  - `scripts/preprocess.py`
  - `config.yaml`

#### 2) Training Strategy
- Tambah opsi scheduler:
  - `reduce_on_plateau` (default),
  - `cosine`,
  - `none`.
- Tambah `clipnorm` untuk gradient clipping.
- Tambah `restore_best_weights` di early stopping.
- File terkait:
  - `scripts/train_cnn_lstm_ae.py`
  - `config.yaml`

#### 3) Evaluation (Cross-Dataset)
- Tambah mode evaluasi:
  - `zero_shot`
  - `few_shot`
- Tambah metode threshold:
  - `percentile`
  - `target_percentile`
  - `target_gaussian`
- Tambah few-shot adaptation:
  - sampling benign target (`few_shot_benign_frac`),
  - optional fine-tuning unsupervised ringan.
- File terkait:
  - `scripts/eval_metrics.py`
  - `config.yaml`

#### 4) Notebook Workflow
- Update `notebooks/colab_ready.ipynb` dan `notebooks/colab_ready_online.ipynb`:
  - cell konfigurasi eksperimen (preprocess/training/evaluation),
  - dukungan run single mode atau dual mode (`zero_shot + few_shot`),
  - cell komparasi otomatis metrik zero-shot vs few-shot.

### Kompatibilitas Metodologi Skripsi
- Model inti tetap: `Hybrid CNN-LSTM Autoencoder`.
- Paradigma utama tetap: unsupervised (normal-only training pada source).
- Perubahan ini merupakan tuning pipeline eksperimen, bukan pergantian metode utama.

### Validasi Implementasi
- Unit test ditambahkan:
  - `tests/test_preprocess_features.py`
  - `tests/test_threshold_modes.py`
- Status:
  - unit tests lulus,
  - compile check scripts lulus.

### Reproducibility
- Gunakan notebook:
  - `notebooks/colab_ready.ipynb` atau
  - `notebooks/colab_ready_online.ipynb`.
- Atau jalankan berurutan:
  1. `python scripts/preprocess.py --config config.yaml`
  2. `python scripts/train_cnn_lstm_ae.py --config config.yaml`
  3. `python scripts/eval_metrics.py --config config.yaml --model models/cnn_lstm_ae/best_model.keras --tag cnn_lstm`
- Untuk perbandingan mode:
  - jalankan dua kali evaluasi dengan `evaluation.mode=zero_shot` dan `evaluation.mode=few_shot`,
  - simpan tag output terpisah (mis. `cnn_lstm_zero_shot` dan `cnn_lstm_few_shot`).

### Catatan
- Angka dampak pasca v1.1.0 belum dicatat di dokumen ini sampai rerun eksperimen selesai.
- Setelah rerun, tambahkan subsection `Post-v1.1.0 Results` (before vs after) untuk pelaporan Bab 4.
