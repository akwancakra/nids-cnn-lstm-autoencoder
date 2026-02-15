# Panduan Notebook (4 Tahapan)

Dokumen ini mengikuti alur di `notebooks/colab_ready.ipynb` dengan penjelasan sederhana.

## Tahap 1 - Preprocess Data

Tujuan:
- Membersihkan data mentah.
- Menyamakan fitur CIC-IDS2017 dan CSE-CIC-IDS2018.
- Membuat data sequence untuk model.

Yang dikerjakan:
1. Baca CSV dari `data/raw/...`.
2. Cari kolom label (`Label`).
3. Bersihkan nilai `NaN/Inf`.
4. Lakukan scaling fitur (fit dari data BENIGN CIC).
5. Ubah data jadi window sequence (`window_size`, `stride`).
6. Simpan ke shard `.npz` + `manifest.json`.

Output penting:
- `data/processed/scaler.pkl`
- `data/processed/feature_columns.json`
- `data/processed/shards/cic/train/manifest.json`
- `data/processed/shards/cic/val/manifest.json`
- `data/processed/shards/cic/test/manifest.json`
- `data/processed/shards/cse/test/manifest.json`

Checklist cepat:
- Tidak ada error `Label column not found`.
- Log menampilkan `Feature intersection count: ...`.
- File `manifest.json` terbentuk.

## Tahap 2 - Train Hybrid CNN-LSTM Autoencoder

Tujuan:
- Melatih model utama pada traffic normal (unsupervised).

Yang dikerjakan:
1. Load shard `cic/train` dan `cic/val`.
2. Bangun arsitektur CNN-LSTM Autoencoder.
3. Train dengan `EarlyStopping` + `ModelCheckpoint`.
4. Simpan model terbaik dan model final.

Output penting:
- `models/cnn_lstm_ae/best_model.keras`
- `models/cnn_lstm_ae/final_model.keras`
- `results/logs/cnn_lstm_history.json`

Checklist cepat:
- `val_loss` turun/stabil.
- `best_model.keras` berhasil tersimpan.

## Tahap 3 - Evaluate (CIC + CSE)

Tujuan:
- Nilai performa model pada CIC (in-distribution) dan CSE (cross-dataset).

Yang dikerjakan:
1. Hitung threshold dari CIC validation BENIGN (percentile di config).
2. Evaluasi CIC test.
3. Evaluasi CSE test.
4. Hitung generalization gap.
5. Simpan metrics + plot.

Output penting (contoh `--tag cnn_lstm`):
- `results/metrics/cnn_lstm_cic_metrics.json`
- `results/metrics/cnn_lstm_cse_metrics.json`
- `results/metrics/cnn_lstm_generalization_gap.json`
- `results/plots/cnn_lstm/roc_cic.png`
- `results/plots/cnn_lstm/roc_cse.png`
- `results/plots/cnn_lstm/cm_cic.png`
- `results/plots/cnn_lstm/cm_cse.png`

Checklist cepat:
- File metrics CIC dan CSE keduanya ada.
- Nilai `fpr` tidak ekstrem.
- Generalization gap masih masuk target penelitianmu.

## Tahap 4 - Baseline (Opsional)

Tujuan:
- Bandingkan model utama dengan baseline LSTM Autoencoder.

Yang dikerjakan:
1. Train `train_lstm_ae.py`.
2. Eval dengan `eval_metrics.py --model models/lstm_ae/best_model.keras --tag lstm_ae`.
3. Bandingkan metrik dengan model utama.

Output penting:
- `models/lstm_ae/best_model.keras`
- `results/metrics/lstm_ae_cic_metrics.json`
- `results/metrics/lstm_ae_cse_metrics.json`
- `results/metrics/lstm_ae_generalization_gap.json`

## Cara Baca Hasil Dengan Cepat

Urutan cek yang paling praktis:
1. `..._cic_metrics.json` -> lihat F1/Recall/FPR.
2. `..._cse_metrics.json` -> lihat drop performa.
3. `..._generalization_gap.json` -> lihat seberapa besar gap.
4. Plot confusion matrix + ROC untuk validasi visual.
