# RunPod Training Guide (NIDS CNN-LSTM Autoencoder)

Panduan ini untuk menjalankan eksperimen skripsi di RunPod secara stabil dan hemat risiko.
Fokus: **Pods on-demand**, training berbasis **TensorFlow**, dan pipeline repo ini.

## 1. Rekomendasi Singkat

- Gunakan **on-demand pod** untuk run utama (lebih stabil dari spot).
- `Any region` boleh untuk training, tapi hindari jika ada kebutuhan lokasi data tertentu.
- Simpan artefak penting secara persisten (model, logs, metrics) sebelum terminate pod.

## 2. Provision Pod

1. Buat pod di RunPod.
2. Pilih GPU sesuai budget (1 GPU sudah cukup untuk project ini).
3. Aktifkan SSH.
4. Siapkan persistent storage yang cukup untuk:
   - dataset raw (`CIC-IDS2017`, `CSE-CIC-IDS2018`)
   - shard processed
   - model checkpoints

## 3. Masuk Pod dan Setup Project

```bash
# SSH ke pod
ssh root@<POD_IP> -p <PORT>

# kerja di workspace
cd /workspace

# clone project
git clone <URL_REPO_KAMU> nids-cnn-lstm-autoencoder
cd nids-cnn-lstm-autoencoder

# buat env + install dependency
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Verifikasi GPU:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 4. Upload Dataset

Pastikan dataset ada di path berikut:

- `data/raw/CIC-IDS2017`
- `data/raw/CSE-CIC-IDS2018`

Jika transfer file besar:
- prioritas pakai metode transfer yang mendukung file besar/stabil
- hindari upload via UI notebook untuk dataset besar

## 5. Konfigurasi Run GPU

Di `config.yaml`, pastikan:

```yaml
training:
  force_cpu: false
```

Catatan:
- Secara default repo kamu sempat memakai `force_cpu: true` untuk lokal.
- Di RunPod GPU, harus diubah ke `false` agar TensorFlow pakai GPU.

## 6. Jalankan Pipeline End-to-End

Urutan standar:

1. Preprocess (membuat shard train/val/test)
2. Training hybrid model
3. Evaluasi CIC dan CSE

```bash
cd /workspace/nids-cnn-lstm-autoencoder
source .venv/bin/activate

# 1) preprocess
python scripts/preprocess.py --config config.yaml

# 2) train hybrid cnn-lstm ae
python scripts/train_cnn_lstm_ae.py --config config.yaml

# 3) evaluate + generalization gap
python scripts/eval_metrics.py \
  --config config.yaml \
  --model models/cnn_lstm_ae/best_model.keras \
  --tag cnn_lstm
```

## 7. Jalankan Aman dari Disconnect SSH

Gunakan `tmux`:

```bash
tmux new -s nids-train
# jalankan command training di dalam tmux
python scripts/train_cnn_lstm_ae.py --config config.yaml

# detach: Ctrl+B lalu D
# attach lagi:
tmux attach -t nids-train
```

Alternatif:

```bash
nohup python scripts/train_cnn_lstm_ae.py --config config.yaml > train.log 2>&1 &
tail -f train.log
```

## 8. Monitoring

```bash
# utilisasi GPU
nvidia-smi

# monitor log training
tail -f train.log
```

Artefak penting:
- model: `models/cnn_lstm_ae/`
- log training: `results/logs/`
- metrik evaluasi: `results/metrics/`
- plot: `results/plots/`

## 9. Download Hasil

Minimal yang diambil:
- `models/cnn_lstm_ae/best_model.keras`
- `models/cnn_lstm_ae/final_model.keras`
- `results/`
- `config.yaml` yang dipakai run

Contoh arsip di pod:

```bash
cd /workspace/nids-cnn-lstm-autoencoder
tar -czf artifacts_runpod.tar.gz models results config.yaml
```

Lalu download ke lokal dengan metode transfer yang kamu pakai.

## 10. Stop vs Terminate

- **Stop**: pod berhenti, resource compute berhenti, tapi storage masih ada.
- **Terminate**: pod dihapus; pastikan artefak sudah dipindah atau ada di volume persisten.

Checklist sebelum terminate:
- [ ] `best_model.keras` sudah aman
- [ ] `results/metrics` dan `results/plots` sudah aman
- [ ] config run final sudah disimpan

## 11. Skenario Eksperimen Skripsi (Direkomendasikan)

1. **Run utama (metodologi final)**:
   - train/val benign-only (sesuai implementasi terbaru di preprocess)
   - test pada CIC mixed + CSE mixed

2. **Run pembanding (opsional ablation)**:
   - mixed-train untuk perbandingan terhadap run utama

Output yang dibandingkan:
- accuracy, precision, recall, F1, FPR
- ROC-AUC
- generalization gap (`*_generalization_gap.json`)
