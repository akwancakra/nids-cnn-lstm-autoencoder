# Tahap 2 - Train Hybrid CNN-LSTM Autoencoder

Script: `scripts/train_cnn_lstm_ae.py`

## Tujuan

- Melatih model utama pada data normal (unsupervised reconstruction learning).

## Input

- Shard train/val dari preprocess.
- Konfigurasi training di `config.yaml`.

## Arsitektur Inti

1. Encoder CNN:
- Conv1D -> BatchNorm -> ReLU -> MaxPool (2 blok).

2. Encoder LSTM:
- LSTM stacked untuk pola temporal.

3. Bottleneck:
- Dense latent vector.

4. Decoder:
- RepeatVector -> LSTM stacked -> TimeDistributed(Dense).

## Objective

- Target training adalah rekonstruksi input: `(x -> x_hat)`.
- Loss: MSE (`mean((x - x_hat)^2)`).

## Callback yang Dipakai

- `EarlyStopping` (hindari overfit).
- `ReduceLROnPlateau` (turunkan learning rate saat stagnan).
- `ModelCheckpoint` (simpan best model).

## Output

- `models/cnn_lstm_ae/best_model.keras`
- `models/cnn_lstm_ae/final_model.keras`
- `results/logs/cnn_lstm_history.json`
- `results/logs/config_snapshot.json`

## Cara Baca Progres

- `loss` dan `val_loss` menurun -> training sehat.
- Jika `val_loss` naik jauh saat `loss` turun -> indikasi overfit.

## Glosarium Singkat

- `Unsupervised`: Training tanpa label attack sebagai target langsung.
- `Autoencoder`: Model yang belajar merekonstruksi input.
- `Bottleneck/Latent`: Representasi ringkas dari input di tengah model.
- `MSE (Mean Squared Error)`: Rata-rata kuadrat selisih input vs output.
- `EarlyStopping`: Hentikan training jika validasi tidak membaik.
- `ModelCheckpoint`: Simpan bobot model terbaik selama training.
