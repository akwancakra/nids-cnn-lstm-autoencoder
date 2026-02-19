# Release v1.1.0-experimental - CNN-LSTM Autoencoder (CIC/CSE Evaluation)

Tanggal rilis: 2026-02-19

## Highlights

- Training dan evaluasi end-to-end selesai pada run terbaru.
- Pipeline preprocess dengan scale guard aktif menghasilkan training stabil (loss tidak meledak).
- Rilis ini ditandai sebagai **experimental** karena performa cross-dataset (CSE) menurun signifikan.

## Training Summary

- epochs: 100
- best_epoch: 97
- best_val_loss: 0.927873
- final_loss: 2.231240
- final_val_loss: 0.938662

## Evaluation Summary (Tag: cnn_lstm)

### CIC-IDS2017 (in-domain)

- accuracy: 0.7975
- precision: 0.9385
- recall: 0.6017
- f1: 0.7333
- roc_auc: 0.8203
- fpr: 0.0339

### CSE-CIC-IDS2018 (cross-domain)

- accuracy: 0.4591
- precision: 0.4716
- recall: 0.1971
- f1: 0.2780
- roc_auc: 0.4421
- fpr: 0.2474

### Generalization Gap

- f1_gap: 0.4553
- auc_gap: 0.3782
- accuracy_gap: 0.3384
- mode: zero_shot
- threshold_method: percentile
- threshold: 5.22420745

## Model Artifacts

- models/cnn_lstm_ae/best_model.keras
- models/cnn_lstm_ae/final_model.keras
- models/cnn_lstm_ae/checkpoints/

## Notes

- Warning `ptxas.exe` terdeteksi saat run, namun non-fatal; TensorFlow fallback ke driver PTX compilation.
- Fokus lanjutan: evaluasi mode `few_shot` dan metode threshold target-domain untuk perbaikan generalisasi CSE.

## Release Artifacts

- cnn-lstm-v1.1.0-experimental-models.zip (85330403 bytes)
- cnn-lstm-v1.1.0-experimental-metrics-and-plots.zip (113460 bytes)
- cnn-lstm-v1.1.0-experimental-logs.zip (4121 bytes)
- cnn-lstm-v1.1.0-experimental-manifest.json

## Checksums (SHA256)

- cnn-lstm-v1.1.0-experimental-models.zip: `7c1155282aab18868d8fb125904962800b690a080002686e05a42bd13016e737`
- cnn-lstm-v1.1.0-experimental-metrics-and-plots.zip: `d5e4209b1c8d59ab30ca44a67df6fe9ac78fe3633eaed0a7e9d8b4b0fa56cac7`
- cnn-lstm-v1.1.0-experimental-logs.zip: `59c3aeb51612ccb017455fe90c04bfc2357bbe72787455010cc9987a0766b0bb`
