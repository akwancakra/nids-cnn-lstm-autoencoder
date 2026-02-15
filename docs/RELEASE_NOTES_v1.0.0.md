# Release v1.0.0 — CNN-LSTM Autoencoder (CIC/CSE Evaluation)

Tanggal rilis: 2026-02-15

## Highlights

- Training dan evaluasi end-to-end berhasil dijalankan pada pipeline notebook lokal.
- Evaluasi cross-dataset selesai untuk CIC-IDS2017 dan CSE-CIC-IDS2018.
- Output metrik, gap generalisasi, dan plot evaluasi tersedia lengkap.

## Model Artifacts

- Model terbaik: models/cnn_lstm_ae/best_model.keras
- Model final: models/cnn_lstm_ae/final_model.keras
- Checkpoint periodik: models/cnn_lstm_ae/checkpoints/

## Evaluation Summary (Tag: cnn_lstm)

### CIC-IDS2017 (in-domain)

- accuracy: 0.8132
- precision: 0.9553
- recall: 0.6256
- f1: 0.7561
- roc_auc: 0.8158
- fpr: 0.0252

### CSE-CIC-IDS2018 (cross-domain)

- accuracy: 0.6514
- precision: 0.6612
- recall: 0.6977
- f1: 0.6790
- roc_auc: 0.6574
- fpr: 0.4005

### Generalization Gap

- f1_gap: 0.0771
- auc_gap: 0.1583
- accuracy_gap: 0.1618

## Files Produced

- results/metrics/cnn_lstm_cic_metrics.json
- results/metrics/cnn_lstm_cse_metrics.json
- results/metrics/cnn_lstm_generalization_gap.json
- results/plots/cnn_lstm/roc_cic.png
- results/plots/cnn_lstm/roc_cse.png
- results/plots/cnn_lstm/cm_cic.png
- results/plots/cnn_lstm/cm_cse.png
- results/plots/cnn_lstm/err_dist_cic.png
- results/plots/cnn_lstm/err_dist_cse.png

## Notes

- Warning ptxas.exe terdeteksi saat run, namun non-fatal; TensorFlow fallback ke driver PTX compilation.
- Evaluasi selesai dengan exit code 0.

## Suggested Release Assets

1. cnn-lstm-v1.0.0-models.zip
2. cnn-lstm-v1.0.0-metrics-and-plots.zip
3. RELEASE_NOTES_v1.0.0.md
