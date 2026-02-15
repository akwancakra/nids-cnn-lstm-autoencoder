# Tahap 4 - Baseline (Opsional tapi Penting)

Scripts:
- `scripts/train_lstm_ae.py`
- `scripts/eval_metrics.py --model models/lstm_ae/best_model.keras --tag lstm_ae`

## Tujuan

- Menyediakan pembanding adil terhadap model hybrid utama.

## Yang Dibandingkan

- CNN-LSTM Autoencoder (utama) vs LSTM Autoencoder (baseline).
- Fokus banding:
  - F1 (CIC)
  - Accuracy/F1 (CSE)
  - FPR
  - Generalization gap

## Output Baseline

- `models/lstm_ae/best_model.keras`
- `results/metrics/lstm_ae_cic_metrics.json`
- `results/metrics/lstm_ae_cse_metrics.json`
- `results/metrics/lstm_ae_generalization_gap.json`

## Interpretasi

- Jika hybrid konsisten lebih baik (terutama di CSE), kontribusi metode makin kuat.
- Jika baseline mirip/better, perlu analisis kenapa kompleksitas hybrid belum memberi gain.

## Glosarium Singkat

- `Baseline`: Model pembanding untuk menilai apakah model utama benar-benar lebih baik.
- `Fair comparison`: Perbandingan dengan pipeline, split data, dan metrik yang sama.
- `Ablation mindset`: Melihat dampak tiap komponen model dengan pembanding yang sederhana.
