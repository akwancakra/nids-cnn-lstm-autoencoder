# Tahap 3 - Evaluate (CIC + CSE)

Script: `scripts/eval_metrics.py`

## Tujuan

- Menilai performa model di:
  - CIC test (in-distribution)
  - CSE test (cross-dataset / OOD)

## Langkah Teknis

1. **Load model**
- Ambil model `best_model.keras`.

2. **Hitung threshold**
- Dari reconstruction error pada CIC validation benign.
- Menggunakan percentile dari `config.yaml` (default 99).

3. **Inferensi**
- Hitung reconstruction error untuk CIC test dan CSE test.
- Prediksi anomali: `score > threshold`.

4. **Hitung metrik**
- Accuracy, Precision, Recall, F1, FPR, ROC-AUC.
- Simpan juga TP, FP, TN, FN.

5. **Hitung generalization gap**
- Selisih metrik CIC vs CSE.

6. **Simpan plot**
- ROC curve, confusion matrix, distribusi error.

## Output (contoh `--tag cnn_lstm`)

- `results/metrics/cnn_lstm_cic_metrics.json`
- `results/metrics/cnn_lstm_cse_metrics.json`
- `results/metrics/cnn_lstm_generalization_gap.json`
- `results/plots/cnn_lstm/roc_cic.png`
- `results/plots/cnn_lstm/roc_cse.png`
- `results/plots/cnn_lstm/cm_cic.png`
- `results/plots/cnn_lstm/cm_cse.png`

## Cara Baca Hasil

- CIC bagus, CSE drop besar -> generalization masih lemah.
- FPR tinggi -> banyak false alarm.
- F1 stabil di CIC+CSE -> generalization lebih baik.

## Glosarium Singkat

- `In-distribution`: Data uji dari domain yang sama dengan data latih (CIC).
- `Cross-dataset / OOD`: Data uji dari domain berbeda (CSE).
- `ROC-AUC`: Ukuran kemampuan model memisahkan normal vs attack di berbagai threshold.
- `TP/FP/TN/FN`: Komponen confusion matrix untuk hitung metrik klasifikasi.
- `Generalization gap`: Selisih performa CIC dan CSE.
