# Evaluation Findings — CNN-LSTM Autoencoder

Tanggal evaluasi: 2026-02-15  
Model: `models/cnn_lstm_ae/best_model.keras`  
Konfigurasi: `config.yaml` (threshold percentile pada CIC validation benign)

## Ringkasan Eksekutif

Model berhasil dilatih dan dievaluasi pada dua domain data:

- **In-domain (CIC-IDS2017)**: performa relatif baik dan stabil.
- **Cross-domain (CSE-CIC-IDS2018)**: performa menurun dengan **false positive rate** yang tinggi.

Temuan utama menunjukkan adanya **domain shift** yang signifikan dari CIC ke CSE.

## Hasil Evaluasi Utama

| Dataset | Accuracy | Precision | Recall |     F1 | ROC-AUC |    FPR |
| ------- | -------: | --------: | -----: | -----: | ------: | -----: |
| CIC     |   0.8132 |    0.9553 | 0.6256 | 0.7561 |  0.8158 | 0.0252 |
| CSE     |   0.6514 |    0.6612 | 0.6977 | 0.6790 |  0.6574 | 0.4005 |

Sumber metrik:

- `results/metrics/cnn_lstm_cic_metrics.json`
- `results/metrics/cnn_lstm_cse_metrics.json`

## Generalization Gap

- **F1 gap**: 0.0771
- **AUC gap**: 0.1583
- **Accuracy gap**: 0.1618

Sumber:

- `results/metrics/cnn_lstm_generalization_gap.json`

## Interpretasi

1. **Performa in-domain kuat** (CIC): AUC dan F1 cukup baik, serta FPR sangat rendah.
2. **Performa cross-domain turun** (CSE): AUC turun cukup jauh dan FPR meningkat tajam.
3. Threshold berbasis CIC cenderung tidak cukup robust ketika dipindahkan ke distribusi CSE.

## Catatan Teknis Selama Run

- Evaluasi selesai sukses (`EXIT CODE 0`), seluruh output metrik dan plot berhasil dibuat.
- Warning `ptxas.exe` muncul tetapi **non-fatal**; TensorFlow fallback ke driver PTX compilation dan evaluasi tetap berjalan.

## File Output yang Dihasilkan

- `results/metrics/cnn_lstm_cic_metrics.json`
- `results/metrics/cnn_lstm_cse_metrics.json`
- `results/metrics/cnn_lstm_generalization_gap.json`
- `results/plots/cnn_lstm/roc_cic.png`
- `results/plots/cnn_lstm/roc_cse.png`

## Rekomendasi Lanjutan

1. **Threshold tuning** khusus target domain (CSE) untuk menurunkan FPR.
2. Uji beberapa skenario threshold (mis. p99.5, p99.7, p99.9) dan bandingkan trade-off recall vs FPR.
3. Pertimbangkan strategi adaptasi domain (normalization alignment / fine-tuning ringan) bila target utama adalah performa lintas dataset.
4. Jalankan baseline (`lstm_ae`) sebagai pembanding resmi agar klaim peningkatan model hybrid lebih kuat.

## Kesimpulan

Model CNN-LSTM Autoencoder sudah valid untuk skenario in-domain CIC, namun generalisasi ke CSE masih perlu perbaikan, terutama untuk menekan false positive rate dan menjaga stabilitas metrik lintas domain.
