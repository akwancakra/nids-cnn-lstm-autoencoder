# Sprint 4 — Staged Experiments dan Lock Konfigurasi Baseline

## Konteks dan Tujuan
Sprint 4 mengimplementasikan framework staged experiments yang mendasari sprint5–n: peningkatan bertahap di preprocess, model arsitektur, dan threshold strategy. Tujuan final menemukan konfigurasi yang dapat memenuhi adaptive recall target (0.40) dengan guardrail CIC FPR ≤ 0.10.

## Konfigurasi Baseline
Evolusi dari Sprint 3: robust scaler (q995_clip20) dengan correlation filter 0.90, hybrid CNN-LSTM AE model dengan CNN filters [64,128], LSTM [128,64], latent dim 32, dropout 0.2, huber loss sebagai default, score mode recon_huber untuk saat evaluasi dengan threshold berbasis source percentile (default p97). Seed 42 dikunci untuk reproducibility.

## Eksperimen yang Dijalankan

### Stage 0 (Repro)
| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cic_f1 | cic_fpr |
|--------|---------|--------|------------|--------|---------|--------|---------|
| s4_00_repro_v4_full | repro baseline | sukses | 0.612 | 0.736 | 0.055 | 0.795 | 0.055 |

### Stage 1 (Preprocess sweep)
| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cic_f1 | cic_fpr |
|--------|---------|--------|------------|--------|---------|--------|---------|
| s4_p01_robust_q995_clip20_corr090 | official baseline | sukses | 0.490 | 0.633 | 0.144 | 0.794 | 0.039 |
| s4_p02_robust_q997_clip20_corr090 | q997 clipping | sukses | 0.250 | 0.380 | 0.162 | 0.803 | 0.039 |
| s4_p03_quantile_q995_clip20_corr090 | quantile scaler | sukses | 0.599 | 0.727 | 0.120 | 0.796 | 0.030 |
| s4_p04_robust_q995_clip15_corr085 | clip 15, corr 0.85 | sukses | 0.196 | 0.313 | 0.140 | 0.809 | 0.038 |
| s4_p05_robust_q995_clip20_corr095 | corr 0.95 | sukses | 0.217 | 0.339 | 0.165 | 0.794 | 0.042 |
| s4_p06_robust_q995_clip20_win20_stride2 | window 20 stride 2 | sukses | 0.052 | 0.098 | 0.042 | 0.763 | 0.043 |

### Stage 2 (Model/architecture)
| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cic_f1 | cic_fpr |
|--------|---------|--------|------------|--------|---------|--------|---------|
| s4_m01_baseline_arch | baseline ref (LSTM-AE hybrid) | sukses | 0.453 | 0.605 | 0.111 | 0.787 | 0.028 |
| s4_m02_dropout_030 | dropout 0.3 | sukses | 0.595 | 0.726 | 0.108 | 0.788 | 0.032 |
| s4_m03_latent_64 | latent dim 64 | sukses | 0.627 | 0.749 | 0.119 | 0.796 | 0.034 |
| s4_m04_cnn_64_128_128 | deeper CNN | sukses | 0.566 | 0.703 | 0.107 | 0.784 | 0.030 |
| s4_m05_lstm_192_96 | deeper LSTM | sukses | 0.628 | 0.741 | 0.165 | 0.805 | 0.028 |
| s4_m06_loss_huber | huber loss (score mode recon_huber) | sukses | 0.639 | 0.758 | 0.118 | 0.795 | 0.032 |
| s4_m07_loss_mse_mae_mix_a07 | hybrid loss α=0.7 | gagal | - | - | - | - | - |
| s4_m08_hybrid_score_alpha07 | recon_latent hybrid (α=0.7) | sukses | 0.012 | 0.023 | 0.018 | 0.040 | 0.026 |

### Stage 3 (Threshold sweep dengan huber loss)
Catatan: Stage 3 menggunakan score mode recon_huber, bukan recon_mse.

| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cic_f1 | cic_fpr | f1_gap |
|--------|---------|--------|------------|--------|---------|--------|---------|--------|
| s4_t01_source_percentile_p93 | percentile p93 | sukses | 0.675 | 0.771 | 0.191 | 0.806 | 0.060 | 0.035 |
| s4_t02_source_percentile_p95 | percentile p95 | sukses | 0.662 | 0.768 | 0.154 | 0.801 | 0.047 | 0.033 |
| s4_t03_source_percentile_p97 | percentile p97 | sukses | 0.639 | 0.758 | 0.118 | 0.795 | 0.032 | 0.037 |
| s4_t04_source_gaussian_k18 | gaussian k=1.8 | sukses | 0.669 | 0.770 | 0.173 | 0.803 | 0.054 | 0.034 |
| s4_t05_source_calib_f1 | calib F1 (max CSE F1, ignore FPR) | sukses | **0.711** | **0.726** | **0.615** | **0.823** | **0.293** | 0.097 |
| s4_t06_source_calib_guardrail_fpr010 | calib guardrail | sukses | **0.690** | **0.767** | 0.271 | 0.811 | **0.093** | **0.044** |

### Stage 4 (Seed confirm dengan huber loss)
Catatan: Stage 4 juga menggunakan score mode recon_huber dan threshold source_calib_guardrail.

| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cic_f1 | cic_fpr |
|--------|---------|--------|------------|--------|---------|--------|---------|
| s4_c01_best_seed42_full | retrain best (huber, guardrail) | sukses | 0.688 | 0.766 | 0.268 | 0.811 | 0.094 |
| s4_c02_best_seed1234_full | seed 1234 (huber, guardrail) | sukses | 0.689 | 0.767 | 0.270 | 0.810 | 0.098 |

## Temuan Utama
Discovery menjawab bahwa staged approach bekerja tetapi mengonfirmasi limitation arsitektur CNN-LSTM AE: best achievable recall ~0.69 masih di bawah level yang diinginkan untuk produksi, meskipun sudah dengan pendekatan threshold optimal (source_calib_guardrail). Huber loss dengan score mode recon_huber memberikan baseline yang solid: s4_m06 recall 0.639, s4_t06 recall 0.690. Modification terbukti bekerja: latent dim 64 (recall 0.627), quantile scaler (recall 0.599), dan huber loss semuanya menghasilkan trade-off yang lebih baik daripada baseline robust scaler. Ops Tidak Terbukti Bekerja: s4_m08 hybrid_score_alpha07 gagal total (recall 0.012), menandakan bahwa menggabungkan reconstruction index dengan latent space score tanpa training adversarial tidak efektif. s4_p06 (window 20 stride 2) terlalu agresif (recall hancur ke 0.052). s4_p04 (clip down ke 15) bertambah recall sebenarnya (0.196) tapi F1 masih rendah (0.313).

## Keputusan Gate
Gate pass: true. Adaptive recall target 0.40 tercapai dengan s4_t06 (0.690 > 0.40). CIC FPR guardrail (≤0.10) juga terpenuhi (0.093). Sprint 4 dianggap berhasil menetapkan baseline configuration yang solid untuk eksperimen selanjutnya. Note: s4_m07 mse_mae_mix loss gagal dieval karena bug infrastructure, bukan karena intrinsic performa buruk.

## Warisan untuk Sprint Berikutnya
Konfigurasi s4_t06 dengan keduanya huber loss (recon_huber score mode) dan source_calib_guardrail_fpr010 threshold locked sebagai final baseline: preprocess robust_q995_clip20_corr090, hybrid model, huber loss, seed 42. Hipotesis bahwa target recall 0.75 dalam Sprint 5 akan sangat agresif dan mungkin memerlukan pendekatan arsitektur yang berbeda (pivot ke USAD). Issue mse_mae_mix loss diidentifikasi sebagai infrastructure bug bukan performa, perlu perbaikan di Sprint 5.
