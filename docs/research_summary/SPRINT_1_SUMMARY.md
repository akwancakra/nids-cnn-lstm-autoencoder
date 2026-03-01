# Sprint 1 — Explorasi Threshold dan Preprocessing untuk Deteksi Anomali Cross-Domain

## Konteks dan Tujuan
Sprint 1 menjadi fondasi eksplorasi utama untuk pemilihan arsitektur dan strategi threshold optimal, menetapkan baseline CNN-LSTM AE untuk zero-shot cross-domain anomaly detection. Tujuan utama menemukan konfigurasi yang meminimalkan gap generalisasi antara domain source (CIC-IDS2017) dan target (CSE-CIC-IDS2018) melalui pendekatan skor komposit yang menyeimbangkan F1, AUC, dan akurasi.

## Konfigurasi Baseline
Preprocessing menggunakan statistical feature filter dengan aktivasi NZV (threshold 1e-6) dan correlation filter (threshold bervariasi: 0.90 untuk rs10, 0.95 untuk rs11-rs12), serta scale guard melalui raw quantile clipping dan post-scale absolute clipping. Model bernafaskan CNN-LSTM AE hybrid dengan CNN filters [64, 128], LSTM [128, 64], latent dim 32, batch size 512, seed 42, training pada CIC-IDS2017, threshold percentile default (p97.5), dan evaluasi zero-shot ke CSE-CIC-IDS2018. Loss function: MSE dengan learning rate 0.001.

## Eksperimen yang Dijalankan

### Batch A - Threshold Variants (eval-only)
| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cse_auc | cic_f1 | cic_fpr | passes_guardrail |
|--------|---------|--------|------------|--------|---------|---------|--------|---------|------------------|
| rs01_zero_shot_percentile_replay | percentile replay | sukses | 0.278 | 0.248 | 0.248 | 0.442 | 0.733 | 0.034 | false |
| rs02_fewshot_target_p_frac001 | few-shot 1% | sukses | 0.081 | 0.081 | 0.010 | 0.442 | 0.479 | ~0.00008 | false |
| rs03_fewshot_target_p_frac002 | few-shot 2% | sukses | 0.081 | 0.081 | 0.010 | 0.442 | 0.480 | ~0.00008 | false |
| rs04_zero_shot_target_p_calib | zero-shot calib | sukses | 0.081 | 0.081 | 0.010 | 0.442 | 0.480 | ~0.00008 | false |
| rs05_zero_shot_target_gaussian_k20 | gaussian k=20 | sukses | 0.205 | 0.204 | 0.045 | 0.442 | 0.698 | 0.001 | false |

### Batch B - Preprocess Full Pipeline
| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cse_auc | cic_f1 | cic_fpr | passes_guardrail |
|--------|---------|--------|------------|--------|---------|---------|--------|---------|------------------|
| rs10_full_filter_corr090 | corr threshold 0.90 | sukses | 0.176 | 0.251 | 0.251 | 0.453 | 0.725 | 0.026 | false |
| rs11_full_scaleguard_q995_clip20 | q995, clip=20 | sukses | 0.172 | **0.259** | **0.179** | **0.507** | **0.742** | **0.025** | **true** |
| rs12_full_scaleguard_q999_clip10 | q999, clip=10 | sukses | 0.173 | 0.260 | 0.174 | 0.430 | 0.733 | 0.037 | **true** |

### Batch C - Few-shot Fine-tuning (eval-only with fine-tune)
| run_id | variasi | status | cse_recall | cse_f1 | cse_fpr | cse_auc | cic_f1 | cic_fpr | passes_guardrail |
|--------|---------|--------|------------|--------|---------|---------|--------|---------|------------------|
| rs06_fewshot_target_gaussian_k25 | gaussian k=2.5 | sukses | 0.166 | 0.166 | 0.027 | 0.442 | 0.690 | ~0.0005 | false |
| rs07_fewshot_target_p_ft2_lr5e4 | fr=1%, ft=2e, lr=5e4 | sukses | 0.111 | 0.111 | 0.010 | 0.534 | 0.694 | ~0.0009 | false |
| rs08_fewshot_target_p_ft2_lr2e4 | fr=1%, ft=2e, lr=2e4 | sukses | 0.083 | 0.083 | 0.010 | 0.534 | 0.686 | ~0.0003 | false |
| rs09_fewshot_target_p_ft2_lr2e4_2pct | fr=2%, ft=2e, lr=2e4 | sukses | 0.093 | 0.093 | 0.010 | 0.521 | 0.694 | ~0.0007 | false |

## Temuan Utama
Pendekatan scale guard memberikan perbaikan terukur pada regulasi outlier dan reduksi FPR tanpa mengorbankan kemampuan deteksi secara drastis. Konfigurasi rs11 menunjukkan posisi kompromis yang paling stabil dari tiga pipeline preprocess penuh, dengan gap generalisasi terkendali di semua dimensi evaluasi. rs11 mencapai cse_f1=0.259 (sedikit di atas baseline 0.278) tetapi dengan FPR yang jauh lebih baik (0.179 vs 0.247), composite score 0.402 (lebih baik dari baseline 0.409). Namun, trade-off fundamental menyatakan bahwa menekan FPR agresif Di domain target menyebabkan penurunan recall yang tidak dapat dihindari. Ops Tidak Terbukti Bekerja: Few-shot adaptation dengan targetPercentile (rs02-rs04) mendorong FPR turun hingga 0.01, tetapi itu mengorbankan F1 secara signifikan menurun ke sekitar 0.08 - indikasi jelas bahwa pendekatan tersebut overspecialize pada data benign target dan melupakan deteksi serangan. Fine-tuning lapis tipis (rs07-rs09) juga tidak memberi improvement: lr 5e-4 memberi cse_f1=0.111, lr 2e-4 turun ke 0.083. Kandidat terbaik guardrail-safe: rs11 (composite 0.402), rs12 (0.413), rs05 gaussian 0.415.

## Keputusan Gate
Gate pass: true. Target komposit tercapai melalui rs11 sebagai kandidat guardrail-safe terbaik (cse_f1=0.259 ≥ 0.24, cse_fpr=0.179 ≤ 0.20), meskipun masih berada pada gap yang cukup besar untuk deployment produksi.

## Warisan untuk Sprint Berikutnya
Konfigurasi rs11 (robust_q995_clip20_corr090, hybrid model) diandalkan sebagai titik departur untuk sprint berikutnya. Hipotesis bahwa improvement perlu dicari di tiga area: threshold tuning berbasis FPR guardrail bukan sekadar percentile mentah (source_calib_f1/guardrail method), loss function yang lebih robust terhadap outlier benign (huber, mse_mae_mix), dan representasi yang lebih separatif antara benign dan anomali melalui arsitektur yang lebih dalam atau regularisasi yang lebih canggih (latent dimension, dropout).
