# Sprint 3 — Transisi ke Mode Anomaly-First dan Threshold Terstruktur

## Konteks dan Tujuan
Sprint 3 adalah turning point yang mengubah fokus dari accuracy agregat ke mode anomaly-first dengan FPR guardrail eksplisit. Tujuan menemukan threshold yang memaksimalkan recall attack sambil menjaga false positive di batas operasional, serta meningkatkan generalisasi ke domain target CSE.

## Konfigurasi Baseline
Pipeline evolve dengan lebih structured: sequence config (seq_len=10, stride=5), quantile scaler fitted pada benign CIC train (2.27M sample), strict file separation untuk CIC (8 train, 8 test) dan CSE (3 test). Model CNN-LSTM AE dengan total parameter 70,866, input shape (10, 34), threshold evolu dengan tiga pendekatan: percentile, gaussian, dan source calibrasi.

## Eksperimen yang Dijalankan
Based on Sprint 3 V4 analysis:

| run_id | variasi | status | metrik utama (CSE) |
|--------|---------|--------|-------------------|
| sprint3_v4_base | threshold percentile 95 | info tersedia | recall=0.0439, f1=0.0708, precision=0.1837 |
| PR-optimal hypothetical | threshold PR-optimal | simulasi | max_f1=0.4362, threshold=0.0034 |
| ROC-optimal hypothetical | threshold Youden-like | simulasi | tpr=0.4859, fpr=0.4526, threshold=0.0223 |

Run Stage-based tidak selesai seperti Sprint 4/5, namun dotenv manual V4 memberikan insight kritis tentang distribusi error dan trade-off threshold.

## Temuan Utama
Discovery paling penting adalah gap yang menakjubkan antara threshold operasional saat ini (recall CSE hanya 4.4%) dan potensi threshold optimal berbasis PR-curves (recall berpotensi mencapai 48.6% tetapi dengan FPR yang meningkatkan). Distribusi error reconstruction di source domain (CIC) memang различается secara statistik (KS=0.43, p-value=0), menunjukkan bahwa model dapat membedakan benign vs attack di domain familiar. Namun, saat diterapkan ke target domain CSE, distribusi error bergeser sedemikian rupa sehingga threshold yang aman untuk CIC menjadi terlalu ketat untuk CSE. Neural network training convergence sudah stabil (val loss turun ke ~0.02), jadi bottleneck bukan under-traning melainkan representasi dan strategi threshold.

## Keputusan Gate
Gate pass: false (konseptual). Target semua metrik >90% tidak tercapai dan realistis diakui secara eksplisit. Sprint 3 established bahwa pendekatan zero-shot threshold source-only punya limitation fundamental untuk domain shift. Keputusan dilanjutkan ke Sprint 4 dengan pendekatan anomaly-first ranked by recall, bukan accuracy balanced.

## Warisan untuk Sprint Berikutnya
Insight bahwa PR-optimal dan ROC-optimal threshold memberikan jalan menuju recall yang lebih tinggi menjadi fondasi untuk Sprint 4. Kongsi bahwa objective harus berubah menjadi memaksimalkan recall dengan guardrail FPR eksplisit. Hipotesis bahwa hybrid scoring (recon_latent combination) dan loss robust (huber, mse_mae_mix) dapat memperbaiki separasi antar kelas di target domain.
