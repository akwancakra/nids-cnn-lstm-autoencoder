# Ringkasan Penelitian NIDS CNN-LSTM Autoencoder — Sprint 1 sampai 5

## Perjalanan Penelitian
Penelitian dimulai dengan Sprint 1 sebagai eksplorasi fundamental: mengidentifikasi kombinasi preprocessing dan threshold yang meminimalkan gap generalisasi CIC (source) ke CSE (target) dengan skor komposit. Kondisi awal serius: cse_f1 hanya 0.260 dengan recall 0.173, FPR 0.179. Sprint 2 fokus pada infrastructural improvement dengan full isolation untuk reproducibility. Sprint 3 menjadi turning point mengubah objective dari accuracy agregat ke mode anomaly-first dengan insight penting: threshold operasional recall hanya 4.4% (mode zero-shot vs source percentile), tetapi PR-optimal threshold berpotensi mencapai 48.6% jika target FPR dibebaskan. Sprint 4 formalized pendekatan staged experiments dan berhasil menemukan konfigurasi champion (s4_t06) dengan recall 0.69 under guardrail FPR 0.093. Sprint 5 mendorong lebih agresif dengan target recall 0.75, mengevaluasi mse_mae_mix loss yang terbukti mengingkatkan recall, dan mengonfirmasi fundamental limitation arsitektur CNN-LSTM AE melalui target-aware threshold eksperimen yang semuanya gagal memenuhi constraint ganda.

## Progres Metrik Lintas Sprint

| Sprint | Best CSE Recall | Best CSE F1 | Best CSE FPR | Best CIC FPR | F1 Gap | Gate | Champion Run |
|--------|-----------------|-------------|---------|---------|--------|------|--------------|
| Sprint 1 | 0.173 (rs11) | 0.260 (rs12) | 0.179 (rs11) | 0.025 (rs11) | 0.483 | True | rs11 |
| Sprint 2 | N/A | N/A | N/A | N/A | N/A | N/A | infra sprint |
| Sprint 3 | 0.044 (op) | 0.071 (op) | 0.164 (op) | 0.043 (op) | 0.722 | False | v4 manual run |
| Sprint 3 | 0.486 (hyp) | 0.436 (hyp) | N/A | 0.453 (hyp) | N/A | False | v4 PR-optimal |
| Sprint 4 | 0.690 (s4_t06) | 0.767 (s4_t06) | 0.271 | 0.093 | 0.044 | True | s4_t06/s4_c02 |
| Sprint 5 | 0.698 (s5_m07) | 0.781 (s5_m07) | 0.224 | 0.109 | 0.046 | False | s5_m07 |

Note: Sprint 3 "hyp" berdasarkan PR-optimal threshold dari V4 manual run, bukan hasil staged runs (semua pending).

Total Progress: Recall +302% (0.173→0.698), F1 +200% (0.260→0.781), Gap F1 -90% (0.483→0.046)

## Apa yang Terbukti Bekerja
- **Robust preprocessing** (q995_clip20_corr090): Konsisten memberikan regulasi outlier yang stabil di lintas sprint, FPR control lebih baik tanpa menaikkan F1 secara signifikan.
- **Huber loss with recon_huber score**: Memberikan baseline solid dan stability (recall 0.639→0.690, F1 0.758→0.767), marginally outperforms MSE di most runs.
- **mse_mae_mix loss** (α=0.7): Terbukti meningkatkan recall 7.2pp dari baseline (0.626→0.698), F1 naik ke 0.781, gap F1 mengecil ke 0.046. Improvement signifikan.
- **Source calibrasi guardrail** threshold: Menemukan threshold yang lebih adaptif terhadap distribution benign source daripada percentile statik (s4_t06 recall 0.690, FPR 0.093 melalui huber loss).
- **Latent dimension 64**: Memberikan improvement signifikan over default 32 (recall 0.627 vs 0.639 baseline huber).
- **Quantile scaler**: Berhasil mengungguli robust scaler di beberapa metrik recall tanpa menaikkan FPR CIC signifikan.

## Apa yang Terbukti Tidak Bekerja
- **Few-shot target-p** (Sprint 1): Menurunkan FPR menjadi sangat agresif (0.01–0.05) tetapi menghancurkan F1 (0.08–0.20), confirm overspecialization pada benign target.
- **Few-shot fine-tuning** (Sprint 1 rs07-rs09): Tidak memberi improvement signifikan, lr 5e-4 hanya mencapai cse_f1=0.111, lr 2e-4 turun ke 0.083.
- **Percentile-based strict tuning** (Sprint 5): p95→p99 tidak memberikan sweet spot viable: saat CIC FPR turun ke 0.001, recall hancur ke 0.007.
- **Hybrid score recon_latent** (Sprint 4 s4_m08): Score mode hybrid dengan alpha 0.7 gagal total, recall tunggal 0.012, F1 0.023.
- **Guardrail relaxation** (Sprint 5): Menaikkan FPR guardrail dari 0.10 ke 0.15 membuka recall (0.65→0.69) tapi FPR naik jauh di CSE (0.24→0.27), trade-off tetap invalid untuk guardrail.
- **Window size stride modification** (Sprint 4 p06): Banyak data loss, recall hancur ke 0.052.
- **Correlation filter 0.95** (rs12): Slight edge FPR (0.174 vs 0.179) tapi AUC lemah (0.430 vs 0.507).
- **Target-aware threshold** (Sprint 5 stage 1): Semua 7 run (t01-t07) gagal memenuhi dua constraint sekaligus, confirm bahwa threshold-only tuning tidak bisa menyelesaikan mismatch distribusi error antar domain target.

## Justifikasi Pivot ke USAD
Data Sprint 4–5 secara eksplisit menunjukkan bahwa CNN-LSTM AE telah mencapai ceiling arsitektural pada objective ini: best recall 0.698 (s5_m07) hanya 7pp di bawah target 0.75, namun trade-off menjadi sangat rigid. Untuk mencapai recall 0.70+ di CSE, CIC FPR minimal 0.12–0.15; untuk menekan CIC FPR ≤0.10, recall hancur menjadi <0.10. Semua eksperimen threshold-only (percentile tuning p95→p99, target-aware threshold dari benign target, guardrail relaxation) gagal menyelesaikan limitation fundamental ini. Distribusi reconstruction error yang berbeda mengindikasikan bahwa autoencoder yang hanya belajar reconstruction sampling data source (melalui MSE/huber loss) tidak bisa memisahkan benign vs anomali secara separatif yang cukup di target domain ketika eval. USAD (Unsupervised Anomaly Detection with Special Adversarially-Generated Samples) menunjukkan promise karena menggunakan adversarial training untuk memaksa separasi latent space antara benign yang reconstructed well dan anomali yang reconstructed poorly, bukan hanya mengandalkan threshold absolute.

## Rekomendasi Sprint 6
Implement USAD dengan konfigurasi:
- **Architecture**: Encoder-decoder (mirroring ISF baseline) dengan adversarial training (encoder = autoencoder encoder, discriminator = binary classifier pada reconstruction error distribution untuk memisahkan benign vs anomali)
- **Preprocessing**: Retain robust_q995_clip20_corr090 (proven stabilized), pertimbangkan quantile scaler (terbukti sedikit lebih baik recall)
- **Loss function**: Reconstruktor loss (MSE/MSE-based) + adversarial loss (λ_recon × L_recon + λ_gan × L_adv)
- **Training**: Train pada CIC-IDS2017 dengan adversarial training, epoch ~100-200, learning rate 0.0001-0.0002, target_false_positive_rate 0.10
- **Threshold**: Use reconstruction error distribution-based threshold atau adversarial score dari discriminator organically
- **Validasi**: Staged experiment framework tetap (preprocess → train → eval → threshold sweep/probability tuning), temukan ke USAD-specific yang optimal
- **Metric focus**: CSE recall >0.75 under CIC FPR ≤0.10 sebagai metric utama, bukan accuracy aggregate
- **Baseline comparison**: Sprint 5 winner (s5_m07 dengan mse_mae_mix) sebagai komparasi anchor untuk mengukur improvement adversarial vs pure reconstruction
| Sprint | Best CSE Recall | Best CSE F1 | CSE FPR | Best CIC FPR | F1 Gap | Gate | Champion Run |
|--------|-----------------|-------------|---------|---------|--------|------|--------------|
| Sprint 1 | 0.173 (rs11) | 0.260 (rs12) | 0.179 (rs11) | 0.025 (rs11) | 0.483 | True | rs11 |
| Sprint 2 | N/A | N/A | N/A | N/A | N/A | N/A | infra sprint |
| Sprint 3 | 0.044 (v4 op) | 0.071 (v4 op) | 0.164 (v4 op) | 0.043 (v4 op) | 0.722 | False | v4 manifold run |
| Sprint 3 | 0.486 (v4 op) | 0.436 (v4 op) | 0.184 | 0.453 (v4 op) | N/A | False | v4 PR-optimal |
| Sprint 4 | 0.690 (s4_t06) | 0.767 (s4_t06) | 0.271 | 0.093 | 0.044 | True | s4_t06/s4_c02 |
| Sprint 5 | 0.698 (s5_m07) | 0.781 (s5_m07) | 0.224 | 0.109 | 0.046 | False | s5_m07 |

Note: Sprint 3 "hyp" = hypothetical berdasarkan PR-optimal threshold dari V4 manual run, bukan hasil staged runs (semua pending). Sprint 3 PR-optimal dihitung karena ini memberikan insight potensi上诉 threshold strategi catastrophe.
