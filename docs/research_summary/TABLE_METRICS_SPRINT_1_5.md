# Tabel Metrik Komprehensif Sprint 1–5

## Ringkasan Best Per Sprint

| Sprint | Champion Run | CSE Recall | CSE Precision | CSE F1 | CSE FPR | CIC F1 | CIC FPR | F1 Gap | Gate Status |
|--------|--------------|------------|---------------|--------|---------|--------|---------|--------|-------------|
| Sprint 1 | rs11_full_scaleguard_q995_clip20 | 0.173 | 0.519 | 0.259 | 0.179 | 0.742 | 0.025 | 0.483 | ✅ Pass |
| Sprint 2 | N/A (infra sprint) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Sprint 3 | V4_manual (percentile p95) | 0.044 | 0.184 | 0.071 | 0.164 | 0.511 | 0.043 | 0.722 | ❌ Fail |
| Sprint 3 | V4_hypothetical (PR-optimal) | **0.486** | N/A | **0.436** | N/A | N/A | **0.453** | N/A | ❌ Fail |
| Sprint 4 | **s4_t06_source_calib_guardrail_fpr010** | **0.690** | **0.863** | **0.767** | 0.271 | 0.811 | **0.093** | **0.044** | ✅ Pass |
| Sprint 5 | **s5_m07_rerun (mse_mae_mix)** | **0.698** | *0.886* | **0.781** | 0.224 | 0.827 | 0.109 | **0.046** | ❌ Fail |
| Sprint 5 | s5_h02_guardrail_fpr015 (relaxed) | 0.693 | 0.865 | 0.769 | 0.268 | 0.847 | 0.147 | 0.078 | ❌ Fail |

*Info tambahan: s5_m07_rerun menggunakan recon_mse score mode dengan mse_mae_mix loss α=0.7

---

## Tabel Per-Stage Sprint 4 (Model Architecture)

| Run ID | Variasi | CSE Recall | CSE F1 | CSE FPR | CIC F1 | CIC FPR | F1 Gap |
|--------|---------|------------|--------|---------|--------|---------|--------|
| s4_m01_baseline_arch | Baseline ref | 0.453 | 0.605 | 0.111 | 0.787 | 0.028 | 0.182 |
| s4_m02_dropout_030 | Dropout 0.3 | 0.595 | 0.726 | 0.108 | 0.788 | 0.032 | 0.062 |
| s4_m03_latent_64 | Latent dim 64 | 0.627 | 0.749 | 0.119 | 0.796 | 0.034 | 0.047 |
| s4_m04_cnn_64_128_128 | Deeper CNN | 0.566 | 0.703 | 0.107 | 0.784 | 0.030 | 0.081 |
| s4_m05_lstm_192_96 | Deeper LSTM | 0.628 | 0.741 | 0.165 | 0.805 | 0.028 | 0.064 |
| **s4_m06_loss_huber** | **Huber loss** | **0.639** | **0.758** | 0.118 | 0.795 | 0.032 | **0.037** |
| s4_m07_loss_mse_mae_mix_a07 | Hybrid loss (gagal) | - | - | - | - | - | - |
| s4_m08_hybrid_score_alpha07 | Recon_latent hybrid | 0.012 | 0.023 | 0.018 | 0.040 | 0.026 | 0.767 |

---

## Tabel Per-Stage Sprint 4 (Threshold Sweep - huber loss mode)

| Run ID | Method | Threshold | CSE Recall | CSE F1 | CSE FPR | CIC F1 | CIC FPR | F1 Gap |
|--------|--------|-----------|------------|--------|---------|--------|---------|--------|
| s4_t01 | Percentile p93 | 0.01344 | 0.675 | 0.771 | 0.191 | 0.806 | 0.060 | **0.035** |
| s4_t02 | Percentile p95 | 0.01465 | 0.662 | 0.768 | 0.154 | 0.801 | 0.047 | 0.033 |
| s4_t03 | Percentile p97 | 0.01641 | 0.639 | 0.758 | 0.118 | 0.795 | 0.032 | 0.037 |
| s4_t04 | Gaussian k=1.8 | 0.01399 | 0.669 | 0.770 | 0.173 | 0.803 | 0.054 | 0.034 |
| s4_t05 | Calib F1 (max CSE F1) | 0.00700 | **0.711** | 0.726 | **0.615** | **0.823** | **0.293** | 0.097 |
| **s4_t06** | **Calib guardrail** | **0.01153** | **0.690** | **0.767** | 0.271 | 0.811 | **0.093** | **0.044** |

---

## Tabel Per-Stage Sprint 5 (Target Threshold Experiments)

| Run ID | Method | Threshold | CSE Recall | CSE F1 | CSE FPR | CIC F1 | CIC FPR | F1 Gap |
|--------|--------|-----------|------------|--------|---------|--------|---------|--------|
| s5_t01 | Target p95 (oracle) | 8.89 | 0.091 | 0.163 | 0.050 | 0.779 | 0.011 | 0.616 |
| s5_t02 | Target p97 (oracle) | 10.73 | 0.044 | 0.084 | 0.030 | 0.772 | 0.005 | 0.688 |
| s5_t03 | Target p98 (oracle) | 12.90 | 0.015 | 0.030 | 0.020 | 0.752 | 0.001 | 0.722 |
| s5_t04 | Target p99 (oracle) | 14.62 | 0.007 | 0.013 | 0.010 | 0.683 | 0.001 | 0.670 |
| s5_t05 | Target p97 semi-blind (5%) | 10.99 | 0.040 | 0.076 | 0.028 | 0.771 | 0.004 | 0.695 |
| s5_t06 | Target p97 semi-blind (10%) | 10.86 | 0.042 | 0.079 | 0.029 | 0.772 | 0.005 | 0.693 |
| s5_t07 | Baseline kontrol (source guardrail) | 4.08 | 0.626 | 0.730 | 0.224 | 0.832 | 0.103 | 0.103 |

---

## Tabel Sprint 5 Stage 3 (Guardrail Relaxation)

| Run ID | FPR Guardrail | Threshold | CSE Recall | CSE F1 | CSE FPR | CIC F1 | CIC FPR | F1 Gap |
|--------|---------------|-----------|------------|--------|---------|--------|---------|--------|
| s5_h01 | FPR ≤ 0.12 | 3.822 | 0.650 | 0.744 | 0.241 | 0.779 | 0.121 | 0.098 |
| s5_h02 | FPR ≤ 0.15 | 3.496 | 0.693 | **0.769** | 0.268 | 0.847 | 0.147 | **0.078** |

---

## Perbandingan Loss Functions

| Loss Function | CSE Recall | CSE F1 | CSE FPR | F1 Gap | Source |
|--------------|------------|--------|---------|--------|--------|
| MSE (baseline) | 0.626 | 0.730 | 0.224 | 0.103 | Sprint 5 baseline |
| **Huber** | 0.639 | 0.758 | 0.118 | 0.037 | Sprint 4 s4_m06 |
| **mse_mae_mix α=0.7** | 0.698 | 0.781 | 0.224 | 0.046 | Sprint 5 s5_m07 (best) |

---

## Perbandingan Scalers

| Scaler Type | CSE Recall Runner-up | Performance Note |
|-------------|---------------------|-----------------|
| Robust (q995) | 0.639 | Sprint 4 s4_m06 huber |
| Quantile (q995) | 0.599 | Sprint 4 s4_p03 quantile - slightly lower recall |

---

## Statistik Progress Lintas Sprint

| Metric | Sprint 1 | Sprint 3 (op) | Sprint 4 (best) | Sprint 5 (best) | Total Change | % Improvement |
|--------|----------|---------------|-----------------|-----------------|--------------|----------------|
| CSE Recall | 0.173 | 0.044 | 0.690 | 0.698 | +0.525 | +303% |
| CSE F1 | 0.259 | 0.071 | 0.767 | 0.781 | +0.522 | +202% |
| CSE FPR | 0.179 | 0.164 | 0.271 | 0.224 | +0.045 | +25% |
| CIC FPR | 0.025 | 0.043 | 0.093 | 0.109 | +0.084 | +336% |
| F1 Gap | 0.483 | 0.722 | 0.044 | 0.046 | -0.437 | -90% (improvement) |

---

## Top 5 Champions (Best Recall dengan FPR Reasonable)

| Rank | Run ID | Sprint | CSE Recall | CSE F1 | CSE FPR | CIC FPR | F1 Gap |
|------|--------|--------|------------|--------|---------|---------|--------|
| 1 | **s5_m07_rerun** (mse_mae_mix) | 5 | 0.698 | 0.781 | 0.224 | 0.109 | 0.046 |
| 2 | s4_t06_source_calib_guardrail | 4 | 0.690 | 0.767 | 0.271 | 0.093 | 0.044 |
| 3 | s4_c02_best_seed1234_full | 4 | 0.689 | 0.767 | 0.270 | 0.098 | 0.044 |
| 4 | s4_c01_best_seed42_full | 4 | 0.688 | 0.766 | 0.268 | 0.094 | 0.045 |
| 5 | s5_h02_guardrail_fpr015 | 5 | 0.693 | 0.769 | 0.268 | 0.147 | 0.078 |

Note: s5_h02 memiliki recall tinggi (0.693) tapi CIC FPR 0.147 melewati guardrail tipis (batas: 0.10)
