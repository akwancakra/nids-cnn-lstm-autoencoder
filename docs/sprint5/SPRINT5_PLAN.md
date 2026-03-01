# Sprint 5 Plan — CSE FPR Reduction & Domain Shift Mitigation

## Tujuan Sprint 5

1. **Mengurangi CSE FPR** sambil menjaga CSE recall tinggi (target recall minimal 0.75).
2. **Menguji target-aware threshold** (`target_percentile`) untuk mitigasi domain shift.
3. **Memperbaiki bug s4_m07** agar run `mse_mae_mix` bisa dieval.
4. **Lock konfigurasi terbaik Sprint 4** sebagai baseline.

## Gate Criteria Sprint 5

| Kriteria          | Sprint 4      | Sprint 5               |
| ----------------- | ------------- | ----------------------- |
| CSE recall min    | adaptive 0.40 | **0.75** (fixed floor)  |
| CIC FPR max       | 0.10          | 0.10 (tetap)            |
| CSE precision min  | 0.30          | 0.30 (tetap)            |

**Trade-off:** Menaikkan target recall ke 0.75 menuntut model/threshold lebih agresif. Jika target-aware threshold menaikkan CSE FPR, perlu didokumentasikan dan dibandingkan dengan baseline zero-shot.

## Gate Fallback Policy

| Urutan | Aksi |
| ------ | ---- |
| 1 | Dokumentasi: catat best CSE recall yang tercapai, gap ke 0.75. |
| 2 | **Pivot:** Turunkan target ke 0.70, ulang gate check. Jika tetap gagal → pivot ke 0.65. |
| 3 | Jika masih gagal: set `pivot_recommendation: USAD`, hentikan eksperimen arsitektur di Sprint 5. |
| 4 | Rekomendasi sprint berikutnya: pertimbangkan DANN / few-shot model adaptation. |

## Struktur Stage (12 run)

| Stage       | Deskripsi                    | Run | Variasi |
| ----------- | ---------------------------- | --- | ------- |
| **Stage 0** | Lock baseline + s4_m07 rerun  | 2   | s5_00_lock_baseline, s5_m07_rerun |
| **Stage 1** | Target threshold             | 7   | Oracle p95–p99, semi-blind p97_sub5/10pct, baseline kontrol |
| **Stage 3** | CIC guardrail relaxation     | 2   | s5_h01 fpr012, s5_h02 fpr015 |
| **Stage 4** | Seed confirmation            | 2   | s5_c01 seed42, s5_c02 seed1234 |

## Baseline Lock (Sprint 4 Best)

- **Preprocess:** robust_q995_clip20_corr090
- **Model:** CNN-LSTM AE hybrid, input_shape=(10, 31)
- **Threshold:** source_calib_guardrail_fpr010
- **Seed:** 42

## Oracle vs Semi-Blind

- **Oracle** (s5_t01–t04): Menggunakan label CSE benign penuh untuk `target_percentile` — upper bound, tidak realistis untuk deployment.
- **Semi-blind** (s5_t05–s5_t06): Subsample CSE benign (5%, 10%) untuk mensimulasikan skenario deployment lebih realistis.
- **Baseline kontrol** (s5_t07): Zero-shot tanpa target info (source_calib_guardrail).
