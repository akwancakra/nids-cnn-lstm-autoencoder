# RINGKASAN_PREPROCESS_EVAL_UJI_SPRINT1

## 1) Tujuan Dokumen
Dokumen ini merangkum hasil sementara dan hasil akhir Sprint 1 untuk tiga area:
- preprocessing data,
- training/evaluasi model,
- pengujian eksperimental lintas konfigurasi.

Periode run utama: 19-20 Februari 2026.

## 2) Ringkasan Pipeline yang Dijalankan
Pipeline eksperimen menggunakan kombinasi:
- `preprocess` -> `train` -> `eval` untuk run full (`rs10`, `rs11`, `rs12`),
- `eval` only untuk run threshold/few-shot (`rs01` s.d. `rs09`).

Dataset dan split yang dipakai:
- CIC-IDS2017 untuk train/val/test internal.
- CSE-CIC-IDS2018 untuk uji generalisasi.

## 3) Implementasi Preprocessing yang Relevan
Perubahan utama preprocessing:
- Statistical feature filter:
  - NZV filter aktif.
  - Correlation filter aktif (nilai threshold bervariasi antar run).
- Scale guard:
  - Raw per-feature clipping berbasis quantile benign (`raw_clip_quantile`).
  - Post-scale clipping absolut (`post_scale_clip_abs`).

Variasi run preprocess full:
- `rs10_full_feature_filter_corr090`
  - Fokus: korelasi lebih ketat (`corr_threshold=0.90`).
  - Scale guard: `q=0.999`, `clip_abs=20`.
- `rs11_full_scaleguard_q995_clip20`
  - Fokus: raw clip lebih ketat (`q=0.995`).
  - Post-scale clip: `20`.
- `rs12_full_scaleguard_q999_clip10`
  - Fokus: post-scale clip lebih ketat (`10`).
  - Raw clip: `q=0.999`.

Catatan runtime:
- Log menunjukkan GPU terdeteksi dan dipakai (RTX 4060).
- Warning `ptxas.exe` muncul, tetapi training tetap berjalan (fallback JIT driver).
- Warning `DtypeWarning` di beberapa file CSE muncul saat preprocess, namun proses tetap selesai.

## 4) Ringkasan Evaluasi Eksperimen (Sprint 1)
Sumber utama metrik: `results/metrics/summary_sprint1.csv`.

Baseline release (`rs00_release_baseline`):
- CIC: `f1=0.7333`, `fpr=0.0339`, `auc=0.8203`
- CSE: `f1=0.2780`, `fpr=0.2474`, `auc=0.4421`
- Gap: `f1_gap=0.4553`, `auc_gap=0.3782`

Hasil run preprocess full:
- `rs10`:
  - CIC `f1=0.7250`
  - CSE `f1=0.2513`, `fpr=0.2511`, `auc=0.4527`
  - cenderung tidak lebih baik dari baseline untuk target gap+FPR.
- `rs11`:
  - CIC `f1=0.7420`
  - CSE `f1=0.2587`, `fpr=0.1788`, `auc=0.5075`
  - `auc_gap` membaik signifikan; kandidat paling stabil untuk guardrail.
- `rs12`:
  - CIC `f1=0.7328`
  - CSE `f1=0.2600`, `fpr=0.1739`, `auc=0.4305`
  - FPR CSE membaik, tetapi AUC CSE turun dibanding `rs11`.

Hasil run threshold/few-shot:
- Beberapa run menurunkan FPR CSE sangat agresif, tetapi CSE F1 turun tajam (`~0.08-0.20`).
- Secara praktis untuk objective balancing, run tersebut belum unggul.

## 5) Hasil Uji Ringkas Berbasis Skor Komposit Gap
Dengan scoring:
- `0.5*|f1_gap| + 0.3*|auc_gap| + 0.2*|accuracy_gap|`
dan guardrail:
- `cse_fpr <= 0.20`
- `cse_f1 >= 0.24`

Kandidat guardrail-safe terbaik saat ini:
- `rs11_full_scaleguard_q995_clip20`
  - score komposit lebih baik dari `rs12` pada kondisi guardrail.

## 6) Kesimpulan Teknis
- Gap generalisasi CIC -> CSE masih besar, tetapi pendekatan preprocess memberikan perbaikan terukur pada beberapa dimensi.
- `rs11` adalah titik kompromi terbaik saat ini (khususnya pada `auc_gap` dan `cse_fpr` dengan `cse_f1` masih layak).
- `rs12` menunjukkan arah baik pada FPR, namun belum konsisten pada AUC.

## 7) Rekomendasi Lanjutan
- Lanjutkan evaluasi dengan threshold sweep eval-only (tanpa retrain) untuk model kandidat `rs10/rs11/rs12`.
- Tetapkan winner berdasarkan:
  - composite gap score,
  - guardrail pass,
  - stabilitas antar rerun.
- Jika sweep tidak memberi peningkatan jelas, lakukan 1 retrain preprocess terpilih (bukan banyak run sekaligus).

## 8) Referensi Artefak
- `results/metrics/summary_sprint1.csv`
- `docs/RESEARCH_REPORT_CSE_F1_SPRINT1.md`
- `research/sprint1/experiments.yaml`
- `research/sprint1/generated_configs/rs10_full_feature_filter_corr090.yaml`
- `research/sprint1/generated_configs/rs11_full_scaleguard_q995_clip20.yaml`
- `research/sprint1/generated_configs/rs12_full_scaleguard_q999_clip10.yaml`
