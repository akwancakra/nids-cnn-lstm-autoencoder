# RESEARCH REPORT CSE F1 - SPRINT 3 V4

- generated_at: 2026-02-27
- source_notebook: `notebooks/do-not-edit/sprint3_sota_upgrade_v4.ipynb`
- environment: Google Colab + Google Drive
- scope: CNN-LSTM Autoencoder, zero-shot, strict split

## 1) Ringkasan Eksekutif

Eksperimen Sprint 3 v4 menunjukkan model sudah punya pemisahan error reconstruction yang cukup baik di domain source (CIC), tetapi generalisasi ke target domain (CSE) masih rendah pada threshold operasional saat ini. Dengan threshold aktif `0.040418` (percentile 95 dari source validation), model cenderung konservatif: false positive rendah di CIC, tetapi banyak miss attack di CSE (recall sangat rendah).

Kesimpulan utama:
- Target seluruh metrik >90% belum tercapai dan masih jauh.
- Bottleneck utama ada di domain shift dan strategi threshold source-only.
- Ada sinyal bahwa recall CSE bisa didorong tinggi jika threshold diturunkan, tetapi precision akan turun (trade-off tajam).

## 2) Perubahan Penting di V4

Preprocess dan setup yang teramati di notebook:
- Sequence config: `seq_len=10`, `stride=5`
- Scaler: `quantile`
- Drift feature drop list dipakai, namun file berisi `0 columns dropped`
- Pemisahan file strict:
  - CIC train files: 8
  - CIC test files: 8
  - CSE test files: 3
- Seleksi fitur:
  - selected features: 34
  - dropped NZV: 12
  - dropped correlation: 31
- Quantile scaler fit pada benign CIC train: 2,271,320 sample

Training setup teramati:
- Train sample: 366,318
- Validation sample: 87,935
- Input shape: `(10, 34)`
- Total parameter: 70,866
- Val loss turun konsisten hingga sekitar `0.0200` pada epoch akhir yang terlihat.

## 3) Hasil Utama

Threshold aktif saat evaluasi:
- method: source percentile 95
- threshold: `0.040417712181806564` (ditampilkan sebagai `0.040418`)

### 3.1 CSE (Target Domain)

Metrik pada threshold aktif:
- Accuracy: **0.6789**
- Precision: **0.1837**
- Recall: **0.0439**
- F1: **0.0708**

PR/ROC diagnostic:
- PR-AUC: **0.2253**
- ROC-AUC: **0.3633**
- PR-optimal threshold (max F1): `0.003367`, max F1: **0.4362**
- ROC-optimal threshold (Youden-like): `0.022348`, TPR: `0.4859`, FPR: `0.4526`

Interpretasi:
- Pada threshold sekarang, model sangat under-detect attack di CSE (recall rendah).
- Menurunkan threshold bisa meningkatkan recall drastis, tapi berdampak besar ke false positive.

### 3.2 CIC (Source Domain)

Metrik pada threshold aktif:
- Accuracy: **0.8371**
- Precision: **0.7184**
- Recall: **0.3968**
- F1: **0.5112**

Confusion (CIC):
- TN: 212,493
- FP: 9,442
- FN: 36,606
- TP: 24,083
- FPR: 0.0425
- TNR: 0.9575

PR/ROC diagnostic:
- PR-AUC: **0.5636**
- ROC-AUC: **0.7359**
- PR-optimal threshold: `0.021003`, max F1: **0.5157**

Interpretasi:
- Di source domain, model cukup baik menahan FPR rendah, tetapi recall masih belum tinggi.
- Threshold aktif memprioritaskan specificity dibanding sensitivity.

## 4) Analisis Error Reconstruction

Statistik CIC error distribution:
- Benign: mean `0.017973`, median `0.015549`, std `0.011070` (n=221,935)
- Attack: mean `0.036782`, median `0.024957`, std `0.024215` (n=60,689)
- KS statistic: `0.4300`, p-value: `0.000000`

Makna:
- Distribusi error benign vs attack memang berbeda secara statistik.
- Namun overlap masih signifikan, sehingga threshold tunggal sulit memberi precision dan recall tinggi secara bersamaan, terutama saat domain shift CIC -> CSE.

## 5) Gap Terhadap KPI Sprint

Target KPI (phase ambisius): semua metrik utama >90%.

Status v4:
- CIC: belum >90% di accuracy/precision/recall/F1.
- CSE: jauh dari >90%, terutama precision/recall/F1.

Penilaian realistis:
- Mencapai semua metrik >90% secara simultan untuk zero-shot cross-domain anomaly detection sangat sulit.
- Tetap mungkin mendekati target dengan kombinasi:
  - scoring yang lebih robust,
  - kalibrasi threshold source-only yang lebih tepat,
  - dan penguatan representasi model agar separasi domain target membaik.

## 6) Temuan Kritis dari V4

1. Training convergence sudah stabil, jadi masalah utama bukan sekadar under-training.
2. Threshold operasional saat ini terlalu konservatif untuk objective anomaly-first.
3. Domain shift menjadi faktor dominan penurunan performa CSE.
4. Ada ruang perbaikan via threshold/scoring (bukti: gap besar antara current F1 vs PR-optimal F1 di CSE), tapi tetap ada trade-off FPR.

## 7) Rekomendasi Aksi Prioritas (V5+)

1. Jadikan objective evaluasi utama berbasis attack-class:
   - prioritaskan `recall_attack` + `precision_attack` + `F1_attack`,
   - tetap laporkan `FPR` sebagai guardrail wajib.
2. Jalankan threshold sweep source-only terstruktur:
   - bandingkan `percentile`, `gaussian`, dan `source_calib_f1/guardrail`,
   - pilih threshold yang memaksimalkan recall dengan batas FPR eksplisit.
3. Aktifkan/kuatkan hybrid score:
   - `hybrid_recon_latent` dengan sweep `hybrid_alpha`,
   - cek apakah separasi CSE membaik tanpa FPR collapse di CIC.
4. Uji loss robust:
   - bandingkan `mse`, `huber`, `mse_mae_mix` untuk mengurangi sensitivitas outlier benign.
5. Pastikan evaluasi resmi full-path:
   - hindari fast/sampled eval untuk angka final report.
6. Tambah seed confirmation:
   - validasi minimal 2 seed pada kandidat terbaik agar performa tidak kebetulan.

## 8) Keputusan Operasional untuk Sprint Lanjutan

- Lanjutkan eksperimen dengan mode **anomaly-first + FPR guardrail**, bukan mengejar accuracy agregat saja.
- Definisikan target bertahap yang lebih feasible:
  - Tahap 1: dorong CSE recall/F1 naik signifikan tanpa FPR CIC meledak.
  - Tahap 2: stabilkan precision lewat score/threshold tuning.
  - Tahap 3: baru optimasi menuju target >90% lintas metrik jika separasi domain sudah membaik.

---

Catatan:
- Beberapa output notebook bersifat truncated pada tampilan, sehingga detail confusion matrix CSE tidak sepenuhnya terlihat di capture ini.
- Meski demikian, metrik inti dan sinyal utama eksperimen v4 sudah cukup kuat untuk menyusun arah perbaikan sprint berikutnya.
