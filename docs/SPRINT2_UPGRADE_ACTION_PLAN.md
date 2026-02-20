# SPRINT2_UPGRADE_ACTION_PLAN

## Objective

Meningkatkan performa cross-dataset pada CSE tanpa melanggar guardrail:
- target utama: `cse_f1 > 0.2780` (melewati baseline `rs00`)
- dengan batas tetap: `cse_fpr <= 0.20`

Baseline kerja:
- branch: `try/3-way-enhancements`
- kandidat guardrail-safe saat ini: `rs11_full_scaleguard_q995_clip20`

---

## Success Criteria

1. Ada minimal 1 kandidat Sprint2 yang:
   - `cse_f1` > `0.2780`
   - `cse_fpr` <= `0.20`
2. Perbandingan hasil bersifat reproducible (seed/config hash jelas).
3. Report eksperimen bisa dipakai untuk keputusan lanjut ke sprint berikutnya tanpa ambiguity.

---

## Workstream Prioritas (Actionable)

## P0 - Threshold & Decision Policy Stabilization

### Target
Memperbaiki trade-off CSE F1 vs FPR tanpa retrain berlebihan.

### Files
- Modify: `scripts/eval_metrics.py`
- Modify: `scripts/research_sprint1.py`
- Modify: `config.yaml`
- Test: `tests/test_threshold_modes.py`

### Tasks
1. Tambah output dual-threshold summary:
   - profile A: guardrail-safe threshold
   - profile B: max-F1 threshold
2. Simpan keduanya dalam metrik output per run (agar keputusan tidak satu dimensi).
3. Di summarizer, tampilkan ranking:
   - by composite gap,
   - by guardrail-safe F1.
4. Tambah test untuk verifikasi pemilihan threshold profile.

### Acceptance
- Per run, tersedia dua keputusan threshold yang eksplisit.
- Ranking tidak lagi hanya “score kecil”, tapi juga mempertimbangkan target F1 practical.

---

## P1 - Few-Shot Adaptation Safety Tuning

### Target
Menghindari collapse CSE F1 saat few-shot.

### Files
- Modify: `scripts/eval_metrics.py`
- Modify: `config.yaml`
- Test: `tests/test_threshold_modes.py`

### Tasks
1. Sweep kecil namun ketat untuk few-shot:
   - `few_shot_benign_frac`: `0.005, 0.01, 0.02`
   - `few_shot_finetune_epochs`: `0, 1, 2`
   - `few_shot_finetune_lr`: `2e-4, 5e-4`
2. Tambahkan guardrail stop:
   - stop candidate bila `cse_f1` turun drastis dibanding baseline run family.
3. Catat delta pre-vs-post adaptation:
   - `delta_f1`, `delta_fpr`, `delta_auc`.

### Acceptance
- Ada insight jelas kapan few-shot membantu vs merusak.
- Eksperimen few-shot tidak lagi buta terhadap degradation.

---

## P2 - Preprocess Family Consolidation (rs11/rs12 Lineage)

### Target
Memperkuat kombinasi preprocess yang sudah terbukti guardrail-safe.

### Files
- Modify: `scripts/preprocess.py`
- Modify: `research/sprint1/generated_configs/rs11_full_scaleguard_q995_clip20.yaml`
- Modify: `research/sprint1/generated_configs/rs12_full_scaleguard_q999_clip10.yaml`
- Test: `tests/test_preprocess_features.py`
- Test: `tests/test_scaler_fit_modes.py`

### Tasks
1. Tambah logging statistik clipping per fitur:
   - proporsi nilai yang ter-clip sebelum dan sesudah scaling.
2. Pastikan feature filter + scale guard menghasilkan manifest statistik konsisten.
3. Buat 1 kandidat preprocess turunan rs11:
   - kombinasi tetap konservatif pada FPR, tapi mencoba recover F1.

### Acceptance
- Artefak preprocess memiliki statistik diagnosis yang cukup untuk debugging domain shift.
- Kandidat turunan rs11 memiliki rasional teknis, bukan trial-and-error murni.

---

## P3 - Standardized Comparison & Branch Delta Check

### Target
Membuat pembuktian improvement yang rapi lintas eksperimen dan lintas branch.

### Files
- Modify: `scripts/research_sprint1.py`
- Add/Update: `results/metrics/repo_comparison_standardized.json`
- Add/Update: `docs/COMPARE_SPRINT1_VS_REPOS.md`
- Optional compare input branch: `feat/zero-shot-strict-research-foundation`

### Tasks
1. Simpan metadata run wajib:
   - seed, config hash, run id, threshold profile.
2. Buat section “branch delta”:
   - `try/3-way-enhancements` vs `feat/zero-shot-strict-research-foundation`
   - minimal pada metrik: CSE F1/FPR/AUC + gap.
3. Update compare report setiap cycle eksperimen utama.

### Acceptance
- Improvement claim selalu punya bukti metrik + metadata run.
- Bisa menjawab “improvement karena apa” secara objektif.

---

## P4 - Borrowed Ideas Integration (Top-3 Relevan)

### Target
Mengadopsi elemen implementasi repo lain yang paling berpotensi.

### Source Repos
- `IntrusionDetectionSystem`
- `Intrusion-Detection-Pipeline`
- `CNN-BiLSTM-Network-Intrusion-Detection-Replication`

### Tasks
1. Port konsep benchmark standardization style dari `IntrusionDetectionSystem` ke output lokal.
2. Port pola evaluasi modular ringkas ala `Intrusion-Detection-Pipeline`.
3. Uji 1 variasi arsitektur regularization ringan terinspirasi CNN-BiLSTM (tanpa mengorbankan objective unsupervised).

### Acceptance
- Ada minimal 1 perubahan konkret dari tiap sumber yang benar-benar diuji.
- Setiap adopsi dinilai impact-nya terhadap KPI, bukan hanya implementasi kosmetik.

---

## Execution Order (Recommended)

1. `P0` -> 2. `P1` -> 3. `P2` -> 4. `P3` -> 5. `P4`

Alasan:
- P0/P1 memberi perbaikan tercepat pada trade-off F1/FPR.
- P2 memperkuat fondasi data.
- P3 menjaga kualitas pembuktian.
- P4 menambah inovasi setelah baseline stabil.

---

## Risk Register

1. **FPR turun tetapi F1 anjlok**  
   Mitigasi: dual-threshold profile + guardrail-aware ranking.

2. **Few-shot overfit ke subset benign target**  
   Mitigasi: batasi epoch, bandingkan delta pre/post adaptation, aktifkan early rejection.

3. **False confidence dari compare antar repo**  
   Mitigasi: pakai `comparability_class` dan label `source_type` (measured vs claimed).

---

## Final Note

Sprint2 bukan mengejar angka terbesar absolut, tapi mengejar **angka yang valid untuk objective skripsi**:
- unsupervised,
- cross-dataset CIC->CSE,
- tetap robust terhadap guardrail operasional.
