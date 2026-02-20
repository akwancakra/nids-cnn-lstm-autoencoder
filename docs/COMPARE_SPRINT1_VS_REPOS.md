# COMPARE_SPRINT1_VS_REPOS

## Scope & Baseline

Perbandingan ini memakai basis:
- repo utama: `nids-cnn-lstm-autoencoder`
- branch: `try/3-way-enhancements`
- hasil acuan Sprint1:  
  - `docs/RESEARCH_REPORT_CSE_F1_SPRINT1.md`
  - `docs/RINGKASAN_PREPROCESS_EVAL_UJI_SPRINT1.md`

Tujuan compare:
1. melihat posisi hasil Sprint1 kamu terhadap `IntrusionDetectionSystem`,
2. menilai “angka besar” repo lain dengan fairness (apple-to-apple vs non-apple-to-apple),
3. menurunkan action item yang realistis untuk Sprint2.

---

## Fairness Rules (Supaya Tidak Bias)

Kelas komparabilitas:
- `A`: protokol dekat dengan target skripsi (unsupervised anomaly, CIC->CSE, zero/few-shot).
- `B`: sebagian relevan, tapi ada dimensi inti yang hilang.
- `C`: bukan apple-to-apple untuk KPI skripsi.

Sumber angka:
- `measured_local`: hasil yang benar-benar kamu jalankan.
- `claimed_readme`: angka klaim di README.
- `claimed_readme_with_code`: ada klaim + code workflow.

---

## 1) Posisi Internal Sprint1 (Measured)

| Run | CIC F1 | CSE F1 | CSE FPR | CSE AUC | Composite Gap Score | Guardrail |
|---|---:|---:|---:|---:|---:|---|
| `rs00_release_baseline` | 0.7333 | 0.2780 | 0.2474 | 0.4421 | 0.4088 | FAIL |
| `rs10_full_feature_filter_corr090` | 0.7250 | 0.2513 | 0.2511 | 0.4527 | 0.4155 | FAIL |
| `rs11_full_scaleguard_q995_clip20` | 0.7420 | 0.2587 | 0.1788 | 0.5075 | 0.4019 | PASS |
| `rs12_full_scaleguard_q999_clip10` | 0.7328 | 0.2600 | 0.1739 | 0.4305 | 0.4133 | PASS |

Guardrail Sprint1:
- `cse_fpr <= 0.20`
- `cse_f1 >= 0.24`

Inti bacaan:
- baseline (`rs00`) masih lebih tinggi di CSE F1, tapi gagal guardrail FPR.
- `rs11` jadi kompromi terbaik saat ini: FPR membaik signifikan sambil CSE F1 tetap lolos guardrail.

---

## 2) Compare dengan `IntrusionDetectionSystem`

### Ringkasan cepat

| Aspek | `nids-cnn-lstm-autoencoder` (Sprint1) | `IntrusionDetectionSystem` |
|---|---|---|
| Target proposal (CIC->CSE, zero/few-shot) | Ya, dieksekusi di Sprint1 | Ya, workflow tersedia via `hybrid_pipeline.py` |
| Paradigma | Unsupervised anomaly (hybrid CNN-LSTM AE) | Dominan autoencoder; ada modul workflow proposal |
| Hasil terukur lokal | Ada (measured) | Belum ada measured lokal yang setara protokol Sprint1 |
| Klaim performa utama | `rs11`: CSE F1 0.2587, CSE FPR 0.1788 | Klaim: 91.3% benign correct, FPR 8.7%, attack detect 98.3% |
| Fairness class | A | B (angka klaim tidak jelas setara CIC->CSE zero-shot) |

### Interpretasi
1. `IntrusionDetectionSystem` kuat sebagai referensi implementasi pipeline (train benign-only -> zero-shot -> few-shot).
2. Angka performa di README terlihat tinggi, tetapi belum bisa disimpulkan “lebih baik” terhadap objective skripsi kamu tanpa re-run protokol yang sama.
3. Teknik dari repo itu tetap layak diadopsi untuk meningkatkan kualitas benchmark dan pelacakan eksperimen.

---

## 3) Compare dengan Repo Lain (Ringkas, Fairness-Aware)

| Repo | Klaim angka menonjol | Kelas | Catatan fairness |
|---|---|---|---|
| `CNN-BiLSTM-Network-Intrusion-Detection-Replication` | NSL binary acc 99.03%, UNSW binary acc 91.95%, UNSW multiclass F1 0.53 | C | Dataset beda (NSL/UNSW), paradigma beda, bukan CIC->CSE zero-shot. |
| `Network-Intrusion-Detection-System` | LSTM acc hingga 99.5% (CIC2018), 99.6% (AWID) | C | Mayoritas in-domain supervised classification. |
| `LSTM-AutoEncoder-Unsupervised-Anomaly-Detection` | AUC 0.79 -> 0.8335 | C | Unsupervised bagus, tapi domain dataset bukan NIDS CIC/CSE. |
| `Intrusion-Detection-Pipeline` | Tidak klaim angka spesifik di README | B | Struktur pipeline sangat berguna, metrik tidak langsung bisa dibandingkan. |
| `CNN-LSTM` | Tidak klaim angka spesifik di README | C | Arsitektur inspiratif, protokol evaluasi tidak standar dengan target kamu. |
| `Efficient-CNN-BiLSTM-for-Network-IDS` | Paper/link repro, tanpa angka lokal rinci | C | Referensi arsitektur, bukan baseline fair untuk KPI skripsi ini. |
| `Intrusion-Detection-CICIDS2017` | Fokus analisis & baseline ML di CIC | C | Berguna untuk baseline, tapi bukan cross-dataset zero-shot. |
| `ML-based-...-CSE-CIC-IDS2018` | Menyebut evaluasi beberapa model, tanpa angka rinci README | C | Dominan supervised klasifikasi. |
| `Unsupervised_Intrusion_Detection` | Fokus unsupervised IDS | B | Konsep relevan, tetapi dataset/testbed tidak setara CIC->CSE. |

---

## 4) Kenapa Repo Lain Bisa Terlihat “Jauh Lebih Tinggi”

Penyebab paling umum:
1. **In-domain evaluation**: train dan test di distribusi data yang sangat mirip.
2. **Supervised setup**: model belajar label attack langsung, lebih mudah capai akurasi tinggi.
3. **Metric selection bias**: menonjolkan accuracy tanpa konteks imbalance/FPR.
4. **Protocol mismatch**: tidak ada cross-dataset zero-shot/few-shot yang ketat seperti objective proposal kamu.

Kesimpulan penting:
- Angka besar di repo lain valid untuk konteks mereka, tapi tidak otomatis berarti lebih unggul untuk skenario zero-day cross-dataset yang kamu kejar.

---

## 5) Implementasi yang Layak Diambil (Top-3 Relevan)

### A. Dari `IntrusionDetectionSystem`
- **Adopt**:
  - API workflow eksplisit train/eval zero-shot/few-shot.
  - Format benchmark yang standardized.
- **Adapt**:
  - threshold baseline `mean + k*std` sebagai pembanding threshold percentile kamu.

### B. Dari `Intrusion-Detection-Pipeline`
- **Adopt**:
  - struktur pipeline modular (loader/cleaner/feature/eval terpisah),
  - evaluasi ringkas terpadu.
- **Adapt**:
  - guardrail-aware summary untuk keputusan kandidat eksperimen.

### C. Dari `CNN-BiLSTM-Network-Intrusion-Detection-Replication`
- **Adopt**:
  - insight arsitektur Conv1D + BiLSTM + regularization.
- **Skip**:
  - direct score comparison sebagai KPI (karena mismatch dataset/protocol).

---

## 6) Bottom Line untuk Kondisi Sekarang

1. Posisi Sprint1 kamu **belum unggul di CSE F1**, tetapi sudah mulai **masuk zona FPR aman** lewat `rs11`/`rs12`.
2. Dibanding `IntrusionDetectionSystem`, gap utama saat ini bukan sekadar arsitektur, tapi **protokol evaluasi terstandar + threshold/few-shot tuning yang stabil**.
3. Fokus Sprint2 paling rasional:
   - pertahankan preprocess family `rs11/rs12` (karena guardrail-safe),
   - naikkan CSE F1 tanpa melepas FPR guardrail,
   - jalankan benchmark lintas-repo dengan schema metrik yang sama.
