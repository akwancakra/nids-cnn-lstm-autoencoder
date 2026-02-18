# NEED TO REVIEW - Skripsi

## Prioritas Tinggi

- [ ] Tegaskan status target metrik sebagai benchmark, bukan syarat mutlak "lulus/gagal penelitian".
  - Prinsip interpretasi:
    - target angka (mis. F1, AUC, FPR) dipakai sebagai acuan evaluasi berbasis literatur dan baseline internal
    - jika target tidak tercapai, penelitian **tetap valid** selama analisis penyebab, trade-off, dan batasan dijelaskan dengan kuat
    - kontribusi utama bisa tetap pada evaluasi generalisasi, temuan domain gap, dan rekomendasi perbaikan model
  - Narasi yang disarankan di naskah:
    - bedakan `target ideal` vs `minimum acceptable performance`
    - hindari kalimat yang membuat penelitian terkesan gagal total hanya karena tidak menyentuh satu angka target
    - tekankan bahwa hasil negatif/di bawah target adalah temuan empiris yang bernilai untuk penelitian lanjutan

- [ ] Finalisasi positioning eksperimen: **zero-shot vs few-shot adaptation**.
  - Definisi yang dipakai:
    - `Zero-shot`: model dilatih di CIC-IDS2017 lalu diuji di CSE-CIC-IDS2018 tanpa fine-tuning/retraining.
    - `Few-shot`: adaptasi unsupervised memakai **1% data BENIGN** dari CSE-CIC-IDS2018, lalu diuji pada sisa data target.
  - Aturan validitas:
    - data adaptasi tidak boleh overlap dengan data evaluasi target (no data leakage)
    - random seed sampling wajib dicatat untuk reproducibility
  - Standar metrik pelaporan:
    - metrik utama: `F1-score` dan `FPR`
    - metrik pendukung: Accuracy, Precision, Recall, AUC-ROC, Generalization Gap
  - Kebijakan threshold:
    - gunakan threshold yang sama dari validasi CIC untuk kedua skenario (zero-shot dan few-shot)
    - jika ada recalibration threshold setelah adaptation, laporkan sebagai varian eksperimen terpisah
  - Aturan interpretasi hasil:
    - few-shot dianggap unggul jika `F1-score` naik dan/atau `FPR` turun terhadap zero-shot
    - jika peningkatan kecil/tidak ada, jelaskan kemungkinan domain gap kecil atau data adaptasi 1% belum cukup
  - Batasan yang wajib ditulis:
    - few-shot membutuhkan akses data benign target sebelum deployment
    - ada trade-off antara kenaikan performa dan kompleksitas operasional adaptasi

- [ ] Konsistensi jumlah fitur dataset di seluruh naskah.
  - Standar yang dipakai:
    - CIC-IDS2017: `79 kolom total (78 fitur prediktor + Label)`
    - CSE-CIC-IDS2018: `80 kolom total (79 fitur prediktor + Label)`
  - Cek dan revisi terutama di:
    - `my-papers/isi-bab-3.md`
    - `my-papers/final-1-3.md`
    - `my-papers/topik-fixed.md`
    - `KNOWLEDGE.md`

- [ ] Konsistensi penyebutan sumber dataset (Official vs Kaggle mirror).
  - Jika data eksperimen pakai Kaggle, tulis eksplisit pada metodologi.
  - Tetap cantumkan sumber resmi sebagai rujukan dataset.
  - Tambahkan alasan pemilihan (aksesibilitas/komputasi) dan keterbatasan penelitian.

- [ ] Tambahkan subbagian "Data Integrity Check".
  - Minimal dokumentasikan:
    - daftar file yang dipakai
    - jumlah kolom per file
    - distribusi label per file
    - checksum/hash file (opsional tapi direkomendasikan)

- [ ] Konsistensi istilah teknis.
  - Pilih satu standar penulisan dan pakai konsisten:
    - `kolom total` vs `fitur prediktor`
    - `records` vs `flows`
    - `Label` vs `label` (di naskah cukup tulis `Label`)

## Prioritas Menengah

- [ ] Jelaskan harmonisasi nama fitur antar dataset.
  - Karena penamaan kolom CIC-IDS2017 dan CSE-CIC-IDS2018 berbeda, perlu dijelaskan bahwa:
    - skema acuan menggunakan CIC-IDS2017
    - kolom CSE di-map ke nama CIC sebelum feature alignment
    - setelah harmonisasi baru dilakukan intersection fitur

- [ ] Perjelas skenario evaluasi agar tidak ambigu.
  - `eval_metrics.py` mengevaluasi CIC + CSE sekaligus dalam satu run.
  - `cross_dataset_eval.py` adalah wrapper dari evaluasi yang sama (beda tag output).

- [ ] Cek konsistensi angka jumlah data.
  - Pastikan satu versi dipakai konsisten untuk:
    - CIC-IDS2017 (`~2.7M` vs `~2.8M`)
    - CSE-CIC-IDS2018 (`~16M` vs `~16.2M`)

## Prioritas Rendah

- [ ] Rapikan redaksi agar klaim hasil tidak terkesan final jika eksperimen belum selesai.
  - Ubah checklist keberhasilan menjadi target (`[ ]`) sampai metrik final benar-benar tersedia.

- [ ] Tambahkan lampiran tabel mapping nama kolom (CSE -> CIC) untuk transparansi metodologi.

## Catatan Teknis Error (Untuk Metodologi/Reproducibility)

- [ ] Dokumentasikan error training DirectML di Windows lokal.
  - Gejala: `InvalidArgumentError` pada `StatelessRandomGetKeyCounter` (multiple OpKernel registrations).
  - Penyebab kemungkinan: konflik backend TensorFlow DirectML saat multi-adapter GPU (RTX + iGPU).
  - Mitigasi saat ini: jalankan training dengan `training.force_cpu: true` di `config.yaml` agar stabil.

- [ ] Dokumentasikan error serialisasi history training.
  - Gejala: `TypeError: Object of type float32 is not JSON serializable` setelah epoch berjalan.
  - Penyebab: `history.history` berisi tipe NumPy (`float32`) yang tidak bisa langsung di-`json.dump`.
  - Mitigasi: cast nilai history ke `float` Python sebelum simpan ke JSON.

- [ ] Dokumentasikan risiko OOM/SIGKILL saat preprocess di Colab.
  - Gejala: proses preprocess mati dengan `SIGKILL: 9`.
  - Penyebab: volume data besar + limit RAM runtime Colab.
  - Mitigasi: kecilkan `chunksize`, `max_rows_per_file`, `sample_frac`, dan `shard_size`.

- [ ] Dokumentasikan issue GPU lokal laptop (Acer Nitro 5 RTX 3050, Windows + DirectML).
  - Gejala:
    - training gagal saat build model dengan error:
      `InvalidArgumentError: Multiple OpKernel registrations ... StatelessRandomGetKeyCounter`.
    - muncul walau adapter sudah dibatasi ke GPU:0.
  - Analisis:
    - ini bukan error path/data pipeline, tetapi bug/ketidakcocokan runtime TensorFlow-DirectML pada setup lokal.
    - warning `cudart64_110.dll not found` bukan akar masalah utama di kasus ini.
  - Keputusan operasional saat ini:
    - Lokal default CPU (`training.force_cpu: true`) demi stabilitas.
    - Jika butuh akselerasi GPU, gunakan Google Colab GPU (CUDA backend).
