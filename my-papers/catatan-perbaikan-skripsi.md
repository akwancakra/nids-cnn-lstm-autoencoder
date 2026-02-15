# Catatan Perbaikan Skripsi

Dokumen ini berisi daftar hal yang perlu dirapikan agar naskah konsisten dengan data lokal dan mudah dipertahankan saat sidang.

## Prioritas Tinggi

### 1) Konsistensi jumlah fitur dataset (CIC-IDS2017 vs CSE-CIC-IDS2018)
- Status: `Perlu diperbaiki`
- Masalah:
  - Di beberapa bagian naskah tertulis **CIC-IDS2017: "lebih dari 80 fitur"**.
  - Hasil cek cepat data lokal menunjukkan:
    - CIC-IDS2017: **79 kolom total** (termasuk `Label`) -> **78 fitur** (tanpa `Label`)
    - CSE-CIC-IDS2018: **80 kolom total** (termasuk `Label`) -> **79 fitur** (tanpa `Label`)
- Dampak:
  - Inkonstistensi angka fitur bisa jadi pertanyaan penguji terkait validitas data.
- Saran revisi redaksi (pilih satu standar dan pakai konsisten di semua bab):
  - Opsi aman: tulis dalam format **"kolom total (termasuk label)"** dan **"fitur prediktor (tanpa label)"**.
  - Contoh:
    - CIC-IDS2017: `79 kolom total, 78 fitur prediktor`
    - CSE-CIC-IDS2018: `80 kolom total, 79 fitur prediktor`
- Lokasi yang perlu dicek/revisi:
  - `my-papers/isi-bab-3.md`
  - `my-papers/final-1-3.md`
  - `my-papers/topik-fixed.md`
  - `KNOWLEDGE.md` (agar konsisten dengan naskah utama)

## Catatan Lanjutan (Untuk Ditambah Bertahap)
- [ ] Konsistensi sumber dataset (resmi vs Kaggle) di bagian metodologi.
- [ ] Konsistensi istilah: `fitur`, `kolom`, `feature`, `records/flows`.
- [ ] Konsistensi angka jumlah data per dataset (2.7 vs 2.8 juta; 16.2 vs 16 juta).
