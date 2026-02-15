# Glossary Teknis (Bahasa Sederhana)

## Fitur
- Kolom input yang dipakai model (X).
- Contoh: `Flow Duration`, `Flow Bytes/s`, `ACK Flag Count`.

## Label
- Kelas target (`BENIGN` atau jenis attack).
- Di unsupervised AE, label tidak dipakai untuk melatih rekonstruksi, tapi dipakai untuk evaluasi.

## Scaling / Scaler
- Menyamakan skala angka antar fitur.
- StandardScaler: `(x - mean_train) / std_train`.

## Fit Scaler
- Menghitung statistik scaler dari data train (di project ini: benign CIC).

## Transform
- Menerapkan scaler yang sudah di-fit ke data lain (val/test/CSE).

## Window / Sequence
- Menggabungkan beberapa baris jadi urutan waktu.
- Bentuk 3D: `(samples, timesteps, features)`.

## Reconstruction
- Model mencoba menyalin ulang input.
- Output model disebut `x_hat`.

## Reconstruction Error
- Selisih antara input asli `x` dan hasil rekonstruksi `x_hat`.
- Error besar -> cenderung anomali.

## Threshold
- Batas skor error untuk memutuskan normal/anomali.
- Contoh: percentile 99 dari benign validation.

## FPR (False Positive Rate)
- Persentase trafik normal yang salah ditandai attack.

## Generalization Gap
- Selisih performa CIC (in-distribution) vs CSE (OOD).
- Gap kecil = model lebih general.

## Shard
- File data kecil-kecil hasil pemecahan dataset besar.
- Tujuan: hemat RAM dan loading lebih stabil.

## Catatan Penggunaan

- Glosarium ini adalah referensi umum.
- Di tiap file tahap (`01`-`04`) ada `Glosarium Singkat` khusus konteks tahap tersebut.
