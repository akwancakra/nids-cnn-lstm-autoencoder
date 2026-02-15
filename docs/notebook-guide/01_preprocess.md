# Tahap 1 - Preprocess (Detail Teknis)

Script: `scripts/preprocess.py`

## Tujuan

- Menyiapkan data agar bisa dipakai model sequence (LSTM/AE).
- Menjamin CIC dan CSE punya fitur yang sebanding.

## Input

- `data/raw/CIC-IDS2017/*.csv`
- `data/raw/CSE-CIC-IDS2018/*.csv`
- Konfigurasi dari `config.yaml`

## Langkah Teknis

1. **Deteksi header & label**
- Header dibaca dengan `skipinitialspace=True` lalu di-`strip()`.
- Label dicari dari kandidat: `Label`, `label`, `Class`, `class`.

2. **Harmonisasi nama kolom CSE -> skema CIC**
- CSE punya naming singkat (`Dst`, `Cnt`, `Byts`) dibanding CIC.
- Script memetakan kolom CSE ke nama acuan CIC sebelum intersection.

3. **Feature intersection**
- Ambil irisan fitur yang tersedia di kedua dataset.
- Drop kolom non-fitur (IP, port, timestamp, protocol, dll sesuai config).

4. **Cleaning nilai**
- Object -> numerik jika memungkinkan.
- `inf/-inf` -> `NaN`.
- `NaN` -> `fillna_value` (default `0.0`).

5. **Label biner**
- `BENIGN` -> `0`
- selain `BENIGN` -> `1`

6. **Fit scaler**
- Scaler di-fit hanya dari **benign CIC**.
- Tujuan: hindari data leakage dan jaga fairness evaluasi OOD.

7. **Windowing sequence**
- Bentuk awal: `(rows, features)`.
- Bentuk akhir: `(samples, window_size, features)`.
- Contoh `window_size=10`, `stride=1`.
- Label window = `1` jika ada attack dalam window.

8. **Split & shard**
- CIC split by file -> train/val/test.
- Train & val hanya benign windows.
- CSE dipakai test OOD.
- Disimpan ke shard `.npz` + `manifest.json`.

## Output

- `data/processed/scaler.pkl`
- `data/processed/feature_columns.json`
- `data/processed/shards/cic/train/manifest.json`
- `data/processed/shards/cic/val/manifest.json`
- `data/processed/shards/cic/test/manifest.json`
- `data/processed/shards/cse/test/manifest.json`

## Cara Verifikasi Cepat

- Log menampilkan `Feature intersection count: ...`.
- Tidak ada error label.
- `manifest.json` berisi `total_samples`, `num_shards`, `input_shape`.

## Glosarium Singkat

- `Feature intersection`: Irisan fitur yang sama-sama tersedia di CIC dan CSE.
- `Data leakage`: Kebocoran informasi test/target ke proses training.
- `OOD (Out-of-Distribution)`: Data uji dari distribusi berbeda dengan data latih.
- `window_size`: Panjang urutan waktu dalam satu sampel sequence.
- `stride`: Jarak pergeseran antar window sequence.
- `manifest.json`: Ringkasan shard (jumlah sampel, input shape, daftar file shard).
