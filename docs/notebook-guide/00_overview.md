# Overview Notebook Pipeline

Folder ini menjelaskan alur `notebooks/colab_ready.ipynb` dalam file terpisah per tahap.

## Urutan Tahapan

1. `01_preprocess.md`
2. `02_train_hybrid.md`
3. `03_evaluate.md`
4. `04_baseline.md`
5. `05_glossary_teknis.md`

## Alur Data Singkat

1. CSV mentah -> dibersihkan + disamakan fitur.
2. Data numerik -> diskalakan.
3. Baris 2D -> sequence 3D (window).
4. Sequence disimpan shard -> dipakai train/eval.
5. Model menghasilkan reconstruction error -> threshold -> prediksi anomali.

## Glosarium Singkat

- `Sequence 3D`: Bentuk data untuk LSTM, formatnya `(samples, timesteps, features)`.
- `Shard`: Pecahan file data besar menjadi file kecil agar hemat RAM.
- `Reconstruction error`: Selisih input asli vs output hasil rekonstruksi model.
- `Threshold`: Nilai batas untuk menentukan normal atau anomali.
