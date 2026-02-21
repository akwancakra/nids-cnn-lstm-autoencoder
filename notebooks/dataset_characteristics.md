# Karakteristik Dataset CIC-IDS2017 vs CSE-CIC-IDS2018

Dokumen ini merangkum karakteristik dataset yang digunakan dalam eksperimen SOTA Upgrade untuk NIDS berbasis CNN-LSTM Autoencoder. Analisis ini diambil dari notebook `karakteristik-dataset-disini.ipynb`.

## 1. Dataset Sumber (Training & Baseline Test): CIC-IDS2017

Dataset ini digunakan sebagai **Source Domain**. Model dilatih hanya menggunakan trafik normal (Benign) dari tahun 2017 untuk mempelajari pola dasar jaringan yang sehat.

*   **Lokasi Raw:** `/content/drive/MyDrive/nids-data/raw/CIC-IDS2017`
*   **Jumlah Fitur Awal:** 79 kolom
*   **Struktur Kolom Utama:**
    *   `Destination Port`: Port tujuan koneksi.
    *   `Flow Duration`: Durasi aliran trafik.
    *   `Total Fwd Packets` / `Total Backward Packets`: Jumlah paket terkirim/diterima.
    *   `Total Length of Fwd/Bwd Packets`: Total ukuran paket.
    *   `Fwd/Bwd Packet Length (Max, Min, Mean, Std)`: Statistik panjang paket.
    *   `Flow Bytes/s`, `Flow Packets/s`: Laju trafik (seringkali mengandung nilai Infinity yang perlu dibersihkan).
    *   `Label`: Label kelas (`BENIGN` atau jenis serangan seperti `DDoS`, `PortScan`, dll).
*   **Kualitas Data:**
    *   **Missing Values:** Beberapa kolom seperti `Flow Bytes/s` memiliki nilai yang hilang atau tidak valid (Infinity/NaN) yang perlu ditangani saat preprocessing.
    *   **Labeling:** Label serangan spesifik (misal: `DoS Hulk`, `SSH-Patator`).

## 2. Dataset Target (Testing Utama): CSE-CIC-IDS2018

Dataset ini digunakan sebagai **Target Domain**. Model diuji kemampuannya mendeteksi serangan pada lingkungan jaringan yang berbeda (tahun 2018).

*   **Lokasi Raw:** `/content/drive/MyDrive/nids-data/raw/CSE-CIC-IDS2018`
*   **Jumlah Fitur Awal:** 80 kolom (Ada sedikit perbedaan nama kolom dibanding 2017).
*   **Perbedaan Kolom (Domain Shift Issue):**
    *   Dataset 2018 memiliki kolom `Timestamp` dengan format berbeda.
    *   Penamaan kolom sedikit berbeda (misal: `Dst Port` vs `Destination Port`, `Tot Fwd Pkts` vs `Total Fwd Packets`).
    *   **Solusi Preprocessing:** Script `preprocess_sota.py` melakukan standarisasi nama kolom (menghapus spasi, menyamakan nama) dan membuang kolom non-fitur seperti `Timestamp`, `Flow ID`, `IP Address` agar kedua dataset memiliki struktur fitur yang identik (Intersection of Columns).

## 3. Strategi Preprocessing & Domain Adaptation

Berdasarkan karakteristik di atas, pipeline SOTA Upgrade menerapkan strategi berikut:

1.  **Column Alignment:** Hanya menggunakan fitur yang **ada di kedua dataset** (Intersection). Kolom unik di satu tahun dibuang.
2.  **Feature Drift Removal:** Menggunakan analisis statistik (KS-Test) untuk membuang fitur yang distribusinya berubah drastis antara 2017 dan 2018 (High Drift), karena fitur ini akan menyebabkan False Positive tinggi pada Autoencoder.
3.  **Benign-Only Training:** Model hanya dilatih pada data `BENIGN` dari CIC-IDS2017.
4.  **Robust Scaling:** Menggunakan `MinMaxScaler(0, 1)` yang di-fit **hanya** pada data Training (2017) untuk mencegah data leakage. Data Test (2018) di-scale menggunakan parameter dari 2017, dengan clipping `[0, 1]` untuk menangani outlier ekstrim di tahun 2018.

## 4. Kesimpulan untuk Eksperimen

*   **Tantangan Utama:** Perubahan karakteristik jaringan (distribusi fitur) dari 2017 ke 2018.
*   **Solusi:** Preprocessing yang ketat (Drift Analysis + Common Features) sangat krusial sebelum data masuk ke model CNN-LSTM.
*   **Ekspektasi:** Performa deteksi pada data 2017 (In-Domain) akan lebih tinggi daripada 2018 (Out-Domain), namun gap tersebut diharapkan mengecil setelah fitur-fitur high-drift dibuang.
