# Analisis GPU Lokal (Windows + RTX 3050) untuk Project NIDS

## Ringkasan

Dokumen ini merangkum masalah training GPU di laptop lokal dan rekomendasi yang paling realistis untuk workflow skripsi.

## Kondisi Environment Saat Ini

- OS: Windows 10 (`10.0.26200`)
- Python: `3.10.6`
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (+ Intel iGPU)
- Package terpasang (berdasarkan pengecekan lokal):
  - `tensorflow==2.10.1`
  - `tensorflow-cpu==2.10.0`
  - `tensorflow-directml-plugin==0.4.0.dev230202`

## Gejala Error yang Muncul

Saat training, proses gagal dengan error:

- `InvalidArgumentError: Multiple OpKernel registrations ... StatelessRandomGetKeyCounter`

Selain itu ada warning:

- `Could not load dynamic library 'cudart64_110.dll'`

Catatan: warning `cudart64_110.dll` bukan akar utama crash di kasus ini; masalah utama ada pada konflik backend/kernel registration.

## Akar Masalah Paling Mungkin

1. Konflik package TensorFlow di environment yang sama.
2. DirectML plugin memang punya keterbatasan kompatibilitas pada beberapa operasi/model.
3. Setup hybrid adapter (NVIDIA + Intel) dapat memperbesar potensi konflik saat runtime.

## Klarifikasi Teknis Penting

1. Native GPU TensorFlow di Windows didukung sampai TensorFlow `2.10`.
2. `tensorflow-gpu` bukan jalur instalasi yang direkomendasikan lagi (paket lama/deprecated).
3. Untuk DirectML, basis instalasi yang benar adalah `tensorflow-cpu==2.10` + `tensorflow-directml-plugin` (bukan dicampur dengan `tensorflow` biasa).

## Opsi Solusi

### Opsi A (Paling Aman untuk Progres Skripsi Saat Ini)

- Lokal: pakai CPU (`training.force_cpu: true`)
- GPU: pakai Google Colab

Kelebihan:
- Stabil, minim waktu debugging environment.
- Fokus ke hasil eksperimen dan penulisan.

Kekurangan:
- Training lokal lebih lambat.

### Opsi B (Coba GPU Lokal via DirectML)

Langkah prinsip:
- Bersihkan env TensorFlow.
- Install hanya:
  - `tensorflow-cpu==2.10.*`
  - `tensorflow-directml-plugin`
- Hindari campur dengan paket `tensorflow` lain dalam env yang sama.

Risiko:
- Masih bisa kena bug DirectML yang sama.

### Opsi C (Future-proof)

- Migrasi ke WSL2 + stack TensorFlow/CUDA yang lebih modern.

Kelebihan:
- Jalur jangka panjang lebih sehat untuk eksperimen DL.

Kekurangan:
- Butuh setup ulang environment.

## Rekomendasi Final untuk Project Ini

Untuk target skripsi dan keterbatasan waktu:

1. Tetap gunakan **CPU lokal** untuk workflow harian/debug.
2. Gunakan **Colab GPU** untuk training besar/final run.
3. Catat kendala GPU lokal ini di bagian keterbatasan penelitian agar transparan dan reproducible.

## Referensi Resmi

- TensorFlow install (pip):  
  https://www.tensorflow.org/install/pip
- TensorFlow source build Windows (compatibility):  
  https://www.tensorflow.org/install/source_windows
- Microsoft DirectML TensorFlow plugin (status + setup):  
  https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin
- Repository DirectML plugin:  
  https://github.com/microsoft/tensorflow-directml-plugin
- PyPI `tensorflow-gpu` (deprecated):  
  https://pypi.org/project/tensorflow-gpu/
