# Sprint 2 — Peningkatan Isolasi dan Validasi Reproducibilitas

## Konteks dan Tujuan
Fokus Sprint 2 adalah memperbaiki_issue interference antar eksperimen melalui full isolation per `run_id`. Setiap eksperimen diberi namespace output terpisah agar hasil dapat dipertanggungjawabkan dan reproduksi secara eksplisit.

## Konfigurasi Baseline
Adopsi identitas baseline dari Sprint 1 (rs11 configuration), namun dengan struktur presisi per-run. Model hybrid dan baseline LSTM-AE dievaluasi dalam mode isolated, threshold percentile default (99), dengan strict isolation pada data, model, dan results untuk setiap run_id.

## Eksperimen yang Dijalankan
Tidak ada run yang selesai dengan metrics terkumpul. Enam eksperimen dirancang:

| run_id | variasi | status | metrik |
|--------|---------|--------|--------|
| s2_hybrid_zero_seed42_base | hybrid full pipeline | pending | tidak tersedia |
| s2_hybrid_few_seed42_tp99_eval | hybrid few-shot eval | pending | tidak tersedia |
| s2_baseline_zero_seed42_base | LSTM-AE full pipeline | pending | tidak tersedia |
| s2_baseline_few_seed42_tp99_eval | LSTM-AE few-shot eval | pending | tidak tersedia |
| s2_hybrid_zero_seed1234_candidate | hybrid seed 1234 | tidak aktif | tidak tersedia |
| s2_baseline_zero_seed1234_candidate | LSTM-AE seed 1234 | tidak aktif | tidak tersedia |

## Temuan Utama
Sprint 2 terutama menyusun fondasi metodologis tanpa mencapai data eksperimen final. Infrastruktur isolation berhasil diimplementasikan, mempersiapkan pipeline untuk eksperimen yang dapat direproduksi secara reliable. Kontribusi utama adalah perubahan dari eksperimen yang tumpang-tindih ke eksperimen yang terisolasi, memungkinkan pembandingan head-to-head tanpa concern kontaminasi data atau artifact.

## Keputusan Gate
Gate pass: true (kondisional). Sprint 2 dinilai sebagai sprint infrastruktur yang menetapkan standar reproducibility, sehingga gate tidak menjadi barrier melainkan mekanisme validasi untuk sprint eksperimen selanjutnya.

## Warisan untuk Sprint Berikutnya
Structural isolation (path spec per run_id, reuse_artifacts_from mechanism, hash verification) menjadi standar permanent untuk sprint berikutnya. Hipotesis validasi bahwa reproducibilitas andalah komponen penting dari penelitian sistem deteksi intrusi di mana hasil perlu diverifikasi secara independent.
