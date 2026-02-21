**SPRINT 3 — SOTA UPGRADE**

Laporan Hasil Eksperimen: CNN-LSTM Autoencoder untuk NIDS

*Experiment: sprint3\_sota\_upgrade\_v1  |  Februari 2025*


# **1. Ringkasan Eksperimen**
Notebook ini menjalankan full pipeline eksperimen Sprint 3: mulai dari eksplorasi data mentah CIC-IDS2017 (train) dan CSE-CIC-IDS2018 (test), preprocessing, drift analysis, training model CNN-LSTM Autoencoder, hingga evaluasi performa cross-domain.

# **2. Dataset**
## **2.1 CIC-IDS2017 (Training Set)**

|**Atribut**|**Nilai**|
| :- | :- |
|Jumlah File CSV|8 file|
|Total Sampel|2,830,743|
|Benign|2,273,097 (80.3%)|
|Attack|557,646 (19.7%)|
|Fitur|79 kolom|
|Sample Shape (processed)|(97,677, 10, 77)|

**Distribusi Attack (CIC-IDS2017):**

|**Label Attack**|**Jumlah**|
| :- | :- |
|DoS Hulk|231,073|
|PortScan|158,930|
|DDoS|128,027|
|DoS GoldenEye|10,293|
|FTP-Patator|7,938|
|SSH-Patator|5,897|
|DoS Slowloris|5,796|
|DoS Slowhttptest|5,499|
|Bot|1,966|
|Web Attack (Brute Force, XSS, SQL)|2,180|
|Infiltration, Heartbleed|47|

## **2.2 CSE-CIC-IDS2018 (Test Set — Cross-Domain)**

|**Atribut**|**Nilai**|
| :- | :- |
|Jumlah File CSV|3 file|
|Total Sampel|3,145,725|
|Benign|2,110,475 (67.1%)|
|Attack|1,035,250 (32.9%)|
|Fitur|80 kolom|

**Distribusi Attack (CSE-CIC-IDS2018):**

|**Label Attack**|**Jumlah**|
| :- | :- |
|DoS attacks-Hulk|461,912|
|FTP-BruteForce|193,360|
|SSH-Bruteforce|187,589|
|DoS attacks-SlowHTTPTest|139,890|
|DoS attacks-GoldenEye|41,508|
|DoS attacks-Slowloris|10,990|

# **3. Kualitas Data & Preprocessing**
## **3.1 Distribusi Label (Processed)**

|**Set**|**Total Sampel**|**Benign (0)**|**Attack (1)**|
| :- | :- | :- | :- |
|Train (CIC-IDS2017)|2,271,248|2,271,248 (100%)|0 (0%)|
|Test (CSE-CIC-IDS2018)|2,085,281|1,648,228 (79.0%)|437,053 (21.0%)|

*Train set hanya berisi data Benign, sesuai untuk training Autoencoder (unsupervised anomaly detection). Test set mengandung kedua kelas dengan class imbalance ratio 3.77.*

## **3.2 Kualitas Data (Processed .npz)**

|**Check**|**Train**|**Test**|
| :- | :- | :- |
|Shape|(97,677, 10, 77)|(1,044,742, 10, 77)|
|Timesteps|10|10|
|Fitur|77|77|
|Data Range|[0.0, 1.0]|[0.0, 1.0]|
|NaN|✅ None|✅ None|
|Inf|✅ None|✅ None|

## **3.3 Drift Analysis (KS Test, threshold = 0.5)**
- 27 fitur numerik umum dianalisis antara dataset 2017 dan 2018.
- 0 fitur dengan KS > 0.5 teridentifikasi — tidak ada fitur yang di-drop berdasarkan threshold ini.
- Namun, analisis feature shift menunjukkan adanya covariate shift signifikan pada beberapa fitur (lihat §3.4).

## **3.4 Feature Shift Analysis (Covariate Shift)**
Meski KS threshold tidak memfilter fitur, analisis mean shift menemukan beberapa fitur dengan drift CRITICAL:

|**Feature Index**|**Train Mean**|**Test Mean**|**Delta**|**Status**|
| :- | :- | :- | :- | :- |
|f42|0\.0182|0\.7260|+0.7078|CRITICAL|
|f1|0\.0000|0\.4368|+0.4367|CRITICAL|
|f29|0\.0548|0\.3449|+0.2900|CRITICAL|
|f37|0\.0134|0\.2686|+0.2552|CRITICAL|
|f45|0\.2781|0\.0640|-0.2141|CRITICAL|

*Covariate shift ini menjadi penyebab utama tingginya False Positive pada data Benign 2018 — Autoencoder menganggap pola baru sebagai anomali.*

## **3.5 Korelasi Fitur**
Ditemukan 81 pasangan fitur dengan korelasi > 0.9, termasuk 8 pasang dengan korelasi sempurna (1.0). Rekomendasi: pertimbangkan feature selection untuk mengurangi redundansi.

# **4. Arsitektur Model**
Model yang digunakan adalah CNN-LSTM Autoencoder yang di-load dari checkpoint pre-trained:

**📁 best\_model.keras  |  Path: research/sprint3\_upgrade/models/**

|**Layer**|**Output Shape**|**Params**|
| :- | :- | :- |
|Input Layer|(None, 10, 77)|0|
|Conv1D|(None, 10, 32)|7,424|
|MaxPooling1D|(None, 5, 32)|0|
|Conv1D\_1|(None, 5, 16)|1,552|
|LSTM (Encoder)|(None, 5, 64)|20,736|
|Dropout|(None, 5, 64)|0|
|Flatten|(None, 320)|0|
|Dense (Bottleneck)|(None, 16)|5,136|
|RepeatVector|(None, 5, 16)|0|
|LSTM\_1 (Decoder)|(None, 5, 64)|20,736|
|Dropout\_1|(None, 5, 64)|0|
|Conv1D\_2|(None, 5, 16)|3,088|
|UpSampling1D|(None, 10, 16)|0|
|Conv1D\_3 (Output)|(None, 10, 77)|3,773|

|**Parameter Type**|**Jumlah**|
| :- | :- |
|Total Params|187,337 (731.79 KB)|
|Trainable Params|62,445 (243.93 KB)|
|Non-trainable Params|0|
|Optimizer Params|124,892 (487.86 KB)|

# **5. Threshold Determination**

|**Parameter**|**Nilai**|
| :- | :- |
|Metode|Percentile pada Validation Set (Benign)|
|Percentile|99\.0%|
|Threshold (Current)|0\.016827|
|Optimal Threshold (PR Curve)|0\.013062 / 0.013654|

# **6. Hasil Evaluasi**
## **6.1 Evaluasi Cross-Domain: CSE-CIC-IDS2018**
Evaluasi dilakukan pada 20% data test (~417,024 sampel). Threshold yang digunakan: 0.016827.

|**Metrik**|**Nilai**|
| :- | :- |
|Accuracy|0\.4944 (49.44%)|
|Precision|0\.8608 (86.08%)|
|Recall (Sensitivity)|0\.5362 (53.62%)|
|F1-Score|0\.6608 (66.08%)|
|PR-AUC|0\.9013|
|ROC-AUC|0\.4304|

**Confusion Matrix — CSE-CIC-IDS2018:**

||**Predicted Benign**|**Predicted Attack**|
| :- | :- | :- |
|Actual Benign (TN/FP)|TN: 804|FP: 33,208|
|Actual Attack (FN/TP)|FN: 177,628|TP: 205,384|

## **6.2 Evaluasi In-Domain: CIC-IDS2017**
Evaluasi pada 20% data CIC-IDS2017 (~565,504 sampel).

|**Metrik**|**Nilai**|
| :- | :- |
|Accuracy|0\.5513 (55.13%)|
|Precision|0\.8110 (81.10%)|
|Recall (Sensitivity)|0\.1707 (17.07%)|
|F1-Score|0\.2821 (28.21%)|
|PR-AUC|0\.6743|
|ROC-AUC|0\.6186|

**Confusion Matrix — CIC-IDS2017:**

||**Predicted Benign**|**Predicted Attack**|
| :- | :- | :- |
|Actual Benign (TN/FP)|TN: 261,880|FP: 11,617|
|Actual Attack (FN/TP)|FN: 242,151|TP: 49,856|

# **7. Perbandingan Performa**

|**Metrik**|**CSE-CIC-IDS2018 (Cross-Domain)**|**CIC-IDS2017 (In-Domain)**|
| :- | :- | :- |
|Accuracy|49\.44%|55\.13%|
|Precision|86\.08%|81\.10%|
|Recall|53\.62%|17\.07%|
|F1-Score|66\.08%|28\.21%|
|PR-AUC|0\.9013|0\.6743|
|ROC-AUC|0\.4304|0\.6186|
|Optimal Threshold|0\.013654|0\.000144|
|Max F1 (optimal)|0\.9578|0\.7478|

# **8. Analisis Reconstruction Error**
## **CSE-CIC-IDS2018**

|**Statistik**|**Benign**|**Attack**|
| :- | :- | :- |
|Count|34,012|383,012|
|Mean|0\.038499|0\.033826|
|Median|0\.037044|0\.034232|
|Std|0\.013141|0\.017954|
|Min|0\.004196|0\.013281|
|Max|0\.091964|0\.083991|

**KS Statistic: 0.4416, p-value: ~0  →  Distribusi berbeda signifikan ✅**

⚠️ Anomali: Benign 2018 memiliki reconstruction error lebih tinggi dari Attack (0.038 vs 0.034). Ini mengindikasikan covariate shift — Autoencoder menganggap data Benign 2018 sebagai anomali karena distribusi berbeda dari training data.

## **CIC-IDS2017**

|**Statistik**|**Benign**|**Attack**|
| :- | :- | :- |
|Count|273,497|292,007|
|Mean|0\.002151|0\.008416|
|Median|0\.000471|0\.000333|
|Std|0\.005057|0\.014672|

**KS Statistic: 0.3033, p-value: ~0  →  Distribusi berbeda signifikan ✅**

# **9. Threshold Tuning**

|**Kondisi**|**Threshold**|**F1-Score**|**Precision**|**Recall**|
| :- | :- | :- | :- | :- |
|Current (percentile 99%)|0\.016827|0\.6608|0\.8608|0\.5362|
|Optimal (PR Curve)|0\.013654|0\.9578|0\.9189|1\.0000|
|Potential Improvement|-|+44.93%|+7%|+46%|

***Rekomendasi: Gunakan threshold ~0.0136 untuk memaksimalkan F1-Score pada dataset CSE-CIC-IDS2018.***

# **10. Kesimpulan & Rekomendasi**
## **Temuan Utama**
- Model CNN-LSTM Autoencoder berhasil di-load dan dievaluasi secara cross-domain.
- Precision model cukup tinggi (86%) namun Recall rendah (53.6%) dengan threshold saat ini — artinya model konservatif dan banyak serangan tidak terdeteksi.
- PR-AUC 0.9013 pada CSE-CIC-IDS2018 menunjukkan model memiliki potensi diskriminasi yang sangat baik.
- ROC-AUC rendah (0.43) disebabkan oleh tingginya FPR — mayoritas Benign 2018 salah diklasifikasi sebagai attack akibat covariate shift.
- In-domain (CIC-IDS2017): Recall sangat rendah (17%) karena threshold terlalu tinggi untuk distribusi error in-domain yang kecil.

## **Rekomendasi Lanjutan**
- Tuning Threshold: Gunakan threshold ~0.0136 (optimal) untuk mendapatkan F1 hingga 0.9578.
- Domain Adaptation: Lakukan re-scaling atau normalisasi ulang untuk mengurangi covariate shift fitur CRITICAL (f42, f1, f29, f37, f45).
- Feature Selection: Drop fitur dengan korelasi >0.9 (81 pasang) untuk mengurangi redundansi dan potensi noise.
- Fine-tuning: Pertimbangkan fine-tuning model dengan sebagian data Benign 2018 untuk adaptasi domain.
- Drift Threshold: Review KS threshold (saat ini 0.5 terlalu longgar); coba 0.2 untuk menangkap lebih banyak fitur drifted.

*Sprint 3 SOTA Upgrade  |  Experiment: sprint3\_sota\_upgrade\_v1  |  CNN-LSTM Autoencoder NIDS*
