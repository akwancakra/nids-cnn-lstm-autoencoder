# BAB 3 METODOLOGI PENELITIAN

## 3.1 Desain Penelitian

Penelitian ini menggunakan Design Research Methodology (DRM) sebagai pendekatan sistematis untuk mengidentifikasi masalah, mengembangkan solusi, dan mengevaluasi efektivitasnya (Blessing & Chakrabarti, 2009). DRM dipilih karena menyediakan kerangka desain yang mendukung pengembangan artefak teknis seperti model deteksi intrusi, dengan empat tahapan yang bersifat iteratif dimana hasil evaluasi dapat memicu perbaikan desain atau penyesuaian metode. Proses iterasi ini terjadi ketika hasil Studi Deskriptif II menunjukkan performa yang belum optimal, sehingga dapat dilakukan refinement pada tahap Studi Preskriptif seperti tuning hyperparameter atau penyesuaian threshold deteksi. Metodologi ini memastikan penelitian berjalan terstruktur, khususnya pada pengembangan model Hybrid CNN-LSTM Autoencoder untuk deteksi zero-day attack dengan pendekatan unsupervised learning. Gambar 3.1 menunjukkan alur DRM yang diterapkan.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                  DESIGN RESEARCH METHODOLOGY (DRM)                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. KLARIFIKASI PENELITIAN (Research Clarification)                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ • Studi Literatur: IDS, Deep Learning (CNN-LSTM), Autoencoder, Zero-Day.     │
│ • Identifikasi Masalah: Keterbatasan IDS konvensional mendeteksi serangan    │
│   baru (zero-day) & masalah generalisasi antar dataset berbeda.              │
│ • Tujuan: Mengembangkan model unsupervised yang robust lintas dataset.       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. STUDI DESKRIPTIF I (Descriptive Study I)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ • Analisis Dataset:                                                          │
│   - Karakteristik CIC-IDS2017 (untuk Training & Baseline Test)               │
│   - Karakteristik CSE-CIC-IDS2018 (untuk Generalization/Zero-Day Test)       │
│ • Analisis Fitur: Identifikasi fitur irisan (intersecting features) agar     │
│   kompatibel untuk validasi lintas dataset.                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. STUDI PRESKRIPTIF (Prescriptive Study)                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────┐ │
│ │    Data Preprocessing│───▶│   Model Development  │───▶│  Model Training  │ │
│ │ • Cleaning & Norm.   │    │ • Hybrid CNN-LSTM    │    │ • Unsupervised   │ │
│ │ • Feature Alignment  │    │ • Autoencoder Arch.  │    │   (Normal Only)  │ │
│ └──────────────────────┘    └──────────────────────┘    └──────────────────┘ │
│                                                                              │
│ ┌──────────────────────┐                                                     │
│ │ Hyperparameter Tuning│◀────────────────────────────────────────────────────┘
│ │ • Grid/Random Search │
│ └──────────────────────┘
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. STUDI DESKRIPTIF II (Descriptive Study II)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ • Evaluasi Baseline: Testing pada CIC-IDS2017 (In-Distribution).             │
│ • Threshold Optimization: Menentukan batas error rekonstruksi optimal.       │
│ • Cross-Dataset Validation: Testing pada CSE-CIC-IDS2018 (Zero-Day/OOD).     │
│ • Analisis & Kesimpulan:                                                     │
│   - Analisis Generalization Gap & Uji Statistik (McNemar's Test)             │
│   - Pembahasan Hasil, Dokumentasi, & Penarikan Kesimpulan                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

Gambar 3.1 Design Research Methodology

### 3.1.1 Klarifikasi Penelitian

Tahap klarifikasi penelitian dilakukan untuk mengidentifikasi inti permasalahan melalui studi literatur yang komprehensif terhadap berbagai referensi ilmiah terkait keamanan siber, deteksi serangan zero-day, algoritma deep learning, dan teknik unsupervised learning. Kajian literatur mencakup arsitektur CNN untuk ekstraksi fitur spasial, LSTM untuk pemodelan temporal, dan Autoencoder untuk deteksi anomali berbasis reconstruction error. Selain itu, dipelajari juga metode evaluasi performa model seperti akurasi, presisi, recall, F1-score, dan AUC-ROC sebagai acuan penilaian keberhasilan pendekatan (Park et al., 2025). Analisis research gap pada tahap ini menemukan bahwa mayoritas penelitian IDS sebelumnya hanya melakukan evaluasi pada single dataset tanpa memvalidasi kemampuan generalisasi model terhadap dataset berbeda, yang menjadi fokus utama penelitian ini.

Berdasarkan analisis gap tersebut, penelitian ini merumuskan dua pertanyaan penelitian utama: (1) Bagaimana performa model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan siber pada dataset CIC-IDS2017 dibandingkan dengan baseline methods? (2) Bagaimana kemampuan generalisasi model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan zero-day melalui cross-dataset validation menggunakan dataset CSE-CIC-IDS2018 dengan perbandingan skenario zero-shot dan few-shot adaptation? Untuk menjawab pertanyaan tersebut, ditetapkan kriteria keberhasilan penelitian yaitu model harus mencapai F1-score minimal 90% pada dataset baseline dan generalization gap tidak boleh melebihi 15% untuk memastikan model robust terhadap data baru.

### 3.1.2 Studi Deskriptif I (Analisis Kebutuhan dan Data)

Tahap ini berfokus pada analisis kebutuhan sistem dan karakteristik data yang akan digunakan. Berdasarkan tinjauan literatur, teridentifikasi beberapa keterbatasan metode eksisting seperti ketergantungan pada supervised learning yang memerlukan label serangan terbaru, evaluasi single-dataset yang tidak membuktikan generalisasi model, dan kompleksitas arsitektur ensemble yang berdampak pada beban komputasi tinggi. Keterbatasan ini menjadi dasar pemilihan pendekatan unsupervised learning dengan cross-dataset validation pada penelitian ini. Analisis kebutuhan mencakup identifikasi spesifikasi teknis seperti environment dengan GPU support, framework TensorFlow, serta tools preprocessing seperti Pandas dan NumPy (Singh & Jang, 2022). Analisis data dilakukan terhadap dua dataset publik, yaitu CIC-IDS2017 dengan 2,7 juta records dan lebih dari 80 fitur, serta CSE-CIC-IDS2018 dengan 16,2 juta records dan 80 fitur. Dataset pertama mencakup trafik normal dan berbagai serangan seperti DoS/DDoS, Web Attacks, Botnet, dan Heartbleed, sedangkan dataset kedua memiliki serangan lebih kompleks. Hasil analisis menunjukkan perlunya alignment fitur antara kedua dataset untuk memungkinkan cross-dataset validation sebagai kontribusi krusial penelitian ini.

### 3.1.3 Studi Preskriptif (Pengembangan Model)

Studi preskriptif merupakan tahap inti yang mencakup seluruh proses teknis pengembangan model Hybrid CNN-LSTM Autoencoder untuk deteksi anomali zero-day. Tahapan ini terdiri dari beberapa subbagian yang saling berkaitan, mulai dari persiapan environment, analisis data awal, preprocessing, implementasi arsitektur model, training dan optimisasi, hingga validasi lintas dataset. Secara keseluruhan, tahapan ini merealisasikan pendekatan solusi yang telah dirumuskan sebelumnya melalui serangkaian eksperimen sistematis.

#### 3.1.3.1 Persiapan Environment

Tahap persiapan dimulai dengan instalasi Python 3.8+ dan setup virtual environment menggunakan conda atau venv. Library dependencies yang digunakan meliputi TensorFlow 2.x, NumPy, Pandas, Scikit-learn, Matplotlib, dan Seaborn untuk mendukung proses training dan analisis data. Konfigurasi GPU driver CUDA dilakukan untuk mempercepat proses training model. Dataset CIC-IDS2017 dan CSE-CIC-IDS2018 diunduh dari situs resmi CIC dan diekstraksi ke dalam struktur direktori yang terorganisir (Park et al., 2025).

#### 3.1.3.2 Exploratory Data Analysis

Exploratory data analysis (EDA) dilakukan untuk memahami karakteristik dataset melalui inspeksi struktur data, identifikasi missing values dan infinite values, serta visualisasi distribusi fitur menggunakan histogram dan boxplot. Analisis korelasi antar fitur dilakukan menggunakan heatmap correlation matrix untuk mengidentifikasi fitur yang highly correlated. Berdasarkan hasil EDA, dilakukan preprocessing yang mencakup handling missing values, handling infinite values, feature alignment antara kedua dataset, dan data cleaning untuk menghapus duplikasi serta outlier ekstrem. Khusus data training, dilakukan filtering untuk mengambil hanya data dengan label "BENIGN" sesuai paradigma unsupervised learning (Almuhanna & Alahmadi, 2025).

#### 3.1.3.3 Preprocessing dan Feature Engineering

Feature engineering mencakup encoding categorical features, normalisasi fitur numerik menggunakan MinMaxScaler atau StandardScaler, dan reshaping data menjadi format 3D (samples, timesteps, features) sesuai input LSTM layer. Data split dilakukan dengan proporsi 70% training, 15% validation, dan 15% testing pada CIC-IDS2017. Pada CSE-CIC-IDS2018, data digunakan untuk dua skenario cross-dataset validation: zero-shot (tanpa adaptasi) dan few-shot adaptation dengan 1% data benign target untuk adaptasi unsupervised, sedangkan sisanya digunakan untuk evaluasi (Singh & Jang, 2022). Alur lengkap pemrosesan data, dari data mentah hingga menjadi input siap latih, diilustrasikan pada Gambar 3.2.

[Gambar 3.2 Alur Preprocessing dan Pembentukan Data Time-Series]

#### 3.1.3.4 Implementasi Arsitektur Model

Implementasi arsitektur Hybrid CNN-LSTM Autoencoder dilakukan menggunakan Keras API dengan struktur encoder yang terdiri dari Conv1D layers untuk ekstraksi fitur spasial diikuti LSTM layers untuk pemodelan temporal, bottleneck layer sebagai latent representation, dan decoder yang merupakan mirror dari encoder untuk rekonstruksi input. Arsitektur ini dirancang khusus untuk menangkap pola spasial dan temporal dalam network traffic data secara simultan. Detail arsitektur model beserta dimensi data pada setiap lapisannya ditampilkan secara rinci pada Gambar 3.3.

[Gambar 3.3 Arsitektur Hybrid CNN-LSTM Autoencoder yang Diusulkan]

#### 3.1.3.5 Training dan Optimisasi Model

Training model menggunakan hanya data normal dengan loss function Mean Squared Error (MSE) untuk mengukur reconstruction error antara input asli dan output rekonstruksi, yang diformulasikan pada Persamaan (8).

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2 \quad (8)
$$

Proses training menggunakan optimizer Adam dengan learning rate 0.001, batch size 128-256, dan early stopping jika validation loss tidak menurun selama 10 epoch berturut-turut. Monitoring dilakukan terhadap training loss dan validation loss untuk mendeteksi overfitting atau underfitting. Evaluasi pada test set CIC-IDS2017 dilakukan dengan menghitung reconstruction error, menentukan threshold berdasarkan 95th atau 99th percentile, dan menghitung metrik evaluasi seperti akurasi, presisi, recall, F1-score, dan AUC-ROC (Baidar et al., 2025).

#### 3.1.3.6 Cross-Dataset Validation

Cross-dataset validation dilakukan dengan menggunakan model yang dilatih pada CIC-IDS2017 untuk memprediksi sampel dari CSE-CIC-IDS2018 dalam dua skenario. Skenario pertama adalah zero-shot, yaitu evaluasi langsung tanpa fine-tuning atau retraining. Skenario kedua adalah few-shot adaptation, yaitu fine-tuning unsupervised menggunakan 1% sampel benign dari domain target, tanpa label attack dan tanpa overlap dengan data evaluasi. Reconstruction error dihitung untuk semua sampel, threshold yang sama dari training digunakan untuk klasifikasi, dan metrik evaluasi dihitung untuk mengukur performa pada dataset berbeda. Analisis generalization gap dilakukan dengan membandingkan performa pada kedua dataset, serta analisis per-attack-type performance untuk melihat jenis serangan yang paling sulit dideteksi. Skema validasi lintas dataset ini, yang menjadi metode utama untuk menguji kemampuan generalisasi dan deteksi serangan zero-day, dapat dilihat secara visual pada Gambar 3.4.

[Gambar 3.4 Skema Eksperimen Validasi Lintas Dataset (Cross-Dataset Validation)]

Dokumentasi hasil mencakup pembuatan visualisasi ROC curve, confusion matrix, dan reconstruction error distribution plot, serta tabel hasil evaluasi yang membandingkan berbagai metrik (Halbouni et al., 2022).

### 3.1.4 Studi Deskriptif II (Evaluasi Model)

Tahap Studi Deskriptif II dilakukan untuk mengevaluasi performa model secara komprehensif dan menganalisis hasil eksperimen. Evaluasi dilakukan dalam beberapa aspek untuk memvalidasi kemampuan model dari berbagai sudut pandang, yang mencakup analisis performa klasifikasi, kemampuan generalisasi, pengujian signifikansi statistik, dan optimisasi threshold deteksi.

#### 3.1.4.1 Analisis Performa Model

Analisis performa model dilakukan dengan mengevaluasi kemampuan Hybrid CNN-LSTM Autoencoder dalam mendeteksi anomali. Evaluasi baseline pada CIC-IDS2017 menggunakan metrik akurasi, presisi, recall, F1-score, dan FPR yang dihitung dari confusion matrix, dengan F1-score dan FPR sebagai metrik utama perbandingan zero-shot vs few-shot. Nilai presisi tinggi menunjukkan alarm model dapat dipercaya dengan false alarm minimal, sedangkan recall tinggi berarti model mampu mendeteksi mayoritas serangan (Park et al., 2025). Analisis distribusi error rekonstruksi dilakukan untuk melihat pemisahan antara data normal dan anomali, dimana idealnya kedua distribusi harus terpisah jauh. Analisis per tipe serangan juga dilakukan untuk mengidentifikasi kelebihan dan kekurangan model dalam mendeteksi berbagai jenis serangan.

#### 3.1.4.2 Analisis Generalisasi

Analisis generalisasi merupakan aspek krusial untuk mengukur kemampuan model pada dataset yang berbeda dari dataset training. Generalization gap dihitung dengan mencari selisih antara performa di dataset CIC-IDS2017 (in-distribution) dan performa di dataset CSE-CIC-IDS2018 (out-of-distribution), yang diformulasikan pada Persamaan (9).

$$
\text{Gap}_{metric} = M_{CIC} - M_{CSE} \quad (9)
$$

Dimana $M_{CIC}$ adalah nilai metrik di CIC-IDS2017 dan $M_{CSE}$ adalah nilai metrik di CSE-CIC-IDS2018. Gap yang kecil menunjukkan model memiliki kemampuan generalisasi yang baik dan tidak overfitting, sedangkan gap besar mengindikasikan model hanya efektif pada data training (Singh & Jang, 2022). Analisis kualitatif dilakukan untuk mengidentifikasi faktor penyebab penurunan performa, seperti perbedaan distribusi fitur atau keberadaan serangan baru seperti Heartbleed di dataset CSE-CIC-IDS2018.

#### 3.1.4.3 Pengujian Statistik

Pengujian statistik dilakukan untuk memastikan perbedaan performa yang diamati signifikan secara statistik dan bukan karena faktor kebetulan. Metode McNemar's test digunakan untuk membandingkan performa model pada kedua dataset dengan membuat tabel kontingensi dari prediksi benar dan salah (Halbouni et al., 2022). Hipotesis null menyatakan tidak ada perbedaan signifikan antara performa model di kedua dataset, sedangkan hipotesis alternatif menyatakan ada perbedaan signifikan. Level signifikansi ditetapkan pada alpha 0.05, dimana p-value kurang dari 0.05 mengindikasikan penolakan hipotesis null dan perbedaan yang signifikan secara statistik (Park et al., 2025). Ilustrasi bagaimana threshold memisahkan distribusi error trafik normal dan serangan ditunjukkan pada Gambar 3.5.

[Gambar 3.5 Ilustrasi Penentuan Threshold Deteksi Berdasarkan Distribusi Error Rekonstruksi]

#### 3.1.4.4 Analisis Threshold

Analisis threshold bertujuan menentukan nilai batas error optimal untuk memisahkan data normal dan anomali. Threshold dipilih berdasarkan distribusi error di data validasi normal menggunakan nilai percentile tertentu seperti 95th atau 99th percentile. Pemilihan ini melibatkan trade-off dimana threshold rendah (95th) menghasilkan TPR tinggi namun FPR juga meningkat, sedangkan threshold tinggi (99th) menurunkan FPR namun juga menurunkan TPR (Singh & Jang, 2022). Optimisasi dilakukan dengan mencoba berbagai nilai threshold dan memilih yang menghasilkan F1-score maksimal seperti pada Persamaan (12).

$$
\theta_{optimal} = \arg\max_{\theta} F1(\theta) \quad (12)
$$

Alternatif lain menggunakan Youden's Index untuk memaksimalkan selisih TPR dan FPR, atau menyesuaikan dengan kebijakan organisasi seperti membatasi FPR dibawah 1% untuk meminimalkan false alarm.

## 3.2 Instrumen Penelitian

Instrumen penelitian merupakan alat ukur yang krusial yang digunakan untuk mengumpulkan data serta mengevaluasi hasil eksperimen dalam penelitian ini, dimana pemilihan instrumen yang tepat sangat penting untuk memastikan validitas dan reliabilitas hasil yang didapatkan. Dalam konteks penelitian deteksi intrusi berbasis deep learning yang dilakukan, instrumen penelitian mencakup berbagai metrik evaluasi kuantitatif yang mana metrik-metrik ini telah divalidasi secara luas dalam domain machine learning. Instrumen evaluasi ini dirancang untuk mengukur tiga aspek utama yaitu performa klasifikasi model, kemampuan deteksi anomali, serta kemampuan generalisasi terhadap data baru yang belum pernah dilihat sebelumnya.

Untuk mengukur kemampuan model dalam melakukan klasifikasi antara trafik normal dan anomali, digunakan beberapa metrik standar. Metrik pertama adalah akurasi yang mengukur proporsi prediksi benar dari total prediksi, yang mana formulanya dapat dilihat pada Persamaan (1) dimana TP adalah True Positive, TN adalah True Negative, FP adalah False Positive, dan FN adalah False Negative.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \quad (1)
$$

Selain itu, digunakan juga metrik presisi yang mengukur proporsi prediksi positif yang benar-benar positif. Hal ini penting untuk mengukur tingkat kepercayaan terhadap alarm yang dihasilkan sistem agar alarm yang palsu tidak terlalu banyak (Park et al., 2025). Formula presisi ditunjukkan pada Persamaan (2).

$$
\text{Precision} = \frac{TP}{TP + FP} \quad (2)
$$

Metrik recall atau sensitivity juga digunakan untuk mengukur proporsi sampel positif yang berhasil terdeteksi, dimana ini sangat krusial dalam konteks IDS karena kegagalan mendeteksi serangan dapat berakibat fatal bagi keamanan jaringan. Formula recall ditunjukkan pada Persamaan (3).

$$
\text{Recall} = \frac{TP}{TP + FN} \quad (3)
$$

F1-score digunakan sebagai harmonic mean antara presisi dan recall, yang mana metrik ini memberikan keseimbangan antara kedua aspek tersebut terutama pada kondisi data yang imbalanced atau tidak seimbang. Formula F1-score ditunjukkan pada Persamaan (4).

$$
F1\text{-}score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (4)
$$

Selanjutnya, instrumen evaluasi anomaly detection berfokus pada kemampuan model untuk membedakan pola normal dan anomali berdasarkan reconstruction error. Metrik utama yang digunakan pada penelitian ini adalah F1-score dan False Positive Rate (FPR), sedangkan AUC-ROC digunakan sebagai metrik pendukung untuk melihat kemampuan pemisahan kelas pada berbagai threshold. ROC curve dibuat dengan memplot True Positive Rate (TPR) terhadap False Positive Rate (FPR) pada berbagai threshold reconstruction error, yang mana ini memberikan visualisasi mengenai trade-off antara detection rate dan false alarm rate (Singh & Jang, 2022). Selain itu, False Positive Rate (FPR) juga dihitung untuk mengukur proporsi trafik normal yang salah diklasifikasikan sebagai anomali menggunakan Persamaan (5) berikut.

$$
FPR = \frac{FP}{FP + TN} \quad (5)
$$

Terakhir, instrumen evaluasi generalization digunakan untuk mengukur kemampuan model dalam mendeteksi serangan pada dataset yang berbeda dari dataset training. Metrik krusial yang digunakan adalah generalization gap yang dihitung sebagai selisih performa antara evaluasi pada test set dataset yang sama (in-distribution) dan evaluasi pada dataset yang berbeda (out-of-distribution) seperti yang ditunjukkan pada Persamaan (7).

$$
\text{Generalization Gap} = P_{in} - P_{out} \quad (7)
$$

Dimana generalization gap yang kecil menunjukkan bahwa model memiliki kemampuan generalisasi yang baik dan tidak overfitting pada karakteristik spesifik dataset training (Park et al., 2025). Statistical significance testing menggunakan metode seperti McNemar's test juga diterapkan untuk memastikan perbedaan performa yang diamati bukan terjadi karena kebetulan.

## 3.3 Alat dan Bahan Penelitian

### 3.3.1 Alat Penelitian

Alat penelitian ini mencakup perangkat keras dan lunak untuk mendukung komputasi intensif _Deep Learning_. Dari sisi _hardware_, dibutuhkan komputer dengan GPU (_Graphics Processing Unit_) berspesifikasi tinggi guna mempercepat proses pelatihan model. Rincian spesifikasi perangkat keras disajikan pada Tabel 3.1, dimana penggunaan platform _cloud_ seperti Google Colab Pro juga dimungkinkan sebagai alternatif lingkungan komputasi yang fleksibel (Park et al., 2025).

**Tabel 3.1 Spesifikasi Hardware yang Digunakan**
| Komponen | Spesifikasi | Fungsi |
| --------- | -------------------------------------------- | ----------------------------------------------------------- |
| GPU | NVIDIA RTX 4060 8GB VRAM dengan CUDA support | Mempercepat proses training model neural network |
| RAM | 32 GB DDR4 | Menangani dataset berukuran besar dan operasi preprocessing |
| Storage | SSD 256 GB | Menyimpan dataset, model checkpoint, dan hasil eksperimen |
| Processor | AMD Ryzen 5 4600G (6 cores / 12 threads) | Mendukung komputasi paralel untuk preprocessing data |
| OS | Windows 11 | Kompatibilitas dengan tools deep learning |

Pada aspek _software_, Visual Studio Code digunakan sebagai _Integrated Development Environment_ (IDE) utama yang menjalankan _kernel_ Jupyter secara lokal untuk memaksimalkan performa GPU RTX 4060. Penggunaan layanan _cloud_ seperti Google Colab disiapkan sebagai lingkungan komputasi alternatif untuk validasi silang. Bahasa Python v3.8+ dengan _framework_ TensorFlow 2.x dipilih untuk membangun arsitektur _Hybrid_ CNN-LSTM Autoencoder. Tabel 3.2 merincikan seluruh perangkat lunak yang digunakan, dimana kombinasi alat _open-source_ ini menjamin aksesibilitas dan kemudahan reproduksi penelitian (Almuhanna & Alahmadi, 2025).

**Tabel 3.2 Software dan Library yang Digunakan**
| Kategori | Software/Library | Versi | Fungsi |
|----------|------------------|-------|--------|
| IDE & Cloud Compute | VS Code (Local Jupyter) / Google Colab | Latest | Lingkungan pengembangan utama & eksekusi komputasi |
| Bahasa Pemrograman | Python | 3.8+ | Bahasa pemrograman utama untuk implementasi |
| Deep Learning Framework | TensorFlow | 2.x | Engine untuk membangun dan melatih model CNN-LSTM |
| Numerical Computing | NumPy | 1.21+ | Operasi numerik dan manipulasi array |
| Data Manipulation | Pandas | 1.3+ | Manipulasi data tabular dan preprocessing |
| Machine Learning | Scikit-learn | 1.0+ | Preprocessing, feature scaling, dan evaluasi metrik |
| Visualization | Matplotlib, Seaborn | 3.5+, 0.11+ | Visualisasi hasil eksperimen dan analisis data |
| Version Control | Git, GitHub | Latest | Manajemen kode dan kolaborasi |
| GPU Support | NVIDIA CUDA (Local/Cloud) | 12.x | Akselerasi komputasi GPU RTX 4060 |

### 3.3.2 Bahan Penelitian

Bahan utama penelitian ini adalah dataset _benchmark_ CIC-IDS2017 dan CSE-CIC-IDS2018 dari _Canadian Institute for Cybersecurity_. Dataset CIC-IDS2017 mencakup 2,8 juta _network flows_ selama lima hari kerja, dimana hari Senin hanya berisi trafik normal. Karakteristik dataset ini, sebagaimana dirinci pada Tabel 3.3, sangat representatif karena memuat berbagai serangan modern (Halbouni et al., 2022).

**Tabel 3.3 Karakteristik Dataset CIC-IDS2017**
| Karakteristik | Deskripsi |
|---------------|------------|
| Sumber | Canadian Institute for Cybersecurity (CIC) |
| Durasi Pengumpulan | 5 hari kerja (Senin - Jumat) |
| Jumlah Total Records | ~2,8 juta network flows |
| Jumlah Fitur | Lebih dari 80 fitur statistik berbasis flow |
| Format File | CSV |
| Tools Ekstraksi | CICFlowMeter |
| Trafik Normal | Hari Senin (Hanya berisi trafik benign) |
| Jenis Serangan | Brute Force (FTP-Patator, SSH-Patator)<br>DoS/DDoS (DoS Hulk, DDoS, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest)<br>Web Attacks (SQL Injection, XSS, Brute Force)<br>Infiltration<br>Botnet<br>Port Scan<br>Heartbleed |
| Link Download | https://www.unb.ca/cic/datasets/ids-2017.html |

Dataset CSE-CIC-IDS2018 digunakan sebagai set pengujian eksternal, yang mana dataset ini memuat 16 juta _flows_ dengan serangan kompleks seperti _Heartbleed_. Penggunaannya bertujuan mensimulasikan skenario serangan _zero-day_ melalui validasi lintas dataset. Tabel 3.4 memperlihatkan detail karakteristiknya, dimana struktur fiturnya memiliki kesamaan dengan dataset 2017 sehingga memudahkan proses penyelarasan data (Singh & Jang, 2022).

**Tabel 3.4 Karakteristik Dataset CSE-CIC-IDS2018**
| Karakteristik | Deskripsi |
|---------------|------------|
| Sumber | Canadian Institute for Cybersecurity (CIC) & CSE Canada |
| Durasi Pengumpulan | 10 hari |
| Jumlah Total Records | ~16,2 juta network flows |
| Jumlah Fitur | 80 fitur statistik berbasis flow |
| Format File | CSV
| Fungsi dalam Penelitian | Test set untuk cross-dataset validation (zero-day simulation) |
| Jenis Serangan | Brute-Force (FTP, SSH, Web)<br>DoS/DDoS attacks (berbagai variasi protokol)<br>Heartbleed vulnerability exploitation<br>Web Attacks<br>Infiltration scenario (APT simulation)<br>Botnet activities |
| Keunggulan | Mencakup serangan yang tidak ada di CIC-IDS2017 (Heartbleed) |
| Link Download | https://www.unb.ca/cic/datasets/ids-2018.html |

Kedua dataset ini dipilih karena dokumentasinya lengkap dan merupakan standar evaluasi IDS saat ini. Kompabilitas antar dataset mempermudah penyelarasan fitur (feature alignment) yang penting untuk menguji kemampuan model dalam mendeteksi serangan baru.
