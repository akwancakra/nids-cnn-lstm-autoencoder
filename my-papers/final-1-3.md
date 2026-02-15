IMPLEMENTASI HYBRID CNN-LSTM AUTOENCODER
UNTUK DETEKSI ANOMALI ZERO-DAY PADA NETWORK-BASED INTRUSION DETECTION SYSTEM (NIDS) ENTERPRISE

PROPOSAL SKRIPSI

diajukan untuk memenuhi sebagian syarat untuk memperoleh gelar Sarjana
Komputer pada Program Studi Rekayasa Perangkat Lunak

Oleh.
Akwan Cakra Tajimalela
NIM 2209098

PRORAM STUDI REKAYASA PERANGKAT LUNAK
KAMPUS UPI DI CIBIRU
UNIVERSITAS PENDIDIKAN INDONESIA
2026

Daftar Isi

DAFTAR ISI vii
DAFTAR TABEL iv
DAFTAR GAMBAR v
BAB I PENDAHULUAN 1
1.1 Latar Belakang 1
1.2 Rumusan Masalah 3
1.3 Tujuan Penelitian 4
1.4 Manfaat Penelitian 4
1.5 Batasan Penelitian 5
1.6 Sistematika Penulisan 5
BAB II TINJAUAN PUSTAKA 7
2.1 State-of-the-art Penelitian 7
2.2 Intrusion Detection System 11
2.2.1 Taxonomy IDS: NIDS vs HIDS 12
2.2.2 Signature-based IDS 13
2.2.3 Anomaly-based IDS 14
2.2.4 Jenis Serangan Jaringan 16
2.3 Zero-Day Attack 18
2.4 Deep Learning untuk Intrusion Detection 19
2.4.1 Supervised vs Unsupervised Learning dalam NIDS 19
2.4.2 Convolutional Neural Network (CNN) 20
2.4.3 Long Short-Term Memory (LSTM) 20
2.5 Autoencoder untuk Anomaly Detection 21
2.5.1 Arsitektur Autoencoder 22
2.5.2 Reconstruction Error sebagai Indikator Anomali 22
2.5.3 Keunggulan Unsupervised Learning untuk Zero-Day Detection 23
2.6 Hybrid CNN-LSTM Autoencoder untuk NIDS 24
2.7 Cross-Dataset Validation 25
2.7.1 Konsep Generalization dan Overfitting 25
2.7.2 Cross-Dataset Validation Strategy 25
2.8 Dataset 26
2.8.1 CIC-IDS2017 26
2.8.2 CSE-CIC-IDS2018 27
2.8.3 Feature Alignment dan Cross-Dataset Strategy 27
2.9 Metrik Evaluasi 28
2.9.1 Classification Metrics 28
2.9.2 ROC Curve dan AUC 29
2.9.3 False Positive Rate (FPR) 29
2.9.4 Generalization Metrics 29
BAB III METODOLOGI PENELITIAN 30
3.1 Desain Penelitian 30
3.1.1 Klarifikasi Penelitian 31
3.1.2 Studi Deskriptif I 31
3.1.3 Studi Preskriptif 32
3.1.4 Studi Deskriptif II 35
3.2 Instrumen Penelitian 38
3.3 Alat dan Bahan Penelitian 40
3.3.1 Alat Penelitian 40
3.3.2 Bahan Penelitian 42
DAFTAR PUSTAKA 45

Daftar Tabel

Table 2.1. Penelitian Terdahulu 9
Tabel 2.2 Klasifikasi Jenis Serangan Jaringan 16
Tabel 3.1 Spesifikasi Hardware Penelitian 40
Tabel 3.2 Software dan Library yang Digunakan 41
Tabel 3.3 Karakteristik Dataset CIC-IDS2017 42
Tabel 3.4 Karakteristik Dataset CSE-CIC-IDS2018 43

Daftar Gambar

Gambar 2.1 Simplified Workflow Signature-based Intrusion Detection System 13
Gambar 2.2 Arsitektur Anomaly-based IDS dengan Two-Phase Detection 15
Gambar 2.3 Lifecycle Zero-Day Attack dan Window of Vulnerability 18
Gambar 2.4 Arsitektur LSTM Cell dan Temporal Sequence Processing 21
Gambar 2.5 Arsitektur Autoencoder dengan Encoder-Decoder untuk Anomaly Detection 22
Gambar 2.6 Arsitektur Hybrid CNN-LSTM Autoencoder untuk Zero-Day Intrusion Detection 24
Gambar 3.1 Design Research Methodology (DRM) 30
Gambar 3.2 Alur Preprocessing dan Pembentukan Data Time-Series 33
Gambar 3.3 Arsitektur Hybrid CNN-LSTM Autoencoder yang Diusulkan 34
Gambar 3.4 Skema Eksperimen Validasi Lintas Dataset (Cross-Dataset Validation) 35
Gambar 3.5 Ilustrasi Penentuan Threshold Deteksi Berdasarkan Distribusi Error. 37

Pendahuluan
Latar Belakang
Vulnerabilitas CVE-2025-55182 atau React2Shell yang baru saja diumumkan pada Desember 2025 menjadi bukti bahwa sistem pertahanan tradisional masih kewalahan menghadapi ancaman zero-day. Dengan skor CVSS sempurna yakni 10.0, celah ini berhasil mengeksploitasi puluhan organisasi dalam waktu kurang dari 24 jam (Zugec, 2025). Data menunjukkan adanya 77.664 alamat IP yang rentan, dimana hal ini memperlihatkan kegagalan fatal dari sistem signature-based dalam merespons ancaman cepat. Eksploitasi ini menyerang komponen React Server Components yang memungkinkan penyerang melakukan Remote Code Execution tanpa autentikasi. Yang lebih mengkhawatirkan lagi adalah ketersediaan 72 alat Proof-of-Concept gratis di hari pertama, yang mana kondisi ini membuat serangan menjadi sangat mudah dilakukan (Cipollone, 2025). Fakta bahwa tingkat keberhasilan eksploitasi mencapai hampir 100% pada konfigurasi default menunjukkan betapa mendesaknya kebutuhan akan metode deteksi yang tidak bergantung pada signature database.
Di sisi lain, eskalasi ancaman siber global terus meningkat dengan proyeksi kerugian mencapai $10,5 triliun pada tahun 2025 (Khalil, 2025). Tahun 2024 saja mencatat 75 kerentanan zero-day baru, dimana 44% diantaranya secara spesifik menargetkan sistem enterprise (Jay, 2025). Yang menarik adalah tren serangan yang kini menyasar perangkat keamanan seperti firewall dan VPN, yang mana perangkat ini seringkali memiliki akses tinggi namun minim pengawasan. Kompleksitas infrastruktur modern yang menggabungkan cloud dan IoT semakin memperluas attack surface yang ada. Masalah semakin pahit ketika melihat kesenjangan antara dwell time rata-rata 10 hari dengan proses eksfiltrasi data yang bisa terjadi dalam 24 jam (Help Net Security, 2024). Kondisi ini mengindikasikan bahwa organisasi tidak lagi bisa mengandalkan mekanisme pertahanan reaktif yang lambat dalam mendeteksi intrusi.
Intrusion Detection System (IDS) pada dasarnya adalah perangkat lunak yang dirancang khusus untuk memonitor lalu lintas jaringan guna mendeteksi aktivitas mencurigakan yang dapat membahayakan sistem (Scarfone & Mell, 2007). Sistem ini bekerja dengan menganalisis trafik secara real-time dan memberikan peringatan dini kepada administrator agar dapat segera ditindaklanjuti (Rahmawati dkk., 2024). Sebagai sistem pertahanan, IDS berperan penting dalam menjaga aspek Confidentiality, Integrity, dan Availability dari serangan siber. Namun demikian, efektivitas IDS tradisional kini mulai dipertanyakan mengingat kemampuannya yang terbatas dalam menghadapi serangan terotomatisasi seperti React2Shell. Serangan modern yang mampu bermutasi dengan cepat menuntut adanya integrasi sistem deteksi yang lebih adaptif, yang mana pendekatan lama dirasa sudah tidak lagi mumpuni untuk menangani ancaman zero-day yang semakin canggih.
Signature-based IDS bekerja dengan cara mencocokkan lalu lintas jaringan dengan signature database serangan yang sudah diketahui (Ahmed dkk., 2025). Meskipun sangat akurat untuk ancaman lama, metode ini terbukti gagal total dengan tingkat deteksi 0% terhadap serangan zero-day karena belum adanya signature yang tersedia saat serangan terjadi (Sridharan dkk., 2025). Ketergantungan pada pembaruan signature menciptakan jeda waktu yang berbahaya, dimana penyerang bisa bebas beraksi selama periode “blind spot” tersebut. Kasus React2Shell menjadi contoh, yang mana puluhan ribu sistem terinfeksi hanya dalam 24 jam sementara vendor keamanan baru merilis signature. Selain beban operasional yang tinggi, kelemahan fundamental ini menegaskan bahwa pendekatan reaktif tidak lagi cukup untuk melindungi aset kritikal enterprise.
Sebagai alternatif, Anomaly-based Intrusion Detection System (AIDS) menawarkan pendekatan proaktif dengan mempelajari pola normal jaringan untuk mendeteksi penyimpangan (Guo, 2023). Sistem hybrid berbasis anomali dilaporkan mampu mencapai deteksi diatas 95% terhadap ancaman baru tanpa perlu menunggu update signature (Sridharan dkk., 2025). Meskipun menjanjikan, deteksi anomali konvensional seringkali kesulitan menangani data jaringan berdimensi tinggi dan masalah ketidakseimbangan kelas. Oleh karena itu, penerapan teknologi deep learning menjadi solusi yang sangat potensial untuk meningkatkan akurasi deteksi (Alsoufi dkk., 2024). Hal ini menunjukkan bahwa kombinasi antara pendekatan anomali dengan deep learning adalah kunci untuk mengatasi keterbatasan IDS tradisional dalam menghadapi serangan yang berkembang cepat.
Integrasi model deep learning terbukti sangat efektif dalam mengenali pola rumit pada lalu lintas jaringan. Arsitektur CNN-LSTM misalnya, memanfaatkan CNN untuk mengekstrak fitur spasial dan LSTM untuk menangkap pola urutan waktu, yang mana kombinasi ini mampu mencapai akurasi 99,64% di dataset CIC-IDS2017 (Halbouni dkk., 2022). Sementara itu, Autoencoder unggul dalam mendeteksi anomali tanpa label serangan dengan cara mengukur reconstruction error dari data input (Alsoufi dkk., 2024). Penggabungan supervised dan unsupervised learning ini menghasilkan sistem pertahanan yang kuat. Dari berbagai penelitian yang ada, dapat dilihat bahwa pendekatan hybrid ini memiliki potensi besar untuk mendeteksi serangan zero-day secara real-time sebelum kerusakan fatal terjadi.
Berdasarkan permasalahan tersebut, penelitian ini mengusulkan Implementasi Hybrid CNN-LSTM Autoencoder Untuk Deteksi Anomali Zero-Day Pada Network-Based Intrusion Detection System (NIDS) Enterprise. Berbeda dengan pendekatan signature-based yang gagal merespons React2Shell, metode ini menggunakan unsupervised learning untuk mengenali deviasi trafik secara mandiri (Almuhanna & Dardouri, 2025). Model ini akan dilatih menggunakan dataset CIC-IDS2017 dan diuji validitasnya pada CSE-CIC-IDS2018 melalui dua skenario cross-dataset validation, yaitu zero-shot (tanpa adaptasi) dan few-shot adaptation (adaptasi unsupervised menggunakan 1% data benign target). Fokus utamanya adalah mengisi celah penelitian sebelumnya yang seringkali hanya menggunakan satu dataset saja. Dengan target akurasi diatas 90% pada data baru, sistem ini diharapkan mampu memberikan perlindungan proaktif dan mengurangi dwell time serangan secara signifikan.

Rumusan Masalah
Berdasarkan latar belakang yang telah dipaparkan, rumusan masalah dalam penelitian ini adalah sebagai berikut:
Bagaimana performa model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan siber pada dataset CIC-IDS2017 dibandingkan dengan baseline methods?
Bagaimana kemampuan generalisasi model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan zero-day melalui cross-dataset validation menggunakan dataset CSE-CIC-IDS2018 dengan perbandingan skenario zero-shot dan few-shot adaptation?

Tujuan Penelitian
Berdasarkan rumusan masalah di atas, berikut adalah tujuan dari penelitian ini:
Mengevaluasi performa model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan siber pada dataset CIC-IDS2017 dan membandingkannya dengan baseline methods.
Menganalisis kemampuan generalisasi model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan zero-day melalui cross-dataset validation menggunakan dataset CSE-CIC-IDS2018 dengan membandingkan skenario zero-shot dan few-shot adaptation.

Manfaat Penelitian
Adapun manfaat dari penelitian yang akan dilakukan, sebagai berikut:
Manfaat Teoritis:
Memberikan kontribusi metode baru dalam pengembangan Anomaly-based Intrusion Detection System menggunakan pendekatan Hybrid CNN-LSTM Autoencoder untuk deteksi zero-day attacks.
Menambah literatur dan pemahaman mengenai penerapan deep learning pada deteksi anomali lalu lintas jaringan tanpa bergantung pada signature database.
Menyediakan bukti empiris mengenai kemampuan generalisasi model deep learning melalui cross-dataset validation dalam domain cybersecurity.
Manfaat Praktis:
Menghasilkan model Intrusion Detection System yang mampu mendeteksi zero-day secara real-time tanpa bergantung pada signature database.
Membantu organisasi enterprise dalam memperkuat pertahanan jaringan dengan mengurangi dwell time deteksi serangan.
Mengurangi beban operasional dan biaya pemeliharaan sistem keamanan dengan meminimalkan ketergantungan pada pembaruan signature secara manual.

Batasan Penelitian
Dalam pelaksanaan penelitian, batasan penelitian yang ditetapkan sebagai berikut:
Dataset yang digunakan terbatas pada CIC-IDS2017 dan CSE-CIC-IDS2018 yang merupakan dataset publik hasil simulasi jaringan enterprise, bukan real-time traffic dari lingkungan produksi.
Arsitektur model yang dikembangkan terbatas pada Hybrid CNN-LSTM Autoencoder tanpa melakukan eksplorasi terhadap arsitektur lainnya seperti Transformer, GRU, atau Attention Mechanism.
Perbandingan performa model dibatasi pada dua baseline methods, yaitu LSTM Autoencoder dan Traditional Machine Learning (Isolation Forest/Random Forest), tanpa membandingkan dengan metode Signature-based IDS atau anomaly detection lainnya.
Proses training model menggunakan pendekatan unsupervised learning yang hanya memanfaatkan data normal traffic (benign), tanpa menggunakan labelled attack data dalam fase training.
Evaluasi performa model dibatasi pada binary classification (Normal vs Attack), dengan metrik utama F1-score dan False Positive Rate (FPR), serta metrik pendukung accuracy, precision, recall, AUC-ROC, dan generalization metrics, tanpa melakukan klasifikasi multi-class untuk jenis serangan spesifik.
Adaptasi model (fine-tuning) dilakukan secara unsupervised menggunakan 1% data benign target tanpa melakukan advanced feature engineering yang kompleks.
Implementasi sistem dilakukan dalam environment offline untuk eksperimen dan evaluasi, bukan deployment real-time pada infrastruktur jaringan enterprise yang aktif.

Sistematika Penulisan
Berikut merupakan sistematika penulisan yang ada pada penelitian ini, diantaranya sebagai berikut:
BAB I PENDAHULUAN
Bab ini menguraikan latar belakang masalah, rumusan masalah penelitian, tujuan yang ingin dicapai, manfaat penelitian, batasan penelitian, dan sistematika penulisan.
BAB II KAJIAN PUSTAKA
Bab ini membahas teori-teori dan konsep yang relevan dengan topik penelitian, serta kajian terhadap penelitian-penelitian sebelumnya yang berkaitan untuk menggambarkan posisi dan kontribusi penelitian ini dalam state-of-the-art.
BAB III METODE PENELITIAN
Bab ini menjelaskan pendekatan dan metode penelitian yang digunakan, serta memaparkan tahapan-tahapan pelaksanaan penelitian secara sistematis dari tahap awal sampai tahap akhir.
BAB IV HASIL DAN PEMBAHASAN
Bab ini menyajikan hasil eksperimen dan temuan penelitian, serta pembahasan mendalam yang menjawab rumusan masalah yang telah ditetapkan.
BAB V PENUTUP
Bab ini memaparkan kesimpulan berdasarkan hasil penelitian, implikasi dari temuan penelitian, serta saran untuk pengembangan penelitian selanjutnya.

TINJAUAN PUSTAKA
State-of-the-art Penelitian
Penelitian ini merujuk pada beberapa penelitian terdahulu yang menjadi acuan penting dalam penelitian ini. Penjelasan di bawah ini menunjukkan research gap yang ada serta posisi state-of-the-art dari topik yang diangkat, sehingga dapat memberikan gambaran yang lebih jelas mengenai kontribusi penelitian ini terhadap pengembangan bidang deteksi intrusi zero-day.
Pada penelitian yang dilakukan oleh Park dkk (2025) bertujuan untuk meneliti deteksi anomali pada paket jaringan dengan fokus utama pada serangan yang belum pernah terlihat sebelumnya (unseen attacks). Penelitian ini menggunakan model Hybrid CNN-BiLSTM Autoencoder berbasis unsupervised learning, dimana model hanya dilatih menggunakan data normal saja pada dataset CIC-IDS2018 tanpa menggunakan data serangan dalam proses training. Proses preprocessing yang dilakukan cukup komprehensif, meliputi penghapusan nilai NaN, normalisasi IP address, hingga Min-Max scaling untuk standarisasi data. Hasil penelitian menunjukkan performa yang sangat baik dengan akurasi mencapai 98,1% dan F1-score sebesar 98,3% pada deteksi unseen attacks. Namun demikian, penelitian ini masih memiliki keterbatasan karena belum membandingkan model dengan metode meta-learning atau menerapkan attention mechanism yang dapat meningkatkan akurasi deteksi secara lebih baik lagi.
Sementara itu, Almuhanna & Dardouri (2025)mengambil pendekatan yang agak berbeda dengan mengembangkan sistem IDS berbasis Hybrid Ensemble yang sangat kompleks. Sistem ini menggabungkan kekuatan lima algoritma sekaligus yaitu XGBoost, Random Forest, GNN, LSTM, dan Autoencoder dalam satu kerangka kerja terintegrasi yang dilatih pada dataset CIC-IDS2017. Untuk menangani masalah data yang tidak seimbang, mereka menerapkan teknik SMOTE balancing yang dikombinasikan dengan feature engineering yang mendalam. Meskipun hasil evaluasi 5-fold cross-validation menunjukkan peforma sempurna dengan akurasi dan F1-score mencapai 100%, kompleksitas arsitektur ini membuat beban komputasi yang sangat tinggi. Kondisi ini mengindikasikan bahwa model tersebut mungkin kurang praktis untuk diimplementasikan pada perangkat jaringan dengan sumber daya terbatas yang membutuhkan respons real-time.
Dalam konteks keamanan jaringan IoT dan 5G yang semakin kompleks Baidar dkk (2025) mengusulkan arsitektur Hybrid Deep Learning yang digabungkan dengan konsep Federated Learning. Mereka memanfaatkan model CNN-BiLSTM Autoencoder dengan strategi supervised learning yang diperkuat oleh mekanisme rekonstruksi mandiri (self-supervised) pada dataset UNSW-NB15. Evaluasi model menunjukkan hasil yang mantap dengan AUC-ROC mencapai 99,59%, yang mana angka ini didukung oleh preprocessing data yang ekstensif mulai dari encoding fitur hingga penanganan outlier. Namun demikian, ketergantungan utama model ini pada data berlabel menjadi kelemahan tersendiri saat menghadapi ancaman zero-day yang dinamis. Hal ini menunjukkan bahwa sekuat apapun model supervised, ia akan selalu kesulitan mengenali jenis serangan baru yang pola atau signature-nya belum tersimpan di dalam database pelatihan.
Penelitian selanjutnya oleh Singh & Jang (2022) menawarkan perspektif lain dengan mengembangkan model MSCNN-LSTM-AE yang dirancang khusus untuk menangkap korelasi spasial dan temporal dari trafik jaringan secara simultan. Menggunakan pendekatan unsupervised learning, model ini mengintegrasikan Multi-Scale CNN untuk ekstraksi fitur spasial dan Isolation Forest sebagai mekanisme koreksi error otomatis pada tiga dataset berbeda termasuk CICDDoS2019. Hasil eksperimen menunjukkan performa yang sangat baik dengan akurasi 99,56% pada deteksi serangan DdoS. Yang perlu dicermati adalah penggunaan threshold rekonstruksi yang bersifat statis dalam penelitian ini, dimana hal tersebut bisa menjadi kaku saat menghadapi fluktuasi trafik jaringan yang dinamis. Kondisi ini menegaskan pentingnya mekanisme threshold yang adaptif agar sistem tidak mudah memicu false alarm akibat perubahan pola normal jaringan yang wajar.
Penelitian lain oleh Halbouni dkk (2022) berfokus pada pengembangan IDS menggunakan arsitektur Hybrid CNN-LSTM standar dengan paradigma supervised learning untuk membedakan trafik benign dan malicious. Model ini menerapkan teknik regularisasi seperti batch normalization dan dropout untuk mencegah overfitting saat dilatih pada dataset CIC-IDS2017 dan UNSW-NB15. Meskipun berhasil mencatat akurasi 99,59% pada CIC-IDS2017, analisis lebih dalam menunjukkan bahwa tingkat deteksi (detection rate) masih rendah untuk kategori serangan minoritas seperti Web Attacks dan Backdoor. Hal ini memperlihatkan bahwa model berbasis deep learning sekalipun masih rentan terhadap bias kelas mayoritas jika tidak ditangani dengan strategi sampling yang tepat. Oleh karena itu, performa tinggi pada akurasi global seringkali menutupi kelemahan model dalam mendeteksi jenis serangan yang langka namun mematikan. Seluruh data terkait state-of-the-art penelitian ini dirangkum dalam Tabel 2.1.

Table 2.1. Penelitian Terdahulu
Penulis Model/Arsitektur Dataset Learning Paradigm Metode Evaluasi & Hasil
Park dkk (2025)
Hybrid CNN-BiLSTM Autoencoder CIDS2018 Unsupervised (trained on normal only) Accuracy: 98.1%
F1-Score: 98.3%
Almuhanna & Dardouri (2025)
Hybrid Ensemble (XGBoost + RF + GNN + LSTM + Autoencoder) CICIDS2017 Supervised + Unsupervised (Hybrid) Accuracy: ~100%
Precision: 100%, Recall: 100%, F1-Score: 100%
5-fold cross-validation
Baidar dkk (2025)
Hybrid Deep Learning-Federated Learning (CNN-BiLSTM-AE) UNSW-NB15 Supervised + Federated Learning Accuracy: 97.12%, Precision: 98.45%, Recall: 96.29%, F1-Score: 97.36%, AUC-ROC: 99.59%
(Singh & Jang, 2022)
MSCNN-LSTM-AE (Multi-Scale CNN-LSTM Autoencoder) NSL-KDD, UNSW-NB15, CICDDoS2019 Unsupervised CICDDoS2019: Accuracy: 99.56%, Precision: 98.91%, Recall: 98.81%, F1-Score: 98.46%
Halbouni dkk (2022)
Hybrid CNN-LSTM CIC-IDS2017, UNSW-NB15, WSN-DS Supervised CIC-IDS2017: 99.59%
UNSW-NB15: 93.68%
WSN-DS: 99.64%
Proposed Hybrid CNN-LSTM Autoencoder CIC-IDS2017 (training)
CSE-CIC-IDS2018 (testing) Unsupervised (normal traffic only) Accuracy, Precision, Recall, F1-Score, AUC-ROC, False Positive Rate (FPR)

Berdasarkan tinjauan mendalam terhadap penelitian-penelitian di atas sebagaimana dirangkum dalam Tabel 2.1, terlihat adanya beberapa kesenjangan atau research gap yang cukup krusial. Pertama, mayoritas penelitian seperti Park dkk (2025) dan Singh & Jang (2022) hanya menguji model mereka pada satu dataset (single dataset scenario), yang mana hal ini belum cukup untuk membuktikan kemampuan generalisasi model terhadap lingkungan jaringan yang berbeda-beda. Kedua, dominasi pendekatan supervised learning pada studi Almuhanna & Dardouri (2025) membuat sistem menjadi reaktif dan sangat bergantung pada ketersediaan data label serangan terbaru. Selain itu, aspek efisiensi komputasi seringkali dikesampingkan demi mengejar akurasi semata, padahal ini adalah faktor kunci untuk implementasi pada real-world. Kondisi ini menunjukkan bahwa masih dibutuhkan studi yang tidak hanya fokus pada akurasi deteksi anomali, tetapi juga pada kemampuan adaptasi model terhadap data baru yang belum pernah dilihat sebelumnya.
Berdasarkan gap yang telah diidentifikasi tersebut, penelitian ini mengusulkan model Hybrid CNN-LSTM Autoencoder berbasis unsupervised learning untuk mengatasi keterbatasan yang ada. Model ini dilatih hanya menggunakan trafik normal dari dataset CIC-IDS2017 dan diuji pada dataset CSE-CIC-IDS2018 untuk mensimulasikan deteksi zero-day melalui cross-dataset validation, yang mana pendekatan ini dapat memvalidasi kemampuan generalisasi model terhadap ancaman baru. Dalam arsitektur yang diusulkan, CNN digunakan untuk ekstraksi fitur spasial dari data jaringan, LSTM untuk pemodelan fitur temporal yang menangkap pola sekuensial, dan arsitektur Autoencoder untuk deteksi anomali berbasis reconstruction error yang tidak memerlukan label serangan. Evaluasi dilakukan secara mendalam menggunakan berbagai metrik seperti Akurasi, Presisi, Recall, F1-Score, AUC-ROC, hingga FPR untuk memastikan model mampu mendeteksi serangan zero-day dengan tingkat false alarm yang rendah dan dapat diimplementasikan pada jaringan enterprise secara efektif.

Intrusion Detection System
Intrusion Detection System (IDS) merupakan sebuah sistem yang dirancang secara khusus untuk memonitor aktivitas jaringan atau sistem komputer, dimana tujuannya adalah untuk mendeteksi tanda-tanda intrusi, pelanggaran kebijakan keamanan, atau aktivitas mencurigakan lainnya yang berpotensi membahayakan integritas sistem (Scarfone & Mell, 2007). Sistem ini memiliki peran yang sangat krusial dalam melindungi tiga pilar keamanan informasi (CIA Triad) dari berbagai bentuk ancaman siber yang terus berkembang, yang mana ia berfungsi sebagai komponen pertahanan berlapis dalam arsitektur keamanan jaringan modern (Aljanabi dkk., 2021; Mohapatra dkk., 2023). Berdasarkan metode deteksinya, IDS dapat diklasifikasikan menjadi dua jenis utama yaitu signature-based dan anomaly-based, yang mana keduanya memiliki karakteristik dan keunggulan masing-masing yang berbeda dalam mendeteksi dan menangani ancaman siber (Ahmed dkk., 2025). Selain itu, pengkategorian IDS juga dapat dilihat dari beberapa aspek lain yang memiliki karakteristik serta aplikasi berbeda dalam lingkungan jaringan. Dari segi implementasi, IDS dapat dibedakan berdasarkan penempatannya yang disesuaikan dengan topologi jaringan organisasi, seperti Network-based IDS (NIDS) yang memantau lalu lintas jaringan dan Host-based IDS (HIDS) yang memantau aktivitas pada host individu.

Taxonomy IDS: NIDS vs HIDS
Intrusion Detection System (IDS) secara umum dapat diklasifikasikan menjadi dua kategori utama, yaitu Network-based Intrusion Detection System (NIDS) dan Host-based Intrusion Detection System (HIDS), dimana pembagian ini didasarkan pada lokasi monitoring serta sumber data yang dianalisis (Almuhanna & Dardouri, 2025; Scarfone & Mell, 2007). NIDS bekerja dengan memonitor lalu lintas jaringan secara real-time dengan cara menganalisis paket data yang melewati titik strategis dalam infrastruktur jaringan, sedangkan HIDS beroperasi pada level host atau endpoint dengan memantau sistem file, log aktivitas, dan proses yang berjalan di dalam host tersebut (Alsoufi dkk., 2024). Pendekatan NIDS memungkinkan identifikasi serangan yang memengaruhi banyak host sekaligus, seperti serangan DdoS atau port scanning, sementara HIDS lebih efektif untuk mendeteksi serangan yang bersifat internal atau berbasis privilege escalation (Al-Jamali dkk., 2025; Baidar dkk., 2025).
NIDS memiliki karakteristik utama berupa analisis lalu lintas jaringan menggunakan metode packet inspection dan flow-based monitoring. Prosesnya mencakup packet capture melalui mekanisme port mirroring, network TAP, atau inline deployment, kemudian data dikumpulkan untuk dianalisis menggunakan pendekatan flow-based (misalnya NetFlow, IPFIX) atau packet-based (deep packet inspection) (Cantone dkk., 2024; Park dkk., 2025). NIDS sangat efektif dalam mendeteksi serangan yang bersifat eksternal atau berbasis jaringan, namun memiliki keterbatasan dalam mengamati aktivitas internal yang terenkripsi atau bersifat host-specific (Alsoufi dkk., 2024). Selain itu, NIDS modern telah mengintegrasikan teknologi machine learning dan deep learning untuk mendeteksi anomali dalam lalu lintas jaringan yang kompleks secara otomatis, sehingga meningkatkan kemampuan deteksi terhadap zero-day attacks (Almuhanna & Dardouri, 2025; Singh & Jang, 2022).
Sebaliknya, HIDS melakukan deteksi berdasarkan event logs, file integrity, dan sistem call monitoring untuk mengidentifikasi perilaku abnormal pada host tertentu (Sridharan dkk., 2025). HIDS menawarkan visibilitas yang lebih dalam terhadap aktivitas sistem internal, seperti modifikasi file atau privilege misuse, namun memiliki keterbatasan dalam mendeteksi serangan yang melibatkan beberapa node jaringan (Guo, 2023). Perbandingan antara keduanya menunjukkan bahwa NIDS lebih unggul dalam cakupan dan skalabilitas, sedangkan HIDS lebih unggul dalam kedalaman analisis per-host. Fokus penelitian modern saat ini lebih mengarah pada pengembangan NIDS berbasis deep learning, hal ini dikarenakan meningkatnya kebutuhan deteksi real-time terhadap trafik jaringan yang beragam (Almuhanna & Dardouri, 2025; Baidar dkk., 2025; Park dkk., 2025).

Signature-based IDS
Signature-based Intrusion Detection System merupakan metode deteksi intrusi yang bekerja dengan cara mencocokkan pola aktivitas jaringan atau sistem dengan database signature yang telah dikenali sebelumnya (Scarfone & Mell, 2007). Signature dalam konteks ini merujuk pada karakteristik spesifik dari serangan siber, seperti byte sequence, header pattern, atau behavioral fingerprint yang unik untuk setiap jenis ancaman (Ahmed dkk., 2025). Mekanisme kerjanya dimulai dengan proses inspeksi terhadap paket data yang melewati jaringan menggunakan teknik pattern matching seperti string matching atau regular expression, kemudian membandingkannya dengan database signature yang terus diperbarui oleh vendor keamanan atau security researchers (Kristi dkk., 2025). Gambar 2.1 mengilustrasikan bagaimana alur kerja lengkap dari signature-based IDS mulai dari packet capture hingga decision making process.

Gambar 2.1 Simplified Workflow Signature-based Intrusion Detection System

Beberapa alat yang umum digunakan dalam implementasi metode ini adalah Snort dan Suricata, yang mana alat ini sering dijadikan objek studi dan rujukan dalam pengembangan sistem NIDS (Ghazi dkk., 2024). Ketika pola yang cocok ditemukan, sistem akan memicu alert dan melakukan respons yang telah dikonfigurasi, seperti blocking, logging, atau notifikasi kepada administrator (Rahmawati dkk., 2024).
Metode signature-based ini memiliki beberapa keunggulan, terutama dalam hal akurasi deteksi yang tinggi untuk serangan yang sudah dikenali dan false positive rate yang rendah karena deteksi didasarkan pada pola yang telah terverifikasi, dengan precision rate mencapai 95-99% untuk known attacks (Ahmed dkk., 2025). Dari sisi efisiensi komputasi, metode ini dapat bekerja secara real-time dengan latency minimal untuk pola serangan yang sudah diketahui, namun efisiensi ini terbatas pada ancaman yang telah teridentifikasi sebelumnya (Achbarou dkk., 2025). Di sisi lain, signature-based IDS memiliki keterbatasan signifikan dalam mendeteksi zero-day attacks atau serangan yang menggunakan teknik evasion seperti polymorphic malware dan obfuscation, dikarenakan signature untuk ancaman tersebut belum tersedia dalam database (Al-Jamali dkk., 2025; Guo, 2023). Selain itu, sistem ini memerlukan pembaruan database signature secara berkala yang mengakibatkan beban operasional tinggi dan adanya vulnerability window yang berkisar 12-48 jam antara waktu disclosure dan distribusi signature patch (Sridharan dkk., 2025).

Anomaly-based IDS
Anomaly-based Intrusion Detection System adalah pendekatan deteksi intrusi yang berfokus pada identifikasi penyimpangan dari pola perilaku normal yang telah dipelajari oleh sistem (Scarfone & Mell, 2007). Berbeda dengan signature-based yang mengandalkan database pola serangan, anomaly-based IDS membangun profil baseline aktivitas normal melalui fase learning atau training period menggunakan data historis lalu lintas jaringan (Almuhanna & Dardouri, 2025). Pembangunan profil baseline yang akurat membutuhkan dataset yang cukup representatif untuk menangkap variasi dalam pola lalu lintas normal. Mekanisme deteksi didasarkan pada asumsi bahwa aktivitas intrusi akan menghasilkan pola yang berbeda secara signifikan dari perilaku normal, baik dari segi volume traffic, frekuensi akses, maupun karakteristik protokol komunikasi (Guo, 2023). Gambar 2.2 menunjukkan arsitektur dua fase dari anomaly-based IDS, yaitu training phase untuk membangun baseline dan detection phase untuk identifikasi anomali.

Gambar 2.2 Arsitektur Anomaly-based IDS dengan Two-Phase Detection

Untuk melakukan deteksi anomali ini, metode statistik seperti Z-score analysis, threshold-based detection, dan probabilistic models sering digunakan dalam pendekatan tradisional, sementara teknik machine learning dan deep learning menjadi tren modern untuk meningkatkan akurasi deteksi (Alsoufi dkk., 2024).
Kelebihan utama pendekatan ini adalah kemampuannya mendeteksi zero-day attacks dan novel attack variants tanpa memerlukan signature yang telah didefinisikan sebelumnya, sehingga memberikan perlindungan proaktif terhadap ancaman yang belum dikenali dengan detection rate 70-85% untuk unknown threats (Al-Jamali dkk., 2025). Namun, anomaly-based IDS juga menghadapi beberapa tantangan, terutama dalam hal tingginya false positive rate yang dapat mencapai 10-30%, karena aktivitas legitimate yang tidak biasa seperti software updates, system maintenance, atau perubahan pola kerja dapat salah diidentifikasi sebagai ancaman (Ahmed dkk., 2025). Efektivitas sistem sangat bergantung pada kualitas data training dan representativitas baseline, dimana profil yang tidak akurat dapat menyebabkan detection rate yang rendah atau missed attacks (Almuhanna & Dardouri, 2025). Untuk mengatasi tantangan ini, penggunaan teknik machine learning seperti Random Forest, SVM, dan deep learning architectures telah terbukti meningkatkan akurasi anomaly detection hingga 95% dengan kemampuan feature extraction dan pattern recognition yang lebih canggih (Alsoufi dkk., 2024).

Jenis Serangan Jaringan
Serangan jaringan dapat diklasifikasikan berdasarkan tujuan, metode eksekusi, dan dampaknya terhadap sistem target (Roy dkk., 2025). Pemahaman terhadap karakteristik setiap kategori serangan sangat penting dalam merancang mekanisme deteksi yang efektif, karena setiap jenis serangan memiliki signature dan behavioral pattern yang berbeda (Ahmed dkk., 2025). Dataset benchmark seperti CIC-IDS2017 dan CSE-CIC-IDS2018 yang digunakan dalam penelitian ini mencakup berbagai kategori serangan, dimanan kategori-kategori ini digunakan untuk menguji kemampuan generalisasi model IDS. Tabel 2.2 menunjukkan klasifikasi jenis serangan jaringan beserta karakteristik dan dampaknya.

Tabel 2.2 Klasifikasi Jenis Serangan Jaringan
Kategori Serangan Jenis Serangan Karakteristik Traffic Target/Dampak Rujukan
DoS/DdoS SYN Flood, UDP Flood, HTTP Flood High volume traffic, bandwidth exhaustion Resource depletion (bandwidth, CPU, memory), service unavailability Kristi dkk., 2025

Penetration Attacks Brute Force, Port Scanning, Vulnerability Exploitation Sequential connection attempts, port probing patterns Unauthorized access, privilege escalation Ahmed dkk., 2025

Web-based Attacks SQL Injection, XSS, RCE Malicious payload in HTTP requests, abnormal query patterns Data theft, code execution, system compromise Rahmawati dkk., 2024

Botnet Attacks Command & Control (C2), DdoS-as-a-Service, Cryptojacking Coordinated multi-source traffic, periodic beaconing Distributed attacks, resource hijacking, spam distribution Alsoufi dkk., 2024

Infiltration Attacks Backdoor, Trojan, APT Low-and-slow traffic, encrypted channels, lateral movement Persistent access, data exfiltration, long-term espionage Almuhanna & Dardouri, 2025

Berdasarkan tren keamanan siber saat ini, serangan Distributed Denial of Service (DdoS) tetap menjadi ancaman yang signifikan dengan skala yang semakin masif. Di samping itu, serangan berbasis web terus berevolusi melalui pemanfaatan kerentanan zero-day yang menargetkan berbagai framework modern, sehingga menjadi tantangan besar bagi sistem keamanan tradisional (Yaddala & Sunkara, 2024). Meningkatnya kompleksitas serangan siber ini menuntut penggunaan IDS berbasis anomali, yang mana harus mampu mendeteksi perubahan perilaku kecil dan ancaman baru pada lalu lintas jaringan secara efektif (Yaddala & Sunkara, 2024).

Zero-Day Attack
Zero-day attack adalah serangan siber yang mengeksploitasi celah keamanan perangkat lunak yang belum diketahui vendor atau belum memiliki patch resmi (Guo, 2023). Istilah “zero-day” merujuk pada fakta bahwa pengembang punya nol hari untuk memperbaiki kerentanan sebelum dieksploitasi, menciptakan window of vulnerability yang sangat berisiko (Al-Jamali dkk., 2025). Karakteristik paling berbahaya dari serangan ini adalah sifatnya yang tidak terdeteksi oleh sistem signature-based, karena belum ada pola yang tercatat untuk ancaman ini (Ahmed dkk., 2025). Dampaknya sangat signifikan, dimana statistik menunjukkan sebagian besar data breaches berawal dari eksploitasi celah yang tak dikenal ini (Guo, 2023).
Siklus hidup zero-day attack dimulai dari penemuan kerentanan, pembuatan kode eksploit, hingga serangan ke target, yang mana proses ini bisa berlangsung berbulan-bulan sebelum terdeteksi (Guo, 2023). Gambar 2.3 mengilustrasikan tahapan lifecycle serangan zero-day ini.

Gambar 2.3 Lifecycle Zero-Day Attack dan Window of Vulnerability

Sistem deteksi berbasis signature secara fundamental tidak efektif menghadapi zero-day attack karena tidak ada referensi pencocokan serangan baru (Sridharan dkk., 2025). Meski akurat untuk serangan lama, kelemahan ini membuat sistem buta terhadap ancaman baru (Ahmed dkk., 2025). Oleh karena itu, tantangan ini mendorong pengembangan deteksi berbasis anomali menggunakan deep learning yang mampu mengenali penyimpangan perilaku tanpa bergantung signature (Al-Jamali dkk., 2025). Model hybrid seperti CNN-LSTM Autoencoder terbukti menjanjikan dalam mendeteksi zero-day attack dengan mempelajari pola normal jaringan secara mendalam (Park dkk., 2025).

Deep Learning untuk Intrusion Detection
Deep learning telah menjadi salah satu pendekatan utama dalam pengembangan Intrusion Detection System modern, dimana hal ini dikarenakan kemampuannya melakukan automatic feature extraction dan pattern recognition dari high-dimensional network traffic data tanpa memerlukan manual feature engineering (Almuhanna & Dardouri, 2025; Alsoufi dkk., 2024). Arsitektur deep neural networks yang terdiri dari banyak hidden layers memungkinkan model menangkap hubungan non-linear yang kompleks pada data lalu lintas, sehingga meningkatkan akurasi deteksi terhadap serangan tingkat lanjut (Park dkk., 2025). Implementasi deep learning dalam NIDS dapat dikategorikan berdasarkan paradigma pembelajarannya yang digunakan, yaitu supervised learning yang memerlukan data berlabel untuk tugas klasifikasi, dan unsupervised learning yang mendeteksi anomali berdasarkan reconstruction error (Alsoufi dkk., 2024). Pendekatan hybrid yang menggabungkan beberapa arsitektur seperti CNN-LSTM atau CNN-Autoencoder telah menunjukkan performa unggul dengan akurasi mencapai 98-100% pada berbagai data uji (Almuhanna & Dardouri, 2025; Park dkk., 2025). Namun, penerapan deep learning juga menghadapi tantangan seperti komputasi yang tinggi, kebutuhan data training dalam jumlah besar, dan interpretability issues (Baidar dkk., 2025).

Supervised vs Unsupervised Learning dalam NIDS
Supervised learning melatih model menggunakan dataset berlabel dimana setiap data memiliki kelas seperti “normal” atau jenis serangan spesifik, sehingga menghasilkan presisi tinggi dengan akurasi 97-99% pada uji in-distribution (Alsoufi dkk., 2024; Baidar dkk., 2025). Namun, pendekatan ini bergantung pada ketersediaan data berlabel yang sulit dan mahal diperoleh dalam domain cybersecurity, serta cenderung mengalami poor generalization terhadap zero-day attacks yang tidak terwakili dalam data latih (Guo, 2023; Park dkk., 2025). Di sisi lain, unsupervised learning tidak memerlukan data berlabel dan bekerja dengan mengidentifikasi anomalies berdasarkan deviasi dari perilaku normal yang dipelajari, sehingga mampu mengidentifikasi serangan baru dan ancaman zero-day tanpa memerlukan sampel serangan (Al-Jamali dkk., 2025; Park dkk., 2025). Namun, pendekatan unsupervised umumnya menghasilkan false positive rate yang lebih tinggi, karena pola lalu lintas normal yang jarang terjadi sering dianggap anomali (Almuhanna & Dardouri, 2025). Pendekatan hybrid yang menggabungkan kedua paradigma yang telah diusulkan untuk mengoptimalkan kekurangan antara akurasi deteksi dan kapabilitas generalisasi (Sridharan dkk., 2025).

Convolutional Neural Network (CNN)
Convolutional Neural Network (CNN) adalah arsitektur deep learning yang kini banyak diadopsi dalam NIDS untuk mengekstraksi fitur spasial dari data lalu lintas jaringan (Alsoufi dkk., 2024). Data jaringan direpresentasikan sebagai matriks, kemudian convolution layer menerapkan filter untuk menangkap pola lokal signifikan secara otomatis (Almuhanna & Dardouri, 2025; Park dkk., 2025).
Keunggulan CNN adalah kemampuannya menyederhanakan fitur dan mengenali pola serangan meskipun posisinya berbeda (translation invariance) (Alsoufi dkk., 2024; Baidar dkk., 2025). Banyak studi menunjukkan NIDS berbasis CNN mencapai akurasi tinggi pada dataset uji (Alsoufi dkk., 2024). Namun, karena kurang optimal menangkap ketergantungan waktu, CNN sering dikombinasikan dengan model sekuensial seperti LSTM (Park dkk., 2025).

Long Short-Term Memory (LSTM)
Long Short-Term Memory (LSTM) adalah jenis RNN khusus yang dirancang memproses data sekuensial dan mengatasi masalah vanishing gradient (Alsoufi dkk., 2024; Park dkk., 2025). Dalam NIDS, LSTM sangat efektif menangkap pola temporal lalu lintas jaringan, dimana urutan paket sering mengandung indikasi serangan krusial (Almuhanna & Dardouri, 2025; Baidar dkk., 2025). Gambar 2.4 mengilustrasikan struktur sel LSTM yang kompleks.

Gambar 2.4 Arsitektur LSTM Cell dan Temporal Sequence Processing

Varian Bidirectional LSTM (BiLSTM) mampu memproses urutan data dua arah, memberikan konteks deteksi yang lebih komperhesif (Park dkk., 2025). Penelitian membuktikan keunggulan LSTM dalam memodelkan ketergantungan jangka panjang menjadikannya handal untuk deteksi serangan berkarakteristik temporal seperti low-rate DdoS (Park dkk., 2025). Kombinasi kekuatan CNN dalam fitur spasial dan LSTM dalam fitur temporal menghasilkan model hybrid yang sangat andal (Baidar dkk., 2025).

Autoencoder untuk Anomaly Detection
Autoencoder adalah jaringan saraf unsupervised yang dilatih merekonstruksi inputnya sendiri melalui kompresi dan dekompresi data via latent space (Alsoufi dkk., 2024). Dalam IDS, autoencoder dilatih hanya dengan data normal agar model mempelajari karakteristik “normalitas” jaringan secara mendalam (Park dkk., 2025). Konsep dasarnya adalah ketika model dihadapkan pada data serangan yang aneh, ia akan gagal merekonstruksinya dengan baik sehingga menghasilkan reconstruction error tinggi (Korniszuk & Sawicki, 2024). Hal ini menjadikan error tersebut indikator kuat intrusi. Pendekatan ini terbukti efektif mendeteksi serangan zero-day karena tidak bergantung pada pengetahuan signature attacks sebelumnya (Dai dkk., 2024).
Berbagai varian seperti Stacked Autoencoder (SAE) dan Denoising Autoencoder (DAE) dikembangkan untuk meningkatkan ketahanan terhadap noise (Devi dkk., 2025). Integrasi Autoencoder dengan arsitektur lain seperti CNN atau LSTM dalam model hybrid juga semakin populer karena menggabungkan ekstraksi fitur canggih dengan deteksi anomali tanpa label (Park dkk., 2025). Meski menjanjikan, tantangan utamanya adalah menentukan threshold optimal untuk membedakan error normal dan anomali (Alsoufi dkk., 2024).

Arsitektur Autoencoder
Struktur Autoencoder terdiri dari dua bagian: encoder yang memampatkan input data ke representasi latent, dan decoder yang merekonstruksi data asli dari representasi tersebut (Alsoufi dkk., 2024). Encoder memaksa model mempelajari fitur esensial melalui pengurangan dimensi, menghilangkan noise yang tak relevan (Almuhanna & Dardouri, 2025; Park dkk., 2025). Decoder kemudian mencoba mengembalikan data ke bentuk aslinya sebaik mungkin (Baidar dkk., 2025). Gambar 2.5 memperlihatkan struktur simetris Autoencoder.

Gambar 2.5 Arsitektur Autoencoder dengan Encoder-Decoder untuk Anomaly Detection

Tujuan pelatihan adalah meminimalkan selisih antara input asli dan output rekonstruksi, biasanya diukur dengan Mean Squared Error (MSE) (Alsoufi dkk., 2024). Konsep bottleneck pada lapisan tengah memastikan model benar-benar mempelajari struktur data, bukan hanya menyalin input (Park dkk., 2025).

Reconstruction Error sebagai Indikator Anomali
Reconstruction error adalah metrik dasar dalam Autoencoder-based anomaly detection yang mengukur discrepancy antara input asli dan output yang direkonstruksi (Alsoufi dkk., 2024). Dalam konteks IDS, reconstruction error dihitung sebagai per-instance loss antara network traffic features yang di-input dan reconstructed features, dimana Autoencoder yang dilatih pada normal traffic akan efisien merekonstruksi normal instances dengan error yang rendah namun gagal merekonstruksi anomalous traffic sehingga menghasilkan error yang tinggi (Alsoufi dkk., 2024; Park dkk., 2025). Threshold untuk reconstruction error dikalibrasi berdasarkan validation set menggunakan pendekatan statistik seperti μ + kσ dari training error distribution atau percentile-based methods (Almuhanna & Dardouri, 2025; Park dkk., 2025). Penelitian menunjukkan bahwa reconstruction error dapat sebagai skor anomali yang efektif untuk deteksi zero-day dengan Area Under Curve (AUC) mencapai 0.98-0.99 pada analisis ROC (Baidar et al., 2025). Tantangan yang dihadapi meliputi penetuan threshold yang optimal antara true positive rate dan false positive rate, serta sensitivitas model terhadap outliers dalam normal training data (Alsoufi dkk., 2024).

Keunggulan Unsupervised Learning untuk Zero-Day Detection
Unsupervised learning dengan Autoencoder menawarkan keunggulan signifikan untuk deteksi zero-day attacks yang tidak memiliki contoh berlabel dalam data latih, dimana hal ini mengurangi kebutuhan untuk data serangan berlabel yang memakan waktu dan memerlukan keahlian khusus di bidang keamanan jaringan yang mendalam (Al-Jamali dkk., 2025; Alsoufi dkk., 2024). Kemampuan model unsupervised untuk menggeneralisasi terhadap jenis unseen attack menjadikannya adaptif terhadap ancaman yang terus berkembang, dimana vairan serangan baru muncul secara terus menerus (Guo, 2023). Karena tidak bergantung pada signature databases atau aturan tetap, sistem tidak memerlukan pembaruan yang sering dan terhindar dari celah waktu antara munculnya serangan dan tersediannya signature yang terdeteksi (Sridharan dkk., 2025). Penelitian empiris menunjukkan bahwa model unsupervised Autoencoder mampu memiliki kinerja yang sebanding dengan model supervised pada data in-distribution, sekaligus menunjukkan kemampuan yang lebih unggul dalam mendeteksi ancaman zero-day (Park dkk., 2025). Fleksibitas untuk melatih ulang model dengan data lalu lintas normal yang terus diperbarui memungkinkan adaptasi terhadap perubahan perilaku jaringan tanpa memerlukan pelabelan ulang data serangan (Baidar dkk., 2025).

Hybrid CNN-LSTM Autoencoder untuk NIDS
Hybrid CNN-LSTM Autoencoder menggabungkan tiga kekuatan utama deep learning: CNN untuk fitur spasial, LSTM untuk pemodelan temporal, dan Autoencoder untuk deteksi anomali tanpa pengawasan (Baidar dkk., 2025; Park dkk., 2025). Membuat integrasi ini menghasilkan sistem deteksi yang jauh lebih unggul, terutama menghadapi serangan zero-day (Osamor & Wellman, 2023). Dalam model ini, CNN menangkap pola lokal lalu lintas, outputnya diproses LSTM untuk urutan waktu, dan akhirnya struktur Encoder-Decoder merekonstruksi data untuk deteksi anomali berdasarkan error (Park dkk., 2025). Pendekatan ini efektif mengatasi kelemahan masing-masing model jika berdiri sendiri (Abdallah dkk., 2021). Gambar 2.6 menggambarkan arsitektur gabungan ini.

Gambar 2.6 Arsitektur Hybrid CNN-LSTM Autoencoder untuk Zero-Day Intrusion Detection

Keunggulan utama model hybrid ini adalah kemampuan mempelajari representasi spasial-temporal kompleks dari lalu lintas normal, yang mana hal ini krusial untuk membedakan variasi normal dan serangan (Baidar dkk., 2025). Eksperimen pada dataset uji menunjukkan model ini mencapai akurasi deteksi sangat tinggi bahkan untuk serangan yang belum pernah dilihat (Park dkk., 2025). Teknik ensemble learning juga dapat diterapkan untuk meningkatkan kinerja dan menekan false positive (Tariq dkk., 2024). Meskipun kompleksitas komputasinya lebih tinggi, akurasi yang ditawarkan menjadikannya solusi yang dapat dipertimbangkan (Akshaya & Padmavathi, 2025).

Cross-Dataset Validation
Cross-dataset validation adalah metode evaluasi yang menguji performa model IDS pada dataset berbeda dari pelatihannya (Cantone dkk., 2024). Banyak model IDS melaporkan akurasi sempurna pada satu dataset tapi gagal total di dataset lain, fenomena yang disebut “false confidence” (Lawall & Zöller, 2025; Xin & Xu, 2025). Pendekatan ini mensimulasikan tantangan di dunia nyata dimana pola jaringan selalu berubah antar lingkungan (Lawall & Zöller, 2025). Tanpa validasi ini, kita tidak bisa menjamin model IDS berfungsi baik saat di-deploy di jaringan sebenarnya (Cantone dkk., 2024). Cross-dataset validation memastikan model dapat berkerja dengan baik pada data baru, bukan hanya pada data yang digunakan saat pelatihan (Layeghy & Portmann, 2023).

Konsep Generalization dan Overfitting
Generalization adalah kemampuan model machine learning bekerja dengan baik pada data baru, indikator utama kecerdasan model yang sesungguhnya (Layeghy dkk., 2023; Verkerken dkk., 2021). Sebaliknya, overfitting terjadi saat model “menghafal” data latih termasuk noise, sehingga kinerjanya hancur pada data sumber lain (Cantone dkk., 2024). Dalam IDS, overfitting merupakan kondisi yang berbahaya karena dapat memberikan rasa aman palsu (false sense of security); model sangat baik di lingkungan laboratorium, tetapi gagal berfungsi secara efektif di kondisi sebenarnya. Penelitian menunjukkan penurunan akurasi drastis pada pengujian lintas dataset, membuktikan banyak model canggih sebenarnya overfitting (Cantone dkk., 2024). Faktor perbedaan distribusi data sangat mempengaruhi kemampuan generalisasi ini (Oyelakin dkk., 2023).

Cross-Dataset Validation Strategy
Strategi validasi ini melatih model pada source dataset dan mengujinya pada target dataset berbeda (Guida dkk., 2023). Evaluasi multi-dataset ini menunjukkan indikasi yang akurat terhadap kekokohan model menghadapi variasi jaringan yang bersifat heterogen (Cantone dkk., 2024). Teknik feature selection yang tepat juga penting untuk meningkatkan generalisasi dengan menghilangkan fitur yang terlalu spesifik (Al-kahla dkk., 2024). Untuk mengatasi tantangan pergeseran distribusi data (distribution shift) antar dataset, strategi validasi ini dapat diperkuat dengan teknik domain adaptation atau fine-tuning ringan pada dataset target. Liao (2023) menunjukkan bahwa Few-Shot Domain Adaptation mampu menjembatani kesenjangan distribusi antar dataset (data drift) secara efektif. Dengan memberikan sedikit sampel data normal dari lingkungan baru (target domain), model dapat "mengkalibrasi" ulang pemahamannya tentang pola normal tanpa perlu dilatih ulang sepenuhnya dari awal (Yue et al., 2024). Pendekatan ini bertujuan untuk membangun model yang memahami esensi serangan, bukan hanya karakteristik dataset.

Dataset
Dalam pengembangan IDS berbasis machine learning, dataset adalah fondasi utama yang menentukan kemampuan model dalam memahmi signature attacks (Cantone dkk., 2024). Kualitas serta tingkat realisme dataset sangat berpengaruh terhadap efektivitas model dalam mengenali serangan di lingkungan yang sebenarnya (Ghurab dkk., 2021). Oleh karena itu, dataset modern harus mencakup skenario serangan terkini dan trafik jaringan yang realistis. Dua dataset standar de-facto saat ini adalah CIC-IDS2017 dan CSE-CIC-IDS2018 karena ukuran dan keragaman serangannya (Sharafaldin dkk., 2018). Namun, setiap dataset punya bias dan error yang bisa menyesatkan model jika tidak ditangani benar (Liu dkk., 2022). Oleh karena itu, strategi preprocessing dan cross-validation sangat penting dalam pengembangan IDS (Cantone dkk., 2024).

CIC-IDS2017
Tantangan utama dalam riset IDS saat ini adalah mencari dataset yang benar-benar realistis. Hal ini sangat penting agar model tidak hanya menunjukkan performa tinggi secara teoretis, tetapi juga efektif di lingkungan yang sebenarnya. Sebagai rujukan utama, CIC-IDS2017 menyediakan trafik jaringan selama lima hari dengan total 2,8 juta instances yang mencakup serangan modern seperti DDoS dan Infiltration (Canadian Institute for Cybersecurity, 2017; Oyelakin dkk., 2023; Sharafaldin dkk., 2018). Namun, beberapa studi mengidentifikasi adanya isu signifikan pada dataset tersebut, di antaranya temuan galat (error) pada 78 fitur hasil ekstraksi serta masalah ketidakseimbangan kelas (class imbalance) yang sangat ekstrem antara trafik benign dan serangan (Gopalan dkk., 2021; Rosay dkk., 2022). Kondisi ini berdampak buruk pada performa model, terutama pada pengujian lintas dataset, di mana akurasinya sering kali turun drastis akibat anomali data tersebut (Barkah dkk., 2023; Cantone dkk., 2024). Oleh karena itu, solusi yang banyak direkomendasikan adalah penerapan praproses (preprocessing) yang ketat disertai validasi ulang fitur agar hasil klasifikasi lebih valid dan memiliki generalisasi yang lebih kuat

CSE-CIC-IDS2018
Sebagai pengembangan dari versi sebelumnya, dataset CSE-CIC-IDS2018 berkembang dari kolaborasi antara peneliti dengan Communications Security Establishment (CSE) dan Amazon Web Services (AWS), yang mana infrastruktur komputasi cloud ini memungkinkan perekaman data skala besar baik dalam format PCAP mentah maupun data yang sudah terproses (Canadian Institute for Cybersecurity & Communications Security Establishment, 2018; Sharafaldin dkk., 2018). Skala masif ini dirancang khusus untuk mengatasi keterbatasan pendahulunya, namun tantangan yang baru justru muncul dalam bentuk fenomena “false confidence” di kalangan peneliti (Karatas dkk., 2020). Walaupun model deep learning sering menunjukkan akurasi internal yang sangat tinggi pada dataset ini, tetapi kinerjanya kadang menurun secara signifikan akibat pergeseran distribusi data saat diuji lintas dataset. Hal ini menunjukkan bahwa model terlalu menyesuaikan diri dengan bias spesifik dataset daripada benar-benar memahami esensi serangan (Cantone dkk., 2024). Temuan ini menegaskan bahwa kuantitas data semata tidak menjamin terciptanya model yang unggul terhadap variasi jaringan yang sebenarnya, sehingga strategi cross-validation yang lebih mendalam menjadi sangat diperlukan (Cantone dkk., 2024).

Feature Alignment dan Cross-Dataset Strategy
Dalam konteks cross-validation lintas dataset, strategi feature alignment menjadi elemen krusial untuk mengatasi perbedaan dimensi dan definisi fitur yang sering muncul antar lingkungan jaringan yang berbeda (Xin & Xu, 2025). Masalah yang tidak cocok ini sering kali diperparah oleh fenomena pergeseran distribusi data (distribution shift), yang mana hal ini dapat menurunkan performa model IDS secara signifikan jika dirancangan untuk dapat beradaptasi terhadap perubahan lingkungan (Cantone dkk., 2024). Sebagai solusi untuk mengatasi hambatan struktural tersebut, model MFEI-IDS yang diusulkan oleh Mao dkk. (2025) menggunakan kombinasi Fully Convolutional Networks dan Transformer untuk menyamakan representasi fitur secara hierarkis. Selain itu, penerapan seleksi fitur berbasis mRMR (minimum Redundancy Maximum Relevance) juga sangat penting untuk memangkas fitur yang rentan terhadap noise, yang mana langkah ini secara signifikan meningkatkan ketahanan model terhadap variabilitas data baru sekaligus menjaga stabilitas deteksi pada skenario operasional jaringan yang sebenarnya (Cantone dkk., 2024).

Metrik Evaluasi
Evaluasi kinerja IDS tidak bisa hanya mengandalkan akurasi, melainkan juga perlu serangkaian metrik komprehensif untuk memahami perilaku model secara penuh (Salih & Abdulazeez, 2021). Mengingat sifat data keamanan yang sangat tidak seimbang dimana serangan adalah kejadian yang langka, metrik seperti Precision, Recall, dan F1-Score jauh lebih relevan (Le dkk., 2024). Selain itu, untuk sistem anomaly detection, analisis kurva ROC dan nilai AUC menjadi standar emas mengukur kemampuan separasi model (Bai & Fossaceca, 2024). False Positive Rate juga metrik operasional yang krusial karena berkaitan langsung dengan kelelahan analis keamanan (alert fatigue) (Mohale & Obagbuwa, 2025).

Classification Metrics
Metrik klasifikasi dasar diturunkan dari confusion matrix yang berfungsi untuk memetakan prediksi benar dan salah. Precision mengukur keandalan prediksi serangan, sedangkan Recall mengukur kemampuan mendeteksi seluruh serangan ada (Salih & Abdulazeez, 2021). F1-Score berfungsi memberikan keseimbangan keduanya, sangat penting untuk menghindari false alarm tanpa melewatkan serangan (Agarwal dkk., 2021). Matthews Correlation Coefficient (MCC) dianggap metrik yang lebih representatif untuk evaluasi pada dataset yang tidak seimbang karena mempertimbangkan semua aspek confusion matrix (Cantone dkk., 2024). Penggunaan metrik ini bersamaan memberikan gambaran lengkap kekuatan model IDS.

ROC Curve dan AUC
Kurva ROC menggambarkan trade-off antara True Positive Rate dan False Positive Rate pada berbagai titik ambang batas (threshold) operasi (Bai & Fossaceca, 2024). Sebagai metrik evaluasi, Area Under Curve (AUC) merangkum kinerja model tersebut dalam satu nilai tunggal. Keunggulan AUC terletak pada stabilitasnya terhadap ketimpangan kelas data (class imbalance), sehingga mampu membedakan aktivitas normal dan serangan dengan baik (Xin & Xu, 2025). Pada skenario deteksi anomali tanpa label (unsupervised), pengembangan varian seperti EM-AUC memungkinkan evaluasi tetap objektif meskipun tanpa tersedia ground truth yang lengkap (Bai & Fossaceca, 2024).

False Positive Rate (FPR)
Tingkat False Positive Rate (FPR) merupakan salah satu tantangan yang paling signifikan dalam operasional Security Operations Center (SOC). Akumulasi peringatan palsu yang tinggi dapat menyebabkan alert fatigue, yang berisiko membuat analis melewatkan ancaman yang valid (Mohale & Obagbuwa, 2025). Pada jaringan IoT, menjaga FPR tetap rendah menjadi prioritas utama untuk mencegah gangguan pada fungsi perangkat (Amrullah, 2025). Penelitian Mohale & Obagbuwa (2025) menunjukkan bahwa penggunaan algoritma XGBoost mampu menekan nilai FPR hingga 0,07. Selain itu, penerapan algoritma Explainable AI (XAI) seperti SHAP dan kalibrasi Platt scaling membantu analis dalam memverifikasi pemicu deteksi secara lebih akurat.

Generalization Metrics
Metrik generalisasi merepresentasikan kemampuan suatu model untuk beroperasi secara efektif pada data yang belum pernah dilihat (unseen data) (Park dkk., 2025). Hal ini sangat krusial untuk memastikan bahwa sistem IDS siap menghadapi variasi serangan sebenarnya, bukan sekadar memberikan performa tinggi pada dataset uji coba (Cantone dkk., 2024). Penurunan performa yang drastis lintas dataset (cross-dataset) menunjukkan adanya masalah overfitting, sehingga kemampuan generalisasi menjadi indikator utama dalam kesiapan implementasi model (Cantone dkk., 2024).

METODOLOGI PENELITIAN
Desain Penelitian
Penelitian ini menggunakan Design Research Methodology (DRM) sebagai pendekatan sistematis untuk mengidentifikasi masalah, mengembangkan solusi, dan mengevaluasi efektivitasnya (Blessing & Chakrabarti, 2009). DRM dipilih karena menyediakan kerangka desain yang mendukung pengembangan artefak teknis seperti model deteksi intrusi, dengan empat tahapan yang bersifat iteratif dimana hasil evaluasi dapat memicu perbaikan desain atau penyesuaian metode. Proses iterasi ini terjadi ketika hasil Studi Deskriptif II menunjukkan performa yang belum optimal, sehingga dapat dilakukan refinement pada tahap Studi Preskriptif seperti tuning hyperparameter atau penyesuaian threshold deteksi. Metodologi ini memastikan penelitian berjalan terstruktur, khususnya pada pengembangan model Hybrid CNN-LSTM Autoencoder untuk deteksi zero-day attack dengan pendekatan unsupervised learning. Gambar 3.1 menunjukkan alur DRM yang diterapkan.

Gambar 3.1 Design Research Methodology (DRM)

Klarifikasi Penelitian
Tahap klarifikasi penelitian dilakukan untuk mengidentifikasi inti permasalahan melalui studi literatur yang komprehensif terhadap berbagai referensi ilmiah terkait keamanan siber, deteksi serangan zero-day, algoritma deep learning, dan teknik unsupervised learning. Kajian literatur mencakup arsitektur CNN untuk ekstraksi fitur spasial, LSTM untuk pemodelan temporal, dan Autoencoder untuk deteksi anomali berbasis reconstruction error. Selain itu, dipelajari juga metode evaluasi performa model seperti akurasi, presisi, recall, F1-score, dan AUC-ROC sebagai acuan penilaian keberhasilan (Park dkk., 2025). Analisis research gap pada tahap ini menemukan bahwa mayoritas penelitian IDS sebelumnya hanya melakukan evaluasi pada single dataset tanpa memvalidasi kemampuan generalisasi model terhadap dataset berbeda, yang menjadi fokus utama penelitian ini.
Berdasarkan analisis gap tersebut, penelitian ini merumuskan dua pertanyaan penelitian utama: (1) Bagaimana performa model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan siber pada dataset CIC-IDS2017 dibandingkan dengan baseline methods? (2) Bagaimana kemampuan generalisasi model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan zero-day melalui cross-dataset validation menggunakan dataset CSE-CIC-IDS2018 dengan perbandingan skenario zero-shot dan few-shot adaptation? Untuk menjawab pertanyaan tersebut, ditetapkan kriteria keberhasilan penelitian yaitu model harus mencapai F1-score minimal 90% pada dataset baseline dan generalization gap tidak boleh melebihi 15% untuk memastikan model dapat beradaptasi dengan data baru.

Studi Deskriptif I
Tahap ini berfokus pada analisis kebutuhan sistem dan karakteristik data yang akan digunakan. Berdasarkan tinjauan literatur, teridentifikasi beberapa keterbatasan metode yang sudah ada seperti ketergantungan pada supervised learning yang memerlukan label serangan terbaru, evaluasi single-dataset yang tidak membuktikan generalisasi model, dan kompleksitas arsitektur ensemble yang berdampak pada beban komputasi yang tinggi. Keterbatasan ini menjadi dasar pemilihan pendekatan unsupervised learning dengan cross-dataset validation pada penelitian ini. Analisis kebutuhan mencakup identifikasi spesifikasi teknis seperti environment dengan dukungan GPU, framework TensorFlow, serta alat preprocessing seperti Pandas dan NumPy (Singh & Jang, 2022). Analisis data dilakukan terhadap dua dataset publik, yaitu CIC-IDS2017 dengan 2,7 juta records dan lebih dari 80 fitur, serta CSE-CIC-IDS2018 dengan 16,2 juta records dan 80 fitur. Dataset pertama mencakup trafik normal dan berbagai serangan seperti DoS/DDoS, Web Attacks, Botnet, dan Heartbleed, sedangkan dataset kedua memiliki serangan lebih kompleks. Hasil analisis menunjukkan perlunya feature alignment antara kedua dataset untuk memungkinkan cross-dataset validation sebagai kontribusi utama pada penelitian ini.

Studi Preskriptif
Studi preskriptif merupakan tahap inti yang mencakup seluruh proses teknis pengembangan model Hybrid CNN-LSTM Autoencoder untuk deteksi anomali zero-day. Tahapan ini terdiri dari beberapa subbagian yang saling berkaitan, mulai dari persiapan environment, analisis data awal, preprocessing, implementasi arsitektur model, training dan optimisasi, hingga validasi lintas dataset. Secara keseluruhan, tahapan ini merealisasikan pendekatan solusi yang telah dirumuskan sebelumnya melalui serangkaian eksperimen sistematis.

Persiapan Environment
Tahap persiapan dimulai dengan instalasi Python 3.8+ dan persiapan virtual environment menggunakan uv atau venv. Library dependencies yang digunakan meliputi TensorFlow 2.x, NumPy, Pandas, Scikit-learn, Matplotlib, dan Seaborn untuk mendukung proses training dan analisis data. Konfigurasi GPU driver CUDA dilakukan untuk mempercepat proses training model. Dataset CIC-IDS2017 dan CSE-CIC-IDS2018 diunduh dari situs resmi CIC dan diekstraksi ke dalam struktur direktori yang terorganisir (Park dkk., 2025).

Exploratory Data Analysis
Exploratory data analysis (EDA) dilakukan untuk memahami karakteristik dataset melalui inspeksi struktur data, identifikasi missing values dan infinite values, serta visualisasi distribusi fitur menggunakan histogram dan boxplot. Analisis korelasi antar fitur dilakukan menggunakan heatmap correlation matrix untuk mengidentifikasi fitur yang berkorelasi tinggi. Berdasarkan hasil EDA, dilakukan preprocessing yang mencakup handling missing values, handling infinite values, feature alignment antara kedua dataset, dan data cleaning untuk menghapus duplikasi serta outlier ekstrem. Khusus data training, dilakukan filtering untuk mengambil hanya data dengan label "BENIGN" sesuai paradigma unsupervised learning (Almuhanna & Dardouri, 2025).

Preprocessing dan Feature Engineering
Feature engineering mencakup encoding categorical features, normalisasi fitur numerik menggunakan MinMaxScaler atau StandardScaler, dan reshaping data menjadi format 3D (samples, timesteps, features) sesuai input LSTM layer. Data split dilakukan dengan proporsi 70% training, 15% validation, dan 15% testing pada CIC-IDS2017. Pada CSE-CIC-IDS2018, data digunakan untuk dua skenario cross-dataset validation: zero-shot (tanpa adaptasi) dan few-shot adaptation dengan 1% data benign target untuk adaptasi unsupervised, sedangkan sisanya digunakan untuk evaluasi (Singh & Jang, 2022). Alur lengkap pemrosesan data, dari data mentah hingga menjadi input siap latih, diilustrasikan pada Gambar 3.2.

Gambar 3.2 Alur Preprocessing dan Pembentukan Data Time-Series

Implementasi Arsitektur Model
Implementasi arsitektur Hybrid CNN-LSTM Autoencoder dilakukan menggunakan Keras API dengan struktur encoder yang terdiri dari Conv1D layers untuk ekstraksi fitur spasial diikuti LSTM layers untuk pemodelan temporal, bottleneck layer sebagai latent representation, dan decoder yang merupakan mirror dari encoder untuk rekonstruksi input. Arsitektur ini dirancang khusus untuk menangkap pola spasial dan temporal dalam network traffic data secara simultan. Detail arsitektur model beserta dimensi data pada setiap lapisannya ditampilkan secara rinci pada Gambar 3.3.

Gambar 3.3 Arsitektur Hybrid CNN-LSTM Autoencoder yang Diusulkan

Training dan Optimisasi Model
Training model menggunakan hanya data normal dengan loss function Mean Squared Error (MSE) untuk mengukur reconstruction error antara input asli dan output rekonstruksi, yang diformulasikan pada Persamaan (1).

MSE=1/n ∑\_(i=1)^n▒(x_i-x ̂_i )^2 (1)

Proses training menggunakan optimizer Adam dengan learning rate 0.001, batch size 128-256, dan early stopping jika validation loss tidak menurun selama 10 epoch berturut-turut. Pengawasan dilakukan terhadap training loss dan validation loss untuk mendeteksi overfitting atau underfitting. Evaluasi pada test set CIC-IDS2017 dilakukan dengan menghitung reconstruction error, menentukan threshold berdasarkan 95th atau 99th persentil, dan menghitung metrik evaluasi seperti akurasi, presisi, recall, F1-score, dan AUC-ROC.

Cross-Dataset Validation
Cross-dataset validation dilakukan dengan menggunakan model yang dilatih pada CIC-IDS2017 untuk memprediksi sampel dari CSE-CIC-IDS2018 dalam dua skenario: zero-shot (tanpa retraining) dan few-shot adaptation (fine-tuning unsupervised menggunakan 1% sampel benign target). Reconstruction error dihitung untuk semua sampel, threshold yang sama dari training digunakan untuk klasifikasi, dan metrik evaluasi dihitung untuk mengukur performa pada dataset berbeda. Pada skenario few-shot, data adaptasi tidak boleh overlap dengan data evaluasi untuk mencegah data leakage. Analisis generalization gap dilakukan dengan membandingkan performa pada kedua dataset, serta analisis per-attack-type performance untuk melihat jenis serangan yang paling sulit dideteksi. Skema validasi lintas dataset ini, yang menjadi metode utama untuk menguji kemampuan generalisasi dan deteksi serangan zero-day, dapat dilihat secara visual pada Gambar 3.4.

Gambar 3.4 Skema Eksperimen Validasi Lintas Dataset (Cross-Dataset Validation)

Dokumentasi hasil mencakup pembuatan visualisasi ROC curve, confusion matrix, dan reconstruction error distribution plot, serta tabel hasil evaluasi yang membandingkan berbagai metrik.

Studi Deskriptif II
Tahap Studi Deskriptif II dilakukan untuk mengevaluasi performa model secara mendalam dan menganalisis hasil eksperimen. Evaluasi dilakukan dalam beberapa aspek untuk memvalidasi kemampuan model dari berbagai sudut pandang, yang mencakup analisis performa klasifikasi, kemampuan generalisasi, pengujian signifikansi statistik, dan optimisasi threshold deteksi.

Analisis Performa Model
Analisis performa model dilakukan dengan mengevaluasi kemampuan Hybrid CNN-LSTM Autoencoder dalam mendeteksi anomali. Evaluasi baseline pada CIC-IDS2017 menggunakan metrik akurasi, presisi, recall, F1-score, dan FPR yang dihitung dari confusion matrix, dengan F1-score dan FPR sebagai metrik utama perbandingan zero-shot vs few-shot. Nilai presisi tinggi menunjukkan alarm model dapat dipercaya dengan false alarm minimal, sedangkan recall tinggi berarti model mampu mendeteksi mayoritas serangan (Park dkk., 2025) Analisis distribusi error rekonstruksi dilakukan untuk melihat pemisahan antara data normal dan anomali, dimana idealnya kedua distribusi harus terpisah jauh. Analisis per tipe serangan juga dilakukan untuk mengidentifikasi kelebihan dan kekurangan model dalam mendeteksi berbagai jenis serangan.

Analisis Generalisasi
Analisis generalisasi merupakan aspek yang penting untuk mengukur kemampuan model pada dataset yang berbeda dari dataset training. Generalization gap dihitung dengan mencari selisih antara performa di dataset CIC-IDS2017 (in-distribution) dan performa di dataset CSE-CIC-IDS2018 (out-of-distribution), yang diformulasikan pada Persamaan (2).

〖"Gap" 〗\_metric=M_CIC-M_CSE (2)

Dimana M_CIC adalah nilai metrik di CIC-IDS2017 dan M_CSE adalah nilai metrik di CSE-CIC-IDS2018. Gap yang kecil menunjukkan model memiliki kemampuan generalisasi yang baik dan tidak overfitting, sedangkan gap besar mengindikasikan model hanya efektif pada data training (Singh & Jang, 2022). Analisis kualitatif dilakukan untuk mengidentifikasi faktor penyebab penurunan performa, seperti perbedaan distribusi fitur atau keberadaan serangan baru seperti Heartbleed di dataset CSE-CIC-IDS2018.

Pengujian Statistik
Pengujian statistik dilakukan untuk memastikan perbedaan performa yang diamati signifikan secara statistik dan bukan karena faktor kebetulan. Metode McNemar's test digunakan untuk membandingkan performa model pada kedua dataset dengan membuat tabel kontingensi dari prediksi benar dan salah (Halbouni et al., 2022). Hipotesis null menyatakan tidak ada perbedaan signifikan antara performa model di kedua dataset, sedangkan hipotesis alternatif menyatakan ada perbedaan signifikan. Level signifikansi ditetapkan pada alpha 0.05, dimana p-value kurang dari 0.05 mengindikasikan penolakan hipotesis null dan perbedaan yang signifikan secara statistik (Park et al., 2025). Ilustrasi bagaimana threshold memisahkan distribusi error trafik normal dan serangan ditunjukkan pada Gambar 3.5.

Gambar 3.5 Ilustrasi Penentuan Threshold Deteksi Berdasarkan Distribusi Error.

Analisis Threshold
Analisis threshold bertujuan menentukan nilai batas error optimal untuk memisahkan data normal dan anomali. Threshold dipilih berdasarkan distribusi error di data validasi normal menggunakan nilai persentil tertentu seperti 95th atau 99th persentil. Pemilihan ini melibatkan trade-off dimana threshold rendah (95th) menghasilkan TPR tinggi namun FPR juga meningkat, sedangkan threshold tinggi (99th) menurunkan FPR namun juga menurunkan TPR (Singh & Jang, 2022). Optimisasi dilakukan dengan mencoba berbagai nilai threshold dan memilih yang menghasilkan F1-score maksimal seperti pada Persamaan (3).

θ_optimal=arg⁡(max⁡)┬θ F1(θ) (3)

Alternatif lain menggunakan Youden's Index untuk memaksimalkan selisih TPR dan FPR, atau menyesuaikan dengan kebijakan organisasi seperti membatasi FPR dibawah 1% untuk meminimalkan false alarm.

Instrumen Penelitian
Instrumen penelitian merupakan alat ukur yang krusial yang digunakan untuk mengumpulkan data serta mengevaluasi hasil eksperimen dalam penelitian ini, dimana pemilihan instrumen yang tepat sangat penting untuk memastikan validitas dan reliabilitas hasil yang didapatkan. Dalam konteks penelitian deteksi intrusi berbasis deep learning yang dilakukan, instrumen penelitian mencakup berbagai metrik evaluasi kuantitatif yang mana metrik-metrik ini telah divalidasi secara luas dalam domain machine learning. Instrumen evaluasi ini dirancang untuk mengukur tiga aspek utama yaitu performa klasifikasi model, kemampuan deteksi anomali, serta kemampuan generalisasi terhadap data baru yang belum pernah dilihat sebelumnya.
Untuk mengukur kemampuan model dalam melakukan klasifikasi antara trafik normal dan anomali, digunakan beberapa metrik standar. Metrik pertama adalah akurasi yang mengukur proporsi prediksi benar dari total prediksi, yang mana formulanya dapat dilihat pada Persamaan (4) dimana TP adalah True Positive, TN adalah True Negative, FP adalah False Positive, dan FN adalah False Negative.

"Accuracy"=(TP+TN)/(TP+TN+FP+FN)(4)

Selain itu, digunakan juga metrik presisi yang mengukur proporsi prediksi positif yang benar-benar positif. Hal ini penting untuk mengukur tingkat kepercayaan terhadap alarm yang dihasilkan sistem agar alarm yang palsu tidak terlalu banyak (Park et al., 2025). Formula presisi ditunjukkan pada Persamaan (5).

"Precision"=TP/(TP+FP) (5)

Metrik recall atau sensitivity juga digunakan untuk mengukur proporsi sampel positif yang berhasil terdeteksi, dimana ini sangat krusial dalam konteks IDS karena kegagalan mendeteksi serangan dapat berakibat fatal bagi keamanan jaringan. Formula recall ditunjukkan pada Persamaan (3).

"Recall"=TP/(TP+FN)(3)

F1-score digunakan sebagai harmonic mean antara presisi dan recall, yang mana metrik ini memberikan keseimbangan antara kedua aspek tersebut terutama pada kondisi data yang imbalanced atau tidak seimbang. Formula F1-score ditunjukkan pada Persamaan (6).

F1"-" score=2×("Precision" ×"Recall" )/("Precision" +"Recall" ) (6)

Selanjutnya, instrumen evaluasi anomaly detection berfokus pada kemampuan model untuk membedakan pola normal dan anomali berdasarkan reconstruction error. Metrik utama yang digunakan pada penelitian ini adalah F1-score dan False Positive Rate (FPR), sedangkan AUC-ROC digunakan sebagai metrik pendukung untuk melihat kemampuan pemisahan kelas pada berbagai threshold. ROC curve sendiri dibuat dengan memplot True Positive Rate (TPR) terhadap False Positive Rate (FPR) pada berbagai threshold reconstruction error, yang mana ini memberikan visualisasi mengenai trade-off antara detection rate dan false alarm rate (Singh & Jang, 2022). Selain itu, False Positive Rate (FPR) juga dihitung untuk mengukur proporsi trafik normal yang salah diklasifikasikan sebagai anomali menggunakan Persamaan (7) berikut.

FPR=FP/(FP+TN) (7)

Terakhir, instrumen evaluasi generalization digunakan untuk mengukur kemampuan model dalam mendeteksi serangan pada dataset yang berbeda dari dataset training. Metrik krusial yang digunakan adalah generalization gap yang dihitung sebagai selisih performa antara evaluasi pada test set dataset yang sama (in-distribution) dan evaluasi pada dataset yang berbeda (out-of-distribution) seperti yang ditunjukkan pada Persamaan (8).

"Generalization Gap"=P_in-P_out (8)

Dimana generalization gap yang kecil menunjukkan bahwa model memiliki kemampuan generalisasi yang baik dan tidak overfitting pada karakteristik spesifik dataset training (Park dkk., 2025). Statistical significance testing menggunakan metode seperti McNemar's test juga diterapkan untuk memastikan perbedaan performa yang diamati bukan terjadi karena kebetulan.

Alat dan Bahan Penelitian
Alat Penelitian
Alat penelitian ini mencakup perangkat keras dan lunak untuk mendukung komputasi intensif Deep Learning. Dari sisi hardware, dibutuhkan komputer dengan GPU (Graphics Processing Unit) berspesifikasi tinggi guna mempercepat proses pelatihan model. Rincian spesifikasi perangkat keras disajikan pada Tabel 3.1, dimana penggunaan platform cloud seperti Google Colab Pro juga dimungkinkan sebagai alternatif lingkungan komputasi yang fleksibel.

Tabel 3.1 Spesifikasi Hardware Penelitian
Komponen Spesifikasi Fungsi
GPU NVIDIA RTX 4060 8GB VRAM dengan CUDA support Mempercepat proses training model neural network
RAM 32 GB DDR4 Menangani dataset berukuran besar dan operasi preprocessing
Storage SSD 256 GB Menyimpan dataset, model checkpoint, dan hasil eksperimen
Processor AMD Ryzen 5 4600G (6 cores / 12 threads) Mendukung komputasi paralel untuk preprocessing data
OS Windows 11 Kompatibilitas dengan tools deep learning

Pada aspek software, Visual Studio Code digunakan sebagai Integrated Development Environment (IDE) utama yang menjalankan kernel Jupyter secara lokal untuk memaksimalkan performa GPU RTX 4060. Penggunaan layanan cloud seperti Google Colab disiapkan sebagai lingkungan komputasi alternatif untuk validasi silang. Bahasa Python v3.8+ dengan framework TensorFlow 2.x dipilih untuk membangun arsitektur Hybrid CNN-LSTM Autoencoder. Tabel 3.2 merincikan seluruh perangkat lunak yang digunakan, dimana kombinasi alat open-source ini menjamin aksesibilitas dan kemudahan reproduksi penelitian.

Tabel 3.2 Software dan Library yang Digunakan
Kategori Software/Library Versi Fungsi
IDE & Cloud Compute VS Code (Local Jupyter) / Google Colab Latest Lingkungan pengembangan utama & eksekusi komputasi
Bahasa Pemrograman Python 3.8+ Bahasa pemrograman utama untuk implementasi
Deep Learning Framework TensorFlow 2.x Engine untuk membangun dan melatih model CNN-LSTM
Numerical Computing NumPy 1.21+ Operasi numerik dan manipulasi array
Data Manipulation Pandas 1.3+ Manipulasi data tabular dan preprocessing
Machine Learning Scikit-learn 1.0+ Preprocessing, feature scaling, dan evaluasi metrik
Visualization Matplotlib, Seaborn 3.5+, 0.11+ Visualisasi hasil eksperimen dan analisis data
Version Control Git, GitHub Latest Manajemen kode dan kolaborasi
GPU Support NVIDIA CUDA (Local/Cloud) 12.x Akselerasi komputasi GPU RTX 4060

Bahan Penelitian
Bahan utama penelitian ini adalah dataset benchmark CIC-IDS2017 dan CSE-CIC-IDS2018 dari Canadian Institute for Cybersecurity. Dataset CIC-IDS2017 mencakup 2,8 juta network flows selama lima hari kerja, dimana hari Senin hanya berisi trafik normal. Karakteristik dataset ini, sebagaimana dirinci pada Tabel 3.3, sangat representatif karena memuat berbagai serangan modern.

Tabel 3.3 Karakteristik Dataset CIC-IDS2017
Karakteristik Deskripsi
Sumber Canadian Institute for Cybersecurity (CIC)
Durasi Pengumpulan 5 hari kerja (Senin - Jumat)
Jumlah Total Records ~2,8 juta network flows
Jumlah Fitur Lebih dari 80 fitur statistik berbasis flow
Format File CSV
Tools Ekstraksi CICFlowMeter
Trafik Normal Hari Senin (Hanya berisi trafik benign)
Jenis Serangan Brute Force (FTP-Patator, SSH-Patator)
DoS/DDoS (DoS Hulk, DDoS, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest)
Web Attacks (SQL Injection, XSS, Brute Force)
Infiltration
Botnet
Port Scan
Heartbleed
Link Download https://www.unb.ca/cic/datasets/ids-2017.html

Dataset CSE-CIC-IDS2018 digunakan sebagai set pengujian eksternal, yang mana dataset ini memuat 16 juta flows dengan serangan kompleks seperti Heartbleed. Penggunaannya bertujuan mensimulasikan skenario serangan zero-day melalui validasi lintas dataset. Tabel 3.4 memperlihatkan detail karakteristiknya, dimana struktur fiturnya memiliki kesamaan dengan dataset 2017 sehingga memudahkan proses penyelarasan data.

Tabel 3.4 Karakteristik Dataset CSE-CIC-IDS2018
Karakteristik Deskripsi
Sumber Canadian Institute for Cybersecurity (CIC) & CSE Canada
Durasi Pengumpulan 10 hari
Jumlah Total Records ~16,2 juta network flows
Jumlah Fitur 80 fitur statistik berbasis flow
Format File CSV
Tools Ekstraksi CICFlowMeter
Fungsi dalam Penelitian Test set untuk cross-dataset validation (zero-day simulation)
Jenis Serangan Brute-Force (FTP, SSH, Web)
DoS/DDoS attacks (berbagai variasi protokol)
Heartbleed vulnerability exploitation
Web Attacks
Infiltration scenario (APT simulation)
Botnet activities
Keunggulan Mencakup serangan yang tidak ada di CIC-IDS2017 (Heartbleed)
Link Download https://www.unb.ca/cic/datasets/ids-2018.html

Kedua dataset ini dipilih karena dokumentasinya lengkap dan merupakan standar evaluasi IDS saat ini. Kompabilitas antar dataset mempermudah penyelarasan fitur (feature alignment) yang penting untuk menguji kemampuan model dalam mendeteksi serangan baru.

Daftar Pustaka
Abdallah, M., An Le Khac, N., Jahromi, H., & Delia Jurcut, A. (2021). A Hybrid CNN-LSTM Based Approach for Anomaly Detection Systems in SDNs. ACM International Conference Proceeding Series. https://doi.org/10.1145/3465481.3469190;PAGE:STRING:ARTICLE/CHAPTER
Achbarou, O., Datsi, T., Bourkoukou, O., & El kiram, A. M. (2025). Enhanced intrusion detection system using feature selection and hybrid learning models for high performance and efficiency in an IOT environment. Journal of Engineering Research. https://doi.org/10.1016/J.JER.2025.10.016
Agarwal, A., Sharma, P., Alshehri, M., Mohamed, A., & Alfarraj, O. (2021). Classification model for accuracy and intrusion detection using machine learning approach. PeerJ Computer Science, 7, e437. https://doi.org/10.7717/PEERJ-CS.437
Ahmed, U., Nazir, M., Sarwar, A., Ali, T., Aggoune, E. H. M., Shahzad, T., & Khan, M. A. (2025). Signature-based intrusion detection using machine learning and deep learning approaches empowered with fuzzy clustering. Scientific Reports 2025 15:1, 15(1), 1726-. https://doi.org/10.1038/s41598-025-85866-7
Akshaya, S., & Padmavathi. (2025). ENHANCING CYBER DEFENSE AGAINST ZERO-DAY ATTACKS USING ENSEMBLE NEURAL NETWORKS. International Journal of Computer Networks and Communications, 17(4), 131–132. https://doi.org/10.5121/ijcnc.2025.17408
Al-Jamali, N., Zarzoor, A. R., & Al-Raweshidy, H. S. (2025). An Effective Technique of Zero-Day Attack Detection in the Internet of Things Network Based on the Conventional Spike Neural Network Learning Method. IET Networks, 14(1), e70019. https://doi.org/10.1049/NTW2.70019;WEBSITE:WEBSITE:IETRESEARCH;WGROUP:STRING:PUBLICATION
Aljanabi, M., Ismail, M. A., & Ali, A. H. (2021). Intrusion Detection Systems, Issues, Challenges, and Needs. International Journal of Computational Intelligence Systems, 14(1), 560–571. https://doi.org/10.2991/IJCIS.D.210105.001
Al-kahla, L., Hussein, M. K., & Alqassab, A. (2024). Feature Selection Techniques in Intrusion Detection: A Comprehensive Review. Iraqi Journal for Computers and Informatics, 50(1), 46–53. https://doi.org/10.25195/IJCI.V50I1.462
Almuhanna, R., & Dardouri, S. (2025). A deep learning/machine learning approach for anomaly based network intrusion detection. Frontiers in Artificial Intelligence, 8, 1625891. https://doi.org/10.3389/FRAI.2025.1625891
Alsoufi, M. A., Siraj, M. M., Ghaleb, F. A., Al-Razgan, M., Al-Asaly, M. S., Alfakih, T., & Saeed, F. (2024). Anomaly-Based Intrusion Detection Model Using Deep Learning for IoT Networks. CMES - Computer Modeling in Engineering and Sciences, 141(1), 823–845. https://doi.org/10.32604/CMES.2024.052112
Amrullah, A. (2025). A Review and Comparative Analysis of Intrusion Detection Systems for Edge Networks in IoT | Intellithings Journal. Intellithings Journal. https://e-jurnal.unisda.ac.id/index.php/intellithings/article/view/8859
Bai, K., & Fossaceca, J. (2024). EM-AUC: A Novel Algorithm for Evaluating Anomaly Based Network Intrusion Detection Systems. Sensors (Basel, Switzerland), 25(1). https://doi.org/10.3390/S25010078
Baidar, R., Maric, S., & Abbas, R. (2025). Hybrid Deep Learning-Federated Learning Powered Intrusion Detection System for IoT/5G Advanced Edge Computing Network. https://arxiv.org/pdf/2509.15555
Barkah, A. S., Selamat, S. R., Abidin, Z. Z., & Wahyudi, R. (2023). Data Generative Model to Detect the Anomalies for IDS Imbalance CICIDS2017 Dataset. TEM Journal, 12(1), 80–89. https://doi.org/10.18421/TEM121-11
Blessing, L., & Chakrabarti, A. (2009). DRM: A Design Research Methodology.
Canadian Institute for Cybersecurity. (2017). Intrusion detection evaluation dataset (CIC-IDS2017). http://cicresearch.ca/CICDataset/CIC-IDS-2017/.
Canadian Institute for Cybersecurity, & Communications Security Establishment. (2018). CSE-CIC-IDS2018 on AWS. University of New Brunswick &amp; Amazon Web Services. https://registry.opendata.aws/cse-cic-ids2018/. https://registry.opendata.aws/cse-cic-ids2018/
Cantone, M., Marrocco, C., & Bria, A. (2024). On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection. IEEE Access. https://doi.org/10.1109/ACCESS.2024.3472907
Cipollone, F. (2025, Desember 5). React4Shell (React2Shell) Is being exploited at scale: Critical Unauthenticated RCE in React RSC Flight (CVE-2025-55182) and Next.js (CVE-2025-66478) - Phoenix Security. https://phoenix.security/react2shell-cve-2025-55182-explotiation/
Dai, Z., Por, L. Y., Chen, Y. L., Yang, J., Ku, C. S., Alizadehsani, R., & Pławiak, P. (2024). An intrusion detection model to detect zero-day attacks in unseen data using machine learning. PLOS ONE, 19(9), e0308469. https://doi.org/10.1371/JOURNAL.PONE.0308469
Devi, A., Prabhakaran, M. K., Prasanna Kumar, J. S., Nithish Kumar, J., Devi, A., Prabhakaran, M. K., Prasanna Kumar, J. S., & Nithish Kumar, J. (2025). Autoencoder-Based Anomaly Detection for Cyber Threat Monitoring. https://services.igi-global.com/resolvedoi/resolve.aspx?doi=10.4018/979-8-3693-9919-4.ch004, 59–80. https://doi.org/10.4018/979-8-3693-9919-4.CH004
Ghazi, D., Hamid, H. S., Zaiter, M. J., Sabri, A., & Behadili, G. (2024). Snort Versus Suricata in Intrusion Detection. Iraqi Journal of Information and Communication Technology, 7(2), 73–88. https://doi.org/10.31987/IJICT.7.2.290
Ghurab, M., Gaphari, G., Alshami, F., Alshamy, R., & Othman, S. (2021). A Detailed Analysis of Benchmark Datasets for Network Intrusion Detection System. Asian Journal of Research in Computer Science, 14–33. https://doi.org/10.9734/AJRCOS/2021/V7I430185
Gopalan, S. S., Ravikumar, D., Linekar, D., Raza, A., & Hasib, M. (2021). Balancing Approaches towards ML for IDS: A Survey for the CSE-CIC IDS Dataset. ICCSPA 2020 - 4th International Conference on Communications, Signal Processing, and their Applications, 2021-January. https://doi.org/10.1109/ICCSPA49915.2021.9385742
Guida, C., Nascita, A., Montieri, A., & Pescape, A. (2023). Cross-Evaluation of Deep Learning-based Network Intrusion Detection Systems. Proceedings - 2023 International Conference on Future Internet of Things and Cloud, FiCloud 2023, 328–335. https://doi.org/10.1109/FICLOUD58648.2023.00055
Guo, Y. (2023). A Survey of Machine Learning-Based Zero-Day Attack Detection: Challenges and Future Directions. Computer communications, 198, 10.1016/j.comcom.2022.11.001. https://doi.org/10.1016/J.COMCOM.2022.11.001
Halbouni, A., Gunawan, T. S., Habaebi, M. H., Halbouni, M., Kartiwi, M., & Ahmad, R. (2022). CNN-LSTM: Hybrid Deep Neural Network for Network Intrusion Detection System. IEEE Access, 10, 99837–99849. https://doi.org/10.1109/ACCESS.2022.3206425
Help Net Security. (2024, April 24). Global attacker median dwell time continues to fall - Help Net Security. https://www.helpnetsecurity.com/2024/04/24/2023-attacker-dwell-time/
Jay. (2025, Juli 28). Zero-Day Attacks Explained: What CISOs Need To Know In 2025 - Bluefire Redteam. https://bluefire-redteam.com/zero-day-attacks-explained-what-cisos-need-to-know-in-2025/
Karatas, G., Demir, O., & Sahingoz, O. K. (2020). Increasing the Performance of Machine Learning-Based IDSs on an Imbalanced and Up-to-Date Dataset. IEEE Access, 8, 32150–32162. https://doi.org/10.1109/ACCESS.2020.2973219
Khalil, M. (2025, September 28). Cybercrime 2025: $10.5T Losses & Shocking New Statistics - Deepstrike. https://deepstrike.io/blog/cybercrime-statistics-2025
Korniszuk, K., & Sawicki, B. (2024). Autoencoder-Based Anomaly Detection in Network Traffic. 2024 25th International Conference on Computational Problems of Electrical Engineering, CPEE 2024. https://doi.org/10.1109/CPEE64152.2024.10720411
Kristi, E., Tobing, A., Eka Septya, R., & Servanda, Y. (2025). Comparative Analysis of Network Security: Firewall, IDS, and AI-Based Defense Against DDoS Attacks. Journal of Artificial Intelligence and Engineering Applications (JAIEA), 4(3), 1818–1822. https://doi.org/10.59934/JAIEA.V4I3.1026
Lawall, A., & Zöller, T. (2025). Impact of Target Network-Specific Data on the Performance of Machine-Learning-Based Intrusion Detection System Models. 2025 5th Intelligent Cybersecurity Conference, ICSC 2025, 290–297. https://doi.org/10.1109/ICSC65596.2025.11140289
Layeghy, S., Baktashmotlagh, M., & Portmann, M. (2023). DI-NIDS: Domain invariant network intrusion detection system. Knowledge-Based Systems, 273, 110626. https://doi.org/10.1016/J.KNOSYS.2023.110626
Layeghy, S., & Portmann, M. (2023). Explainable Cross-domain Evaluation of ML-based Network Intrusion Detection Systems. Computers and Electrical Engineering, 108, 108692. https://doi.org/10.1016/J.COMPELECENG.2023.108692
Le, T. T. H., Shin, Y., Kim, M., & Kim, H. (2024). Towards unbalanced multiclass intrusion detection with hybrid sampling methods and ensemble classification. Applied Soft Computing, 157. https://doi.org/10.1016/j.asoc.2024.111517
Liu, L., Engelen, G., Lynar, T., Essam, D., & Joosen, W. (2022). Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018. 2022 IEEE Conference on Communications and Network Security, CNS 2022, 254–263. https://doi.org/10.1109/CNS56114.2022.9947235
Mao, J., Yang, X., Hu, B., Lu, Y., Yin, G., Mao, J., Yang, X., Hu, B., Lu, Y., & Yin, G. (2025). Intrusion Detection System Based on Multi-Level Feature Extraction and Inductive Network. Electronics 2025, Vol. 14, 14(1). https://doi.org/10.3390/ELECTRONICS14010189
Mohale, V. Z., & Obagbuwa, I. C. (2025). Evaluating machine learning-based intrusion detection systems with explainable AI: enhancing transparency and interpretability. Frontiers in Computer Science, 7, 1520741. https://doi.org/10.3389/FCOMP.2025.1520741/BIBTEX
Mohapatra, A., Jain, N., & Rudra, B. (2023). Intrusion Detection System in Networks Employing a Double-Layer Architecture Using Machine Learning Algorithms. 2023 14th International Conference on Computing Communication and Networking Technologies, ICCCNT 2023. https://doi.org/10.1109/ICCCNT56998.2023.10307540
Osamor, F., & Wellman, B. (2023). A Deep Learning-Based Hybrid Model for Optimal Anomaly Detection. Proceedings - 2023 Congress in Computer Science, Computer Engineering, and Applied Computing, CSCE 2023, 650–656. https://doi.org/10.1109/CSCE60160.2023.00111
Oyelakin, A., Ameen A.O, Ogundele T.S, Salau-Ibrahim T, Abdulrauf U.T, Olufadi H.I, Ajiboye I.K, Muhammad-Thani S, & Adeniji I. A. (2023). Overview and Exploratory Analyses of CICIDS 2017 Intrusion Detection Dataset. Journal of Systems Engineering and Information Technology (JOSEIT), 2(2), 45–52. https://doi.org/10.29207/joseit.v2i2.5411
Park, H., Shin, D., Park, C., Jang, J., Shin, D., Park, H., Shin, D., Park, C., Jang, J., & Shin, D. (2025). Unsupervised Machine Learning Methods for Anomaly Detection in Network Packets. Electronics 2025, Vol. 14, 14(14). https://doi.org/10.3390/ELECTRONICS14142779
Rahmawati, T., Karna, N., Shin, S. Y., & Putra, M. A. P. (2024). Enhancing Network Security Through Real-Time Threat Detection with Intrusion Prevention System (Case Study on Web Attack). Jurnal Ilmiah Teknik Elektro Komputer dan Informatika, 10(4), 1004–1020. https://doi.org/10.26555/JITEKI.V10I4.30380
Rosay, A., Cheval, E., Carlier, F., & Leroux, P. (2022). Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017. International Conference on Information Systems Security and Privacy, 25–36. https://doi.org/10.5220/0010774000003120
Roy, N., Tiwari, R. G., Roy, S., Agarwal, A. K., Garg, A., & Gupta, N. (2025). The Evolving Landscape of Network Threats: Classification, Defense Challenges, and Future Directions. Proceedings of 8th International Conference on Computing Methodologies and Communication, ICCMC 2025, 504–510. https://doi.org/10.1109/ICCMC65190.2025.11140963
Salih, A., & Abdulazeez, A. (2021). Evaluation of Classification Algorithms for Intrusion Detection System: A Review. Journal of Soft Computing and Data Mining, 2(1), 31–40. https://doi.org/10.30880/jscdm.2021.02.01.004
Scarfone, K., & Mell, P. (2007). Special Publication 800-94 Guide to Intrusion Detection and Prevention Systems (IDPS) Recommendations of the National Institute of Standards and Technology. https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=50951
Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP 2018 - Proceedings of the 4th International Conference on Information Systems Security and Privacy, 2018-January, 108–116. https://doi.org/10.5220/0006639801080116
Singh, A., & Jang, J. (2022). Autoencoder-based Unsupervised Intrusion Detection using Multi-Scale Convolutional Recurrent Networks. https://arxiv.org/pdf/2204.03779
Sridharan, S., Patil, S., Shobha, T., & Pai, P. (2025). Hybrid Machine Learning–Based Intrusion Detection for Zero-Day Attack Prevention in Digital Education Networks. International Journal of Safety and Security Engineering, 15(8). https://doi.org/10.18280/IJSSE.150815
Tariq, A. H. I. E., Tariq, M. B. I. E., & Lu, S. (2024). Hybrid AI-Driven Techniques for Enhancing ZeroDay Exploit Detection in Intrusion Detection System (IDS). 2024 3rd International Conference on Artificial Intelligence, Internet of Things and Cloud Computing Technology, AIoTC 2024, 156–160. https://doi.org/10.1109/AIOTC63215.2024.10748333
Verkerken, M., D’hooge, L., Wauters, T., Volckaert, B., & De Turck, F. (2021). Towards Model Generalization for Intrusion Detection: Unsupervised Machine Learning Techniques. Journal of Network and Systems Management 2021 30:1, 30(1), 12-. https://doi.org/10.1007/S10922-021-09615-7
Xin, C., & Xu, K. (2025). Cross-Dataset Transformer-IDS with Calibration and AUC Optimization (Evaluated on NSL-KDD, UNSW-NB15, CIC-IDS2017). Journal of Cyber Security, 7(1), 483–503. https://doi.org/10.32604/jcs.2025.071627
Yaddala, M., & Sunkara, Y. (2024). Comprehensive Survey of Web Security Threats in 2024. International Journal of Scientific Research in Engineering and Management. https://doi.org/10.55041/IJSREM38614
Zugec, M. (2025, Desember 4). Technical Advisory: React2Shell Critical Unauthenticated RCE in React (CVE-2025-55182) - Bitdefender. https://businessinsights.bitdefender.com/advisory-react2shell-critical-unauthenticated-rce-in-react-cve-2025-55182
