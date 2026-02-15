# BAB 1 PENDAHULUAN

## 1.1 Latar Belakang

Vulnerabilitas CVE-2025-55182 atau React2Shell yang baru saja diumumkan pada Desember 2025 menjadi bukti nyata bahwa sistem pertahanan tradisional masih kewalahan menghadapi ancaman zero-day. Dengan skor CVSS sempurna 10.0, celah ini berhasil mengeksploitasi puluhan organisasi dalam waktu kurang dari 24 jam (Bitdefender, 2025). Data menunjukkan adanya 77.664 IP address yang rentan, dimana hal ini memperlihatkan kegagalan fatal dari sistem signature-based dalam merespons ancaman cepat. Eksploitasi ini menyerang komponen React Server Components yang memungkinkan penyerang melakukan Remote Code Execution tanpa autentikasi. Yang lebih mengkhawatirkan adalah ketersediaan 72 alat Proof-of-Concept gratis di hari pertama, yang mana kondisi ini membuat serangan menjadi sangat mudah dilakukan (Phoenix Security, 2026). Fakta bahwa tingkat keberhasilan eksploitasi mencapai hampir 100% pada konfigurasi default menunjukkan betapa mendesaknya kebutuhan akan metode deteksi yang tidak bergantung pada signature database.

Di sisi lain, eskalasi ancaman siber global terus meningkat dengan proyeksi kerugian mencapai $10,5 triliun pada tahun 2025 (Deepstrike, 2025). Tahun 2024 saja mencatat 75 kerentanan zero-day baru, dimana 44% diantaranya secara spesifik menargetkan sistem enterprise (Bluefire Redteam, 2025). Yang menarik adalah tren serangan yang kini menyasar perangkat keamanan seperti firewall dan VPN, yang mana perangkat ini seringkali memiliki akses tinggi namun minim pengawasan. Kompleksitas infrastruktur modern yang menggabungkan cloud dan IoT semakin memperluas attack surface yang ada. Masalah semakin pelik ketika melihat kesenjangan antara dwell time rata-rata 10 hari dengan proses eksfiltrasi data yang bisa terjadi dalam 24 jam (Help Net Security, 2024). Kondisi ini mengindikasikan bahwa organisasi tidak lagi bisa mengandalkan mekanisme pertahanan reaktif yang lambat dalam mendeteksi intrusi.

Intrusion Detection System (IDS) pada dasarnya adalah perangkat lunak yang dirancang khusus untuk memonitor lalu lintas jaringan guna mendeteksi aktivitas mencurigakan yang dapat membahayakan sistem (Scarfone & Mell, 2007). Sistem ini bekerja dengan menganalisis trafik secara real-time dan memberikan peringatan dini kepada administrator agar dapat segera ditindaklanjuti (Rahmawati et al., 2024). Sebagai garda pertahanan, IDS berperan vital dalam menjaga aspek Confidentiality, Integrity, dan Availability dari serangan siber. Namun demikian, efektivitas IDS tradisional kini mulai dipertanyakan mengingat kemampuannya yang terbatas dalam menghadapi serangan terotomatisasi seperti React2Shell. Serangan modern yang mampu bermutasi dengan cepat menuntut adanya integrasi teknik deteksi yang lebih adaptif, yang mana pendekatan lama dirasa sudah tidak lagi mumpuni untuk menangani ancaman zero-day yang semakin canggih.

Signature-based IDS bekerja dengan cara mencocokkan lalu lintas jaringan dengan database pola serangan yang sudah diketahui (Ahmed et al., 2025). Meskipun sangat akurat untuk ancaman lama, metode ini terbukti gagal total dengan tingkat deteksi 0% terhadap serangan zero-day karena belum adanya signature yang tersedia saat serangan terjadi (Sridharan et al., 2025). Ketergantungan pada pembaruan signature menciptakan jeda waktu berbahaya, dimana penyerang bisa bebas beraksi selama periode "blind spot" tersebut. Kasus React2Shell menjadi contoh nyata, yang mana puluhan ribu sistem terinfeksi hanya dalam 24 jam sementara vendor keamanan baru merilis signature. Selain beban operasional yang tinggi, kelemahan fundamental ini menegaskan bahwa pendekatan reaktif tidak lagi cukup untuk melindungi aset kritikal enterprise.

Sebagai alternatif, Anomaly-based Intrusion Detection System (AIDS) menawarkan pendekatan proaktif dengan mempelajari pola normal jaringan untuk mendeteksi penyimpangan (Gu, 2023). Sistem hybrid berbasis anomali dilaporkan mampu mencapai deteksi diatas 95% terhadap ancaman baru tanpa perlu menunggu update signature (Sridharan et al., 2025). Meskipun menjanjikan, deteksi anomali konvensional seringkali kesulitan menangani data jaringan berdimensi tinggi dan masalah ketidakseimbangan kelas. Oleh karena itu, penerapan teknologi deep learning menjadi solusi yang sangat potensial untuk meningkatkan akurasi deteksi (Alsoufi et al., 2024). Hal ini menunjukkan bahwa kombinasi antara pendekatan anomali dengan deep learning adalah kunci untuk mengatasi keterbatasan IDS tradisional dalam menghadapi serangan yang berkembang cepat.

Integrasi model deep learning terbukti sangat efektif dalam mengenali pola rumit pada lalu lintas jaringan. Arsitektur CNN-LSTM misalnya, memanfaatkan CNN untuk mengekstrak fitur spasial dan LSTM untuk menangkap pola urutan waktu, yang mana kombinasi ini mampu mencapai akurasi 99,64% di dataset CIC-IDS2017 (Halbouni dkk., 2022). Sementara itu, Autoencoder unggul dalam mendeteksi anomali tanpa label serangan dengan cara mengukur reconstruction error dari data input (Alsoufi et al., 2024). Penggabungan supervised dan unsupervised learning ini menghasilkan sistem pertahanan yang tangguh. Dari berbagai penelitian yang ada, dapat dilihat bahwa pendekatan hybrid ini memiliki potensi besar untuk mendeteksi serangan zero-day secara real-time sebelum kerusakan fatal terjadi.

Berdasarkan permasalahan tersebut, penelitian ini mengusulkan implementasi Hybrid CNN-LSTM Autoencoder untuk mendeteksi anomali zero-day pada lingkungan enterprise. Berbeda dengan pendekatan signature-based yang gagal merespons React2Shell, metode ini menggunakan unsupervised learning untuk mengenali deviasi trafik secara mandiri (Almuhanna dkk., 2025). Model ini akan dilatih menggunakan dataset CIC-IDS2017 dan diuji validitasnya pada CSE-CIC-IDS2018 melalui dua skenario cross-dataset validation, yaitu zero-shot (tanpa adaptasi) dan few-shot adaptation (adaptasi unsupervised menggunakan 1% data benign target). Fokus utamanya adalah mengisi celah penelitian sebelumnya yang seringkali hanya menggunakan satu dataset saja. Dengan target akurasi diatas 90% pada data baru, sistem ini diharapkan mampu memberikan perlindungan proaktif dan mengurangi dwell time serangan secara signifikan.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah dipaparkan, rumusan masalah dalam penelitian ini adalah sebagai berikut:

1. Bagaimana performa model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan siber pada dataset CIC-IDS2017 dibandingkan dengan baseline methods?

2. Bagaimana kemampuan generalisasi model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan zero-day melalui cross-dataset validation menggunakan dataset CSE-CIC-IDS2018 dengan perbandingan skenario zero-shot dan few-shot adaptation?

## 1.3 Tujuan Penelitian

Berdasarkan rumusan masalah di atas, berikut adalah tujuan dari penelitian ini:

1. Mengevaluasi performa model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan siber pada dataset CIC-IDS2017 dan membandingkannya dengan baseline methods.

2. Menganalisis kemampuan generalisasi model Hybrid CNN-LSTM Autoencoder dalam mendeteksi serangan zero-day melalui cross-dataset validation menggunakan dataset CSE-CIC-IDS2018 dengan membandingkan skenario zero-shot dan few-shot adaptation.

## 1.4 Manfaat Penelitian

Adapun manfaat dari penelitian yang akan dilakukan, sebagai berikut:

1. Manfaat Teoretis:
   a. Memberikan kontribusi metode baru dalam pengembangan anomaly-based Intrusion Detection System menggunakan pendekatan Hybrid CNN-LSTM Autoencoder untuk deteksi zero-day attacks. b. Menambah literatur dan pemahaman mengenai penerapan deep learning untuk deteksi anomali zero-day pada network traffic time-series data.
   b. Menambah literatur dan pemahaman mengenai penerapan deep learning pada deteksi anomali lalu lintas jaringan tanpa bergantung pada signature database. d. Menjadi referensi bagi penelitian serupa di bidang keamanan siber dan kecerdasan buatan, khususnya dalam implementasi unsupervised learning untuk threat detection.
   c. Menyediakan bukti empiris mengenai kemampuan generalisasi model deep learning melalui cross-dataset validation dalam domain cybersecurity.

2. Manfaat Praktis:
   a. Menghasilkan model Intrusion Detection System yang mampu mendeteksi zero-day secara real-time tanpa bergantung pada signature database.
   b. Membantu organisasi enterprise dalam memperkuat pertahanan jaringan dengan mengurangi dwell time deteksi serangan.
   c. Mengurangi beban operasional dan biaya pemeliharaan sistem keamanan dengan meminimalkan ketergantungan pada pembaruan signature secara manual.

## 1.5 Batasan Penelitian

Dalam pelaksanaan penelitian, batasan yang ditetapkan sebagai berikut:

1. Dataset yang digunakan terbatas pada CIC-IDS2017 dan CSE-CIC-IDS2018 yang merupakan dataset publik hasil simulasi jaringan enterprise, bukan real-time traffic dari lingkungan produksi.
2. Arsitektur model yang dikembangkan terbatas pada Hybrid CNN-LSTM Autoencoder tanpa melakukan eksplorasi terhadap arsitektur lainnya seperti Transformer, GRU, atau Attention Mechanism.
3. Perbandingan performa model dibatasi pada dua baseline methods, yaitu LSTM Autoencoder dan Traditional Machine Learning (Isolation Forest/Random Forest), tanpa membandingkan dengan metode Signature-based IDS atau anomaly detection lainnya.
4. Proses training model menggunakan pendekatan unsupervised learning yang hanya memanfaatkan data normal traffic (benign), tanpa menggunakan labelled attack data dalam fase training.
5. Evaluasi performa model dibatasi pada binary classification (Normal vs Attack), dengan metrik utama F1-score dan False Positive Rate (FPR), serta metrik pendukung accuracy, precision, recall, AUC-ROC, dan generalization metrics, tanpa melakukan klasifikasi multi-class untuk jenis serangan spesifik.
6. Few-shot adaptation dibatasi pada fine-tuning unsupervised menggunakan 1% data benign dari domain target (CSE-CIC-IDS2018), tanpa menggunakan label attack dan tanpa overlap dengan data evaluasi.
7. Implementasi sistem dilakukan dalam environment offline untuk eksperimen dan evaluasi, bukan deployment real-time pada infrastruktur jaringan enterprise yang aktif.

## 1.6 Struktur Organisasi Skripsi

Penulisan skripsi ini disusun dengan sistematika sebagai berikut:

1. **BAB I PENDAHULUAN**  
   Bab ini menguraikan latar belakang masalah, rumusan masalah penelitian, tujuan yang ingin dicapai, manfaat penelitian, batasan penelitian, dan sistematika penulisan.

2. **BAB II TINJAUAN PUSTAKA**  
   Bab ini membahas teori-teori dan konsep yang relevan dengan topik penelitian, serta kajian terhadap penelitian-penelitian sebelumnya yang berkaitan untuk menggambarkan posisi dan kontribusi penelitian ini dalam state-of-the-art.

3. **BAB III METODOLOGI PENELITIAN**  
   Bab ini menjelaskan pendekatan dan metode penelitian yang digunakan, serta memaparkan tahapan-tahapan pelaksanaan penelitian secara sistematis dari tahap awal sampai tahap akhir.

4. **BAB IV HASIL DAN PEMBAHASAN**  
   Bab ini menyajikan hasil eksperimen dan temuan penelitian, serta pembahasan mendalam yang menjawab rumusan masalah yang telah ditetapkan.

5. **BAB V PENUTUP**  
   Bab ini memaparkan kesimpulan berdasarkan hasil penelitian, implikasi dari temuan penelitian, serta saran untuk pengembangan penelitian selanjutnya.
