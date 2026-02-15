# State-of-the-Art Analysis: IDS & Zero-Day Detection

## 1. Hybrid DL/ML for Anomaly-Based NIDS (2025)

**Almuhanna et al.** | DOI: 10.3389/frai.2025.1625891 | Frontiers in AI

**Model:** Ensemble (GNN + LSTM + Autoencoder + XGBoost + RF) dengan weighted soft voting  
**Learning:** Hybrid (Supervised + Unsupervised + Sequential)  
**Dataset:** Dataset skala besar (tidak disebutkan spesifik)

**Limitations:**

- Tidak mengevaluasi explicit zero-day scenarios
- Perlu evaluasi berkelanjutan pada varied datasets
- Challenge: overhead komputasi, latensi, resource constraints (edge/IoT)
- Label noise memerlukan data cleaning yang cermat

---

## 2. CNN-LSTM Hybrid NIDS (2022)

**Halbouni et al.** | DOI: 10.1109/ACCESS.2022.3206 | IEEE Access

**Model:** CNN (spatial features) + LSTM (temporal features) + Batch Norm + Dropout, 3x repetisi  
**Learning:** Supervised (One-Hot Encoding)  
**Datasets:** CIC-IDS2017, UNSW-NB15, WSN-DS | Split: 80/20 | K-Fold CV

**Limitations:**

- Tingkat deteksi rendah: web attacks, worm, backdoor, analysis
- Struggle dengan ancaman baru/zero-day
- Imbalanced dataset → high false alarm rate

---

## 3. SAE-CNN for IoT (2024)

**Alsoufi et al.** | DOI: 10.32604/cmes.2024.052112 | Computer Modeling in Engineering & Sciences

**Model:** Sparse Autoencoder (dimensionality reduction) → CNN (binary classification)  
**Learning:** Hybrid (Unsupervised SAE + Supervised CNN)  
**Dataset:** BOT-IoT | 80/20 split | 5.87M train, 733K test | SMOTE balancing | **99.9% accuracy**

**Limitations:**

- Perlu validasi pada alternative datasets untuk generalisasi
- Optimisasi architecture & hyperparameter untuk efisiensi
- Perlu peningkatan adaptive capability untuk evolving attacks

---

## 4. Zero-Day Attack Detection Survey (2023)

**Guo** | DOI: 10.1016/j.comcom.2022.11.001 | Computer Communications

**Models Surveyed:**

- One-Class SVM, Autoencoder, Kitsune (ensemble AE)
- Random Forest + SVM hybrid
- GAN (diskriminator sebagai detektor)
- Transfer Learning (feature-based, domain adaptation)

**Learning:** Unsupervised, Supervised, Hybrid, TL  
**Datasets:** CIC-IDS2017, NSL-KDD, CSE-CIC-IDS2018, Kaggle Malware  
**Zero-Day Simulation:** Withhold attack classes, inject noise, mix unseen attacks

**Key Challenges:**

- **Data availability:** Zero-day samples unknown beforehand
- **Assumption:** Zero-day mirip existing attacks (need validation)
- **Feature engineering:** Memerlukan strong domain knowledge
- **Evaluation:** Tidak ada true zero-day test data, lack comprehensive benchmarks
- **Limited datasets:** 7/9 studies hanya gunakan 5 public datasets
- **Variable performance:** Akurasi bervariasi across attack types
- Transfer Learning tidak cocok untuk real-time detection

---

## 5. Hybrid ML-IDS untuk Education Networks (2025)

**Sridharan et al.** | DOI: 10.18280/ijsse.150815 | Int'l J. Safety & Security Engineering

**Model:** Deep Autoencoder (anomaly) + Random Forest (known attacks) + Hybrid decision logic  
**Learning:** Hybrid (Unsupervised AE + Supervised RF)  
**Datasets:** Digital Education Network Traffic + UNSW-NB15 + CIC-IDS2017 | CV tuning  
**Performance:** **98.7% detection rate** untuk serangan baru, low false alarm

**Limitations:**

- Kebanyakan IDS tidak tailored untuk education domain (unique traffic patterns)
- Education cybersecurity tertinggal, resource constraints
- Traditional ML poor generalization ke unseen attacks
- Signature-based IDS tidak fleksibel untuk new threats
- Anomaly-based high false positive rate
- Large labeled training sets sulit didapat

---

## 6. Signature-Based IDS + Fuzzy Clustering (2025)

**Ahmed et al.** | DOI: 10.1038/s41598-025-85866-7 | Scientific Reports

**Models:** SVM, KNN, RF, DT, LSTM (2 layers×20 units), ANN (Sequential, 2 hidden×20 neurons)  
**Learning:** Supervised + Semi-supervised (One-Class SVM)  
**Datasets:** UNSW-NB15, KDDCUP 99 | 80/20 split | Oversampling/undersampling  
**Best:** RF & SVM (flexibility, explainability), LSTM & ANN (sequential dependencies)

**Limitations:**

- **Tidak cocok untuk zero-day/advanced threats** (effective hanya untuk known risks)
- **Tidak dirancang untuk real-time detection** (critical untuk high-speed networks)
- Model mungkin tidak represent real-world diversity
- Fuzzy clustering struggle dengan highly skewed/dynamic data
- High computational cost LSTM & ANN untuk large-scale data

---

## 7. ZD-CSNN for IoT Zero-Day (2025)

**Al-Jamali et al.** | DOI: 10.1049/ntw2.70019 | IET Networks

**Model:** Convolutional Spiking Neural Network

- 3 tiers: Data Preprocessing (LDA) → CSNN Detection (2 levels, LIF neurons, 2 conv layers) → Evaluation
- Time-based classifier untuk zero-day threats

**Learning:** Hybrid (Supervised + Unsupervised)  
**Dataset:** CicioT2023 (105 IoT devices, 32 attack scenarios) | 70/30 split  
**Performance:** Outperforms ZD-DL & ZD-CNN | Metrics: Accuracy, F1, FP, FN

**Limitations:**

- Edge deployment details masih dalam development
- Perlu improve detection untuk web-based attacks (less temporal dynamic)
- Future: integrate real-time detection untuk live IoT
- Focus: evaluate computational efficiency pada resource-constrained edge devices
- Motivated oleh IoT constraints: limited processing, real-time response, lightweight

---

## 8. NIST IDPS Guide (2007)

**Scarfone et al.** | NIST Special Publication 800-94

**Types:** Network-based, Wireless, Network Behavior Analysis, Host-based  
**Comprehensive guideline** untuk design, deploy, configure, secure, monitor, maintain IDPS  
**Emphasis:** Multiple IDPS technologies untuk comprehensive detection & prevention

**Limitations:**

- **Network-based:** Tidak detect encrypted traffic, difficulty under high load, vulnerable
- **Wireless:** Tidak detect passive monitoring, offline processing; vulnerable evasion & DoS
- **NBA:** Detection lag (batch flow data transfers)
- **Host-based:** Alert delays, centralized reporting delays, significant resource usage, conflicts
- **General:** Tidak ada open standards atau up-to-date comprehensive public test suites
- Sulit compare products

---

## 9. Suricata-Based IPS untuk Web Attacks (2024)

**Rahmawati et al.** | DOI: 10.26555/jiteki.v10i4.30380 | JITEKI

**System:** Suricata IPS + ELK Stack (Elasticsearch, Logstash, Kibana) + Filebeat | Docker container  
**Detection:** Signature-based (predefined rules), bukan ML/DL  
**Test:** DVWA (SQL Injection, XSS, Command Injection) | 300 payloads  
**Performance:** **100% blocking success**, **0.902s average detection time**

**Limitations:**

- **Focus predefined rules, bukan zero-day anomaly detection**
- Target: low-resource computers, bukan enterprise-level
- Perlu optimize rules & log rotation untuk low-spec hardware (bottlenecks di high traffic)
- Future: adaptive rules dengan ML/behavioral analysis, expand historical log storage

---

## Summary Comparison Table

| Paper             | Year | Model Type                    | Learning   | Zero-Day      | Real-Time  | Best Performance      |
| ----------------- | ---- | ----------------------------- | ---------- | ------------- | ---------- | --------------------- |
| Hybrid DL/ML NIDS | 2025 | Ensemble (GNN+LSTM+AE+XGB+RF) | Hybrid     | Not tested    | Need opt   | Near-perfect          |
| CNN-LSTM          | 2022 | CNN-LSTM                      | Supervised | Limited       | Yes        | High accuracy, low FP |
| SAE-CNN IoT       | 2024 | SAE+CNN                       | Hybrid     | Yes (anomaly) | Yes        | **99.9% accuracy**    |
| Zero-Day Survey   | 2023 | Survey (Multiple)             | All        | Focus         | Varies     | Varies                |
| Education IDS     | 2025 | AE+RF                         | Hybrid     | Yes           | Yes        | **98.7% detection**   |
| Fuzzy IDS         | 2025 | SVM/RF/LSTM/ANN               | Supervised | **No**        | **No**     | Known attacks only    |
| ZD-CSNN IoT       | 2025 | Spiking NN                    | Hybrid     | **Yes**       | Edge focus | Outperforms baselines |
| NIST Guide        | 2007 | Guideline                     | N/A        | N/A           | N/A        | N/A                   |
| Suricata IPS      | 2024 | Rule-Based                    | None       | **No**        | **Yes**    | **100% block, 0.9s**  |

---

## Common Research Gaps

### 1. Zero-Day Detection

- Lack of real zero-day samples untuk training/testing
- Assumption: zero-day mirip known attacks (need validation)
- Limited dataset diversity & availability
- Difficulty evaluating true zero-day capability

### 2. Dataset Issues

- Reliance pada synthetic/outdated datasets (NSL-KDD, KDDCUP 99)
- Label noise & imbalanced data
- Limited cross-dataset validation
- Lack comprehensive benchmarks

### 3. Deployment Challenges

- Computational overhead & latency (especially edge/IoT)
- Resource constraints untuk DL models
- Scalability untuk high-speed networks
- Integration dengan existing security infrastructure

### 4. Model Generalization

- Performance varies across attack types
- Poor detection untuk certain categories (web, worm, backdoor)
- Need continuous evaluation pada evolving threats
- Limited adaptivity untuk new patterns

### 5. Evaluation & Benchmarking

- Lack standardized protocols
- Inconsistent metrics across studies
- Limited fair comparison antar methods
- Need comprehensive public test suites

---

## Promising Approaches & Future Directions

**Effective Techniques:**

1. **Hybrid Models:** Supervised (known) + Unsupervised (anomalies/zero-day)
2. **Ensemble Methods:** Multiple algorithms untuk complementary strengths
3. **Deep Learning:** LSTM (temporal), CNN (spatial), Autoencoder (anomaly)
4. **Transfer Learning:** Domain adaptation untuk limited labeled data
5. **Spiking Neural Networks:** Efficient edge deployment

**Future Work:**

1. Develop standardized benchmarks & evaluation protocols
2. Create realistic, diverse, up-to-date datasets
3. Focus lightweight models untuk resource-constrained environments
4. Enhance adaptivity & continuous learning
5. Integrate explainability untuk trust & debugging
6. Develop hybrid signature + anomaly systems
7. Address encrypted traffic analysis
8. Improve real-time processing capabilities

---

**Key Insight:** Hybrid approaches combining supervised (known attacks) + unsupervised (anomalies) show most promise untuk zero-day detection, dengan SAE-CNN (99.9%) dan Education IDS (98.7%) achieving strong results. Rule-based systems (Suricata: 100% block) effective untuk known attacks tapi tidak scalable untuk zero-day threats.
