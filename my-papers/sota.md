Oke bro, **ini versi final 6 kolom** yang bikin learning paradigm lu stand out! 🔥

## Tabel 2.1 State-of-the-Art Penelitian Terdahulu

| Penulis                     | Model/Arsitektur                                          | Dataset                                                 | Learning Paradigm                      | Metode Evaluasi & Hasil                                                                     |
| --------------------------- | --------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------- |
| Park et al. (2025)          | Hybrid CNN-BiLSTM Autoencoder                             | CICIDS2018                                              | Unsupervised (trained on normal only)  | Accuracy: 98.1%<br>F1-Score: 98.3%                                                          |
| Almuhanna & Alahmadi (2025) | Hybrid Ensemble (XGBoost + RF + GNN + LSTM + Autoencoder) | CICIDS2017                                              | Supervised + Unsupervised (Hybrid)     | Accuracy: ~100%<br>Precision: 100%, Recall: 100%, F1-Score: 100%<br>5-fold cross-validation |
| Halbouni et al. (2022)      | Hybrid CNN-LSTM                                           | CIC-IDS2017, UNSW-NB15, WSN-DS                          | Supervised                             | CIC-IDS2017: 99.59%<br>UNSW-NB15: 93.68%<br>WSN-DS: 99.64%                                  |
| Singh & Jang (2022)         | MSCNN-LSTM-AE (Multi-Scale CNN-LSTM Autoencoder)          | NSL-KDD, UNSW-NB15, CICDDoS2019                         | Unsupervised                           | CICDDoS2019: Accuracy: 99.56%, Precision: 98.91%, Recall: 98.81%, F1-Score: 98.46%          |
| Baidar et al. (2025)        | Hybrid Deep Learning-Federated Learning (CNN-BiLSTM-AE)   | UNSW-NB15                                               | Supervised + Federated Learning        | Accuracy: 97.12%, Precision: 98.45%, Recall: 96.29%, F1-Score: 97.36%, AUC-ROC: 99.59%      |
| **Proposed**                | **Hybrid CNN-LSTM Autoencoder**                           | **CIC-IDS2017 (training)<br>CSE-CIC-IDS2018 (testing)** | **Unsupervised (normal traffic only)** | **Accuracy, Precision, Recall, F1-Score, AUC-ROC, False Positive Rate (FPR)**               |

---

Full

| Penulis                     | Task                                      | Model/Arsitektur                                          | Dataset                                                 | Learning Paradigm                      | Metode Evaluasi & Hasil                                                                     |
| --------------------------- | ----------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------- |
| Park et al. (2025) [1]      | Zero-day intrusion detection              | Hybrid CNN-BiLSTM Autoencoder                             | CICIDS2018                                              | Unsupervised (trained on normal only)  | Accuracy: 98.1%<br>F1-Score: 98.3%                                                          |
| Almuhanna & Alahmadi (2025) | Anomaly-based network intrusion detection | Hybrid Ensemble (XGBoost + RF + GNN + LSTM + Autoencoder) | CICIDS2017                                              | Supervised + Unsupervised (Hybrid)     | Accuracy: ~100%<br>Precision: 100%, Recall: 100%, F1-Score: 100%<br>5-fold cross-validation |
| Halbouni et al. (2022)      | Network intrusion detection               | Hybrid CNN-LSTM                                           | CIC-IDS2017, UNSW-NB15, WSN-DS                          | Supervised                             | CIC-IDS2017: 99.59%<br>UNSW-NB15: 93.68%<br>WSN-DS: 99.64%                                  |
| Singh & Jang (2022) [2]     | Unsupervised anomaly detection            | MSCNN-LSTM-AE (Multi-Scale CNN-LSTM Autoencoder)          | NSL-KDD, UNSW-NB15, CICDDoS2019                         | Unsupervised                           | CICDDoS2019: Accuracy: 99.56%, Precision: 98.91%, Recall: 98.81%, F1-Score: 98.46%          |
| Baidar et al. (2025) [3]    | IoT/5G intrusion detection                | Hybrid Deep Learning-Federated Learning (CNN-BiLSTM-AE)   | UNSW-NB15                                               | Supervised + Federated Learning        | Accuracy: 97.12%, Precision: 98.45%, Recall: 96.29%, F1-Score: 97.36%, AUC-ROC: 99.59%      |
| **Proposed**                | **Zero-day intrusion detection**          | **Hybrid CNN-LSTM Autoencoder**                           | **CIC-IDS2017 (training)<br>CSE-CIC-IDS2018 (testing)** | **Unsupervised (normal traffic only)** | **Accuracy, Precision, Recall, F1-Score, AUC-ROC, False Positive Rate (FPR)**               |

---

## Key Advantages dengan 6 Kolom Ini:

### 1. **Learning Paradigm Jelas Terpisah** ✅

- **Park & Proposed**: Pure unsupervised (realistic untuk zero-day)[1]
- **Almuhanna & Halbouni**: Hybrid (supervised + unsupervised)
- **Baidar**: Supervised + FL (butuh attack labels)
- **Singh**: Unsupervised tapi no cross-dataset[4]

### 2. **Dataset Kolom Lebih Clean** ✅

- Fokus pada **nama dataset** aja
- **Cross-dataset validation lu** super jelas: CIC-IDS2017 → CSE-CIC-IDS2018

### 3. **Research Positioning Kuat** ✅

Dari tabel langsung keliatan:

- Lu **salah satu dari 3** yang pure unsupervised (Park, Singh, Proposed)
- Tapi **HANYA LU** yang pakai cross-dataset validation
- **HANYA LU** yang include FPR dalam evaluasi

### 4. **Visual Comparison** ✅

Reader bisa langsung compare:

- **Supervised** (Baidar) → Butuh attack labels ❌ untuk zero-day
- **Hybrid** (Almuhanna, Halbouni) → Masih pakai supervised components
- **Pure Unsupervised** (Park, Singh, **Proposed**) → Realistic untuk zero-day ✅

---

## Perfect! Siap dipake bro! 🚀

Struktur 6 kolom ini **optimal** karena:

1. Penulis (wajib)
2. Task (konteks)
3. Model (technical approach)
4. Dataset (data yang dipakai)
5. **Learning Paradigm** ← **Core differentiator lu!**
6. Evaluasi & Hasil (performa)

Tinggal lu jelasin gap-nya di paragraf setelah tabel! 🔥

[1](https://www.academia.edu/143943862/Hybrid_CNN_LSTM_Model_for_Real_Time_Detection_of_Zero_Day_Attacks_in_Heterogeneous_IoT_Networks)
[2](https://ssrpublisher.com/wp-content/uploads/2025/09/Neural-Networks-and-Deep-Learning-Models-in-Intrusion-Detection-Systems.pdf)
[3](https://ictactjournals.in/paper/IJCT_Vol_16_Iss_3_Paper_15_3677_3685.pdf)
[4](https://arxiv.org/pdf/2204.03779.pdf)
