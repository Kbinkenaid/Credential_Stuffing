# Credential Stuffing Attack Detection using Anomaly Detection

This project demonstrates the detection of **credential stuffing attacks** using machine learning-based **anomaly detection** techniques, specifically **Isolation Forest** and **One-Class SVM**, applied to the KDD Cup 1999 dataset.

---

## 🧠 Overview

Credential stuffing involves using stolen credentials to gain unauthorized access to systems. This project treats such behavior as anomalous by training models on *normal traffic only* and testing their ability to flag malicious attempts as anomalies.

---

## 📊 Models Used

### ✅ Isolation Forest
- Detects anomalies based on tree-based partitioning
- Automatically estimates contamination level

### ✅ One-Class SVM
- Uses kernel methods to model the boundary of normal data
- Flags outliers as potential attacks

---

## 📂 Dataset

Uses the [KDD Cup 1999 dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html):
- `kdd_train.csv` for training
- `kdd_test.csv` for evaluation

Attack types treated as **credential stuffing** include:
- `guess_passwd`, `ftp_write`, `multihop`, `phf`, `spy`, `warezclient`, `warezmaster`, `imap`

All attacks (including those not listed) are treated as anomalies (`-1`), and normal data is labeled as `1`.

---

## ⚙️ Preprocessing

- Label encoding of categorical fields (`protocol_type`, `service`, `flag`)
- Min-Max normalization of numerical values
- Training performed only on **normal traffic**

---

## 🚀 How to Run

1. Download the dataset files `kdd_train.csv` and `kdd_test.csv` and place them in your working directory.
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python detect_credential_stuffing.py
```

4. Output will include classification reports and accuracy metrics for both models.
5. Confusion matrices will be saved as:
   - `confusion_matrix_isoforest.png`
   - `confusion_matrix_oneclasssvm.png`

---

## 📦 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- This approach simulates real-world scenarios where only benign traffic is known in advance.
- Visualization of confusion matrices helps interpret the model performance.
