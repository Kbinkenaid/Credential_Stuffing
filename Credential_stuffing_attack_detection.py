# -*- coding: utf-8 -*-

# Open a log file to capture output
log_file = open("output_log.txt", "w")
def log(message):
    print(message)
    log_file.write(str(message) + "\n")
    log_file.flush()  # Ensure output is written immediately

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""### Step 1: Load Dataset"""

# Load datasets
log("Starting to load datasets...")
train_df = pd.read_csv("kdd_train.csv")
log("Train dataset loaded successfully.")
test_df = pd.read_csv("kdd_test.csv")
log("Test dataset loaded successfully.")

# First 5 rows of train dataset
log("Displaying first 5 rows of train dataset:")
log(train_df.head())

# First 5 rows of test dataset
log("Displaying first 5 rows of test dataset:")
log(test_df.head())

# Describe the train dataset
log("Describing train dataset:")
log(train_df.describe())

# Describe the test dataset
log("Describing test dataset:")
log(test_df.describe())

"""### Step 2: Preprocess"""

# Use standard column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

train_df.columns = column_names
test_df.columns = column_names

# labels in the train dataset
train_df['label'].value_counts()

# labels in the test dataset
test_df['label'].value_counts()

# Distribution of labels in train dataset
log("Creating distribution plot for train dataset...")
sns.countplot(train_df["label"])
log("Train dataset distribution plot created.")

# Distribution of labels in test dataset
log("Creating distribution plot for test dataset...")
sns.countplot(test_df["label"])
log("Test dataset distribution plot created.")

"""### Step 3: Filter only normal and credential stuffing attacks (e.g., brute force, guess_passwd)"""

# Selecting features only for credential stuffing attack
credential_attacks = ['guess_passwd', 'ftp_write', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'imap']

def simplify_label(label):
    if label == 'normal':
        return 1  # Normal
    elif label in credential_attacks:
        return -1  # Anomaly
    else:
        return -1  # Treat other attacks also as anomaly to prevent from class imbalance issue

train_df['target'] = train_df['label'].apply(simplify_label)
test_df['target'] = test_df['label'].apply(simplify_label)

# Combine train + test for consistent preprocessing
combined = pd.concat([train_df, test_df], axis=0)

# Encode categorical
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Normalize numerical features
scaler = MinMaxScaler()
features = combined.drop(columns=['label', 'target'])
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)\

# Split back to train/test
X_train = features_scaled[:len(train_df)][train_df['target'] == 1]  # Use only normal for training
X_test = features_scaled[len(train_df):]
y_test = test_df['target'].values

"""### Step 4: Testing ML models with Isolation Forest and One-Class SVM"""

# Anomaly Detection with Isolation Forest
log("\n==== Isolation Forest Results: ====")
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X_train)
y_pred_if = iso_forest.predict(X_test)
log("Classification Report:\n" + classification_report(y_test, y_pred_if, target_names=["Anomaly", "Normal"]))
log("Accuracy: " + str(accuracy_score(y_test, y_pred_if)))
log("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_if)))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_if, labels=[-1, 1])

# Plotting
plt.figure(figsize=(6, 4))
sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues',
            xticklabels=["Anomaly", "Normal"], yticklabels=["Anomaly", "Normal"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Isolation Forest (Credential Stuffing Attack Detection)")
plt.tight_layout()
plt.savefig("confusion_matrix_isoforest.png", dpi=300)
plt.show()

# Step 5: Anomaly Detection with One-Class SVM
log("\n==== One-Class SVM Results: ====")
svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm.fit(X_train)
y_pred_svm = svm.predict(X_test)
log("Classification Report:\n" + classification_report(y_test, y_pred_if, target_names=["Anomaly", "Normal"]))
log("Accuracy: " + str(accuracy_score(y_test, y_pred_svm)))
log("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_svm)))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_svm, labels=[-1, 1])
log("Confusion Matrix:\n" + str(cm))

# Plotting
plt.figure(figsize=(6, 4))
sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues',
            xticklabels=["Anomaly", "Normal"], yticklabels=["Anomaly", "Normal"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - One-Class SVM (Credential Stuffing Attack Detection)")
plt.tight_layout()
plt.savefig("confusion_matrix_oneclasssvm.png", dpi=300)
plt.show()

# Close the log file
log("Script execution completed.")
log_file.close()
