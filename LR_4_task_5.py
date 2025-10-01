import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------------
# 1. Завантаження CSV
# -------------------------------
df = pd.read_csv("data_metrics.csv")
print("Перші рядки з файлу:")
print(df.head())

# -------------------------------
# 2. Отримуємо дані з CSV
# -------------------------------
y_true = df["actual_label"].values
y_scores_RF = df["model_RF"].values
y_scores_LR = df["model_LR"].values

# Поріг 0.5 для початкових передбачень
y_pred_RF = (y_scores_RF >= 0.5).astype(int)
y_pred_LR = (y_scores_LR >= 0.5).astype(int)

# -------------------------------
# 3. Метрики (Goncharenko)
# -------------------------------
def goncharenko_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def goncharenko_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def goncharenko_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def goncharenko_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def goncharenko_conf_matrix_values(y_true, y_pred):
    TP = goncharenko_TP(y_true, y_pred)
    FN = goncharenko_FN(y_true, y_pred)
    FP = goncharenko_FP(y_true, y_pred)
    TN = goncharenko_TN(y_true, y_pred)
    return TP, FN, FP, TN

def goncharenko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = goncharenko_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def goncharenko_accuracy(y_true, y_pred):
    TP, FN, FP, TN = goncharenko_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def goncharenko_recall(y_true, y_pred):
    TP, FN, FP, TN = goncharenko_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def goncharenko_precision(y_true, y_pred):
    TP, FN, FP, TN = goncharenko_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

# -------------------------------
# 7. F1-score
# -------------------------------
def goncharenko_f1_score(y_true, y_pred):
    recall = goncharenko_recall(y_true, y_pred)
    precision = goncharenko_precision(y_true, y_pred)
    return (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

# -------------------------------
# 4. Перевірка результатів для порогу 0.5
# -------------------------------
print("\nConfusion Matrix RF:")
print(goncharenko_confusion_matrix(y_true, y_pred_RF))
print("\nConfusion Matrix LR:")
print(goncharenko_confusion_matrix(y_true, y_pred_LR))

print("\nМетрики для RF:")
print("Accuracy:", goncharenko_accuracy(y_true, y_pred_RF))
print("Recall:", goncharenko_recall(y_true, y_pred_RF))
print("Precision:", goncharenko_precision(y_true, y_pred_RF))
print("F1:", goncharenko_f1_score(y_true, y_pred_RF))

print("\nМетрики для LR:")
print("Accuracy:", goncharenko_accuracy(y_true, y_pred_LR))
print("Recall:", goncharenko_recall(y_true, y_pred_LR))
print("Precision:", goncharenko_precision(y_true, y_pred_LR))
print("F1:", goncharenko_f1_score(y_true, y_pred_LR))

# -------------------------------
# 8. Тестування різних порогів
# -------------------------------
thresholds = [0.5, 0.25]

for thresh in thresholds:
    print(f"\nМетрики для порога = {thresh}")
    pred_RF_thresh = (y_scores_RF >= thresh).astype(int)
    pred_LR_thresh = (y_scores_LR >= thresh).astype(int)

    print("Random Forest:")
    print("Accuracy:", goncharenko_accuracy(y_true, pred_RF_thresh))
    print("Recall:", goncharenko_recall(y_true, pred_RF_thresh))
    print("Precision:", goncharenko_precision(y_true, pred_RF_thresh))
    print("F1:", goncharenko_f1_score(y_true, pred_RF_thresh))

    print("Logistic Regression:")
    print("Accuracy:", goncharenko_accuracy(y_true, pred_LR_thresh))
    print("Recall:", goncharenko_recall(y_true, pred_LR_thresh))
    print("Precision:", goncharenko_precision(y_true, pred_LR_thresh))
    print("F1:", goncharenko_f1_score(y_true, pred_LR_thresh))

# -------------------------------
# ROC-криві та AUC
# -------------------------------
fpr_RF, tpr_RF, _ = roc_curve(y_true, y_scores_RF)
fpr_LR, tpr_LR, _ = roc_curve(y_true, y_scores_LR)

auc_RF = roc_auc_score(y_true, y_scores_RF)
auc_LR = roc_auc_score(y_true, y_scores_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-криві моделей")
plt.grid(True)
plt.tight_layout()
plt.show()
