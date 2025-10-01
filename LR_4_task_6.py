import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# -----------------------------
# 1. Завантажуємо дані
# -----------------------------
data = np.loadtxt("data_multivar_nb.txt")
X = data[:, :-1]  # всі колонки, крім останньої
y = data[:, -1]   # остання колонка - мітка класу

# -----------------------------
# 2. Розділяємо на train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 3. SVM
# -----------------------------
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Метрики для SVM
print("Метрики для SVM:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_svm), 3))
print("Recall:", round(recall_score(y_test, y_pred_svm), 3))
print("Precision:", round(precision_score(y_test, y_pred_svm), 3))
print("F1-score:", round(f1_score(y_test, y_pred_svm), 3))
print()

# -----------------------------
# 4. Naive Bayes
# -----------------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Метрики для Naive Bayes
print("Метрики для Naive Bayes:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_nb), 3))
print("Recall:", round(recall_score(y_test, y_pred_nb), 3))
print("Precision:", round(precision_score(y_test, y_pred_nb), 3))
print("F1-score:", round(f1_score(y_test, y_pred_nb), 3))



