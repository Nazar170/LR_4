# Створення наївного байєсівського класифікатора
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# Вхідний файл, який містить дані
input_file = 'аываываыв.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Створення наївного байєсовського класифікатора
classifier = GaussianNB()

# Тренування класифікатора
classifier.fit(X, y)

# Прогнозування значень для тренувальних даних
y_pred = classifier.predict(X)

# Обчислення якості класифікатора
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

# Візуалізація результатів роботи класифікатора
visualize_classifier(classifier, X, y)

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print("Розмір тренувальної вибірки:", X_train.shape)
print("Розмір тестової вибірки:", X_test.shape)

# Створення та тренування нового класифікатора
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# Обчислення якості класифікатора
accuracy_test = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy_test, 2), "%")

# Візуалізація роботи класифікатора на тестових даних
visualize_classifier(classifier_new, X_test, y_test, title='Наївний байєсівський класифікатор (Тестові дані)')

# Кількість блоків для перехресної перевірки
num_folds = 3

# Створення нового класифікатора для перехресної перевірки
classifier_cv = GaussianNB()

# Обчислення якості (accuracy) з використанням перехресної перевірки
accuracy_values = cross_val_score(classifier_cv, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy (cross-validation): " + str(round(100 * accuracy_values.mean(), 2)) + "%")

# Обчислення точності (precision) з використанням перехресної перевірки
precision_values = cross_val_score(classifier_cv, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision (cross-validation): " + str(round(100 * precision_values.mean(), 2)) + "%")

# Обчислення повноти (recall) з використанням перехресної перевірки
recall_values = cross_val_score(classifier_cv, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall (cross-validation): " + str(round(100 * recall_values.mean(), 2)) + "%")

# Обчислення F1-міри з використанням перехресної перевірки
f1_values = cross_val_score(classifier_cv, X, y, scoring='f1_weighted', cv=num_folds)
print("F1 (cross-validation): " + str(round(100 * f1_values.mean(), 2)) + "%")