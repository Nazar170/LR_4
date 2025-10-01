# Створіть новий файл Python та імпортуйте наведені нижче пакети.
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from utilities import visualize_classifier

# Визначимо зразок вхідних даних за допомогою двовимірних векторів і відповідних міток.
# Визначення зразка вхідних даних
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5],
              [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9],
              [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

print("Вхідні дані X:")
print(X)
print("\nМітки y:")
print(y)

# Ми тренуватимемо класифікатор, використовуючи ці позначені дані.
# Створимо об'єкт логістичного класифікатора.
# Створення логістичного класифікатора
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

# Навчимо класифікатор, використовуючи певні дані.
# Тренування класифікатора
classifier.fit(X, y)

print("\nКласифікатор навчено!")

# Візуалізуємо результати роботи класифікатора, відстеживши межі
visualize_classifier(classifier, X, y, title='Логістична регресія (Logistic Regression)')