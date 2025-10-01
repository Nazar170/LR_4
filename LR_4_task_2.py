# Імпорт необхідних бібліотек
import numpy as np
from sklearn import preprocessing

# Визначення міток
# Надання позначок вхідних даних
input_labels = ['red', 'Black', 'red', 'green', 'black', 'yellow', 'white']

print("Вхідні мітки:")
print(input_labels)

# Створення кодувальника та встановлення відповідності між мітками та числами
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Виведення відображення слів на числа
print("\nВідображення міток на числа (Label mapping):")
for i, item in enumerate(encoder.classes_):
    print(item, '->', i)


# Перевірка кодувальника на тестових даних


# Перетворимо набір випадково впорядкованих міток, щоб перевірити роботу кодувальника
test_labels = ['green', 'red', 'Black']
encoded_values = encoder.transform(test_labels)
print("Мітки =", test_labels)
print("Закодовані значення =", list(encoded_values))


# Декодуємо випадковий набір чисел
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("Закодовані значення =", encoded_values)
print("Декодовані мітки =", list(decoded_list))