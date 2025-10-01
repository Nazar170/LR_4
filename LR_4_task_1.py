# Імпорт необхідних бібліотек
import numpy as np
from sklearn import preprocessing

# Визначення вихідних даних
input_data = np.array([[1.3, -3.9, 6.5],
                       [-4.9, -2.2, 1.3],
                       [2.2, 6.5, -6.1],
                       [-5.4, -1.4, 2.2]])

print("Вихідні дані:")
print(input_data)

#  Бінарізація

data_binarized = preprocessing.Binarizer(threshold=1.1).transform(input_data)
print("Бінаризовані дані (поріг = 1.1):")
print(data_binarized)


#  Виключення середнього (Standardization)



# Виведення статистик ДО обробки
print("\nДО обробки:")
print("Середнє значення =", input_data.mean(axis=0))
print("Стандартне відхилення =", input_data.std(axis=0))

# Виключення середнього та масштабування до одиничної дисперсії
data_scaled = preprocessing.scale(input_data)
print("\nПІСЛЯ обробки:")
print("Середнє значення =", data_scaled.mean(axis=0))
print("Стандартне відхилення =", data_scaled.std(axis=0))



#  Масштабування (Min-Max Scaling)

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("Масштабовані дані (MinMax [0, 1]):")
print(data_scaled_minmax)


#  Нормалізація (Normalization)


# Нормалізація L1
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
print("L1-нормалізовані дані:")
print(data_normalized_l1)



# Нормалізація L2
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL2-нормалізовані дані:")
print(data_normalized_l2)




