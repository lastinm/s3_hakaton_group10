import pandas as pd
from sklearn.model_selection import train_test_split

# Параметры
input_file = './data/tokenized_reviews.csv'  # Путь к входному файлу
train_file = './data/train_data.csv'  # Путь для сохранения тренировочного набора
test_file = './data/test_data.csv'  # Путь для сохранения тестового набора
test_size = 0.2  # Размер тестовой выборки

# Чтение данных из файла
data = pd.read_csv(input_file, sep='|', header=None, encoding='utf-8', names=['address', 'name_ru', 'rating', 'rubrics', 'text', 'tokens'])

# Удаление ненужных столбцов
data = data[['name_ru', 'rating', 'rubrics', 'tokens']]  # Оставляем только нужные столбцы

# Деление данных на тренировочную и тестовую выборки с использованием train_test_split
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# Сохранение результатов в файлы
train_data.to_csv(train_file, sep='|', header=False, index=False)
test_data.to_csv(test_file, sep='|', header=False, index=False)

print(f"Тренировочный набор сохранен в {train_file}, тестовый набор сохранен в {test_file}.")