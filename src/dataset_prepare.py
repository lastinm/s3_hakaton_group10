import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

# Загрузка данных из файла
input_file = './data/geo-reviews-dataset-2023.tskv'

# Определяем функцию для обработки строк
def process_line(line):
    # Разбиваем строку по символу табуляции
    parts = line.strip().split("\t")
    entry = {}
    for part in parts:
        # Разделяем ключ и значение
        key_value = part.split("=", 1)  # разделяем по первому "="
        if len(key_value) != 2:
            continue  # если нет корректной пары, пропускаем
        key, value = key_value
        key = key.strip()
        value = value.strip()
        
        # Проверяем, является ли значение рейтинга float и преобразуем в int
        if key == "rating":  # Замените "rating" на фактический ключ рейтинга
            try:
                entry[key] = int(float(value))  # Преобразуем в float, затем в int
            except ValueError:
                entry[key] = None  # В случае ошибки сохраняем None
        else:
            entry[key] = value  # Остальные значения сохраняем
        
    return entry

# Чтение файла
with open(input_file, 'r', encoding='utf-8') as f:
    # Читаем все строки и обрабатываем их
    data = [process_line(line) for line in f if line.strip()]

# Удаление пустых записей (если такие имеются)
data = [entry for entry in data if entry]

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Удаление строк с пропущенными значениями
df.dropna(inplace=True)

# ===== Токенизация текстовых данных
# Пример для поля name_ru
df['name_ru'] = df['name_ru'].str.lower().str.split()

# ====== LabelEncoder
# Регулярное выражение для разделения строк по , и ;
df_expanded = df['rubrics'].str.replace(';', ',', regex=False).str.split(',', expand=True).stack().reset_index(drop=True)

# Удаление пробелов от начала и конца после разделения
df_expanded = df_expanded.str.strip()

# Создание объекта LabelEncoder
label_encoder = LabelEncoder()

# Обучение и преобразование категорий
encoded_labels = label_encoder.fit_transform(df_expanded)

# Создание DataFrame с закодированными метками
encoded_df = pd.DataFrame({'rubrics_encoded': encoded_labels})

# Добавление закодированного столбца обратно в исходный DataFrame
# Для этого необходимо создать общий индекс для связывания с исходным DataFrame
df_exploded = df.loc[df.index.repeat(df['rubrics'].str.count(';') + 1)].reset_index(drop=True)
df_exploded['rubrics'] = df_expanded

# Объединяем закодированные метки с взрывным DataFrame
df_exploded = df_exploded.join(encoded_df)

# Удаляем столбец 'rubrics'
df_exploded = df_exploded.drop(columns=['rubrics'])

# Сохраняем в CSV
output_file = './data/geo-reviews-dataset-2023.csv'
df_exploded.to_csv(output_file, index=False, encoding='utf-8', sep='|')


