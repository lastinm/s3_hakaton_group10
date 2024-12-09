import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# 1. Импорт необходимых библиотек

data_file = './data/geo-reviews-dataset-2023.csv'
chunk_size = 10000  # Размер блока данных
train_file = './data/train_dataset.csv'
test_file = './data/test_dataset.csv'

# Инициализация токенизатора вне цикла
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(text):
    # Токенизация текста
    return tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# Подсчет общего количества отзывов для корректного разделения на обучающий и тестовый набор
total_reviews = 0

# Сначала мы пройдемся по всем данным, чтобы получить общее количество отзывов
for chunk in pd.read_csv(data_file, chunksize=chunk_size):
    total_reviews += len(chunk)

# Обработка данных по частям и сохранение результатов
train_tokens = []
test_tokens = []

# Считываем данные в два временных файла
for chunk in pd.read_csv(data_file, chunksize=chunk_size):
    print(chunk.head())  # Проверяем текущий чанк

    # Применяем токенизацию к каждому отзыву в текущем блоке
    chunk_tokens = chunk['text'].apply(tokenize_text)

    # Преобразуем тензоры и сохраняем токены в временные файлы
    for token in chunk_tokens:
        tokenized_text = token['input_ids'].tolist()  # Получаем токены в виде списка
        train_tokens.append(tokenized_text)
        
        # Записываем в файл один за другим
        with open(train_file, 'a') as f:
            f.write(','.join(map(str, tokenized_text)) + '\n')

# Загружаем токены в DataFrame
train_df = pd.read_csv(train_file, header=None)

# Создание обучающего и тестового наборов
train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Сохраняем обучающий и тестовый наборы
train_data.to_csv(train_file, index=False, header=False, mode='w')  # Записываем снова, чтобы очистить временные данные
test_data.to_csv(test_file, index=False, header=False)

# Печатаем размеры наборов
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

