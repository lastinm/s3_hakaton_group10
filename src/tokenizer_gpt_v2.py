import pandas as pd
from transformers import GPT2Tokenizer

# Инициализация токенизатора
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Параметры
input_file = './data/geo-reviews-dataset-2023.csv'  # Путь к вашему входному файлу с отзывами
output_file = './data/tokenized_reviews.csv'  # Путь к выходному файлу для токенов
chunksize = 1000  # Размер чанка для обработки

# Открытие файла для записи токенов и оригинальных данных
with open(output_file, 'a', encoding='utf-8') as out_file:
    # Чтение данных по частям
    for chunk in pd.read_csv(input_file, sep='|', chunksize=chunksize, encoding='utf-8'):
        # Токенизация отзывов
        for index, row in chunk.iterrows():
            name_ru = row['name_ru']
            rating = row['rating']
            rubrics = row['rubrics']
            text = row['text']
   

            # Токенизация текста с добавлением специальных токенов
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True)

            # Преобразование токенов в строку
            tokens_str = ' '.join(map(str, tokens))

            # Запись в файл: токены, название точки, рейтинг, рубрики
            out_file.write(f"{name_ru}|{rating}|{rubrics}|{tokens_str}\n")

print("Токенизация завершена и данные сохранены в файле:", output_file)
