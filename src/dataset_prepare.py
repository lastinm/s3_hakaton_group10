import pandas as pd

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

# Сохраняем в CSV
output_file = './data/geo-reviews-dataset-2023.csv'
df.to_csv(output_file, index=False, encoding='utf-8', sep='|')


