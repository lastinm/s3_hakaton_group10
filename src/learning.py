# Импорт необходимых библиотек
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch

# Проверка доступности CUDA
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU")
# else:
#     device = torch.device("cpu")
#     print("Using CPU")
device = torch.device('cpu')  # Устанавливаем устройство на CPU

# Загрузка предобученного токенизатора и модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)  # Перемещение модели на GPU
# tensor = torch.randn(3, 3).to(device)  # Создаем тензор на CPU

# Загрузка токенизированных данных из файла
with open('./data/train_data.csv', 'r', encoding='utf-8') as f:
       tokens = f.read().strip().split('\n')

# Соединение токенов в текст
text_data = " ".join(tokens)

# Создание датасета
dataset = Dataset.from_dict({"text": [text_data]})

# Предполагаем, что колонка 'text' уже содержит токенизированные данные, 
# поэтому мы просто создаем данные в формате, который понимает модель.
# Если в колонке 'text' хранятся токены в виде строки, их нужно преобразовать.
def encode_function(examples):
    return {"input_ids": [list(map(int, tokens.split())) for tokens in examples['text']]}

# Применение функции преобразования
encoded_datasets = dataset['train'].map(encode_function, batched=True)

# Определение аргументов для обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Создание экземпляра Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets,
)

# Обучение модели
trainer.train()