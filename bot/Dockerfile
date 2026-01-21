FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем пользователя, чтобы не запускать от root (хорошая практика)
# Но для простоты доступа к файлам можно оставить root