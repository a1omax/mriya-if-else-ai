FROM python:3.11-slim

WORKDIR /app

# Встановлення системних залежностей
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копіювання requirements та встановлення залежностей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання коду додатку
COPY . .

# Відкриття порту
EXPOSE 8501

# Команда запуску
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
