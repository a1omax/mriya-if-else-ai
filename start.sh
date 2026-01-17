#!/bin/bash

echo "🚀 Запуск Мрія AI Tutor..."
echo ""

# Перевірка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не встановлено. Встановіть Docker та спробуйте знову."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не встановлено. Встановіть Docker Compose та спробуйте знову."
    exit 1
fi

echo "✅ Docker та Docker Compose знайдено"
echo ""

# Зупинка попередніх контейнерів
echo "🛑 Зупинка попередніх контейнерів..."
docker-compose down

# Збірка та запуск
echo "🔨 Збірка контейнерів..."
docker-compose up --build -d

echo ""
echo "✅ Сервіси запущено!"
echo ""
echo "📡 Доступні URL:"
echo "   - Frontend (Streamlit): http://localhost:8501"
echo "   - Backend API: http://localhost:8000"
echo "   - API Documentation: http://localhost:8000/docs"
echo ""
echo "📊 Перегляд логів:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Зупинка сервісів:"
echo "   docker-compose down"
echo ""
