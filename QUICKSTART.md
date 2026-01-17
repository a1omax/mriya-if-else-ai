# 🚀 QUICKSTART - Mriia AI Tutor

## Быстрый запуск за 3 шага

### Шаг 1: Распаковать

```bash
tar -xzf mriia_hackathon_full.tar.gz
cd mriia_hackathon
```

### Шаг 2: Добавить данные (опционально)

Если у вас есть Parquet файлы, положите их в `data/`:

```bash
cp /path/to/*.parquet data/
```

Или создайте структуру:

```bash
mkdir -p data/public_data/"Lapathon2026 Mriia public files"
cp /path/to/*.parquet data/public_data/"Lapathon2026 Mriia public files"/
```

### Шаг 3: Запустить

#### Вариант А: Docker (рекомендуется)

```bash
docker-compose up --build
```

#### Вариант Б: Локально

```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
python main.py

# Terminal 2: Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Шаг 4: Открыть в браузере

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🧪 Проверка работы

```bash
# Запустить тестовый скрипт
python test_api.py
```

Ожидаемый результат:
```
✅ PASS - Health Check
✅ PASS - Subjects List
✅ PASS - Generate Material
✅ PASS - Assess Student

Total: 4/4 tests passed
🎉 All tests passed!
```

---

## 🔧 Настройка

Все настройки в `.env`:

```bash
# Lapa LLM API
LLM_API_KEY=sk-J94Etria-0A2EMmH1xp-eg

# RAG
USE_RAG=true
RAG_TOP_K=3

# Данные
CODABENCH_INPUT_DIR=./data
```

---

## 📊 Использование API

### Генерация материала

```bash
curl -X POST http://localhost:8000/api/generate-material \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Квадратні рівняння",
    "grade": 8,
    "subject": "algebra",
    "use_rag": true
  }'
```

### Оценка студента

```bash
curl -X POST http://localhost:8000/api/assess-student \
  -H "Content-Type: application/json" \
  -d '{
    "student_answers": [
      {"question_id": "q1", "selected_answer_index": 1}
    ],
    "exercises": [
      {
        "question_id": "q1",
        "question_text": "Test question",
        "answers": ["A", "B", "C", "D"],
        "correct_answer_indices": [1]
      }
    ]
  }'
```

---

## 🐛 Troubleshooting

### Backend не запускается

```bash
# Проверьте логи
docker-compose logs backend

# Или запустите вручную
cd backend
python main.py
```

### Frontend не подключается к backend

Проверьте что backend запущен:
```bash
curl http://localhost:8000/health
```

### Нет данных RAG

Это нормально! Backend будет работать без данных, но:
- RAG поиск будет пустым
- LLM все равно будет генерировать контент

Чтобы добавить данные:
```bash
mkdir -p data/public_data
# Скопируйте Parquet файлы
```

---

## 📖 Дополнительная документация

- `README.md` - Полная документация
- `docs/HOW_TO_USE_YOUR_CODE.md` - Гайд по коду
- `backend/agent.py` - Ваш LLM код
- `backend/hybrid_retriever.py` - Ваш RAG код

---

## ✅ Чеклист готовности

- [ ] Распаковал архив
- [ ] Запустил Docker / локально
- [ ] Открыл http://localhost:8501
- [ ] Проверил http://localhost:8000/health
- [ ] Запустил test_api.py
- [ ] Все тесты прошли ✅

**Готово! Система работает! 🎉**
