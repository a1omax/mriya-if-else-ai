# 🎓 Mriia AI Tutor — Lapathon 2026

**Повне рішення на базі коду з Lapathon3.zip**

## ✅ Що включено

### Backend (FastAPI)

* ✅ **agent.py** — код з LLM та RAG
* ✅ **hybrid_retriever.py** — просунутий retriever
* ✅ **main.py** — FastAPI-обгортка над основною логікою

### Frontend (Streamlit)

* ✅ UI для вчителя та учня
* ✅ Два режими роботи

### Дані

* ✅ Підтримка Parquet-файлів
* ✅ Сумісність з наявною структурою даних

## 🚀 Швидкий старт

### Варіант 1: Docker

```bash
# 1. Розпакувати
tar -xzf mriia_hackathon.tar.gz
cd mriia_hackathon

# 2. Помістити Parquet-файли в data/
cp /шлях/до/*.parquet data/

# 3. Запустити
docker-compose up --build

# 4. Відкрити
http://localhost:8501  # Frontend
http://localhost:8000  # Backend API
http://localhost:8000/docs  # Swagger
```

### Варіант 2: Локально

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend (в іншому терміналі)
cd frontend  
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Структура проєкту

```
mriia_hackathon/
├── backend/
│   ├── agent.py              # Основна логіка з архіву
│   ├── hybrid_retriever.py   # Retriever з архіву
│   ├── main.py               # FastAPI-обгортка
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── data/                     # Parquet-файли
├── .env                      # Налаштування LLM
├── docker-compose.yml
└── README.md
```

## 🔧 Як це працює

### 1. RAG на основі існуючого коду

```python
from agent import _retrieve_context, _get_llm_reasoning

context = _retrieve_context(
    question_text=topic,
    subject=subject,
    top_k=3
)
```

### 2. Використання LLM-клієнта

```python
llm = _get_llm_reasoning()

from langchain_core.messages import SystemMessage, HumanMessage

response = llm.invoke([
    SystemMessage(content="System prompt..."),
    HumanMessage(content="User query...")
])
```

### 3. HybridRetriever

```python
from hybrid_retriever import HybridRetriever, HybridConfig

retriever = HybridRetriever(config=HybridConfig())
# використовується в API endpoints
```

## 📊 API Endpoints

### POST /api/generate-material

Генерує навчальний матеріал з використанням:

* RAG для пошуку контексту
* LLM для генерації відповіді

Запит:

```json
{
  "topic": "Квадратні рівняння",
  "grade": 8,
  "subject": "algebra",
  "use_rag": true
}
```

Відповідь:

```json
{
  "summary": "...",
  "explanation": "...",
  "exercises": [...],
  "rag_used": true
}
```

### POST /api/assess-student

Оцінює відповіді учня за допомогою LLM.

## 🔑 Конфігурація

Усі параметри зберігаються в `.env`:

```bash
LLM_API_URL=https://api.lapathoniia.top/v1/chat/completions
LLM_API_KEY=sk-********************************
LLM_MODEL=lapa

RAG_TOP_K=3
USE_RAG=true
```

## 📦 Розміщення даних

Parquet-файли мають бути в каталозі `data/`:

```
data/
├── toc_for_hackathon_with_subtopics.parquet
├── pages_for_hackathon.parquet
└── lms_questions_dev.parquet
```

Backend автоматично знаходить файли під час запуску.

## 🧪 Тестування

```bash
# Перевірка стану сервісу
curl http://localhost:8000/health

# Генерація матеріалу
curl -X POST http://localhost:8000/api/generate-material \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Квадратні рівняння",
    "grade": 8,
    "subject": "algebra"
  }'
```

## 📝 Логи

Приклад логів під час запуску:

```
🚀 Starting Mriia AI Tutor
📚 Loading RAG data...
✅ Loaded 4 RAG datasets
🔍 Initializing HybridRetriever...
✅ HybridRetriever initialized
🤖 Initializing LLM client...
✅ LLM client initialized
✅ Backend ready!
```

## 🔍 Основні компоненти

### agent.py

* `_load_rag_data()` — завантаження Parquet
* `_retrieve_context()` — RAG-пошук
* `_get_llm_reasoning()` — LLM-клієнт
* `_get_llm_classification()` — класифікація
* `SUBJECT_MAP` — мапінг предметів

### hybrid_retriever.py

* `HybridRetriever`
* `HybridConfig`
* FAISS-індекси
* BM25 sparse retrieval
* Reranking

## ✨ Переваги рішення

1. Повторне використання наявного коду
2. Мінімальні зміни в логіці
3. Чітка та зрозуміла структура
4. Простота підтримки й розширення
5. Єдина конфігурація для Backend і Frontend

## 🎯 Відповідність вимогам хакатону

✅ Два основні API endpoint’и
✅ Робота з підручниками через RAG
✅ Генерація навчального матеріалу
✅ Генерація тестів
✅ Перевірка відповідей
✅ Персоналізовані рекомендації
✅ Frontend + Backend
✅ Docker
✅ Документація

## 💡 Додатково

### Swagger-документація

[http://localhost:8000/docs](http://localhost:8000/docs)

### Streamlit UI

[http://localhost:8501](http://localhost:8501)

---

**Рішення готове до використання та розширення 🚀**
