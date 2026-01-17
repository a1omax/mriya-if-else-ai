# 🎓 Mriia AI Tutor - Lapathon 2026

**Полное решение на базе ВАШЕГО кода из Lapathon3.zip**

## ✅ Что включено

### Backend (FastAPI)
- ✅ **agent.py** - ВАШ код с LLM и RAG
- ✅ **hybrid_retriever.py** - ВАШ продвинутый retriever
- ✅ **main.py** - FastAPI обертка над вашим кодом

### Frontend (Streamlit)
- ✅ UI для вчителя и учня
- ✅ Два режима работы

### Данные
- ✅ Поддержка Parquet файлов
- ✅ Совместимость с вашей структурой данных

## 🚀 Быстрый старт

### Вариант 1: Docker

```bash
# 1. Распаковать
tar -xzf mriia_hackathon.tar.gz
cd mriia_hackathon

# 2. Поместить Parquet файлы в data/
cp /путь/к/*.parquet data/

# 3. Запустить
docker-compose up --build

# 4. Открыть
http://localhost:8501  # Frontend
http://localhost:8000  # Backend API
http://localhost:8000/docs  # Swagger
```

### Вариант 2: Локально

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend (в другом терминале)
cd frontend  
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Структура проекта

```
mriia_hackathon/
├── backend/
│   ├── agent.py              # ← ВАШ КОД из архива
│   ├── hybrid_retriever.py   # ← ВАШ КОД из архива
│   ├── main.py              # ← FastAPI обертка
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── data/                    # Поместите Parquet файлы сюда
├── .env                     # Настройки LLM
├── docker-compose.yml
└── README.md
```

## 🔧 Как это работает

### 1. Используется ВАШ код для RAG

```python
# В main.py импортируем ВАШ КОД:
from agent import _retrieve_context, _get_llm_reasoning

# Используем вашу RAG функцию:
context = _retrieve_context(
    question_text=topic,
    subject=subject,
    top_k=3
)
```

### 2. Используется ВАШ LLM client

```python
# Инициализация ВАШИМ способом:
llm = _get_llm_reasoning()

# Вызов LLM:
from langchain_core.messages import SystemMessage, HumanMessage

response = llm.invoke([
    SystemMessage(content="System prompt..."),
    HumanMessage(content="User query...")
])
```

### 3. Используется ВАШ HybridRetriever

```python
from hybrid_retriever import HybridRetriever, HybridConfig

retriever = HybridRetriever(config=HybridConfig())
# ... используется в endpoints
```

## 📊 API Endpoints

### POST /api/generate-material

Генерирует обучающий материал используя:
- `agent._retrieve_context()` для RAG
- `agent._get_llm_reasoning()` для генерации

Запрос:
```json
{
  "topic": "Квадратні рівняння",
  "grade": 8,
  "subject": "algebra",
  "use_rag": true
}
```

Ответ:
```json
{
  "summary": "...",
  "explanation": "...",
  "exercises": [...],
  "rag_used": true
}
```

### POST /api/assess-student

Оценивает ученика через ваш LLM.

## 🔑 Конфигурация

Все настройки в `.env` взяты из ВАШЕГО `agent.py`:

```bash
# Lapa LLM (из вашего кода)
LLM_API_URL=https://api.lapathoniia.top/v1/chat/completions
LLM_API_KEY=sk-J94Etria-0A2EMmH1xp-eg
LLM_MODEL=lapa

# RAG (из вашего кода)
RAG_TOP_K=3
USE_RAG=true
```

## 📦 Размещение данных

Parquet файлы должны быть в `data/`:

```
data/
├── Lapathon2026 Mriia public files/
│   ├── gemini-embedding-001__toc_for_hackathon_with_subtopics.parquet
│   ├── gemini-embedding-001__pages_for_hackathon.parquet
│   └── ...
```

Или просто:
```
data/
├── toc_for_hackathon_with_subtopics.parquet
├── pages_for_hackathon.parquet
└── lms_questions_dev.parquet
```

Backend автоматически найдет файлы используя логику из `agent._load_rag_data()`

## 🧪 Тестирование

```bash
# Health check
curl http://localhost:8000/health

# Генерация материала
curl -X POST http://localhost:8000/api/generate-material \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Квадратні рівняння",
    "grade": 8,
    "subject": "algebra"
  }'
```

## 📝 Логи

Backend логирует все операции:

```
🚀 Starting Mriia AI Tutor - Based on YOUR code
📚 Loading RAG data using agent.py...
✅ Loaded 4 RAG datasets
🔍 Initializing HybridRetriever...
✅ HybridRetriever initialized
🤖 Initializing LLM client...
✅ LLM client initialized
✅ Backend ready!
```

## 🔍 Что используется из ВАШЕГО кода

### Из agent.py:
- ✅ `_load_rag_data()` - загрузка Parquet
- ✅ `_retrieve_context()` - RAG поиск
- ✅ `_get_llm_reasoning()` - LLM client
- ✅ `_get_llm_classification()` - классификация
- ✅ `SUBJECT_MAP` - маппинг предметов
- ✅ Все настройки LLM (URL, key, model)

### Из hybrid_retriever.py:
- ✅ `HybridRetriever` - класс retriever
- ✅ `HybridConfig` - конфигурация
- ✅ FAISS индексы
- ✅ BM25 sparse retrieval
- ✅ Reranking

## ✨ Преимущества этого подхода

1. **Использует ВАШУ работу** - весь RAG и LLM код
2. **Минимум нового кода** - только FastAPI обертка
3. **Проверенное решение** - ваш код уже работает
4. **Легко поддерживать** - понятная структура
5. **Все настройки сохранены** - из вашего .env

## 🎯 Соответствие требованиям хакатона

✅ Два основных endpoint'а
✅ Работа с подручниками через RAG
✅ Генерация материала через LLM
✅ Генерация тестов
✅ Проверка ответов
✅ Персонализированные рекомендации
✅ Frontend + Backend
✅ Docker support
✅ Документация

## 💡 Дополнительно

### Просмотр Swagger документации

http://localhost:8000/docs

### Streamlit UI

http://localhost:8501

---

**Готово! Используйте ВАШИ agent.py и hybrid_retriever.py через FastAPI endpoints! 🚀**
