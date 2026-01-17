# 🎯 Как адаптировать ваш код из Lapathon3.zip под задачу хакатона

## ❌ Что я сделал НЕПРАВИЛЬНО

Я создал решение "с нуля", НЕ используя ваш существующий код из `Lapathon3.zip`.

## ✅ Что нужно сделать ПРАВИЛЬНО

Использовать **ВАШ КОД** как основу и адаптировать его под два endpoint'а хакатона.

---

## 📦 Что у вас УЖЕНЕСТЬ в архиве

### 1. `agent.py` - Ключевой файл с LLM вызовами
**Что там есть:**
- ✅ Настоящие вызовы Lapa LLM через `langchain_openai.ChatOpenAI`
- ✅ RAG функция `_retrieve_context()` с поиском по текстовому сходству
- ✅ LangGraph пайплайн с nodes
- ✅ Загрузка Parquet данных
- ✅ Работа с эмбеддингами

**Как использовать:**
```python
# Используйте их LLM клиент:
from agent import _get_llm_reasoning, _get_llm_classification

llm = _get_llm_reasoning()
response = llm.invoke([
    SystemMessage(content="Ти - вчитель..."),
    HumanMessage(content="Створи матеріал...")
])
```

### 2. `hybrid_retriever.py` - Продвинутый RAG
**Что там есть:**
- ✅ FAISS векторный поиск
- ✅ BM25 sparse retrieval
- ✅ Reranking с cross-encoder
- ✅ Hybrid fusion (dense + sparse)

**Как использовать:**
```python
from hybrid_retriever import HybridRetriever, HybridConfig

retriever = HybridRetriever(config=HybridConfig())
retriever.load_data(toc_df, pages_df)
retriever.build_indexes()

results = retriever.retrieve(query="квадратні рівняння", top_k=5)
```

### 3. Другие полезные файлы:
- `eval.py` - примеры работы с questions
- `test_hybrid_retriever.py` - примеры использования
- `data_analysis.py` - анализ данных

---

## 🔧 План адаптации

### ШАГ 1: Создайте структуру проекта

```bash
mriia_hackathon/
├── backend/
│   ├── agent.py              # ← Копируем из архива
│   ├── hybrid_retriever.py   # ← Копируем из архива
│   ├── api_server.py         # ← НОВЫЙ: FastAPI endpoints
│   ├── tutor_service.py      # ← НОВЫЙ: Логика тьютора
│   └── requirements.txt
├── frontend/
│   └── app.py               # Streamlit UI
└── data/                    # Parquet файлы
```

### ШАГ 2: Создайте API Server (api_server.py)

```python
"""
FastAPI сервер используя ВАШ КОД из agent.py и hybrid_retriever.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

# ИМПОРТИРУЕМ ВАШ КОД
from agent import _get_llm_reasoning, _load_rag_data, _retrieve_context
from hybrid_retriever import HybridRetriever, HybridConfig

app = FastAPI(title="Mriia AI Tutor - Based on Your Code")

# Глобальные объекты
retriever = None
llm_client = None

@app.on_event("startup")
async def startup():
    """Инициализация используя ваш код"""
    global retriever, llm_client
    
    # 1. Загружаем данные ВАШИМ способом
    rag_data = _load_rag_data()
    
    # 2. Создаем retriever из ВАШЕГО hybrid_retriever.py
    retriever = HybridRetriever(config=HybridConfig(
        final_top_k=5,
        use_reranking=True
    ))
    
    # Загружаем данные в retriever
    if "toc_gemini" in rag_data:
        toc_df = rag_data["toc_gemini"]
        retriever.load_toc_data(toc_df)
    
    if "pages_gemini" in rag_data:
        pages_df = rag_data["pages_gemini"]
        retriever.load_pages_data(pages_df)
    
    retriever.build_indexes()
    
    # 3. Инициализируем LLM ВАШИМ способом
    llm_client = _get_llm_reasoning()
    
    print("✅ Система инициализирована используя ваш код!")


class GenerateRequest(BaseModel):
    topic: str
    grade: int
    subject: str


@app.post("/api/generate-material")
async def generate_material(request: GenerateRequest):
    """
    Endpoint 1: Генерация материала
    Использует ВАШ retriever и LLM client
    """
    
    # 1. RAG - используем ВАШУ функцию
    context = _retrieve_context(
        question_text=request.topic,
        subject=request.subject,
        top_k=3
    )
    
    # ИЛИ используем ваш hybrid retriever
    # results = retriever.retrieve(query=request.topic, top_k=5)
    # context = "\n".join([r.content for r in results])
    
    # 2. LLM генерация - используем ВАШЕГО клиента
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content=f"Ти - вчитель для {request.grade} класу"),
        HumanMessage(content=f"""Створи навчальний матеріал:
        
Тема: {request.topic}
Контекст: {context[:1000]}

Створи:
1. Короткий конспект
2. Детальне пояснення
3. 10 тестових питань

Формат:
КОНСПЕКТ: ...
ПОЯСНЕННЯ: ...
ТЕСТИ: ...
""")
    ]
    
    response = llm_client.invoke(messages)
    content = response.content
    
    # Парсинг и возврат
    return {
        "summary": content,  # Парсите это
        "exercises": [],     # Парсите tests из content
        "context_used": context[:500]
    }


# Аналогично для /api/assess-student
```

### ШАГ 3: Создайте Tutor Service (tutor_service.py)

```python
"""
Сервис тьютора на базе ВАШЕГО кода
"""

from agent import _get_llm_reasoning, _retrieve_context
from hybrid_retriever import HybridRetriever
from langchain_core.messages import SystemMessage, HumanMessage

class TutorService:
    """Обертка над вашим кодом для удобства"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = _get_llm_reasoning()
    
    def generate_content(self, topic: str, grade: int, subject: str):
        """
        Генерирует контент используя:
        1. Ваш retriever для RAG
        2. Ваш LLM client для генерации
        """
        
        # RAG - ваш код
        context = _retrieve_context(topic, subject)
        
        # LLM - ваш код
        messages = [
            SystemMessage(content=f"Вчитель для {grade} класу"),
            HumanMessage(content=f"Тема: {topic}\nКонтекст: {context}")
        ]
        
        response = self.llm.invoke(messages)
        
        return self._parse_response(response.content)
    
    def generate_exercises(self, topic: str, count: int = 10):
        """Генерирует тесты через ваш LLM"""
        messages = [
            SystemMessage(content="Створи тестові завдання"),
            HumanMessage(content=f"Тема: {topic}, кількість: {count}")
        ]
        
        response = self.llm.invoke(messages)
        return self._parse_exercises(response.content)
    
    def assess_student(self, answers, exercises):
        """Оценивает через ваш LLM"""
        # Используйте ваш LLM для анализа
        pass
```

### ШАГ 4: Requirements.txt из ВАШЕГО кода

```txt
# Основные (из вашего pyproject.toml)
langchain==1.2.4
langchain-openai==1.1.7
langchain-core==1.2.7
langgraph==1.0.6

# RAG (из вашего hybrid_retriever.py)
faiss-cpu==1.7.4
sentence-transformers==5.2.0
rank-bm25==0.2.2

# Data
pandas==2.3.3
pyarrow==22.0.0
numpy==2.4.1

# FastAPI (новое для endpoints)
fastapi==0.128.0
uvicorn==0.40.0
pydantic==2.12.5
```

---

## 🎯 Основные адаптации

### Адаптация 1: RAG функция

**ВАША функция** (`agent.py:155`):
```python
def _retrieve_context(question_text: str, subject: str, top_k: int = RAG_TOP_K) -> str:
    # ... ваш код поиска по text similarity
```

**Как использовать для endpoint:**
```python
@app.post("/api/generate-material")
async def generate_material(request):
    # Просто вызываем вашу функцию!
    context = _retrieve_context(
        question_text=request.topic,
        subject=request.subject
    )
    
    # Используем context для LLM
```

### Адаптация 2: LLM Client

**ВАШ LLM client** (`agent.py:254`):
```python
def _get_llm_reasoning() -> ChatOpenAI:
    return ChatOpenAI(
        model="lapa",
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
        temperature=0.1,
        max_tokens=512
    )
```

**Как использовать:**
```python
llm = _get_llm_reasoning()

# Генерация контента
response = llm.invoke([
    SystemMessage(content="System prompt..."),
    HumanMessage(content="User query...")
])

content = response.content  # Ответ LLM
```

### Адаптация 3: Hybrid Retriever

**ВАШ retriever** (`hybrid_retriever.py`):
```python
from hybrid_retriever import HybridRetriever, HybridConfig

retriever = HybridRetriever(config=HybridConfig())
retriever.load_data(toc_df=toc, pages_df=pages)
retriever.build_indexes()

results = retriever.retrieve(query="квадратні рівняння")
```

**Используйте в endpoint:**
```python
@app.post("/api/generate-material")
async def generate_material(request):
    # ВАШИ retrieval results
    results = retriever.retrieve(
        query=request.topic,
        top_k=5,
        filters={"grade": request.grade}
    )
    
    # Формируем context
    context = "\n\n".join([r.content for r in results])
    
    # Передаем в LLM
```

---

## 📝 Итоговый правильный подход

### Файлы которые берем ИЗ ВАШЕГО АРХИВА:
1. ✅ `agent.py` - LLM clients и RAG
2. ✅ `hybrid_retriever.py` - Продвинутый retrieval
3. ✅ `requirements.txt` / `pyproject.toml` - Dependencies

### Файлы которые СОЗДАЕМ НОВЫЕ:
1. ➕ `api_server.py` - FastAPI с endpoints
2. ➕ `tutor_service.py` - Обертка над вашим кодом
3. ➕ `frontend/app.py` - Streamlit UI

### Логика работы:

```
User Request → FastAPI endpoint
                    ↓
            TutorService (новый)
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   ВАШ agent.py          ВАШ hybrid_retriever.py
   (_retrieve_context)   (HybridRetriever)
        ↓                       ↓
   ВАШ LLM client        ВАШИ FAISS indexes
   (_get_llm_reasoning)
        ↓
    Response
```

---

## ✅ Чеклист правильной реализации

- [ ] Скопировать `agent.py` и `hybrid_retriever.py` из архива
- [ ] Импортировать `_get_llm_reasoning()` для LLM вызовов
- [ ] Импортировать `_retrieve_context()` для RAG
- [ ] Создать FastAPI обертку вокруг вашего кода
- [ ] Использовать ваши requirements (langchain, faiss, etc)
- [ ] НЕ переписывать RAG логику - использовать вашу
- [ ] НЕ переписывать LLM вызовы - использовать ваши

---

## 🚀 Быстрый старт с ВАШИМ кодом

```bash
# 1. Распаковать ваш архив
unzip Lapathon3.zip -d lapathon_base

# 2. Создать проект
mkdir -p mriia_hackathon/backend
cd mriia_hackathon/backend

# 3. Скопировать ВАШИ файлы
cp ../../lapathon_base/agent.py .
cp ../../lapathon_base/hybrid_retriever.py .

# 4. Создать FastAPI обертку (api_server.py)
# Используя примеры выше

# 5. Установить зависимости
pip install -r requirements.txt  # Из вашего архива

# 6. Запустить
python api_server.py
```

---

## 💡 Почему это ПРАВИЛЬНЫЙ подход

1. **Используется ВАША работа** - agent.py, hybrid_retriever.py
2. **Не изобретается велосипед** - RAG и LLM уже есть
3. **Минимум нового кода** - только FastAPI обертка
4. **Проверенное решение** - ваш код уже протестирован
5. **Легко поддерживать** - понятная структура

---

**ВЫВОД:** Используйте ваш существующий код как библиотеку, добавьте только FastAPI endpoints!
