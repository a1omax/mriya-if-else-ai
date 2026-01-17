"""
Mriia AI Tutor - FastAPI Backend
Использует КОД ИЗ agent.py и hybrid_retriever.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
from pathlib import Path
from datetime import datetime
import logging

# ИМПОРТИРУЕМ ВАШ КОД
from agent import (
    _get_llm_reasoning,
    _get_llm_classification, 
    _load_rag_data,
    _retrieve_context,
    SUBJECT_MAP
)
from hybrid_retriever import HybridRetriever, HybridConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Mriia AI Tutor API",
    description="AI Tutor using agent.py and hybrid_retriever.py",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные объекты
retriever: Optional[HybridRetriever] = None
llm_reasoning = None
rag_data = None

# ============================================================================
# Pydantic Models
# ============================================================================

class TeacherRequest(BaseModel):
    topic: str
    grade: int = Field(..., ge=8, le=9)
    subject: str
    use_rag: bool = True

class Exercise(BaseModel):
    question_id: str
    question_text: str
    test_type: str = "single_choice"
    answers: List[str]
    correct_answer_indices: List[int]
    difficulty: str = "medium"
    metadata: Dict[str, Any] = {}

class LearningMaterial(BaseModel):
    topic: str
    grade: int
    subject: str
    summary: str
    explanation: str
    key_concepts: List[str]
    source_references: List[Dict[str, Any]]
    exercises: List[Exercise]
    rag_used: bool = False
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class StudentAnswer(BaseModel):
    question_id: str
    selected_answer_index: int

class StudentProfile(BaseModel):
    student_id: int
    grade: int
    recent_scores: List[Dict[str, Any]] = []

class Correction(BaseModel):
    question_id: str
    is_correct: bool
    student_answer: str
    correct_answer: str
    explanation: str

class Recommendation(BaseModel):
    topic: str
    reason: str
    priority: str = "medium"

class AssessmentRequest(BaseModel):
    student_answers: List[StudentAnswer]
    exercises: List[Exercise]
    student_profile: Optional[StudentProfile] = None

class AssessmentResponse(BaseModel):
    score: float
    total_questions: int
    correct_answers: int
    corrections: List[Correction]
    performance_analysis: str
    recommendations: List[Recommendation]
    next_steps: List[str]

# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Инициализация используя ВАШ КОД"""
    global retriever, llm_reasoning, rag_data
    
    logger.info("="*80)
    logger.info("🚀 Starting Mriia AI Tutor - Based on YOUR code")
    logger.info("="*80)
    
    # 1. Загружаем данные ВАШИМ способом из agent.py
    logger.info("📚 Loading RAG data using agent.py...")
    try:
        rag_data = _load_rag_data()
        logger.info(f"✅ Loaded {len(rag_data)} RAG datasets")
    except Exception as e:
        logger.warning(f"⚠️ Could not load RAG data: {e}")
        rag_data = {}
    
    # 2. Инициализируем Hybrid Retriever из ВАШЕГО кода
    logger.info("🔍 Initializing HybridRetriever...")
    try:
        retriever = HybridRetriever(config=HybridConfig(
            final_top_k=5,
            use_reranking=True,
            use_query_expansion=True
        ))
        
        # Загружаем данные в retriever если есть
        if "toc_gemini" in rag_data:
            logger.info("Loading TOC data into retriever...")
            # retriever.load_toc_data(rag_data["toc_gemini"])
        
        logger.info("✅ HybridRetriever initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize retriever: {e}")
        retriever = None
    
    # 3. Инициализируем LLM ВАШИМ способом из agent.py  
    logger.info("🤖 Initializing LLM client...")
    try:
        llm_reasoning = _get_llm_reasoning()
        logger.info("✅ LLM client initialized")
    except Exception as e:
        logger.error(f"❌ Could not initialize LLM: {e}")
        llm_reasoning = None
    
    logger.info("="*80)
    logger.info("✅ Backend ready!")
    logger.info("="*80)

# ============================================================================
# Helper Functions
# ============================================================================

def parse_llm_response(response_text: str) -> Dict[str, str]:
    """Парсит ответ LLM на компоненты"""
    parts = {}
    
    if "КОНСПЕКТ:" in response_text:
        parts_split = response_text.split("КОНСПЕКТ:")
        if len(parts_split) > 1:
            content = parts_split[1]
            if "ПОЯСНЕННЯ:" in content:
                summary, rest = content.split("ПОЯСНЕННЯ:", 1)
                parts["summary"] = summary.strip()
                if "ТЕСТИ:" in rest:
                    explanation, tests = rest.split("ТЕСТИ:", 1)
                    parts["explanation"] = explanation.strip()
                    parts["tests"] = tests.strip()
                else:
                    parts["explanation"] = rest.strip()
            else:
                parts["summary"] = content.strip()
    
    return parts

def parse_exercises_from_text(text: str) -> List[Dict]:
    """Парсит тестовые задания из текста LLM"""
    exercises = []
    
    # Простой парсинг - ищем блоки с ПИТАННЯ
    blocks = text.split("ПИТАННЯ:")
    
    for i, block in enumerate(blocks[1:], 1):  # Пропускаем первый пустой
        try:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            
            question = lines[0] if lines else ""
            answers = []
            correct = 0
            
            for line in lines[1:]:
                if line.startswith("А)") or line.startswith("A)"):
                    answers.append(line[2:].strip())
                elif line.startswith("Б)") or line.startswith("B)"):
                    answers.append(line[2:].strip())
                elif line.startswith("В)") or line.startswith("C)"):
                    answers.append(line[2:].strip())
                elif line.startswith("Г)") or line.startswith("D)"):
                    answers.append(line[2:].strip())
                elif "ПРАВИЛЬНА" in line.upper():
                    ans_letter = line.split(":")[-1].strip().upper()
                    correct = {"А": 0, "Б": 1, "В": 2, "Г": 3, 
                             "A": 0, "B": 1, "C": 2, "D": 3}.get(ans_letter, 0)
            
            if question and len(answers) == 4:
                exercises.append({
                    "question_id": f"gen_{i}",
                    "question_text": question,
                    "test_type": "single_choice",
                    "answers": answers,
                    "correct_answer_indices": [correct],
                    "difficulty": "medium",
                    "metadata": {"generated": True}
                })
        except:
            continue
    
    return exercises

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Mriia AI Tutor v3.0",
        "description": "Based on YOUR agent.py and hybrid_retriever.py",
        "llm_ready": llm_reasoning is not None,
        "rag_ready": rag_data is not None and len(rag_data) > 0
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llm_initialized": llm_reasoning is not None,
        "rag_loaded": rag_data is not None,
        "retriever_ready": retriever is not None
    }

@app.post("/api/generate-material", response_model=LearningMaterial)
async def generate_material(request: TeacherRequest):
    """
    Генерация материала используя ВАШ КОД:
    1. agent._retrieve_context() для RAG
    2. agent._get_llm_reasoning() для LLM
    """
    
    logger.info(f"📚 Generate request: {request.topic}, {request.subject}, grade {request.grade}")
    
    if llm_reasoning is None:
        raise HTTPException(500, "LLM not initialized")
    
    try:
        # ШАГ 1: RAG - используем ВАШУ функцию из agent.py
        context = ""
        if request.use_rag:
            logger.info("🔍 Using _retrieve_context from agent.py...")
            context = _retrieve_context(
                question_text=request.topic,
                subject=request.subject,
                top_k=3
            )
            logger.info(f"✅ Retrieved {len(context)} chars of context")
        
        # ШАГ 2: LLM генерация - используем ВАШЕГО клиента
        from langchain_core.messages import SystemMessage, HumanMessage
        
        subject_name = SUBJECT_MAP.get(request.subject, request.subject)
        
        system_prompt = f"""Ти - досвідчений вчитель для {request.grade} класу.
Предмет: {subject_name}
Створюй навчальні матеріали українською мовою."""

        user_prompt = f"""Створи навчальний матеріал на тему: "{request.topic}"

{f"Контекст з підручника:\n{context[:1500]}\n" if context else ""}

Створи у форматі:

КОНСПЕКТ:
[200-300 слів короткого конспекту]

ПОЯСНЕННЯ:
[400-600 слів детального пояснення з прикладами]

ТЕСТИ:
ПИТАННЯ: [текст питання 1]
А) [варіант А]
Б) [варіант Б]
В) [варіант В]
Г) [варіант Г]
ПРАВИЛЬНА_ВІДПОВІДЬ: [А/Б/В/Г]

[... ще 9 питань ...]
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        logger.info("🤖 Calling LLM...")
        response = llm_reasoning.invoke(messages)
        content_text = response.content
        
        logger.info(f"✅ LLM returned {len(content_text)} chars")
        
        # ШАГ 3: Парсинг ответа
        parsed = parse_llm_response(content_text)
        
        summary = parsed.get("summary", content_text[:500])
        explanation = parsed.get("explanation", content_text[500:1500] if len(content_text) > 500 else "")
        
        # Парсим упражнения
        exercises = []
        if "tests" in parsed:
            exercises = parse_exercises_from_text(parsed["tests"])
        
        # Если не распарсились, создаем минимум
        if len(exercises) < 3:
            exercises = parse_exercises_from_text(content_text)
        
        # Ключевые концепции
        key_concepts = [
            line.strip("- •").strip()
            for line in summary.split("\n")
            if line.strip().startswith(("-", "•"))
        ][:5]
        
        # Источники
        source_refs = []
        if context:
            source_refs.append({
                "type": "rag",
                "preview": context[:200],
                "length": len(context)
            })
        
        return LearningMaterial(
            topic=request.topic,
            grade=request.grade,
            subject=request.subject,
            summary=summary,
            explanation=explanation,
            key_concepts=key_concepts,
            source_references=source_refs,
            exercises=[Exercise(**ex) for ex in exercises[:10]],
            rag_used=len(context) > 0
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(500, f"Generation error: {str(e)}")

@app.post("/api/assess-student", response_model=AssessmentResponse)
async def assess_student(request: AssessmentRequest):
    """Оценивание ученика используя ВАШ LLM"""
    
    logger.info(f"📊 Assessment: {len(request.student_answers)} answers")
    
    if llm_reasoning is None:
        raise HTTPException(500, "LLM not initialized")
    
    try:
        # Проверка ответов
        corrections = []
        correct_count = 0
        
        ex_map = {ex.question_id: ex for ex in request.exercises}
        
        for ans in request.student_answers:
            ex = ex_map.get(ans.question_id)
            if not ex:
                continue
            
            is_correct = ans.selected_answer_index in ex.correct_answer_indices
            if is_correct:
                correct_count += 1
            
            student_ans = ex.answers[ans.selected_answer_index] if ans.selected_answer_index < len(ex.answers) else "?"
            correct_ans = ex.answers[ex.correct_answer_indices[0]] if ex.correct_answer_indices else "?"
            
            corrections.append(Correction(
                question_id=ans.question_id,
                is_correct=is_correct,
                student_answer=student_ans,
                correct_answer=correct_ans,
                explanation="✅ Правильно!" if is_correct else f"❌ Правильна відповідь: {correct_ans}"
            ))
        
        score = (correct_count / len(request.student_answers) * 100) if request.student_answers else 0
        
        # LLM рекомендации
        from langchain_core.messages import SystemMessage, HumanMessage
        
        rec_prompt = f"""Проаналізуй результати учня:
- Всього питань: {len(request.student_answers)}
- Правильних: {correct_count}
- Відсоток: {score:.1f}%

Створи:
1. АНАЛІЗ (2-3 речення)
2. РЕКОМЕНДАЦІЇ (3-5 пунктів що покращити)
3. НАСТУПНІ_КРОКИ (3 конкретні дії)

Формат:
АНАЛІЗ: ...
РЕКОМЕНДАЦІЇ:
- ...
НАСТУПНІ_КРОКИ:
- ...
"""
        
        messages = [
            SystemMessage(content="Ти - педагог який дає конструктивні рекомендації"),
            HumanMessage(content=rec_prompt)
        ]
        
        response = llm_reasoning.invoke(messages)
        rec_text = response.content
        
        # Парсинг рекомендаций
        analysis = rec_text.split("РЕКОМЕНДАЦІЇ:")[0].replace("АНАЛІЗ:", "").strip()
        
        recommendations = []
        if "РЕКОМЕНДАЦІЇ:" in rec_text:
            rec_part = rec_text.split("РЕКОМЕНДАЦІЇ:")[1]
            if "НАСТУПНІ_КРОКИ:" in rec_part:
                rec_part = rec_part.split("НАСТУПНІ_КРОКИ:")[0]
            
            for line in rec_part.split("\n"):
                if line.strip().startswith("-"):
                    recommendations.append(Recommendation(
                        topic="Покращення",
                        reason=line.strip("- ").strip(),
                        priority="high" if score < 60 else "medium"
                    ))
        
        next_steps = []
        if "НАСТУПНІ_КРОКИ:" in rec_text:
            steps_part = rec_text.split("НАСТУПНІ_КРОКИ:")[1]
            for line in steps_part.split("\n"):
                if line.strip().startswith("-"):
                    next_steps.append(line.strip("- ").strip())
        
        return AssessmentResponse(
            score=score,
            total_questions=len(request.student_answers),
            correct_answers=correct_count,
            corrections=corrections,
            performance_analysis=analysis,
            recommendations=recommendations[:5],
            next_steps=next_steps[:5]
        )
        
    except Exception as e:
        logger.error(f"❌ Assessment error: {e}", exc_info=True)
        raise HTTPException(500, f"Assessment error: {str(e)}")

@app.get("/api/subjects")
async def get_subjects():
    return {
        "subjects": [
            {"id": "algebra", "name": "Алгебра", "grades": [8, 9]},
            {"id": "ukrainian_language", "name": "Українська мова", "grades": [8, 9]},
            {"id": "history_ukraine", "name": "Історія України", "grades": [8, 9]}
        ]
    }

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
