"""
Enhanced LangGraph agent for CodaBench with 2-stage pipeline:
1. Retrieval + Reasoning: Find relevant context and reason about the question
2. Classification: Use reasoning to select the final answer (A/B/C/D)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# ============================================================================
# Configuration
# ============================================================================

LLM_API_URL = os.getenv("LLM_API_URL", "https://api.lapathoniia.top/v1/chat/completions")
base_url = LLM_API_URL.rstrip("/")
LLM_API_BASE = os.getenv("LLM_API_BASE") or (
    base_url[: -len("/chat/completions")] if base_url.endswith("/chat/completions") else base_url
)
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("LAPA_API_KEY", "sk-J94Etria-0A2EMmH1xp-eg")
LLM_MODEL = os.getenv("LLM_MODEL", "lapa")
LLM_MAX_TOKENS_REASONING = int(os.getenv("LLM_MAX_TOKENS_REASONING", "512"))
LLM_MAX_TOKENS_CLASSIFICATION = int(os.getenv("LLM_MAX_TOKENS_CLASSIFICATION", "1"))
LLM_GUIDED_CHOICE = os.getenv("LLM_GUIDED_CHOICE", "true").strip().lower() in {"1", "true", "yes"}
GUIDED_CHOICES = ["A", "B", "C", "D"]

# RAG Configuration
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
USE_RAG = os.getenv("USE_RAG", "true").strip().lower() in {"1", "true", "yes"}

INPUT_DIR = Path(os.getenv("CODABENCH_INPUT_DIR", "/app/input_data"))
PUBLIC_DATA_DIR = INPUT_DIR / "public_data"
PRIVATE_TEST_PATH = INPUT_DIR / "lms_questions_test.parquet"

# Subject mapping (Ukrainian)
SUBJECT_MAP = {
    131: "ukrainian_language",
    "Українська мова": "ukrainian_language",
    "українська мова": "ukrainian_language",
    132: "algebra",
    "Алгебра": "algebra",
    "алгебра": "algebra",
    133: "history_ukraine",
    "Історія України": "history_ukraine",
    "історія україни": "history_ukraine",
}

# ============================================================================
# Global State
# ============================================================================

_LLM_CLIENT: Optional[ChatOpenAI] = None
_LLM_CLIENT_REASONING: Optional[ChatOpenAI] = None
_GRAPH = None
_RAG_DATA: Optional[Dict[str, pd.DataFrame]] = None


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    question: Any
    answer: str
    hidden_data: dict
    reasoning: str
    context: str
    subject: str


# ============================================================================
# RAG Data Loading
# ============================================================================

def _load_rag_data() -> Dict[str, pd.DataFrame]:
    """Load RAG data from parquet files."""
    global _RAG_DATA
    if _RAG_DATA is not None:
        return _RAG_DATA
    
    _RAG_DATA = {}
    base_dir = PUBLIC_DATA_DIR / "Lapathon2026 Mriia public files"
    
    # Try different path patterns
    paths_to_try = {
        "toc_gemini": [
            base_dir / "gemini-embedding-001__toc_for_hackathon_with_subtopics.parquet",
            base_dir / "gemini-embedding-001" / "toc_for_hackathon_with_subtopics.parquet",
        ],
        "pages_gemini": [
            base_dir / "gemini-embedding-001__pages_for_hackathon.parquet",
            base_dir / "gemini-embedding-001" / "pages_for_hackathon.parquet",
        ],
        "toc_qwen": [
            base_dir / "text-embedding-qwen__toc_for_hackathon_with_subtopics.parquet",
            base_dir / "text-embedding-qwen" / "toc_for_hackathon_with_subtopics.parquet",
        ],
        "pages_qwen": [
            base_dir / "text-embedding-qwen__pages_for_hackathon.parquet",
            base_dir / "text-embedding-qwen" / "pages_for_hackathon.parquet",
        ],
    }
    
    for key, path_list in paths_to_try.items():
        for path in path_list:
            if path.exists():
                try:
                    _RAG_DATA[key] = pd.read_parquet(path)
                    print(f"Loaded {key} from {path}", file=sys.stderr)
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}", file=sys.stderr)
    
    return _RAG_DATA


def _get_embedding_column(df: pd.DataFrame) -> Optional[str]:
    """Find the embedding column in a dataframe."""
    embedding_cols = [col for col in df.columns if "embedding" in col.lower()]
    if embedding_cols:
        return embedding_cols[0]
    return None


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _text_similarity(text1: str, text2: str) -> float:
    """Simple text similarity based on word overlap (Jaccard-like)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def _retrieve_context(question_text: str, subject: str, top_k: int = RAG_TOP_K) -> str:
    """Retrieve relevant context from RAG data based on text similarity."""
    if not USE_RAG:
        return ""
    
    rag_data = _load_rag_data()
    if not rag_data:
        return ""
    
    all_results = []
    
    # Search in pages data (contains more detailed content)
    for key in ["pages_gemini", "pages_qwen"]:
        if key not in rag_data:
            continue
        df = rag_data[key]
        
        # Filter by subject if possible
        if "global_discipline_name" in df.columns:
            subject_lower = subject.lower() if subject else ""
            if "укр" in subject_lower or subject == "ukrainian_language":
                df_filtered = df[df["global_discipline_name"].str.lower().str.contains("укр", na=False)]
            elif "алг" in subject_lower or "матем" in subject_lower or subject == "algebra":
                df_filtered = df[df["global_discipline_name"].str.lower().str.contains("алг|матем", na=False, regex=True)]
            elif "істор" in subject_lower or subject == "history_ukraine":
                df_filtered = df[df["global_discipline_name"].str.lower().str.contains("істор", na=False)]
            else:
                df_filtered = df
            
            if len(df_filtered) > 0:
                df = df_filtered
        
        # Compute text similarity for each row
        text_col = None
        for col in ["topic_text", "page_text", "topic_summary"]:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            continue
        
        for idx, row in df.iterrows():
            text = str(row.get(text_col, ""))
            if not text or len(text) < 20:
                continue
            
            sim = _text_similarity(question_text, text)
            if sim > 0.05:  # Minimum threshold
                all_results.append({
                    "text": text[:2000],  # Limit text length
                    "title": str(row.get("topic_title", row.get("section_title", ""))),
                    "book": str(row.get("book_name", "")),
                    "similarity": sim,
                })
    
    # Also search in TOC data (contains summaries and subtopics)
    for key in ["toc_gemini", "toc_qwen"]:
        if key not in rag_data:
            continue
        df = rag_data[key]
        
        text_col = "page_text" if "page_text" in df.columns else None
        if text_col is None:
            continue
        
        for idx, row in df.iterrows():
            text = str(row.get(text_col, ""))
            if not text or len(text) < 20:
                continue
            
            sim = _text_similarity(question_text, text)
            if sim > 0.05:
                all_results.append({
                    "text": text[:1500],
                    "title": str(row.get("topic_title", row.get("section_title", ""))),
                    "book": str(row.get("book_name", "")),
                    "similarity": sim,
                })
    
    # Sort by similarity and take top_k
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = all_results[:top_k]
    
    if not top_results:
        return ""
    
    # Format context
    context_parts = []
    for i, res in enumerate(top_results, 1):
        context_parts.append(f"[Джерело {i}] {res['title']}\n{res['text']}")
    
    return "\n\n---\n\n".join(context_parts)


# ============================================================================
# LLM Clients
# ============================================================================

def _get_llm_reasoning() -> ChatOpenAI:
    """Get LLM client for reasoning (higher max_tokens)."""
    global _LLM_CLIENT_REASONING
    if _LLM_CLIENT_REASONING:
        return _LLM_CLIENT_REASONING
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set")
    _LLM_CLIENT_REASONING = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
        temperature=0.1,
        max_tokens=LLM_MAX_TOKENS_REASONING,
        timeout=60,
    )
    return _LLM_CLIENT_REASONING


def _get_llm_classification() -> ChatOpenAI:
    """Get LLM client for classification (single token output)."""
    global _LLM_CLIENT
    if _LLM_CLIENT:
        return _LLM_CLIENT
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set")
    model_kwargs = {"extra_body": {"guided_choice": GUIDED_CHOICES}} if LLM_GUIDED_CHOICE else {}
    _LLM_CLIENT = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
        temperature=0,
        max_tokens=LLM_MAX_TOKENS_CLASSIFICATION,
        timeout=30,
        model_kwargs=model_kwargs,
    )
    return _LLM_CLIENT


def _normalize_choice(text: str) -> str:
    """Extract A/B/C/D from response."""
    if not text:
        return ""
    match = re.search(r"[A-D]", text.upper())
    return match.group(0) if match else text.strip()


# ============================================================================
# Prompt Formatting
# ============================================================================

def _get_subject_from_question(question: Any) -> str:
    """Extract subject from question data."""
    if isinstance(question, dict):
        # Try different possible fields
        for field in ["subject", "discipline", "global_discipline_name", "global_discipline_id"]:
            if field in question:
                val = question[field]
                if val in SUBJECT_MAP:
                    return SUBJECT_MAP[val]
                if isinstance(val, str):
                    val_lower = val.lower()
                    if "укр" in val_lower:
                        return "ukrainian_language"
                    elif "алг" in val_lower or "матем" in val_lower:
                        return "algebra"
                    elif "істор" in val_lower:
                        return "history_ukraine"
    return "unknown"


def _format_options(options: Any) -> str:
    """Format answer options."""
    if not options:
        return ""
    lines = []
    if isinstance(options, dict):
        for key in sorted(options.keys()):
            lines.append(f"{key}) {options[key]}")
    else:
        letters = "ABCD"
        try:
            option_list = list(options)
        except TypeError:
            option_list = [str(options)]
        for idx, text in enumerate(option_list):
            label = letters[idx] if idx < len(letters) else str(idx + 1)
            lines.append(f"{label}) {text}")
    return "\n".join(lines)


def _build_question_text(question: Any) -> tuple[str, str]:
    """Build question text and options string."""
    if isinstance(question, dict):
        text = question.get("question_text") or question.get("text") or ""
        options = question.get("answers") or question.get("options") or []
        return str(text).strip(), _format_options(options)
    return str(question or "").strip(), ""


# ============================================================================
# Pipeline Nodes
# ============================================================================

def retrieve_context_node(state: AgentState) -> AgentState:
    """Node 1: Retrieve relevant context from RAG data."""
    question = state["question"]
    question_text, _ = _build_question_text(question)
    subject = _get_subject_from_question(question)
    
    try:
        context = _retrieve_context(question_text, subject, top_k=RAG_TOP_K)
    except Exception as e:
        print(f"RAG retrieval error: {e}", file=sys.stderr)
        context = ""
    
    return {
        **state,
        "context": context,
        "subject": subject,
    }


def reasoning_node(state: AgentState) -> AgentState:
    """Node 2: Generate reasoning about the question."""
    question = state["question"]
    context = state.get("context", "")
    subject = state.get("subject", "unknown")
    hidden_data = state.get("hidden_data", {})
    
    question_text, options_text = _build_question_text(question)
    
    # Build reasoning prompt
    subject_names = {
        "ukrainian_language": "Українська мова",
        "algebra": "Алгебра",
        "history_ukraine": "Історія України",
    }
    subject_name = subject_names.get(subject, "Загальний предмет")
    
    system_prompt = f"""Ти - експерт з предмету "{subject_name}" для 8-9 класів української школи.
Твоє завдання - проаналізувати питання тесту та знайти правильну відповідь.

Проаналізуй питання крок за кроком:
1. Визнач, що саме запитується
2. Якщо є контекст з підручників, використай його
3. Застосуй свої знання з предмету
4. Обґрунтуй, чому одна відповідь правильна, а інші - ні
5. Зроби висновок щодо правильної відповіді (A, B, C або D)

Відповідай українською мовою. Будь лаконічним, але точним."""

    user_prompt_parts = []
    
    if context:
        user_prompt_parts.append(f"КОНТЕКСТ З ПІДРУЧНИКІВ:\n{context}\n")
    
    if hidden_data:
        user_prompt_parts.append(f"ДОДАТКОВА ІНФОРМАЦІЯ:\n{json.dumps(hidden_data, ensure_ascii=False, indent=2)}\n")
    
    user_prompt_parts.append(f"ПИТАННЯ:\n{question_text}")
    
    if options_text:
        user_prompt_parts.append(f"\nВАРІАНТИ ВІДПОВІДЕЙ:\n{options_text}")
    
    user_prompt_parts.append("\nПроаналізуй це питання та визнач правильну відповідь:")
    
    user_prompt = "\n".join(user_prompt_parts)
    
    try:
        response = _get_llm_reasoning().invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        reasoning = getattr(response, "content", "")
    except Exception as e:
        print(f"Reasoning LLM error: {e}", file=sys.stderr)
        reasoning = ""
    
    return {
        **state,
        "reasoning": reasoning,
    }


def classification_node(state: AgentState) -> AgentState:
    """Node 3: Use reasoning to make final classification."""
    question = state["question"]
    reasoning = state.get("reasoning", "")
    context = state.get("context", "")
    hidden_data = state.get("hidden_data", {})
    
    question_text, options_text = _build_question_text(question)
    
    # Build classification prompt
    system_prompt = """Ти - система класифікації для тестів з множинним вибором.
На основі наданого аналізу, вибери правильну відповідь.
Відповідай ТІЛЬКИ однією літерою: A, B, C або D."""

    user_prompt_parts = []
    
    if reasoning:
        user_prompt_parts.append(f"АНАЛІЗ ПИТАННЯ:\n{reasoning}\n")
    
    user_prompt_parts.append(f"ПИТАННЯ:\n{question_text}")
    
    if options_text:
        user_prompt_parts.append(f"\nВАРІАНТИ:\n{options_text}")
    
    user_prompt_parts.append("\nПравильна відповідь (тільки літера):")
    
    user_prompt = "\n".join(user_prompt_parts)
    
    try:
        response = _get_llm_classification().invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        answer = _normalize_choice(getattr(response, "content", ""))
    except Exception as e:
        print(f"Classification LLM error: {e}", file=sys.stderr)
        # Fallback: try to extract answer from reasoning
        if reasoning:
            answer = _normalize_choice(reasoning)
        else:
            answer = ""
    
    return {
        **state,
        "answer": answer,
    }


# ============================================================================
# Simplified Single-Call Node (Fallback)
# ============================================================================

def call_llm_simple(state: AgentState) -> AgentState:
    """Simple single-call fallback (original behavior)."""
    question = state["question"]
    hidden_data = state.get("hidden_data", {})
    
    question_text, options_text = _build_question_text(question)
    
    system = (
        "You are a helpful assistant for multiple-choice questions. "
        "Reply with only the option letter (A, B, C, or D)."
    )
    if hidden_data:
        system += "\n\nContext:\n" + json.dumps(hidden_data, indent=2)
    
    prompt = question_text
    if options_text:
        prompt += f"\n\nOptions:\n{options_text}\n\nAnswer (letter only):"
    
    try:
        response = _get_llm_classification().invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt)
        ])
        answer = _normalize_choice(getattr(response, "content", ""))
    except Exception as e:
        print(f"Simple LLM error: {e}", file=sys.stderr)
        answer = ""
    
    return {
        **state,
        "answer": answer,
    }


# ============================================================================
# Graph Building
# ============================================================================

def log_input_data() -> None:
    """Log available input data for debugging."""
    try:
        _load_rag_data()
    except Exception as e:
        print(f"Error pre-loading RAG data: {e}", file=sys.stderr)


def build() -> StateGraph:
    """Build the LangGraph workflow with 2-stage pipeline."""
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    
    log_input_data()
    
    workflow = StateGraph(AgentState)
    
    # Add nodes for 2-stage pipeline
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("classification", classification_node)
    
    # Set entry point and edges
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "reasoning")
    workflow.add_edge("reasoning", "classification")
    workflow.add_edge("classification", END)
    
    _GRAPH = workflow.compile()
    return _GRAPH


def invoke(inputs: Dict[str, Any]) -> Any:
    """Invoke the graph with inputs."""
    # Ensure all required keys are present
    if "hidden_data" not in inputs:
        inputs["hidden_data"] = {}
    if "answer" not in inputs:
        inputs["answer"] = ""
    if "reasoning" not in inputs:
        inputs["reasoning"] = ""
    if "context" not in inputs:
        inputs["context"] = ""
    if "subject" not in inputs:
        inputs["subject"] = ""
    
    return build().invoke(inputs)


# Build the graph on module load
graph = build()