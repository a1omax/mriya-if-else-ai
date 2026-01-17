"""
Мрія AI Tutor - Frontend
Streamlit інтерфейс для вчителів та учнів
"""

import streamlit as st
import requests
import json
from typing import List, Dict
from datetime import datetime

# Конфігурація
API_BASE_URL = "http://localhost:8000"

# Стилізація
st.set_page_config(
    page_title="Мрія AI Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .exercise-card {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Ініціалізація session state
if 'generated_material' not in st.session_state:
    st.session_state.generated_material = None
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}
if 'assessment_result' not in st.session_state:
    st.session_state.assessment_result = None

# ============================================================================
# Helper Functions
# ============================================================================

def call_api(endpoint: str, method: str = "GET", data: dict = None):
    """Виклик API"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Помилка з'єднання з API: {str(e)}")
        return None


def display_exercise_card(exercise: dict, index: int, show_answer: bool = False):
    """Відображення картки з вправою"""
    with st.container():
        st.markdown(f"### Завдання {index + 1}")
        st.write(exercise['question_text'])
        
        # Варіанти відповіді
        answer_labels = ["А", "Б", "В", "Г"]
        
        if not show_answer:
            # Режим відповіді учня
            selected = st.radio(
                "Виберіть відповідь:",
                options=range(len(exercise['answers'])),
                format_func=lambda x: f"{answer_labels[x]}. {exercise['answers'][x]}",
                key=f"q_{exercise['question_id']}",
                index=None
            )
            
            if selected is not None:
                st.session_state.student_answers[exercise['question_id']] = selected
        else:
            # Режим перегляду з правильними відповідями
            correct_idx = exercise['correct_answer_indices'][0]
            for idx, answer in enumerate(exercise['answers']):
                label = answer_labels[idx]
                if idx == correct_idx:
                    st.markdown(f"✅ **{label}. {answer}** (правильна відповідь)")
                else:
                    st.markdown(f"{label}. {answer}")
        
        st.markdown("---")


def display_material_section(material: dict):
    """Відображення навчального матеріалу"""
    st.markdown("<div class='sub-header'>📖 Навчальний матеріал</div>", unsafe_allow_html=True)
    
    # Конспект
    with st.expander("📝 Короткий конспект", expanded=True):
        st.markdown(material['summary'])
    
    # Детальне пояснення
    with st.expander("📚 Детальне пояснення", expanded=False):
        st.markdown(material['explanation'])
    
    # Ключові поняття
    with st.expander("🔑 Ключові поняття", expanded=False):
        for concept in material['key_concepts']:
            st.markdown(f"- {concept}")
    
    # Джерела
    with st.expander("📌 Джерела", expanded=False):
        for ref in material['source_references']:
            st.write(f"**{ref['title']}**")
            if 'subtopics' in ref:
                for subtopic in ref['subtopics']:
                    st.write(f"  - {subtopic}")


def display_exercises_section(exercises: list):
    """Відображення тестових завдань"""
    st.markdown("<div class='sub-header'>✍️ Тестові завдання</div>", unsafe_allow_html=True)
    st.info(f"📊 Всього завдань: {len(exercises)}")
    
    for idx, exercise in enumerate(exercises):
        display_exercise_card(exercise, idx)


def display_assessment_section(result: dict):
    """Відображення результатів оцінювання"""
    st.markdown("<div class='sub-header'>📊 Результати оцінювання</div>", unsafe_allow_html=True)
    
    # Загальний результат
    score = result['score']
    correct = result['correct_answers']
    total = result['total_questions']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Правильних відповідей", f"{correct}/{total}")
    with col2:
        st.metric("Відсоток виконання", f"{score:.1f}%")
    with col3:
        if score >= 90:
            st.metric("Оцінка", "Відмінно ⭐")
        elif score >= 75:
            st.metric("Оцінка", "Добре 👍")
        elif score >= 60:
            st.metric("Оцінка", "Задовільно 📝")
        else:
            st.metric("Оцінка", "Потребує покращення 📖")
    
    # Аналіз успішності
    with st.expander("📈 Аналіз успішності", expanded=True):
        st.write(result['performance_analysis'])
        if result.get('compared_to_class'):
            st.info(result['compared_to_class'])
    
    # Детальна перевірка відповідей
    with st.expander("✅ Перевірка відповідей", expanded=True):
        for corr in result['corrections']:
            if corr['is_correct']:
                st.markdown(f"""
                <div class='success-box'>
                    <strong>✅ Правильно!</strong><br>
                    Ваша відповідь: {corr['student_answer']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='error-box'>
                    <strong>❌ Помилка</strong><br>
                    Ваша відповідь: {corr['student_answer']}<br>
                    Правильна відповідь: {corr['correct_answer']}<br>
                    {corr['explanation']}
                </div>
                """, unsafe_allow_html=True)
    
    # Рекомендації
    with st.expander("💡 Рекомендації", expanded=True):
        for rec in result['recommendations']:
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            icon = priority_icon.get(rec['priority'], "⚪")
            
            st.markdown(f"""
            {icon} **{rec['topic']}**  
            {rec['reason']}
            """)
            
            if rec.get('suggested_exercises'):
                st.write("Рекомендовані вправи:")
                for exercise in rec['suggested_exercises']:
                    st.write(f"  - {exercise}")
    
    # Наступні кроки
    with st.expander("🎯 Наступні кроки", expanded=True):
        for step in result['next_steps']:
            st.markdown(f"- {step}")


# ============================================================================
# Main App
# ============================================================================

def main():
    # Заголовок
    st.markdown("<h1 class='main-header'>📚 Мрія AI Tutor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Інтелектуальна система підтримки навчання для 8-9 класів</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Mriia+Logo", width=150)
        st.markdown("---")
        
        mode = st.radio(
            "Режим роботи:",
            ["🧑‍🏫 Генерація матеріалів", "👨‍🎓 Проходження тесту"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ Інформація")
        st.info("""
        **Мрія AI Tutor** - це інтелектуальна система, яка допомагає:
        
        - 🧑‍🏫 Вчителям: генерувати навчальні матеріали та тести
        - 👨‍🎓 Учням: отримувати персоналізовані рекомендації
        
        **Предмети:**
        - Алгебра
        - Українська мова
        - Історія України
        
        **Класи:** 8, 9
        """)
    
    # ========================================================================
    # Режим 1: Генерація матеріалів (для вчителя)
    # ========================================================================
    if mode == "🧑‍🏫 Генерація матеріалів":
        st.markdown("## 🧑‍🏫 Генерація навчальних матеріалів")
        st.write("Введіть параметри для генерації навчального матеріалу та тестових завдань")
        
        col1, col2 = st.columns(2)
        
        with col1:
            subject = st.selectbox(
                "Предмет:",
                ["algebra", "ukrainian_language", "history_ukraine"],
                format_func=lambda x: {
                    "algebra": "Алгебра",
                    "ukrainian_language": "Українська мова",
                    "history_ukraine": "Історія України"
                }[x]
            )
        
        with col2:
            grade = st.selectbox("Клас:", [8, 9])
        
        topic = st.text_input(
            "Тема уроку:",
            placeholder="Наприклад: Квадратні рівняння, Словосполучення, Козацька доба"
        )
        
        if st.button("🚀 Згенерувати матеріал", type="primary", use_container_width=True):
            if not topic:
                st.warning("⚠️ Будь ласка, введіть тему уроку")
            else:
                with st.spinner("⏳ Генерація матеріалу... Це може зайняти кілька секунд"):
                    # Виклик API
                    request_data = {
                        "topic": topic,
                        "grade": grade,
                        "subject": subject
                    }
                    
                    result = call_api("/api/generate-material", method="POST", data=request_data)
                    
                    if result:
                        st.session_state.generated_material = result
                        st.success("✅ Матеріал успішно згенеровано!")
        
        # Відображення згенерованого матеріалу
        if st.session_state.generated_material:
            material = st.session_state.generated_material
            
            st.markdown("---")
            display_material_section(material)
            
            st.markdown("---")
            display_exercises_section(material['exercises'])
            
            # Кнопка скидання
            if st.button("🔄 Згенерувати нову тему"):
                st.session_state.generated_material = None
                st.session_state.student_answers = {}
                st.session_state.assessment_result = None
                st.rerun()
    
    # ========================================================================
    # Режим 2: Проходження тесту (для учня)
    # ========================================================================
    else:
        st.markdown("## 👨‍🎓 Проходження тестування")
        
        if not st.session_state.generated_material:
            st.warning("⚠️ Спочатку потрібно згенерувати матеріал у режимі вчителя")
            if st.button("➡️ Перейти до генерації матеріалів"):
                st.rerun()
        else:
            material = st.session_state.generated_material
            
            # Інформація про тест
            st.info(f"""
            📚 **Тема:** {material['topic']}  
            📊 **Предмет:** {material['subject']}  
            🎓 **Клас:** {material['grade']}  
            ✍️ **Завдань:** {len(material['exercises'])}
            """)
            
            # Опціональна інформація про учня
            with st.expander("👤 Профіль учня (опціонально)", expanded=False):
                student_id = st.number_input("ID учня:", min_value=1, value=12345)
                school_id = st.number_input("ID школи:", min_value=1, value=1)
            
            st.markdown("---")
            
            # Вкладки для навчання та тестування
            tab1, tab2 = st.tabs(["📖 Вивчити матеріал", "✍️ Пройти тест"])
            
            with tab1:
                display_material_section(material)
            
            with tab2:
                if not st.session_state.assessment_result:
                    st.write("Дайте відповіді на всі питання та натисніть кнопку для перевірки")
                    
                    # Відображення питань
                    for idx, exercise in enumerate(material['exercises']):
                        display_exercise_card(exercise, idx, show_answer=False)
                    
                    # Кнопка відправки
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("📤 Відправити відповіді", type="primary", use_container_width=True):
                            # Перевірка, чи всі питання відповіли
                            answered = len(st.session_state.student_answers)
                            total = len(material['exercises'])
                            
                            if answered < total:
                                st.warning(f"⚠️ Відповідайте на всі питання! Відповіли: {answered}/{total}")
                            else:
                                with st.spinner("⏳ Перевірка відповідей..."):
                                    # Формуємо запит
                                    student_answers = [
                                        {
                                            "question_id": qid,
                                            "selected_answer_index": ans
                                        }
                                        for qid, ans in st.session_state.student_answers.items()
                                    ]
                                    
                                    request_data = {
                                        "student_answers": student_answers,
                                        "exercises": material['exercises'],
                                        "student_profile": {
                                            "student_id": student_id,
                                            "grade": material['grade'],
                                            "school_id": school_id
                                        }
                                    }
                                    
                                    result = call_api("/api/assess-student", method="POST", data=request_data)
                                    
                                    if result:
                                        st.session_state.assessment_result = result
                                        st.rerun()
                else:
                    # Відображення результатів
                    display_assessment_section(st.session_state.assessment_result)
                    
                    # Кнопка для нового тесту
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔄 Пройти новий тест", use_container_width=True):
                            st.session_state.generated_material = None
                            st.session_state.student_answers = {}
                            st.session_state.assessment_result = None
                            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #95a5a6; padding: 2rem 0;'>
        <p>Мрія AI Tutor | Lapathon 2026 | Створено з ❤️ для українських учнів та вчителів</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
