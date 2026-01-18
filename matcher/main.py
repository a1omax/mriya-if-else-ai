"""
Main entry point for Student Analysis Agent
With vLLM and SentenceTransformer embeddings support
"""
import json
from typing import Dict, Any, Optional

from agent import create_agent, GraphState, analyze_with_local_model
from knowledge_graph import KnowledgeGraphManager, TopicMatcher, create_topic_links
from vllm_client import VLLMConfig, VLLMClient, get_vllm_launch_command, print_vllm_setup_instructions


def run_analysis(
    model_output: Optional[str] = None,
    knowledge_graph_json: Optional[str] = None,
    user_prompt: Optional[str] = None,
    # vLLM options (used if model_output is None)
    use_vllm: bool = False,
    question: Optional[str] = None,
    correct_answer: Optional[str] = None,
    student_answer: Optional[str] = None,
    student_explanation: Optional[str] = None,
    vllm_host: str = "localhost",
    vllm_port: int = 8000,
    model_path: str = "models/supervised_v1",
    adapter_path: str = "models/adapter_v1",
    # Embedding options
    use_embeddings: bool = True,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Dict[str, Any]:
    """
    Run the student analysis agent
    
    Args:
        model_output: Pre-generated model output (with JSON analysis)
        knowledge_graph_json: JSON string of the knowledge graph
        user_prompt: Original user prompt (optional)
        use_vllm: Whether to use vLLM for analysis (if model_output not provided)
        question: Question asked (for vLLM)
        correct_answer: Correct answer (for vLLM)
        student_answer: Student's answer (for vLLM)
        student_explanation: Student's explanation (for vLLM)
        vllm_host: vLLM server host
        vllm_port: vLLM server port
        model_path: Path to the base model
        adapter_path: Path to the LoRA adapter
        use_embeddings: Whether to use SentenceTransformer embeddings
        embedding_model: SentenceTransformer model name
    
    Returns:
        Dictionary with analysis results and recommendations
    """
    # Create and run the agent
    agent = create_agent()
    
    initial_state: GraphState = {
        "user_prompt": user_prompt or "",
        "question": question,
        "correct_answer": correct_answer,
        "student_answer": student_answer,
        "student_explanation": student_explanation,
        "model_output": model_output,
        "knowledge_graph_json": knowledge_graph_json,
        "parsed_response": None,
        "analysis_result": None,
        "kg_manager": None,
        "embedding_manager": None,
        "semantic_matcher": None,
        "topic_matches": [],
        "topic_links": None,
        "semantic_matches": None,
        "final_recommendations": None,
        "messages": [],
        "processing_steps": [],
        "errors": [],
        "use_vllm": use_vllm,
        "vllm_config": {
            "host": vllm_host,
            "port": vllm_port,
            "model_path": model_path,
            "adapter_path": adapter_path
        },
        "use_embeddings": use_embeddings,
        "embedding_model": embedding_model
    }
    
    # Run the graph
    result = agent.invoke(initial_state)
    
    return {
        "recommendations": result.get("final_recommendations"),
        "processing_steps": result.get("processing_steps"),
        "errors": result.get("errors"),
        "analysis_result": result.get("analysis_result"),
        "topic_matches": result.get("topic_matches"),
        "semantic_matches": result.get("semantic_matches")
    }


def main():
    """Example usage with the provided data"""
    
    # Print vLLM setup instructions
    print_vllm_setup_instructions()
    
    # Example model output (from your data)
    model_output = """<think>Аналізую відповідь учня на питання. Учень допустив помилку типу 'None'. Слабкі концепції: Правила перетворення прямої мови в непряму, особливо для спонукальних речень., Розуміння структури речень з непрямою мовою. Відповідь потребує суттєвого доопрацювання.</think>
{
  "is_correct": false,
  "score": 0,
  "summary": "Відповідь учня невідома, тому оцінити її неможливо. Учень не надав жодної конкретної відповіді на поставлене питання.",
  "mistakes": [],
  "weak_terms": [
    "Непряма мова",
    "Спонукальне речення",
    "Трансформація речень"
  ],
  "weak_concepts": [
    "Правила перетворення прямої мови в непряму, особливо для спонукальних речень.",
    "Розуміння структури речень з непрямою мовою."
  ],
  "weak_skills": [
    "Аналіз речень",
    "Трансформація речень з прямої мови в непряму",
    "Застосування правил граматики"
  ],
  "correct_aspects": [],
  "suggested_topics": [
    "§ 5. Непряма мова. Заміна прямої мови непрямою",
    "Типи речень за метою висловлювання (зокрема, спонукальні)",
    "Сполучники в складнопідрядних реченнях"
  ],
  "explanation": "Правильна відповідь на питання вимагає знання правил перетворення прямої мови в непряму, особливо коли йдеться про спонукальні речення. У такому випадку, пряма мова, що виражає прохання або наказ, перетворюється на підрядне речення, що вводиться сполучником 'щоб'. Наприклад, речення «Принеси, будь ласка, квіти», — попросила мама Оксану, слід перетворити на: Мама попросила Оксану, щоб вона принесла квіти.",
  "encouragement": "Не засмучуйся! Розуміння правил перетворення прямої мови в непряму потребує практики. Переглянь ще раз теоретичний матеріал та спробуй виконати більше вправ. Ти обов'язково зможеш це освоїти!"
}"""

    # Example knowledge graph (from your data)
    knowledge_graph = {
        "nodes": [
            {
                "id": "56f52f5f3a71bc0ce9539f26ffb0a90f",
                "node_type": "book",
                "name": "«Українська мова» підручник для 9 класу загальноосвітніх навчальних закладів — Заболотний В. В., Заболотний О. В.",
                "grade": 9,
                "discipline": "Українська мова"
            },
            {
                "id": "2969158886450936084",
                "node_type": "section",
                "title": "ПРЯМА І НЕПРЯМА МОВА",
                "book_id": "56f52f5f3a71bc0ce9539f26ffb0a90f",
                "start_page": 26,
                "end_page": 47
            },
            {
                "id": "-6813754816033091347",
                "node_type": "topic",
                "title": "§ 5. Непряма мова. Заміна прямої мови непрямою",
                "type": "theoretical",
                "summary": "Непряма мова – це чуже мовлення, яке передають не дослівно, а лише зі збереженням основного змісту. Речення з непрямою мовою за будовою є складним реченням, частини якого з'єднано за допомогою сполучників або сполучних слів (займенників, прислівників), при цьому непряму мову завжди наводять після слів автора. Заміна прямої мови непрямою відбувається по-різному для розповідних, спонукальних та питальних речень. Під час заміни пропускаються вигуки, частки, зрідка вставні слова; дієслова в наказовій формі замінюються іншими формами; звертання пропускаються або стають членами речення; займенники першої та другої особи замінюються формами третьої особи. У деяких випадках пряму мову недоцільно заміняти непрямою, оскільки це може спотворити зміст висловлення.",
                "section_id": "2969158886450936084",
                "book_id": "56f52f5f3a71bc0ce9539f26ffb0a90f",
                "start_page": 30.0,
                "end_page": 33.0
            },
            {
                "id": "topic_sentence_types",
                "node_type": "topic",
                "title": "Типи речень за метою висловлювання",
                "type": "theoretical",
                "summary": "Речення поділяються на розповідні, питальні та спонукальні за метою висловлювання. Спонукальні речення виражають наказ, прохання, побажання.",
                "section_id": "section_syntax",
                "book_id": "56f52f5f3a71bc0ce9539f26ffb0a90f"
            },
            {
                "id": "topic_conjunctions",
                "node_type": "topic",
                "title": "Сполучники в складнопідрядних реченнях",
                "type": "theoretical",
                "summary": "Сполучники служать для з'єднання частин складнопідрядного речення. Сполучник 'щоб' використовується для вираження мети.",
                "book_id": "56f52f5f3a71bc0ce9539f26ffb0a90f"
            }
        ],
        "edges": [
            {
                "source": "topic:-6813754816033091347",
                "target": "term:непряма мова",
                "relation": "MENTIONS_TERM",
                "confidence": 0.95
            },
            {
                "source": "topic:-6813754816033091347",
                "target": "term:спонукальне речення",
                "relation": "MENTIONS_TERM",
                "confidence": 0.85
            },
            {
                "source": "topic:-6813754816033091347",
                "target": "term:пряма мова",
                "relation": "MENTIONS_TERM",
                "confidence": 0.9
            },
            {
                "source": "2969158886450936084",
                "target": "topic:-6813754816033091347",
                "relation": "CONTAINS_TOPIC"
            },
            {
                "source": "56f52f5f3a71bc0ce9539f26ffb0a90f",
                "target": "2969158886450936084",
                "relation": "CONTAINS_SECTION"
            }
        ]
    }

    user_prompt = "Питання: Як правильно замінити пряму мову непрямою в реченні «Принеси, будь ласка, квіти», — попросила мама Оксану, враховуючи правила трансформації спонукальних речень?"

    # Run the analysis (with pre-generated output, embeddings enabled)
    print("\n" + "=" * 80)
    print("Running analysis with pre-generated model output...")
    print("=" * 80)
    
    result = run_analysis(
        model_output=model_output,
        knowledge_graph_json=json.dumps(knowledge_graph, ensure_ascii=False),
        user_prompt=user_prompt,
        use_embeddings=False  # Set to True if sentence-transformers is installed
    )

    # Pretty print results
    print("\n📋 Processing Steps:")
    for step in result.get("processing_steps", []):
        print(f"  ✓ {step}")
    
    if result.get("errors"):
        print("\n⚠️ Errors:")
        for error in result["errors"]:
            print(f"  • {error}")
    
    recommendations = result.get("recommendations", {})
    
    if recommendations:
        print("\n📊 Student Assessment:")
        assessment = recommendations.get("student_assessment", {})
        print(f"  • Correct: {assessment.get('is_correct')}")
        print(f"  • Score: {assessment.get('score')}/100")
        print(f"  • Summary: {assessment.get('summary')}")
        
        print("\n🎯 Identified Weaknesses:")
        weaknesses = recommendations.get("identified_weaknesses", {})
        if weaknesses.get("terms"):
            print(f"  Terms: {', '.join(weaknesses['terms'])}")
        if weaknesses.get("concepts"):
            print(f"  Concepts: {weaknesses['concepts'][0][:50]}...")
        if weaknesses.get("skills"):
            print(f"  Skills: {', '.join(weaknesses['skills'])}")
        
        print("\n📚 Topic Recommendations:")
        for i, topic in enumerate(recommendations.get("topic_recommendations", [])[:5], 1):
            print(f"  {i}. {topic.get('suggested')}")
            print(f"     → Matched: {topic.get('curriculum_topic')}")
            print(f"     → Score: {topic.get('combined_score', 0):.2f}")
        
        print("\n📖 Learning Path:")
        for item in recommendations.get("learning_path", []):
            print(f"  • {item.get('topic')}")
            if item.get('section'):
                print(f"    Section: {item['section']}")
            if item.get('pages'):
                print(f"    Pages: {item['pages']}")
        
        print(f"\n💬 Encouragement: {assessment.get('encouragement')}")
    
    print("\n" + "=" * 80)
    
    # Return full result for programmatic use
    return result


def run_with_vllm(
    question: str,
    correct_answer: str,
    student_answer: str,
    student_explanation: str = "",
    knowledge_graph: Optional[Dict] = None,
    vllm_host: str = "localhost",
    vllm_port: int = 8000
) -> Dict[str, Any]:
    """
    Run analysis using local vLLM model
    
    Example:
        result = run_with_vllm(
            question="Як замінити пряму мову непрямою?",
            correct_answer="Мама попросила Оксану, щоб вона принесла квіти.",
            student_answer="Не знаю",
            knowledge_graph=my_kg
        )
    """
    return run_analysis(
        use_vllm=True,
        question=question,
        correct_answer=correct_answer,
        student_answer=student_answer,
        student_explanation=student_explanation,
        knowledge_graph_json=json.dumps(knowledge_graph) if knowledge_graph else None,
        vllm_host=vllm_host,
        vllm_port=vllm_port
    )


if __name__ == "__main__":
    result = main()
    
    # Optionally save results to file
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        # Convert non-serializable objects
        output = {
            "recommendations": result.get("recommendations"),
            "processing_steps": result.get("processing_steps"),
            "errors": result.get("errors"),
            "analysis_result": result.get("analysis_result"),
            "topic_matches": result.get("topic_matches")
        }
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Results saved to analysis_results.json")
    
    # Print vLLM command
    print("\n" + "=" * 80)
    print("To use with your local model, run vLLM with:")
    print("=" * 80)
    print(get_vllm_launch_command())