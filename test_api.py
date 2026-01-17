#!/usr/bin/env python3
"""
Тестовый скрипт для проверки API
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Проверка health endpoint"""
    print("🔍 Testing /health...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"✅ Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_subjects():
    """Проверка списка предметов"""
    print("\n🔍 Testing /api/subjects...")
    try:
        response = requests.get(f"{API_BASE}/api/subjects")
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"   Subjects: {len(data['subjects'])}")
        for subj in data['subjects']:
            print(f"   - {subj['name']} ({subj['id']})")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_generate_material():
    """Проверка генерации материала"""
    print("\n🔍 Testing /api/generate-material...")
    
    payload = {
        "topic": "Квадратні рівняння",
        "grade": 8,
        "subject": "algebra",
        "use_rag": True
    }
    
    try:
        print(f"   Request: {json.dumps(payload, ensure_ascii=False)}")
        response = requests.post(
            f"{API_BASE}/api/generate-material",
            json=payload,
            timeout=60
        )
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Topic: {data['topic']}")
            print(f"   Summary length: {len(data['summary'])} chars")
            print(f"   Explanation length: {len(data['explanation'])} chars")
            print(f"   Exercises: {len(data['exercises'])}")
            print(f"   RAG used: {data['rag_used']}")
            
            # Показываем первое упражнение
            if data['exercises']:
                ex = data['exercises'][0]
                print(f"\n   📝 Example exercise:")
                print(f"      Q: {ex['question_text'][:100]}...")
                print(f"      Answers: {len(ex['answers'])}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_assess_student():
    """Проверка оценки студента"""
    print("\n🔍 Testing /api/assess-student...")
    
    # Создаем mock данные
    exercises = [
        {
            "question_id": "q1",
            "question_text": "Скільки коренів має рівняння x² = 4?",
            "test_type": "single_choice",
            "answers": ["Один", "Два", "Три", "Жодного"],
            "correct_answer_indices": [1],
            "difficulty": "easy",
            "metadata": {}
        },
        {
            "question_id": "q2",
            "question_text": "Чому дорівнює x² + 2x + 1?",
            "test_type": "single_choice",
            "answers": ["(x+1)²", "(x-1)²", "(x+2)²", "x²+1"],
            "correct_answer_indices": [0],
            "difficulty": "medium",
            "metadata": {}
        }
    ]
    
    student_answers = [
        {"question_id": "q1", "selected_answer_index": 1},  # Правильно
        {"question_id": "q2", "selected_answer_index": 2},  # Неправильно
    ]
    
    payload = {
        "student_answers": student_answers,
        "exercises": exercises,
        "student_profile": {
            "student_id": 1,
            "grade": 8,
            "recent_scores": []
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/assess-student",
            json=payload,
            timeout=60
        )
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Score: {data['score']:.1f}%")
            print(f"   Correct: {data['correct_answers']}/{data['total_questions']}")
            print(f"   Recommendations: {len(data['recommendations'])}")
            print(f"   Next steps: {len(data['next_steps'])}")
            
            # Показываем анализ
            print(f"\n   📊 Analysis:")
            print(f"      {data['performance_analysis'][:200]}...")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*80)
    print("🧪 MRIIA AI TUTOR - API TEST")
    print("="*80)
    
    # Проверяем что сервер запущен
    print("\n⏳ Waiting for server to start...")
    for i in range(10):
        try:
            requests.get(f"{API_BASE}/health", timeout=2)
            print("✅ Server is ready!")
            break
        except:
            if i == 9:
                print("❌ Server not responding. Please start the server first:")
                print("   cd backend && python main.py")
                return
            time.sleep(2)
    
    # Запускаем тесты
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Subjects List", test_subjects()))
    results.append(("Generate Material", test_generate_material()))
    results.append(("Assess Student", test_assess_student()))
    
    # Итоги
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed")

if __name__ == "__main__":
    main()
