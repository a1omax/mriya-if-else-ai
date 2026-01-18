"""
vLLM Client for local model inference with structured JSON output
Uses guided decoding / JSON format enforcement for reliable structured outputs
"""
import json
import requests
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import Enum

class MistakeSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Mistake(BaseModel):
    """A single mistake made by the student"""
    type: str = Field(description="Type of mistake (conceptual, factual, grammatical, etc.)")
    description: str = Field(description="Detailed description of the mistake")
    severity: MistakeSeverity = Field(description="Severity level: low, medium, or high")


class StudentEvaluation(BaseModel):
    """Structured evaluation of a student's response"""
    is_correct: bool = Field(description="Whether the answer is correct")
    score: int = Field(ge=0, le=100, description="Score from 0 to 100")
    summary: str = Field(description="Brief summary of the evaluation (1-2 sentences)")
    mistakes: List[Mistake] = Field(default_factory=list, description="List of mistakes made")
    weak_terms: List[str] = Field(default_factory=list, description="Terms the student misunderstood")
    weak_concepts: List[str] = Field(default_factory=list, description="Concepts needing work")
    weak_skills: List[str] = Field(default_factory=list, description="Skills needing improvement")
    correct_aspects: List[str] = Field(default_factory=list, description="What the student got right")
    suggested_topics: List[str] = Field(default_factory=list, description="Topics for review")
    explanation: str = Field(description="Detailed explanation of the correct answer")
    encouragement: str = Field(description="Supportive message for the student")


# Generate JSON schema from Pydantic model
EVALUATION_JSON_SCHEMA = StudentEvaluation.model_json_schema()


SYSTEM_PROMPT_EVALUATION = """Ти — досвідчений вчитель-експерт з оцінювання навчальних досягнень учнів.
Твоя задача — проаналізувати відповідь учня та надати детальну оцінку у форматі JSON.
Формат відповіді:
{
    "is_correct": true/false,
    "score": 0-100,
    "summary": "Короткий підсумок (1-2 речення)",
    "mistakes": [
        {"type": "тип помилки", "description": "детальний опис помилки", "severity": "low/medium/high"}
    ],
    "weak_terms": ["терміни, які учень не зрозумів або використав неправильно"],
    "weak_concepts": ["ширші концепції, які потребують роботи"],
    "weak_skills": ["навички, які потребують покращення"],
    "correct_aspects": ["що учень зробив правильно"],
    "suggested_topics": ["теми для повторення"],
    "explanation": "Детальне пояснення правильної відповіді",
    "encouragement": "Підтримуюче повідомлення для учня"
}
Будь справедливим, конструктивним та підтримуючим у своїх оцінках.
Відповідай ТІЛЬКИ валідним JSON без додаткового тексту."""

SYSTEM_PROMPT_EVALUATION_WITH_REASONING = """Ти — досвідчений вчитель-експерт з оцінювання навчальних досягнень учнів.
Твоя задача — проаналізувати відповідь учня та надати детальну оцінку у форматі JSON.
Перед тим як дати оцінку, згенеруй свої розгорнуті розмірковування стосовно відповіді учня, виділи розмірковування <think> токеном.
Формат відповіді:
{
    "is_correct": true/false,
    "score": 0-100,
    "summary": "Короткий підсумок (1-2 речення)",
    "mistakes": [
        {"type": "тип помилки", "description": "детальний опис помилки", "severity": "low/medium/high"}
    ],
    "weak_terms": ["терміни, які учень не зрозумів або використав неправильно"],
    "weak_concepts": ["ширші концепції, які потребують роботи"],
    "weak_skills": ["навички, які потребують покращення"],
    "correct_aspects": ["що учень зробив правильно"],
    "suggested_topics": ["теми для повторення"],
    "explanation": "Детальне пояснення правильної відповіді",
    "encouragement": "Підтримуюче повідомлення для учня"
}
Будь справедливим, конструктивним та підтримуючим у своїх оцінках."""


@dataclass
class VLLMConfig:
    """Configuration for vLLM server"""
    host: str = "localhost"
    port: int = 8000
    model_path: str = "models/supervised_v1"
    adapter_path: str = "models/adapter_v1"
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"
    
    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


class VLLMClient:
    """
    Client for interacting with vLLM server using OpenAI-compatible API
    Supports structured JSON output via guided decoding
    """
    
    def __init__(
        self,
        config: Optional[VLLMConfig] = None,
        use_reasoning: bool = True,
        use_guided_json: bool = True,
        structured_output_mode: str = "guided_json",  # "guided_json", "response_format", or "none"
        timeout: int = 300,  # Increased to 5 minutes
        stream: bool = False
    ):
        """
        Initialize vLLM client
        
        Args:
            config: VLLMConfig instance
            use_reasoning: Whether to use the reasoning prompt (with <think> tags)
            use_guided_json: Whether to use guided JSON decoding for structured output
            structured_output_mode: Method for structured output:
                - "guided_json": Use vLLM's guided_json parameter
                - "response_format": Use OpenAI-style response_format
                - "none": No structured output enforcement
            timeout: Request timeout in seconds (default: 300 for guided decoding)
            stream: Whether to use streaming responses
        """
        self.config = config or VLLMConfig()
        self.use_reasoning = use_reasoning
        self.use_guided_json = use_guided_json
        self.structured_output_mode = structured_output_mode if use_guided_json else "none"
        self.timeout = timeout
        self.stream = stream
        
        # Use appropriate system prompt
        self.system_prompt = (
            SYSTEM_PROMPT_EVALUATION_WITH_REASONING 
            if use_reasoning 
            else SYSTEM_PROMPT_EVALUATION
        )
    
    def check_health(self) -> bool:
        """Check if vLLM server is running"""
        try:
            response = requests.get(self.config.health_url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        json_schema: Optional[Dict] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Generate completion from the model
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            json_schema: Optional JSON schema for guided decoding
            timeout: Override default timeout
            
        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.config.model_path,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": self.stream
        }
        
        # Add structured output based on mode - use only ONE method
        if json_schema and self.structured_output_mode == "guided_json":
            # vLLM native guided decoding
            payload["guided_json"] = json_schema
        elif json_schema and self.structured_output_mode == "response_format":
            # OpenAI-compatible response_format
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "student_evaluation",
                    "schema": json_schema
                }
            }
        # If mode is "none", don't add any structured output constraints
        
        request_timeout = timeout or self.timeout
        
        try:
            if self.stream:
                return self._generate_streaming(payload, request_timeout)
            else:
                response = requests.post(
                    self.config.completions_url,
                    json=payload,
                    timeout=request_timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request timed out after {request_timeout}s. "
                f"Try increasing timeout or disabling guided_json for faster responses."
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to vLLM server at {self.config.base_url}. "
                f"Make sure the server is running."
            )
        except Exception as e:
            raise RuntimeError(f"vLLM request failed: {e}")
    
    def _generate_streaming(self, payload: Dict, timeout: int) -> str:
        """Generate with streaming responses"""
        payload["stream"] = True
        
        full_response = ""
        
        try:
            with requests.post(
                self.config.completions_url,
                json=payload,
                timeout=timeout,
                stream=True
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                full_response += content
                            except json.JSONDecodeError:
                                continue
            
            return full_response
            
        except Exception as e:
            if full_response:
                return full_response  # Return partial response if available
            raise e
    
    def generate_structured(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: Optional[int] = None
    ) -> StudentEvaluation:
        """
        Generate structured evaluation using JSON schema enforcement
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Optional timeout override
            
        Returns:
            StudentEvaluation object
        """
        # Generate with JSON schema
        raw_output = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=EVALUATION_JSON_SCHEMA if self.use_guided_json else None,
            timeout=timeout
        )
        
        # Parse and validate
        return self._parse_evaluation(raw_output)
    
    def _parse_evaluation(self, raw_output: str) -> StudentEvaluation:
        """
        Parse raw output into StudentEvaluation object
        Handles both JSON-only and <think>JSON formats
        """
        import re
        
        json_str = raw_output
        
        # Try to extract JSON after </think> tags
        think_pattern = r'</think>\s*(\{[\s\S]*\})'
        match = re.search(think_pattern, raw_output)
        if match:
            json_str = match.group(1)
        else:
            # Try to find any JSON object
            json_pattern = r'\{[\s\S]*"is_correct"[\s\S]*\}'
            match = re.search(json_pattern, raw_output)
            if match:
                json_str = match.group(0)
        
        try:
            data = json.loads(json_str)
            return StudentEvaluation(**data)
        except (json.JSONDecodeError, Exception) as e:
            # Return a default evaluation with error info
            return StudentEvaluation(
                is_correct=False,
                score=0,
                summary=f"Failed to parse model output: {str(e)}",
                mistakes=[],
                weak_terms=[],
                weak_concepts=[],
                weak_skills=[],
                correct_aspects=[],
                suggested_topics=[],
                explanation=f"Raw output: {raw_output[:500]}...",
                encouragement="Please try again."
            )
    
    def analyze_student_response(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
        student_explanation: str = "",
        return_structured: bool = False,
        timeout: Optional[int] = None
    ) -> Union[str, StudentEvaluation]:
        """
        Analyze a student's response
        
        Args:
            question: The question asked
            correct_answer: The correct answer
            student_answer: Student's answer
            student_explanation: Student's explanation (optional)
            return_structured: If True, returns StudentEvaluation object
            timeout: Optional timeout override
            
        Returns:
            Model's analysis (JSON string or StudentEvaluation object)
        """
        prompt = f"""Проаналізуй відповідь учня на наступне питання:

Питання: {question}

Правильна відповідь: {correct_answer}

Відповідь учня: {student_answer}

Пояснення учня: {student_explanation or "Не надано"}

Оціни відповідь учня та надай детальний аналіз у форматі JSON."""
        
        if return_structured:
            return self.generate_structured(prompt, timeout=timeout)
        else:
            return self.generate(
                prompt,
                json_schema=EVALUATION_JSON_SCHEMA if self.use_guided_json else None,
                timeout=timeout
            )
    
    def analyze_batch(
        self,
        responses: List[Dict[str, str]],
        return_structured: bool = False
    ) -> List[Union[str, StudentEvaluation]]:
        """
        Analyze multiple student responses
        
        Args:
            responses: List of dicts with question, correct_answer, student_answer, student_explanation
            return_structured: If True, returns StudentEvaluation objects
            
        Returns:
            List of analysis results
        """
        results = []
        for resp in responses:
            try:
                result = self.analyze_student_response(
                    question=resp.get("question", ""),
                    correct_answer=resp.get("correct_answer", ""),
                    student_answer=resp.get("student_answer", ""),
                    student_explanation=resp.get("student_explanation", ""),
                    return_structured=return_structured
                )
                results.append(result)
            except Exception as e:
                if return_structured:
                    results.append(StudentEvaluation(
                        is_correct=False,
                        score=0,
                        summary=f"Error: {str(e)}",
                        mistakes=[],
                        weak_terms=[],
                        weak_concepts=[],
                        weak_skills=[],
                        correct_aspects=[],
                        suggested_topics=[],
                        explanation="",
                        encouragement=""
                    ))
                else:
                    results.append(f"Error: {str(e)}")
        
        return results


class LMFormatEnforcerClient(VLLMClient):
    """
    Extended client that uses lm-format-enforcer for guaranteed JSON output
    Requires: pip install lm-format-enforcer
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_enforcer()
    
    def _check_enforcer(self):
        """Check if lm-format-enforcer is available"""
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.vllm import build_vllm_logits_processor
            self.enforcer_available = True
        except ImportError:
            self.enforcer_available = False
            print("Warning: lm-format-enforcer not installed. Using basic JSON parsing.")
            print("Install with: pip install lm-format-enforcer")
    
    def generate_with_enforcer(
        self,
        prompt: str,
        schema: Dict,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate with lm-format-enforcer for guaranteed valid JSON
        
        Note: This requires direct model access, not API.
        For API usage, use the guided_json parameter instead.
        """
        if not self.enforcer_available:
            # Fallback to regular generation
            raw = self.generate(prompt, max_tokens, temperature, json_schema=schema)
            return json.loads(raw)
        
        # For API usage, we use guided_json which is the API equivalent
        raw = self.generate(prompt, max_tokens, temperature, json_schema=schema)
        return json.loads(raw)


def get_evaluation_schema() -> Dict:
    """Get the JSON schema for student evaluation"""
    return EVALUATION_JSON_SCHEMA


def get_vllm_launch_command(
    model_path: str = "models/supervised_v1",
    adapter_path: str = "models/adapter_v1",
    port: int = 8000,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    dtype: str = "auto"
) -> str:
    """
    Generate the vLLM server launch command with guided decoding support
    """
    cmd = f"""python -m vllm.entrypoints.openai.api_server \\
    --model {model_path} \\
    --enable-lora \\
    --lora-modules adapter={adapter_path} \\
    --port {port} \\
    --host {host} \\
    --tensor-parallel-size {tensor_parallel_size} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --max-model-len {max_model_len} \\
    --dtype {dtype} \\
    --guided-decoding-backend outlines \\
    --trust-remote-code"""
    
    return cmd


def print_vllm_setup_instructions():
    """Print setup instructions for vLLM with guided decoding"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    vLLM Setup with Guided Decoding                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  1. Install vLLM with guided decoding support:                                ║
║     pip install vllm outlines lm-format-enforcer                              ║
║                                                                               ║
║  2. Start the vLLM server:                                                    ║
║                                                                               ║
║     python -m vllm.entrypoints.openai.api_server \\                           ║
║         --model models/supervised_v1 \\                                        ║
║         --enable-lora \\                                                       ║
║         --lora-modules adapter=models/adapter_v1 \\                            ║
║         --port 8000 \\                                                         ║
║         --guided-decoding-backend outlines \\                                  ║
║         --trust-remote-code                                                   ║
║                                                                               ║
║  3. The client will automatically use guided_json for structured output       ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_vllm_setup_instructions()
    
    print("\nJSON Schema for Student Evaluation:")
    print(json.dumps(EVALUATION_JSON_SCHEMA, indent=2, ensure_ascii=False))
    
    print("\n\nExample usage:")
    print("""
from vllm_client import VLLMClient, StudentEvaluation

# Initialize client with guided JSON
client = VLLMClient(use_guided_json=True)

# Get structured response
evaluation: StudentEvaluation = client.analyze_student_response(
    question="Що таке підмет?",
    correct_answer="Підмет - головний член речення, що означає предмет...",
    student_answer="Це слово в реченні",
    return_structured=True
)

print(f"Score: {evaluation.score}")
print(f"Weak concepts: {evaluation.weak_concepts}")
print(f"Suggested topics: {evaluation.suggested_topics}")
""")