"""
LangGraph Agent for Student Solution Analysis and Topic Recommendation
Updated with SentenceTransformer embeddings and vLLM integration
"""
import json
import re
from typing import Annotated, TypedDict, Literal, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from knowledge_graph import (
    KnowledgeGraphManager, TopicMatcher, 
    create_topic_links
)
from vllm_client import (
    VLLMClient, VLLMConfig, StudentEvaluation,
    SYSTEM_PROMPT_EVALUATION_WITH_REASONING, EVALUATION_JSON_SCHEMA
)


class GraphState(TypedDict):
    """State for the LangGraph"""
    # Input
    user_prompt: str
    question: Optional[str]
    correct_answer: Optional[str]
    student_answer: Optional[str]
    student_explanation: Optional[str]
    model_output: Optional[str]
    knowledge_graph_json: Optional[str]
    
    # Processing
    parsed_response: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    
    # Knowledge graph and embeddings
    kg_manager: Optional[Any]
    embedding_manager: Optional[Any]
    semantic_matcher: Optional[Any]
    topic_matches: List[Dict[str, Any]]
    topic_links: Optional[Dict[str, Any]]
    semantic_matches: Optional[Dict[str, Any]]
    
    # Output
    final_recommendations: Optional[Dict[str, Any]]
    
    # Metadata
    messages: Annotated[list, add_messages]
    processing_steps: List[str]
    errors: List[str]
    
    # Config
    use_vllm: bool
    vllm_config: Optional[Dict[str, Any]]
    use_embeddings: bool
    embedding_model: Optional[str]


def analyze_with_vllm(state: GraphState) -> GraphState:
    """
    Node 0: Analyze student response using vLLM (optional)
    Supports structured JSON output via guided decoding
    """
    processing_steps = state.get("processing_steps", [])
    errors = state.get("errors", [])
    processing_steps.append("analyze_with_vllm")
    
    # Skip if model_output already provided or vLLM not enabled
    if state.get("model_output") or not state.get("use_vllm", False):
        return {**state, "processing_steps": processing_steps, "errors": errors}
    
    # Check if we have the required inputs
    question = state.get("question")
    student_answer = state.get("student_answer")
    
    if not question or not student_answer:
        errors.append("Missing question or student_answer for vLLM analysis")
        return {**state, "processing_steps": processing_steps, "errors": errors}
    
    try:
        # Initialize vLLM client
        vllm_config_dict = state.get("vllm_config", {})
        config = VLLMConfig(
            host=vllm_config_dict.get("host", "localhost"),
            port=vllm_config_dict.get("port", 8000),
            model_path=vllm_config_dict.get("model_path", "models/supervised_v1"),
            adapter_path=vllm_config_dict.get("adapter_path", "models/adapter_v1")
        )
        
        # Use guided JSON for structured output
        structured_output_mode = vllm_config_dict.get("structured_output_mode", "none")
        use_guided_json = structured_output_mode != "none"
        timeout = vllm_config_dict.get("timeout", 300)
        
        client = VLLMClient(
            config=config, 
            use_reasoning=True,
            use_guided_json=use_guided_json,
            structured_output_mode=structured_output_mode,
            timeout=timeout
        )
        
        # Check if server is running
        if not client.check_health():
            errors.append(f"vLLM server not available at {config.base_url}")
            return {**state, "processing_steps": processing_steps, "errors": errors}
        
        # Generate analysis - try structured first, fall back to raw
        try:
            evaluation = client.analyze_student_response(
                question=question,
                correct_answer=state.get("correct_answer", ""),
                student_answer=student_answer,
                student_explanation=state.get("student_explanation", ""),
                return_structured=True
            )
            
            # Convert StudentEvaluation to dict for compatibility
            model_output = evaluation.model_dump_json(indent=2)
            
            # Also store parsed response directly
            return {
                **state,
                "model_output": model_output,
                "parsed_response": evaluation.model_dump(),
                "processing_steps": processing_steps,
                "errors": errors
            }
            
        except Exception as e:
            # Fall back to raw output
            errors.append(f"Structured output failed, using raw: {str(e)}")
            model_output = client.analyze_student_response(
                question=question,
                correct_answer=state.get("correct_answer", ""),
                student_answer=student_answer,
                student_explanation=state.get("student_explanation", ""),
                return_structured=False
            )
            
            return {
                **state,
                "model_output": model_output,
                "processing_steps": processing_steps,
                "errors": errors
            }
        
    except Exception as e:
        errors.append(f"vLLM analysis failed: {str(e)}")
        return {**state, "processing_steps": processing_steps, "errors": errors}


def parse_model_output(state: GraphState) -> GraphState:
    """
    Node 1: Parse the model output to extract JSON analysis
    """
    processing_steps = state.get("processing_steps", [])
    errors = state.get("errors", [])
    processing_steps.append("parse_model_output")
    
    model_output = state.get("model_output", "")
    
    if not model_output:
        errors.append("No model output provided")
        return {**state, "processing_steps": processing_steps, "errors": errors}
    
    # Try to extract JSON from the output
    parsed_response = None
    
    # Pattern 1: Look for JSON after </think> tags
    think_pattern = r'</think>\s*(\{[\s\S]*\})'
    match = re.search(think_pattern, model_output)
    if match:
        try:
            parsed_response = json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Pattern 2: Look for any JSON object with is_correct field
    if not parsed_response:
        json_pattern = r'\{[\s\S]*"is_correct"[\s\S]*\}'
        match = re.search(json_pattern, model_output)
        if match:
            try:
                parsed_response = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    
    # Pattern 3: Try to find JSON between code blocks
    if not parsed_response:
        code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        match = re.search(code_block_pattern, model_output)
        if match:
            try:
                parsed_response = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    
    # Pattern 4: Try the entire output as JSON
    if not parsed_response:
        try:
            parsed_response = json.loads(model_output)
        except json.JSONDecodeError:
            pass
    
    if parsed_response:
        return {
            **state, 
            "parsed_response": parsed_response,
            "processing_steps": processing_steps,
            "errors": errors
        }
    else:
        errors.append("Failed to parse JSON from model output")
        return {**state, "processing_steps": processing_steps, "errors": errors}


def validate_analysis(state: GraphState) -> GraphState:
    """
    Node 2: Validate and structure the parsed analysis
    """
    processing_steps = state.get("processing_steps", [])
    errors = state.get("errors", [])
    processing_steps.append("validate_analysis")
    
    parsed = state.get("parsed_response")
    
    if not parsed:
        errors.append("No parsed response to validate")
        return {**state, "processing_steps": processing_steps, "errors": errors}
    
    # Validate required fields and provide defaults
    required_fields = [
        "is_correct", "score", "summary", "mistakes", "weak_terms",
        "weak_concepts", "weak_skills", "correct_aspects", 
        "suggested_topics", "explanation", "encouragement"
    ]
    
    analysis_result = {}
    for field in required_fields:
        if field in parsed:
            analysis_result[field] = parsed[field]
        else:
            # Provide defaults
            if field in ["mistakes", "weak_terms", "weak_concepts", 
                        "weak_skills", "correct_aspects", "suggested_topics"]:
                analysis_result[field] = []
            elif field == "is_correct":
                analysis_result[field] = False
            elif field == "score":
                analysis_result[field] = 0
            else:
                analysis_result[field] = ""
    
    # Validate score range
    if not 0 <= analysis_result.get("score", 0) <= 100:
        analysis_result["score"] = max(0, min(100, analysis_result.get("score", 0)))
    
    return {
        **state,
        "analysis_result": analysis_result,
        "processing_steps": processing_steps,
        "errors": errors
    }


def load_knowledge_graph_and_embeddings(state: GraphState) -> GraphState:
    """
    Node 3: Load knowledge graph and create embeddings
    """
    processing_steps = state.get("processing_steps", [])
    errors = state.get("errors", [])
    processing_steps.append("load_knowledge_graph_and_embeddings")
    
    kg_json = state.get("knowledge_graph_json")
    
    if not kg_json:
        errors.append("No knowledge graph provided")
        return {**state, "processing_steps": processing_steps, "errors": errors}
    
    try:
        kg_data = json.loads(kg_json) if isinstance(kg_json, str) else kg_json
        
        # Create KnowledgeGraph object
        from models import Edge, KnowledgeGraph
        
        nodes = kg_data.get("nodes", [])
        edges_data = kg_data.get("edges", [])
        
        edges = []
        for e in edges_data:
            try:
                edges.append(Edge(
                    source=e["source"],
                    target=e["target"],
                    relation=e["relation"],
                    confidence=e.get("confidence")
                ))
            except Exception:
                continue
        
        kg = KnowledgeGraph(nodes=nodes, edges=edges)
        kg_manager = KnowledgeGraphManager(kg)
        
        # Initialize embeddings if enabled
        embedding_manager = None
        semantic_matcher = None
        
        if state.get("use_embeddings", True):
            try:
                from embeddings import EmbeddingManager, SemanticTopicMatcher
                
                model_name = state.get(
                    "embedding_model", 
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                
                embedding_manager = EmbeddingManager(model_name=model_name)
                embedding_manager.embed_knowledge_graph(nodes)
                
                semantic_matcher = SemanticTopicMatcher(embedding_manager)
                
                processing_steps.append("embeddings_created")
                
            except ImportError as e:
                errors.append(f"Embeddings not available: {e}. Install sentence-transformers.")
            except Exception as e:
                errors.append(f"Failed to create embeddings: {e}")
        
        return {
            **state,
            "kg_manager": kg_manager,
            "embedding_manager": embedding_manager,
            "semantic_matcher": semantic_matcher,
            "processing_steps": processing_steps,
            "errors": errors
        }
        
    except Exception as e:
        errors.append(f"Failed to load knowledge graph: {str(e)}")
        return {**state, "processing_steps": processing_steps, "errors": errors}


def match_topics_semantically(state: GraphState) -> GraphState:
    """
    Node 4: Match suggested topics using semantic embeddings
    """
    processing_steps = state.get("processing_steps", [])
    errors = state.get("errors", [])
    processing_steps.append("match_topics_semantically")
    
    analysis = state.get("analysis_result")
    semantic_matcher = state.get("semantic_matcher")
    kg_manager = state.get("kg_manager")
    
    if not analysis:
        errors.append("No analysis result available for topic matching")
        return {**state, "processing_steps": processing_steps, "errors": errors}
    
    suggested_topics = analysis.get("suggested_topics", [])
    weak_terms = analysis.get("weak_terms", [])
    weak_concepts = analysis.get("weak_concepts", [])
    
    topic_matches = []
    semantic_matches = {
        "topic_matches": [],
        "term_matches": {},
        "concept_matches": {}
    }
    
    # Use semantic matcher if available
    if semantic_matcher:
        # Match suggested topics
        for suggested in suggested_topics:
            matches = semantic_matcher.find_matching_topics(suggested, top_k=3)
            for match in matches:
                topic_matches.append({
                    "suggested_topic": suggested,
                    "matched_topic": match
                })
            semantic_matches["topic_matches"].append({
                "suggested": suggested,
                "matches": matches
            })
        
        # Match weak terms
        semantic_matches["term_matches"] = semantic_matcher.find_related_terms(
            weak_terms, top_k=3
        )
        
        # Match weak concepts
        semantic_matches["concept_matches"] = semantic_matcher.find_related_concepts(
            weak_concepts, top_k=3
        )
    
    # Fallback to fuzzy matching if no semantic matcher
    elif kg_manager:
        matcher = TopicMatcher(kg_manager)
        for suggested in suggested_topics:
            matches = matcher.find_best_matches(suggested)
            for match in matches:
                match = matcher.enrich_match_with_relations(match)
                topic_matches.append({
                    "suggested_topic": suggested,
                    "matched_topic": match.dict()
                })
    
    return {
        **state,
        "topic_matches": topic_matches,
        "semantic_matches": semantic_matches,
        "processing_steps": processing_steps,
        "errors": errors
    }


def link_concepts_to_graph(state: GraphState) -> GraphState:
    """
    Node 5: Link weak terms and concepts to knowledge graph
    """
    processing_steps = state.get("processing_steps", [])
    errors = state.get("errors", [])
    processing_steps.append("link_concepts_to_graph")
    
    analysis = state.get("analysis_result")
    kg_manager = state.get("kg_manager")
    
    if not analysis or not kg_manager:
        return {**state, "processing_steps": processing_steps, "errors": errors}
    
    weak_terms = analysis.get("weak_terms", [])
    weak_concepts = analysis.get("weak_concepts", [])
    suggested_topics = analysis.get("suggested_topics", [])
    
    # Create comprehensive topic links
    topic_links = create_topic_links(
        kg_manager,
        suggested_topics,
        weak_terms,
        weak_concepts
    )
    
    return {
        **state,
        "topic_links": topic_links,
        "processing_steps": processing_steps,
        "errors": errors
    }


def generate_recommendations(state: GraphState) -> GraphState:
    """
    Node 6: Generate final recommendations combining all analysis
    """
    processing_steps = state.get("processing_steps", [])
    errors = state.get("errors", [])
    processing_steps.append("generate_recommendations")
    
    analysis = state.get("analysis_result", {})
    if analysis is None:
        analysis = {}
    topic_matches = state.get("topic_matches", [])
    if topic_matches is None:
        topic_matches = []
    topic_links = state.get("topic_links", {})
    if topic_links is None:
        topic_links = {}
    semantic_matches = state.get("semantic_matches", {})
    if semantic_matches is None:
        semantic_matches = {}

    # Build final recommendations
    recommendations = {
        "student_assessment": {
            "is_correct": analysis.get("is_correct", False),
            "score": analysis.get("score", 0),
            "summary": analysis.get("summary", ""),
            "explanation": analysis.get("explanation", ""),
            "encouragement": analysis.get("encouragement", "")
        },
        "identified_weaknesses": {
            "terms": analysis.get("weak_terms", []),
            "concepts": analysis.get("weak_concepts", []),
            "skills": analysis.get("weak_skills", []),
            "mistakes": analysis.get("mistakes", [])
        },
        "topic_recommendations": [],
        "semantic_topic_matches": semantic_matches.get("topic_matches", []) if semantic_matches else [],
        "related_terms": semantic_matches.get("term_matches", {}) if semantic_matches else {},
        "related_concepts": semantic_matches.get("concept_matches", {}) if semantic_matches else {},
        "learning_resources": [],
        "learning_path": topic_links.get("recommended_learning_path", []) if topic_links else []
    }
    
    # Process topic matches
    seen_topics = set()
    for match in topic_matches:
        matched = match.get("matched_topic", {})
        topic_id = matched.get("topic_id", "")
        
        if topic_id and topic_id not in seen_topics:
            seen_topics.add(topic_id)
            recommendations["topic_recommendations"].append({
                "suggested": match.get("suggested_topic", ""),
                "curriculum_topic": matched.get("topic_title", ""),
                "topic_id": topic_id,
                "semantic_score": matched.get("semantic_score", 0),
                "fuzzy_score": matched.get("fuzzy_score", 0),
                "combined_score": matched.get("combined_score", matched.get("similarity_score", 0)),
                "match_type": matched.get("match_type", ""),
                "summary": matched.get("summary", ""),
                "related_resources": matched.get("related_nodes", [])
            })
    
    # Sort by combined score
    recommendations["topic_recommendations"].sort(
        key=lambda x: x.get("combined_score", 0), 
        reverse=True
    )
    
    # Add linked terms and concepts as resources
    if topic_links:
        for term_link in topic_links.get("term_links", []):
            node = term_link.get("node", {})
            recommendations["learning_resources"].append({
                "type": "term",
                "name": node.get("title", node.get("name", "")),
                "matched_from": term_link.get("matched_term", ""),
                "node_id": node.get("id", "")
            })
        
        for concept_link in topic_links.get("concept_links", []):
            node = concept_link.get("node", {})
            recommendations["learning_resources"].append({
                "type": "concept",
                "name": node.get("title", node.get("name", "")),
                "matched_from": concept_link.get("matched_term", ""),
                "node_id": node.get("id", "")
            })
    
    return {
        **state,
        "final_recommendations": recommendations,
        "processing_steps": processing_steps,
        "errors": errors
    }


def should_continue_to_kg(state: GraphState) -> Literal["load_kg", "skip_kg"]:
    """Router: Decide if we should process knowledge graph"""
    if state.get("knowledge_graph_json"):
        return "load_kg"
    return "skip_kg"


def should_use_vllm(state: GraphState) -> Literal["use_vllm", "skip_vllm"]:
    """Router: Decide if we should use vLLM for analysis"""
    if state.get("use_vllm", False) and not state.get("model_output"):
        return "use_vllm"
    return "skip_vllm"


def build_student_analysis_graph() -> StateGraph:
    """
    Build the complete LangGraph for student analysis
    """
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("analyze_vllm", analyze_with_vllm)
    workflow.add_node("parse_output", parse_model_output)
    workflow.add_node("validate_analysis", validate_analysis)
    workflow.add_node("load_kg_embeddings", load_knowledge_graph_and_embeddings)
    workflow.add_node("match_topics", match_topics_semantically)
    workflow.add_node("link_concepts", link_concepts_to_graph)
    workflow.add_node("generate_recommendations", generate_recommendations)
    
    # Define edges
    workflow.set_entry_point("analyze_vllm")
    
    # vLLM -> parse
    workflow.add_edge("analyze_vllm", "parse_output")
    workflow.add_edge("parse_output", "validate_analysis")
    
    # Conditional edge based on whether KG is provided
    workflow.add_conditional_edges(
        "validate_analysis",
        should_continue_to_kg,
        {
            "load_kg": "load_kg_embeddings",
            "skip_kg": "generate_recommendations"
        }
    )
    
    workflow.add_edge("load_kg_embeddings", "match_topics")
    workflow.add_edge("match_topics", "link_concepts")
    workflow.add_edge("link_concepts", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    
    return workflow


def create_agent():
    """Create and compile the agent"""
    workflow = build_student_analysis_graph()
    return workflow.compile()


# Convenience function for direct vLLM analysis
def analyze_with_local_model(
    question: str,
    correct_answer: str,
    student_answer: str,
    student_explanation: str = "",
    vllm_host: str = "localhost",
    vllm_port: int = 8000
) -> str:
    """
    Directly analyze using vLLM without the full agent
    """
    config = VLLMConfig(host=vllm_host, port=vllm_port)
    client = VLLMClient(config=config, use_reasoning=True)
    
    return client.analyze_student_response(
        question=question,
        correct_answer=correct_answer,
        student_answer=student_answer,
        student_explanation=student_explanation
    )