"""
Gradio Demo for Student Analysis Agent
Interactive interface to test student response analysis with knowledge graph integration

Assumes vLLM server is already running independently.
"""
import json
import os
import gradio as gr
from typing import Optional, Dict, Any, Tuple
import traceback

# Import agent components
from agent import create_agent, GraphState
from vllm_client import VLLMClient, VLLMConfig

# Default paths - adjust these for your setup
DEFAULT_KG_PATH = "../data/knowledge_graph.json"
DEFAULT_VLLM_HOST = "localhost"
DEFAULT_VLLM_PORT = 8000
DEFAULT_MODEL_PATH = "../models/supervised_v1"
DEFAULT_ADAPTER_PATH = "../models/adapter_v1"
DEFAULT_TIMEOUT = 300  # 5 minutes for guided decoding

# Global state for cached knowledge graph
CACHED_KG = None
CACHED_KG_PATH = None


def load_knowledge_graph(kg_path: str, force_reload: bool = False) -> Tuple[Optional[Dict], str]:
    """Load knowledge graph from file with caching"""
    global CACHED_KG, CACHED_KG_PATH
    
    try:
        # Return cached if available and path matches
        if not force_reload and CACHED_KG is not None and CACHED_KG_PATH == kg_path:
            nodes_count = len(CACHED_KG.get("nodes", []))
            edges_count = len(CACHED_KG.get("edges", []))
            return CACHED_KG, f"✅ Using cached KG: {nodes_count} nodes, {edges_count} edges"
        
        if not os.path.exists(kg_path):
            return None, f"❌ File not found: {kg_path}"
        
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        nodes_count = len(kg_data.get("nodes", []))
        edges_count = len(kg_data.get("edges", []))
        
        # Count by type
        type_counts = {}
        for node in kg_data.get("nodes", []):
            node_type = node.get("node_type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        # Cache it
        CACHED_KG = kg_data
        CACHED_KG_PATH = kg_path
        
        status = f"✅ Loaded knowledge graph:\n"
        status += f"   • {nodes_count} nodes, {edges_count} edges\n"
        status += f"   • Types: {', '.join(f'{k}={v}' for k, v in type_counts.items())}"
        
        return kg_data, status
    except Exception as e:
        return None, f"❌ Error loading KG: {str(e)}"


def check_vllm_status(host: str, port: int) -> Tuple[bool, str]:
    """Check if vLLM server is running"""
    try:
        config = VLLMConfig(host=host, port=int(port))
        client = VLLMClient(config=config)
        
        if client.check_health():
            return True, f"✅ vLLM server is running at {config.base_url}"
        else:
            return False, f"⚠️ vLLM server not responding at {config.base_url}"
    except Exception as e:
        return False, f"❌ Error connecting to vLLM: {str(e)}"


def format_recommendations(result: Dict[str, Any]) -> str:
    """Format recommendations as markdown"""
    if not result or not result.get("recommendations"):
        return "No recommendations available"
    
    rec = result["recommendations"]
    
    md = "## 📊 Student Assessment\n\n"
    
    assessment = rec.get("student_assessment", {})
    is_correct = "✅ Correct" if assessment.get("is_correct") else "❌ Incorrect"
    md += f"**Result:** {is_correct}\n\n"
    md += f"**Score:** {assessment.get('score', 0)}/100\n\n"
    md += f"**Summary:** {assessment.get('summary', 'N/A')}\n\n"
    
    md += "---\n\n"
    md += "## 🎯 Identified Weaknesses\n\n"
    
    weaknesses = rec.get("identified_weaknesses", {})
    
    if weaknesses.get("terms"):
        md += "**Weak Terms:**\n"
        for term in weaknesses["terms"]:
            md += f"- {term}\n"
        md += "\n"
    
    if weaknesses.get("concepts"):
        md += "**Weak Concepts:**\n"
        for concept in weaknesses["concepts"]:
            md += f"- {concept}\n"
        md += "\n"
    
    if weaknesses.get("skills"):
        md += "**Weak Skills:**\n"
        for skill in weaknesses["skills"]:
            md += f"- {skill}\n"
        md += "\n"
    
    if weaknesses.get("mistakes"):
        md += "**Mistakes:**\n"
        for mistake in weaknesses["mistakes"]:
            if isinstance(mistake, dict):
                md += f"- [{mistake.get('severity', 'N/A')}] {mistake.get('type', 'N/A')}: {mistake.get('description', 'N/A')}\n"
            else:
                md += f"- {mistake}\n"
        md += "\n"
    
    md += "---\n\n"
    md += "## 📚 Topic Recommendations\n\n"
    
    topic_recs = rec.get("topic_recommendations", [])
    if topic_recs:
        for i, topic in enumerate(topic_recs[:5], 1):
            md += f"### {i}. {topic.get('curriculum_topic', 'N/A')}\n"
            md += f"- **Suggested from:** {topic.get('suggested', 'N/A')}\n"
            score = topic.get('combined_score', 0)
            if isinstance(score, (int, float)):
                md += f"- **Match Score:** {score:.2%}\n"
            if topic.get('semantic_score'):
                md += f"- **Semantic Score:** {topic.get('semantic_score', 0):.2%}\n"
            if topic.get('summary'):
                md += f"- **Summary:** {topic.get('summary', '')[:200]}...\n"
            md += "\n"
    else:
        md += "*No topic recommendations found*\n\n"
    
    md += "---\n\n"
    md += "## 📖 Learning Path\n\n"
    
    learning_path = rec.get("learning_path", [])
    if learning_path:
        for item in learning_path:
            md += f"### 📑 {item.get('topic', 'N/A')}\n"
            if item.get('section'):
                md += f"- **Section:** {item['section']}\n"
            if item.get('book'):
                md += f"- **Book:** {item['book']}\n"
            if item.get('pages'):
                md += f"- **Pages:** {item['pages']}\n"
            md += "\n"
    else:
        md += "*No learning path generated*\n\n"
    
    md += "---\n\n"
    md += "## 💬 Explanation\n\n"
    md += f"{assessment.get('explanation', 'N/A')}\n\n"
    
    md += "---\n\n"
    md += "## 🌟 Encouragement\n\n"
    md += f"*{assessment.get('encouragement', 'Keep learning!')}*\n"
    
    return md


def run_analysis_gradio(
    question: str,
    correct_answer: str,
    student_answer: str,
    student_explanation: str,
    vllm_host: str,
    vllm_port: int,
    use_embeddings: bool,
    structured_output_mode: str,
    timeout: int,
    kg_path: str,
    pre_generated_output: str
) -> Tuple[str, str, str]:
    """
    Main analysis function for Gradio
    
    Returns:
        Tuple of (recommendations_md, raw_json, status)
    """
    try:
        # Check vLLM status first if no pre-generated output
        use_vllm = True
        if pre_generated_output and pre_generated_output.strip():
            use_vllm = False
            model_output = pre_generated_output
        else:
            model_output = None
            is_running, status_msg = check_vllm_status(vllm_host, int(vllm_port))
            if not is_running:
                return (
                    f"## ⚠️ vLLM Not Available\n\n{status_msg}\n\nPlease check your vLLM server or provide pre-generated output.",
                    "{}",
                    status_msg
                )
        
        # Load knowledge graph
        kg_data, kg_status = load_knowledge_graph(kg_path)
        
        # Create agent
        agent = create_agent()
        
        # Prepare initial state
        initial_state: GraphState = {
            "user_prompt": question,
            "question": question,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "student_explanation": student_explanation,
            "model_output": model_output,
            "knowledge_graph_json": json.dumps(kg_data) if kg_data else None,
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
                "port": int(vllm_port),
                "model_path": DEFAULT_MODEL_PATH,
                "adapter_path": DEFAULT_ADAPTER_PATH,
                "structured_output_mode": structured_output_mode,
                "timeout": int(timeout)
            },
            "use_embeddings": use_embeddings,
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
        
        # Run agent
        result = agent.invoke(initial_state)
        
        # Format outputs
        recommendations_md = format_recommendations({
            "recommendations": result.get("final_recommendations")
        })
        
        raw_output = {
            "recommendations": result.get("final_recommendations"),
            "analysis_result": result.get("analysis_result"),
            "topic_matches": result.get("topic_matches"),
            "processing_steps": result.get("processing_steps"),
            "errors": result.get("errors")
        }
        
        errors = result.get("errors", [])
        steps = result.get("processing_steps", [])
        
        status = f"**✅ Analysis Complete**\n\n"
        status += f"**Steps:** {' → '.join(steps)}\n\n"
        status += f"**KG Status:** {kg_status}\n\n"
        if errors:
            status += f"**Warnings:** {', '.join(errors)}"
        
        return (
            recommendations_md,
            json.dumps(raw_output, ensure_ascii=False, indent=2),
            status
        )
        
    except Exception as e:
        error_msg = f"## ❌ Error\n\n```\n{str(e)}\n```\n\n```\n{traceback.format_exc()}\n```"
        return (error_msg, "{}", f"❌ Error: {str(e)}")


def create_demo():
    """Create and return Gradio demo interface"""
    
    # Check initial status
    initial_vllm_ok, initial_vllm_status = check_vllm_status(DEFAULT_VLLM_HOST, DEFAULT_VLLM_PORT)
    _, initial_kg_status = load_knowledge_graph(DEFAULT_KG_PATH)
    
    with gr.Blocks(
        title="🎓 Student Analysis Agent",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .status-ok { color: green; font-weight: bold; }
        .status-error { color: red; font-weight: bold; }
        .compact-row { gap: 10px; }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🎓 Student Analysis Agent
            
            Analyze student responses using AI with knowledge graph integration.
            Evaluates answers, identifies weaknesses, and recommends learning materials.
            """,
            elem_classes="main-header"
        )
        
        # Status bar at top
        with gr.Row():
            with gr.Column(scale=1):
                vllm_status_display = gr.Markdown(
                    value=f"**vLLM:** {initial_vllm_status}",
                    elem_classes="status-ok" if initial_vllm_ok else "status-error"
                )
            with gr.Column(scale=1):
                kg_status_display = gr.Markdown(
                    value=f"**KG:** {initial_kg_status}"
                )
            with gr.Column(scale=1):
                refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
        
        gr.Markdown("---")
        
        with gr.Tabs():
            # Tab 1: Main Analysis
            with gr.TabItem("📝 Analyze Student Response"):
                with gr.Row():
                    with gr.Column(scale=1):
                        question_input = gr.Textbox(
                            label="📋 Question",
                            placeholder="Enter the question asked to the student...",
                            lines=3,
                            value="Як правильно замінити пряму мову непрямою в реченні «Принеси, будь ласка, квіти», — попросила мама Оксану, враховуючи правила трансформації спонукальних речень?"
                        )
                        
                        correct_answer_input = gr.Textbox(
                            label="✅ Correct Answer",
                            placeholder="Enter the correct answer...",
                            lines=2,
                            value="Мама попросила Оксану, щоб вона принесла квіти."
                        )
                        
                        student_answer_input = gr.Textbox(
                            label="📝 Student's Answer",
                            placeholder="Enter the student's answer...",
                            lines=2,
                            value=""
                        )
                        
                        student_explanation_input = gr.Textbox(
                            label="💭 Student's Explanation (optional)",
                            placeholder="Enter the student's explanation if any...",
                            lines=2,
                            value=""
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button("🔍 Analyze Response", variant="primary", size="lg", scale=2)
                            clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)
                    
                    with gr.Column(scale=1):
                        recommendations_output = gr.Markdown(
                            label="Results",
                            value="*Click 'Analyze Response' to see results*"
                        )
                
                with gr.Accordion("📊 Raw JSON Output & Status", open=False):
                    status_output = gr.Markdown(label="Status")
                    json_output = gr.Code(
                        label="Raw JSON",
                        language="json"
                    )
            
            # Tab 2: Settings
            with gr.TabItem("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔌 vLLM Connection")
                        vllm_host_input = gr.Textbox(
                            label="Host",
                            value=DEFAULT_VLLM_HOST
                        )
                        vllm_port_input = gr.Number(
                            label="Port",
                            value=DEFAULT_VLLM_PORT
                        )
                        timeout_input = gr.Slider(
                            label="Timeout (seconds)",
                            minimum=60,
                            maximum=600,
                            value=DEFAULT_TIMEOUT,
                            step=30,
                            info="Increase if getting timeout errors"
                        )
                        check_vllm_btn = gr.Button("🔌 Test Connection")
                        vllm_test_result = gr.Markdown("")
                    
                    with gr.Column():
                        gr.Markdown("### 📂 Knowledge Graph")
                        kg_path_input = gr.Textbox(
                            label="Path",
                            value=DEFAULT_KG_PATH
                        )
                        load_kg_btn = gr.Button("📂 Reload KG")
                        kg_load_result = gr.Markdown("")
                        
                        gr.Markdown("### 🧠 Options")
                        use_embeddings_checkbox = gr.Checkbox(
                            label="Use Semantic Embeddings (SentenceTransformer)",
                            value=True,
                            info="Better topic matching but slower"
                        )
                        structured_output_mode = gr.Dropdown(
                            label="Structured Output Mode",
                            choices=["none", "guided_json", "response_format"],
                            value="none",
                            info="none=fastest, guided_json=vLLM native, response_format=OpenAI style"
                        )
                
                with gr.Accordion("🔧 Advanced: Use Pre-generated Output", open=False):
                    gr.Markdown("If vLLM is not available, paste model output here:")
                    pre_generated_input = gr.Textbox(
                        label="Pre-generated Model Output",
                        placeholder="Paste JSON output with <think> tags...",
                        lines=10
                    )
                
                # Settings event handlers
                check_vllm_btn.click(
                    fn=lambda h, p: check_vllm_status(h, int(p))[1],
                    inputs=[vllm_host_input, vllm_port_input],
                    outputs=vllm_test_result
                )
                
                load_kg_btn.click(
                    fn=lambda p: load_knowledge_graph(p, force_reload=True)[1],
                    inputs=kg_path_input,
                    outputs=kg_load_result
                )
            
            # Tab 3: Knowledge Graph Explorer
            with gr.TabItem("🗺️ Explore Knowledge Graph"):
                with gr.Row():
                    with gr.Column(scale=1):
                        kg_search_input = gr.Textbox(
                            label="🔍 Search Topics",
                            placeholder="Enter search term..."
                        )
                        kg_type_filter = gr.Dropdown(
                            label="Filter by Type",
                            choices=["all", "topic", "section", "book", "term"],
                            value="topic"
                        )
                        explore_btn = gr.Button("🔍 Search")
                    
                    with gr.Column(scale=2):
                        kg_results = gr.Dataframe(
                            headers=["Type", "Title/Name", "ID"],
                            label="Results",
                            wrap=True
                        )
                
                def search_kg(search_term: str, type_filter: str):
                    kg_data, _ = load_knowledge_graph(DEFAULT_KG_PATH)
                    if not kg_data:
                        return []
                    
                    results = []
                    search_lower = search_term.lower() if search_term else ""
                    
                    for node in kg_data.get("nodes", []):
                        node_type = node.get("node_type", "unknown")
                        
                        # Type filter
                        if type_filter != "all" and node_type != type_filter:
                            continue
                        
                        title = node.get("title", node.get("name", ""))
                        
                        # Search filter
                        if search_lower and search_lower not in title.lower():
                            continue
                        
                        results.append([
                            node_type,
                            title[:80] + ("..." if len(title) > 80 else ""),
                            str(node.get("id", ""))[:30] + "..."
                        ])
                    
                    return results[:100]  # Limit results
                
                explore_btn.click(
                    fn=search_kg,
                    inputs=[kg_search_input, kg_type_filter],
                    outputs=kg_results
                )
            
            # Tab 4: Examples
            with gr.TabItem("📚 Examples"):
                gr.Markdown("""
                ### Example Inputs
                
                Click on any example to load it into the analysis form.
                """)
                
                gr.Examples(
                    examples=[
                        [
                            "Як правильно замінити пряму мову непрямою в реченні «Принеси, будь ласка, квіти», — попросила мама Оксану?",
                            "Мама попросила Оксану, щоб вона принесла квіти.",
                            "",
                            ""
                        ],
                        [
                            "Що таке складнопідрядне речення?",
                            "Складнопідрядне речення — це складне речення, частини якого поєднані підрядним зв'язком за допомогою сполучників або сполучних слів.",
                            "Це речення з двох частин.",
                            "Я так думаю."
                        ],
                        [
                            "Визначте тип речення за метою висловлювання: «Принеси мені книгу!»",
                            "Спонукальне речення, оскільки воно виражає наказ або прохання.",
                            "Питальне",
                            ""
                        ],
                        [
                            "Яка роль сполучника 'щоб' у складнопідрядному реченні?",
                            "Сполучник 'щоб' вводить підрядне речення мети або з'ясувальне підрядне речення.",
                            "Він з'єднує слова.",
                            "Не знаю точно."
                        ]
                    ],
                    inputs=[
                        question_input,
                        correct_answer_input,
                        student_answer_input,
                        student_explanation_input
                    ],
                    label="Click to load example"
                )
        
        # Event handlers
        def refresh_all_status(host, port, kg_path):
            ok, vllm_stat = check_vllm_status(host, int(port))
            _, kg_stat = load_knowledge_graph(kg_path)
            return f"**vLLM:** {vllm_stat}", f"**KG:** {kg_stat}"
        
        refresh_status_btn.click(
            fn=refresh_all_status,
            inputs=[vllm_host_input, vllm_port_input, kg_path_input],
            outputs=[vllm_status_display, kg_status_display]
        )
        
        def clear_inputs():
            return "", "", "", ""
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[question_input, correct_answer_input, student_answer_input, student_explanation_input]
        )
        
        # Main analyze button
        analyze_btn.click(
            fn=run_analysis_gradio,
            inputs=[
                question_input,
                correct_answer_input,
                student_answer_input,
                student_explanation_input,
                vllm_host_input,
                vllm_port_input,
                use_embeddings_checkbox,
                structured_output_mode,
                timeout_input,
                kg_path_input,
                pre_generated_input
            ],
            outputs=[
                recommendations_output,
                json_output,
                status_output
            ]
        )
    
    return demo


# Standalone execution
if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Student Analysis Agent - Gradio Demo")
    print("=" * 60)
    print(f"vLLM expected at: http://{DEFAULT_VLLM_HOST}:{DEFAULT_VLLM_PORT}")
    print(f"Knowledge Graph: {DEFAULT_KG_PATH}")
    print("=" * 60)
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )