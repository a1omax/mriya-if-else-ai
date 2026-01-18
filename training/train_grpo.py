"""
GRPO Training Script with Think Format and Correctness Rewards

This script trains a language model using Group Relative Policy Optimization (GRPO)
with two reward functions:
1. Think format reward: Encourages <think>...</think> output format
2. Correctness reward: Evaluates the answer following </think> tags

Usage:
    # Basic training
    python grpo_train.py \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --data_path data/train.json \
        --output_dir ./grpo_output \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1

    # With LoRA and 4-bit quantization
    python grpo_train.py \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --data_path data/train.json \
        --output_dir ./grpo_output \
        --use_peft \
        --load_in_4bit \
        --learning_rate 1e-5

    # With vLLM acceleration (recommended for faster generation)
    accelerate launch --config_file accelerate_config.yaml grpo_train.py \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --data_path data/train.json \
        --output_dir ./grpo_output \
        --use_vllm \
        --vllm_mode colocate \
        --use_peft
"""

import re
import json
import logging
import random
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import torch
from datasets import Dataset, DatasetDict

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Extended script arguments for GRPO training."""
    
    data_path: str = field(
        default="data/train.json",
        metadata={"help": "Path to the JSON/JSONL dataset file"},
    )
    val_split: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )
    think_format_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for think format reward"},
    )
    correctness_reward_weight: float = field(
        default=2.0,
        metadata={"help": "Weight for correctness reward"},
    )
    use_fuzzy_matching: bool = field(
        default=True,
        metadata={"help": "Use fuzzy matching for correctness evaluation"},
    )
    fuzzy_threshold: float = field(
        default=0.8,
        metadata={"help": "Threshold for fuzzy matching (0-1)"},
    )


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    data = []
    
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def create_dataset(
    file_path: str,
    val_split: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """Create train/val dataset from file."""
    data = load_data(file_path)
    random.seed(seed)
    random.shuffle(data)
    
    # Normalize data format
    normalized_data = {
        'system_prompt': [],
        'user_prompt': [],
        'answer': [],
    }
    
    for item in data:
        # Handle various field names for flexibility
        system_prompt = item.get('system_prompt', item.get('system', ''))
        user_prompt = item.get('user_prompt', item.get('user', item.get('input', item.get('question', ''))))
        answer = item.get('answer', item.get('output', item.get('response', item.get('target', ''))))
        
        if user_prompt and answer:
            normalized_data['system_prompt'].append(str(system_prompt))
            normalized_data['user_prompt'].append(str(user_prompt))
            normalized_data['answer'].append(str(answer))
    
    dataset = Dataset.from_dict(normalized_data)
    dataset = dataset.shuffle(seed=seed)
    
    # Split into train and validation
    split_dataset = dataset.train_test_split(test_size=val_split, seed=seed)
    
    return DatasetDict({
        'train': split_dataset['train'],
        'val': split_dataset['test']
    })


def extract_think_content(text: str) -> tuple[str, str]:
    """
    Extract content inside <think>...</think> tags and the answer after.
    
    Returns:
        tuple: (think_content, answer_content)
    """
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, text, re.DOTALL)
    
    if match:
        think_content = match.group(1).strip()
        # Get everything after </think>
        answer_start = match.end()
        answer_content = text[answer_start:].strip()
        return think_content, answer_content
    
    return "", text.strip()


def think_format_reward(
    completions: List[List[Dict[str, str]]],
    answer: List[str],  # Ground truth (not used here, but required by signature)
    **kwargs
) -> List[float]:
    """
    Reward function that checks for proper <think>...</think> format.
    
    Rewards:
    - 1.0: Has both <think> and </think> tags with content inside and answer after
    - 0.5: Has the tags but missing content or answer
    - 0.0: Missing or malformed tags
    """
    rewards = []
    
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        
        has_open_tag = '<think>' in content
        has_close_tag = '</think>' in content
        
        if has_open_tag and has_close_tag:
            think_content, answer_content = extract_think_content(content)
            
            if think_content and answer_content:
                # Perfect format: has thinking and answer
                rewards.append(1.0)
            elif think_content or answer_content:
                # Partial: has tags but missing one part
                rewards.append(0.5)
            else:
                # Has tags but both parts are empty
                rewards.append(0.25)
        elif has_open_tag or has_close_tag:
            # Only one tag present
            rewards.append(0.1)
        else:
            # No tags at all
            rewards.append(0.0)
    
    return rewards


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Convert to lowercase
    text = text.lower()
    # Remove common punctuation variations
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def fuzzy_match(pred: str, target: str, threshold: float = 0.8) -> float:
    """
    Compute fuzzy match score between prediction and target.
    
    Returns a score between 0 and 1.
    """
    pred_norm = normalize_text(pred)
    target_norm = normalize_text(target)
    
    if not target_norm:
        return 1.0 if not pred_norm else 0.0
    
    if not pred_norm:
        return 0.0
    
    # Exact match after normalization
    if pred_norm == target_norm:
        return 1.0
    
    # Check if target is contained in prediction (or vice versa)
    if target_norm in pred_norm or pred_norm in target_norm:
        return 0.9
    
    # Use sequence matcher for fuzzy comparison
    ratio = SequenceMatcher(None, pred_norm, target_norm).ratio()
    
    return ratio


def exact_match(pred: str, target: str) -> float:
    """Check for exact match after normalization."""
    pred_norm = normalize_text(pred)
    target_norm = normalize_text(target)
    return 1.0 if pred_norm == target_norm else 0.0


def correctness_reward(
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    use_fuzzy: bool = True,
    fuzzy_threshold: float = 0.8,
    **kwargs
) -> List[float]:
    """
    Reward function that evaluates the correctness of the answer after </think>.
    
    The answer is expected to follow immediately after the </think> tag.
    
    Args:
        completions: Model completions
        answer: Ground truth answers
        use_fuzzy: Whether to use fuzzy matching
        fuzzy_threshold: Threshold for fuzzy matching
    
    Returns:
        List of reward scores (0-1)
    """
    rewards = []
    
    for completion, target in zip(completions, answer):
        content = completion[0]["content"] if completion else ""
        
        # Extract the answer part (after </think>)
        _, pred_answer = extract_think_content(content)
        
        if not pred_answer:
            # No answer provided
            rewards.append(0.0)
            continue
        
        if use_fuzzy:
            score = fuzzy_match(pred_answer, target, fuzzy_threshold)
        else:
            score = exact_match(pred_answer, target)
        
        rewards.append(score)
    
    return rewards


SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that thinks step by step before answering.

When responding:
1. First, think through the problem inside <think>...</think> tags
2. Then, provide your final answer immediately after the </think> tag

Example format:
<think>
Let me analyze this step by step...
[Your reasoning here]
</think>
[Your final answer here]"""


def make_conversation(example: Dict[str, str]) -> Dict[str, Any]:
    """Convert dataset example to conversation format for GRPO."""
    system_prompt = example.get('system_prompt', '')
    user_prompt = example['user_prompt']
    
    # Use custom system prompt if provided, otherwise use template
    if system_prompt:
        full_system = f"{SYSTEM_PROMPT_TEMPLATE}\n\nAdditional context:\n{system_prompt}"
    else:
        full_system = SYSTEM_PROMPT_TEMPLATE
    
    return {
        "prompt": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ],
    }

def create_weighted_think_format_reward(weight: float = 1.0):
    """Create a think format reward function with configurable weight."""
    def weighted_reward(completions, answer, **kwargs):
        base_rewards = think_format_reward(completions, answer, **kwargs)
        return [r * weight for r in base_rewards]
    
    weighted_reward.__name__ = "think_format_reward"
    return weighted_reward


def create_weighted_correctness_reward(
    weight: float = 2.0,
    use_fuzzy: bool = True,
    fuzzy_threshold: float = 0.8
):
    """Create a correctness reward function with configurable weight and matching."""
    def weighted_reward(completions, answer, **kwargs):
        base_rewards = correctness_reward(
            completions, answer,
            use_fuzzy=use_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
            **kwargs
        )
        return [r * weight for r in base_rewards]
    
    weighted_reward.__name__ = "correctness_reward"
    return weighted_reward


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    model_name = model_args.model_name_or_path
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Use PEFT/LoRA: {model_args.use_peft}")
    
    # Set up model initialization kwargs
    dtype = (
        model_args.torch_dtype 
        if model_args.torch_dtype in ["auto", None] 
        else getattr(torch, model_args.torch_dtype)
    )
    
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
    )
    
    # Quantization config
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config
    

    logger.info(f"Loading dataset from: {script_args.data_path}")
    dataset = create_dataset(
        script_args.data_path,
        val_split=script_args.val_split,
        seed=script_args.seed
    )
    
    train_dataset = dataset['train']
    eval_dataset = dataset['val'] if training_args.eval_strategy != "no" else None
    
    logger.info(f"Train size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Val size: {len(eval_dataset)}")
    
    # Convert to conversation format
    train_dataset = train_dataset.map(make_conversation)
    if eval_dataset:
        eval_dataset = eval_dataset.map(make_conversation)
    
    # Keep 'answer' column for reward functions, remove others
    columns_to_remove = ['system_prompt', 'user_prompt']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    if eval_dataset:
        eval_dataset = eval_dataset.remove_columns(columns_to_remove)
    
    logger.info(f"Think format reward weight: {script_args.think_format_reward_weight}")
    logger.info(f"Correctness reward weight: {script_args.correctness_reward_weight}")
    logger.info(f"Using fuzzy matching: {script_args.use_fuzzy_matching}")
    
    reward_funcs = [
        create_weighted_think_format_reward(script_args.think_format_reward_weight),
        create_weighted_correctness_reward(
            weight=script_args.correctness_reward_weight,
            use_fuzzy=script_args.use_fuzzy_matching,
            fuzzy_threshold=script_args.fuzzy_threshold,
        ),
    ]
    
    logger.info("Setting up GRPO trainer...")
    
    trainer = GRPOTrainer(
        model=model_name,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )
    
    logger.info("Starting GRPO training...")
    trainer.train()
    

    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if hasattr(trainer, 'processing_class') and trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(training_args.output_dir)
    
    # Push to Hub if requested
    if training_args.push_to_hub:
        logger.info("Pushing model to Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    
    logger.info("Training complete!")