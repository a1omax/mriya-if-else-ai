"""
Training script using TRL's SFTTrainer with TRL argument parsers.

# Full training with DeepSpeed
deepspeed --num_gpus=2 train_trl.py \
    --model_name_or_path lapa-llm/lapa-v0.1.2-instruct \
    --data_path data/supervised_data.json \
    --output_dir models \
    --deepspeed zero_configs/ds_zero2.json

# LoRA training with DeepSpeed
deepspeed --num_gpus=2 train_trl.py \
    --model_name_or_path lapa-llm/lapa-v0.1.2-instruct \
    --data_path data/supervised_data.json \
    --output_dir models \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --deepspeed zero_configs/ds_zero2.json

# With quantization (QLoRA)
python train_trl.py \
    --model_name_or_path lapa-llm/lapa-v0.1.2-instruct \
    --data_path data/supervised_data.json \
    --output_dir models \
    --use_peft \
    --load_in_4bit
"""

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTScriptArguments(ScriptArguments):
    """Extended script arguments for SFT training."""
    
    data_path: str = field(
        default="data/supervised_data.json",
        metadata={"help": "Path to the JSON/JSONL dataset file"},
    )
    val_split: float = field(
        default=0.2,
        metadata={"help": "Validation split ratio"},
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


def make_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dataset example to conversation format for SFT.
    
    Keeps system prompt and user prompt separate in the messages format.
    """
    system_prompt = example.get('system_prompt', '')
    user_prompt = example.get('user_prompt', '')
    answer = example.get('answer', '')
    
    messages = []
    
    # Add system message if present
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add user message
    messages.append({"role": "user", "content": user_prompt})
    
    # Add assistant message
    messages.append({"role": "assistant", "content": answer})
    
    return {"messages": messages}


def create_dataset(
    file_path: str,
    val_split: float = 0.2,
    seed: int = 42
) -> DatasetDict:
    """Create train/val dataset from file."""
    data = load_data(file_path)
    random.seed(seed)
    random.shuffle(data)
    
    # Filter out invalid examples
    valid_data = []
    for item in data:
        user_prompt = item.get('user_prompt', '')
        answer = item.get('answer', '')
        
        if user_prompt and answer:
            valid_data.append(item)
    
    if len(valid_data) < len(data):
        logger.warning(f"Filtered out {len(data) - len(valid_data)} invalid examples")
    
    # Convert to Dataset and apply conversation format
    dataset = Dataset.from_list(valid_data)
    dataset = dataset.map(make_conversation, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    
    # Split into train and validation
    split_dataset = dataset.train_test_split(test_size=val_split, seed=seed)
    
    return DatasetDict({
        'train': split_dataset['train'],
        'val': split_dataset['test']
    })


def main(script_args: SFTScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    """Main training function using TRL's SFTTrainer."""
    
    model_name = model_args.model_name_or_path
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Use PEFT/LoRA: {model_args.use_peft}")
    
    # Set up model initialization kwargs
    dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
    )
    
    # Quantization config
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
        logger.info("Using quantization config")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    logger.info(f"Loading dataset from: {script_args.data_path}")
    dataset = create_dataset(
        script_args.data_path,
        val_split=script_args.val_split,
        seed=training_args.seed,
    )
    
    train_dataset = dataset['train']
    eval_dataset = dataset['val'] if training_args.eval_strategy != "no" else None
    
    logger.info(f"Train size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Val size: {len(eval_dataset)}")
    
    peft_config = get_peft_config(model_args)
    if peft_config:
        logger.info(f"Using LoRA with r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    
    logger.info("Setting up SFT trainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    

    logger.info("Starting SFT training...")
    trainer.train()
    
    trainer.accelerator.print("✅ Training completed.")
    
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}")
    
    # Save tokenizer
    if hasattr(trainer, 'processing_class') and trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(training_args.output_dir)
    
    # Push to Hub if requested
    if training_args.push_to_hub:
        logger.info("Pushing model to Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to Hub: https://huggingface.co/{trainer.hub_model_id}")
    
    logger.info("Training complete!")
    return trainer


if __name__ == "__main__":
    # Parse arguments using TRL's parser
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Run training
    main(script_args, training_args, model_args)