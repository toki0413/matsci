#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for MatSci-Agent.

Trains a lightweight adapter on top of a base LLM using the Sobko
computational chemistry dataset. This improves the model's domain
knowledge without full fine-tuning.

Usage:
    python scripts/lora_finetune.py \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --data-path ../Sobko_MCP_project/advanced_optimization/lora_training_full.jsonl \
        --output-dir ./lora_adapters/sobko-qc

Requirements:
    pip install transformers peft accelerate bitsandbytes datasets
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_alpaca_data(path: str) -> list[dict[str, Any]]:
    """Load Alpaca-format JSONL data."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_prompt(record: dict[str, Any]) -> str:
    """Format a record into a chat-style prompt."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    output = record.get("output", "")

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    return prompt + output


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune for computational chemistry")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model identifier (HuggingFace hub or local path)")
    parser.add_argument("--data-path", type=str,
                        default="../Sobko_MCP_project/advanced_optimization/lora_training_full.jsonl",
                        help="Path to Alpaca-format training data")
    parser.add_argument("--output-dir", type=str, default="./lora_adapters/sobko-qc",
                        help="Directory to save LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--use-qlora", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA) for memory efficiency")
    args = parser.parse_args()

    print("=" * 60)
    print("MatSci-Agent LoRA Fine-tuning")
    print("=" * 60)

    # Lazy imports to avoid dependency errors if not installed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install transformers peft accelerate bitsandbytes datasets")
        return

    # Load data
    print(f"\nLoading training data from {args.data_path} ...")
    records = load_alpaca_data(args.data_path)
    print(f"  Loaded {len(records)} training examples")

    # Prepare dataset
    formatted_texts = [format_prompt(r) for r in records]
    dataset = Dataset.from_dict({"text": formatted_texts})

    # Split train/val
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = dataset["train"]
    val_ds = dataset["test"]
    print(f"  Train: {len(train_ds)}, Validation: {len(val_ds)}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=["text"])

    # Load model
    print(f"\nLoading model: {args.model_name} ...")
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    if args.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("  Using 4-bit QLoRA for memory efficiency")
        except ImportError:
            print("  WARNING: bitsandbytes not available, falling back to full precision")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
    )

    from transformers import Trainer, DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("\nStarting training ...")
    trainer.train()

    # Save adapter
    print(f"\nSaving LoRA adapter to {output_dir} ...")
    model.save_pretrained(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    # Save metadata
    metadata = {
        "base_model": args.model_name,
        "training_examples": len(records),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "data_source": "Sobko_MCP_project",
    }
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Adapter saved to: {output_dir / 'final_adapter'}")
    print("=" * 60)
    print("\nTo use the adapter in MatSci-Agent:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    print(f"  model = PeftModel.from_pretrained(model, '{output_dir / 'final_adapter'}')")


if __name__ == "__main__":
    main()
