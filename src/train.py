"""Fine-tune a language model on financial instruction data."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def format_example(example: dict) -> str:
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    if user_input:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return prompt


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize(batch):
        texts = []
        for i in range(len(batch["instruction"])):
            texts.append(
                format_example(
                    {
                        "instruction": batch["instruction"][i],
                        "input": batch.get("input", [""] * len(batch["instruction"]))[i],
                        "output": batch["output"][i],
                    }
                )
            )
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on financial data")
    parser.add_argument("--model-name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--train-file", type=Path, required=True, help="Path to training JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-lora", action="store_true", default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
