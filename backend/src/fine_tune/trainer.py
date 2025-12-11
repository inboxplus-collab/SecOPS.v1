"""Transformer-based fine-tuning utilities."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class FineTuneEngine:
    """Run small causal language model fine-tunes using Hugging Face Transformers."""

    def __init__(
        self,
        *,
        base_model: str = "distilgpt2",
        output_root: str = "models/fine_tuned",
        max_length: int = 512,
    ) -> None:
        self.base_model = base_model
        self.output_root = Path(output_root)
        self.max_length = max_length

        self.output_root.mkdir(parents=True, exist_ok=True)

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _prepare_dataset(self, dataset_path: Path, tokenizer: AutoTokenizer):
        raw = load_dataset("json", data_files=str(dataset_path))

        def tokenize(example: Dict[str, str]) -> Dict[str, Any]:
            text = f"{example['prompt']}\n\n{example['completion']}"
            tokenized = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return raw.map(tokenize)

    def start_job(
        self,
        dataset_path: str,
        *,
        learning_rate: float = 5e-5,
        epochs: int = 1,
        batch_size: int = 4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        seed: int = 42,
        report_to: Optional[str] = None,
    ) -> str:
        """Fine-tune a lightweight causal LM on the provided JSONL dataset."""

        dataset_file = Path(dataset_path)
        run_id = f"ft-{uuid.uuid4().hex[:8]}"
        output_dir = self.output_root / run_id

        tokenizer = self._load_tokenizer()
        tokenized = self._prepare_dataset(dataset_file, tokenizer)

        model = AutoModelForCausalLM.from_pretrained(self.base_model)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            seed=seed,
            logging_steps=5,
            save_steps=50,
            report_to=report_to if report_to else [],
            overwrite_output_dir=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return run_id
