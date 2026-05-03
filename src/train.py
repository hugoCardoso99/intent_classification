"""
Fine-tune RoBERTa-base for maritime/port logistics intent classification using PEFT/LoRA.

Instead of updating all ~125M parameters, LoRA injects small trainable
rank-decomposition matrices into the attention layers. This typically
trains only ~0.5-1% of the total parameters while achieving comparable
accuracy to full fine-tuning.

Uses class weights derived from the real-world intent distribution so the
model penalizes errors on rare intents more heavily.

Domain: Maritime/Port Logistics - specialized vocabulary including
Incoterms, vessel classes, HS codes, berth assignments, pilotage, customs.

Usage:
    python src/train.py [--epochs 5] [--batch_size 16] [--lr 2e-4]
    python src/train.py --lora_r 16 --lora_alpha 32  # higher rank

The fine-tuned LoRA adapter is saved to ./models/roberta-intent/
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train intent classifier with LoRA")
    parser.add_argument("--model_name", default="roberta-base", help="HuggingFace model name")
    parser.add_argument("--max_length", type=int, default=64, help="Max token length")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (higher for LoRA)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    return parser.parse_args()


def load_data(path):
    """Load CSV and convert to HuggingFace Dataset."""
    df = pd.read_csv(path)
    return Dataset.from_pandas(df[["text", "label"]])


def print_trainable_params(model):
    """Print the number of trainable vs total parameters."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute accuracy, macro F1, and micro F1 for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(base_dir, "data", "train.csv")
    test_path = os.path.join(base_dir, "data", "test.csv")
    output_dir = os.path.join(base_dir, "models", "roberta-intent")
    log_dir = os.path.join(base_dir, "models", "logs")

    label_map_path = os.path.join(base_dir, "data", "label_map.csv")
    label_df = pd.read_csv(label_map_path)
    id2label = dict(zip(label_df["label"], label_df["intent_name"]))
    label2id = dict(zip(label_df["intent_name"], label_df["label"]))
    num_labels = len(id2label)

    print(f"Number of intents: {num_labels}")
    print(f"Labels: {list(id2label.values())}")

    weights_path = os.path.join(base_dir, "data", "class_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            weight_dict = json.load(f)
        weight_list = [weight_dict[str(i)] for i in range(num_labels)]
        class_weights = torch.tensor(weight_list, dtype=torch.float32)
        print(f"\nClass weights loaded:")
        for i, w in enumerate(weight_list):
            print(f"  {id2label[i]:25s} weight={w:.4f}")
    else:
        class_weights = None
        print("\nNo class_weights.json found - training without class weights.")

    train_dataset = load_data(train_path)
    eval_dataset = load_data(test_path)
    print(f"\nTrain: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(base_model, lora_config)

    print(f"\nLoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Target modules: {lora_config.target_modules}")
    print_trainable_params(model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=log_dir,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting LoRA training...\n")
    trainer.train()

    # Save only the LoRA adapter (few MB) + tokenizer.
    # The base model is referenced by name and loaded from HF cache at inference.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapter saved to {output_dir}")

    print("\nFinal evaluation on test set:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(results_dir, "training_metrics.csv"), index=False)
    print(f"Metrics saved to {results_dir}/training_metrics.csv")


if __name__ == "__main__":
    main()
