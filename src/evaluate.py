"""
Evaluate the fine-tuned intent classifier on the test set.

Loads the base model + LoRA adapter via peft for inference.
Results are saved to results/with_adapter/ or results/base_model/.

Usage:
    python src/evaluate.py                  # with LoRA adapter
    python src/evaluate.py --no_adapter     # bare base model (baseline)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate intent classifier")
    parser.add_argument("--model_path", default=None,
                        help="Path to LoRA adapter (default: models/roberta-intent)")
    parser.add_argument("--base_model", default="roberta-base",
                        help="Base model name on HuggingFace")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Skip LoRA adapter, evaluate bare base model only")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def predict_batch(texts, model, tokenizer, max_length, batch_size, device):
    """Run inference and return predicted labels + probabilities."""
    all_preds = []
    all_probs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved -> {output_path}")


def plot_per_class_f1(report_dict, labels, output_path):
    """Bar chart of per-class F1 scores."""
    f1_scores = [report_dict[label]["f1-score"] for label in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, f1_scores, color="steelblue")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class F1 chart saved -> {output_path}")


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = args.model_path or os.path.join(base_dir, "models", "roberta-intent")
    test_path = os.path.join(base_dir, "data", "test.csv")
    subfolder = "base_model" if args.no_adapter else "with_adapter"
    results_dir = os.path.join(base_dir, "results", subfolder)
    os.makedirs(results_dir, exist_ok=True)

    label_map_path = os.path.join(base_dir, "data", "label_map.csv")
    label_df = pd.read_csv(label_map_path)
    id2label = dict(zip(label_df["label"], label_df["intent_name"]))
    label2id = {v: k for k, v in id2label.items()}
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model -- with or without LoRA adapter
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(id2label),
        id2label=id2label, label2id=label2id,
    )

    if args.no_adapter:
        model = base_model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model.eval()
        print(f"Base model only (no adapter): {args.base_model}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = PeftModel.from_pretrained(base_model, model_path).to(device)
        model.eval()
        print(f"Base model: {args.base_model}")
        print(f"LoRA adapter loaded from {model_path}")

    print(f"Results will be saved to {results_dir}")

    test_df = pd.read_csv(test_path)
    texts = test_df["text"].tolist()
    true_labels = test_df["label"].values
    print(f"Test set: {len(texts)} examples")

    pred_labels, pred_probs = predict_batch(
        texts, model, tokenizer, args.max_length, args.batch_size, device
    )

    report = classification_report(
        true_labels, pred_labels, target_names=label_names, digits=4
    )
    report_dict = classification_report(
        true_labels, pred_labels, target_names=label_names, output_dict=True
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    overall_acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted")
    print(f"Overall Accuracy:  {overall_acc:.4f}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write(f"\nOverall Accuracy:  {overall_acc:.4f}\n")
        f.write(f"Macro F1 Score:    {macro_f1:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
    print(f"\nReport saved -> {report_path}")

    results_df = test_df.copy()
    results_df["predicted_label"] = pred_labels
    results_df["predicted_intent"] = [id2label[p] for p in pred_labels]
    results_df["correct"] = results_df["label"] == results_df["predicted_label"]
    results_df["confidence"] = [pred_probs[i, pred_labels[i]] for i in range(len(pred_labels))]
    results_df.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)
    print(f"Predictions saved -> {results_dir}/predictions.csv")

    plot_confusion_matrix(
        true_labels, pred_labels, label_names,
        os.path.join(results_dir, "confusion_matrix.png"),
    )
    plot_per_class_f1(
        report_dict, label_names,
        os.path.join(results_dir, "per_class_f1.png"),
    )

    errors = results_df[~results_df["correct"]]
    if len(errors) > 0:
        error_pct = len(errors) / len(results_df) * 100
        print(f"\n{len(errors)} misclassified examples ({error_pct:.1f}%)")
        print("\nSample errors:")
        for _, row in errors.head(10).iterrows():
            print(f"  Text: \"{row['text']}\"")
            print(f"    True: {row['intent_name']} | Pred: {row['predicted_intent']} "
                  f"(conf: {row['confidence']:.3f})")
            print()
    else:
        print("\nNo misclassifications!")


if __name__ == "__main__":
    main()
