"""
Interactive inference with the fine-tuned intent classifier.

Usage:
    python src/predict.py                        # with LoRA adapter
    python src/predict.py --no_adapter           # bare base model (baseline)
    python src/predict.py --input utterances.txt # batch mode

Supports:
    - Interactive mode: type utterances one by one
    - Batch mode:       pass --input file.txt (one utterance per line)
    - Baseline mode:    pass --no_adapter to skip the LoRA adapter
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict intents")
    parser.add_argument("--model_path", default=None, help="Path to LoRA adapter")
    parser.add_argument("--base_model", default="roberta-base", help="Base model name on HuggingFace")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Skip LoRA adapter, use bare base model only")
    parser.add_argument("--input", default=None, help="Text file with one utterance per line")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=3, help="Show top-k predictions")
    return parser.parse_args()


class IntentClassifier:
    """Wrapper for easy inference with the fine-tuned model."""

    def __init__(self, model_path, base_model="roberta-base", max_length=64,
                 no_adapter=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=10)

        if no_adapter:
            self.model = base.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = PeftModel.from_pretrained(base, model_path).to(self.device)

        self.model.eval()
        self.max_length = max_length
        self.id2label = self.model.config.id2label

    def predict(self, text, top_k=3):
        """Classify a single utterance. Returns list of dicts with intent and confidence."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        top_indices = torch.topk(probs, min(top_k, len(probs))).indices.cpu().numpy()
        results = []
        for idx in top_indices:
            results.append({
                "intent": self.id2label[int(idx)],
                "confidence": float(probs[idx]),
            })
        return results

    def predict_batch(self, texts, top_k=1):
        """Classify a batch of utterances."""
        return [self.predict(text, top_k) for text in texts]


def interactive_mode(classifier, top_k):
    """Run interactive prediction loop."""
    print("\nIntent Classifier - Interactive Mode")
    print("Type an utterance and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        results = classifier.predict(text, top_k=top_k)
        print(f"  -> {results[0]['intent']} ({results[0]['confidence']:.1%})")
        if top_k > 1 and len(results) > 1:
            for r in results[1:]:
                print(f"     {r['intent']} ({r['confidence']:.1%})")
        print()


def batch_mode(classifier, input_path, top_k):
    """Classify all lines in a text file."""
    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"\nClassifying {len(lines)} utterances from {input_path}\n")

    for text in lines:
        results = classifier.predict(text, top_k=top_k)
        print(f"  \"{text}\"")
        print(f"    -> {results[0]['intent']} ({results[0]['confidence']:.1%})")
        if top_k > 1:
            for r in results[1:]:
                print(f"       {r['intent']} ({r['confidence']:.1%})")
        print()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = args.model_path or os.path.join(base_dir, "models", "roberta-intent")

    if not args.no_adapter and not os.path.exists(model_path):
        print(f"Error: LoRA adapter not found at {model_path}")
        print("Run 'python src/train.py' first to fine-tune the model.")
        sys.exit(1)

    classifier = IntentClassifier(
        model_path, base_model=args.base_model,
        max_length=args.max_length, no_adapter=args.no_adapter,
    )

    if args.no_adapter:
        print(f"Base model only (no adapter): {args.base_model}")
    else:
        print(f"Base model: {args.base_model}")
        print(f"LoRA adapter loaded from {model_path}")

    if args.input:
        batch_mode(classifier, args.input, args.top_k)
    else:
        interactive_mode(classifier, args.top_k)


if __name__ == "__main__":
    main()
