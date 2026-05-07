"""
Model version comparison using LLM-as-a-Judge with McNemar's test.

Compares two model versions on the same production sample using independent
LLM scoring (pseudo-ground-truth), then tests for statistically significant
differences via McNemar's test. Includes a pairwise tiebreaker for ambiguous cases.

Usage:
    # Compare current adapter vs a previous version
    python src/compare_models.py \
        --model_a models/roberta-intent \
        --model_b models/roberta-intent-v2 \
        --data data/test.csv

    # Compare adapter vs base model on production data
    python src/compare_models.py \
        --model_a models/roberta-intent \
        --no_adapter_b \
        --data production_sample.csv

    # Use a specific test file and significance level
    python src/compare_models.py \
        --model_a models/roberta-intent \
        --model_b models/roberta-intent-v2 \
        --data data/test.csv \
        --alpha 0.01

Environment:
    GEMINI_API_KEY (or other provider key) — see llm_judge.py
"""

import argparse
import json
import os
import sys
from typing import Optional

from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_judge import IntentJudge, get_provider


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, base_model: str = "roberta-base",
               no_adapter: bool = False, num_labels: int = 10):
    """Load a model (with or without LoRA adapter) for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels
    )

    if no_adapter:
        model = base.to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = PeftModel.from_pretrained(base, model_path).to(device)

    model.eval()

    # Load label mapping
    id2label = model.config.id2label if model.config.id2label else {}
    return model, tokenizer, device, id2label


def predict_single(text: str, model, tokenizer, device,
                   id2label: dict, max_length: int = 64) -> tuple[str, float]:
    """Predict intent for a single text. Returns (intent_name, confidence)."""
    inputs = tokenizer(
        text, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    intent = id2label.get(pred_idx, str(pred_idx))
    return intent, confidence


def predict_batch(texts: list[str], model, tokenizer, device,
                  id2label: dict, max_length: int = 64,
                  batch_size: int = 32) -> tuple[list[str], list[float]]:
    """Predict intents for a batch of texts."""
    all_intents = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        for j, pred_idx in enumerate(preds.cpu().numpy()):
            all_intents.append(id2label.get(int(pred_idx), str(pred_idx)))
            all_confs.append(float(probs[j, pred_idx].item()))

    return all_intents, all_confs


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

class McNemarResult(BaseModel):
    """Result of McNemar's test comparing two classifiers."""
    n_both_correct: int = Field(description="Both models correct")
    n_a_only: int = Field(description="A correct, B wrong")
    n_b_only: int = Field(description="A wrong, B correct")
    n_both_wrong: int = Field(description="Both models wrong")
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    winner: Optional[str] = Field(default=None, description="'A', 'B', or None")
    effect_summary: str

    def to_dict(self) -> dict:
        return {
            "contingency_table": {
                "both_correct": self.n_both_correct,
                "a_correct_b_wrong": self.n_a_only,
                "a_wrong_b_correct": self.n_b_only,
                "both_wrong": self.n_both_wrong,
            },
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "significant": self.significant,
            "winner": self.winner,
            "effect_summary": self.effect_summary,
        }


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray,
                 alpha: float = 0.05) -> McNemarResult:
    """Run McNemar's test on paired binary correctness arrays.

    Args:
        a_correct: boolean array -- True where model A was correct
        b_correct: boolean array -- True where model B was correct
        alpha: significance level

    Uses exact binomial test when discordant pairs < 25,
    chi-squared approximation otherwise (with continuity correction).
    """
    # Contingency table
    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    discordant = a_only + b_only

    if discordant == 0:
        return McNemarResult(
            n_both_correct=both_correct, n_a_only=a_only,
            n_b_only=b_only, n_both_wrong=both_wrong,
            statistic=0.0, p_value=1.0,
            significant=False, alpha=alpha, winner=None,
            effect_summary="Models are identical on all examples.",
        )

    if discordant < 25:
        # Exact binomial test
        from scipy.stats import binom_test
        try:
            p_value = binom_test(a_only, discordant, 0.5)
        except Exception:
            # scipy >= 1.7 deprecation fallback
            from scipy.stats import binomtest
            result = binomtest(a_only, discordant, 0.5)
            p_value = result.pvalue
        statistic = float(a_only)  # not a chi-sq, but the count itself
    else:
        # Chi-squared with continuity correction
        statistic = (abs(a_only - b_only) - 1) ** 2 / discordant
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(statistic, df=1)

    significant = p_value < alpha

    # Determine winner
    if significant:
        if a_only > b_only:
            winner = "A"
            summary = (
                f"Model A is significantly better (p={p_value:.4f} < {alpha}). "
                f"A was correct on {a_only} examples where B was wrong, "
                f"vs {b_only} the other way."
            )
        else:
            winner = "B"
            summary = (
                f"Model B is significantly better (p={p_value:.4f} < {alpha}). "
                f"B was correct on {b_only} examples where A was wrong, "
                f"vs {a_only} the other way."
            )
    else:
        winner = None
        summary = (
            f"No significant difference (p={p_value:.4f} >= {alpha}). "
            f"Discordant pairs: A-only={a_only}, B-only={b_only}."
        )

    return McNemarResult(
        n_both_correct=both_correct, n_a_only=a_only,
        n_b_only=b_only, n_both_wrong=both_wrong,
        statistic=round(statistic, 4), p_value=p_value,
        significant=significant, alpha=alpha,
        winner=winner, effect_summary=summary,
    )


# ---------------------------------------------------------------------------
# Full comparison pipeline
# ---------------------------------------------------------------------------

def compare_models(texts: list[str],
                   model_a, tokenizer_a, device_a, id2label_a,
                   model_b, tokenizer_b, device_b, id2label_b,
                   judge: IntentJudge,
                   alpha: float = 0.05,
                   pairwise_on_ambiguous: bool = True) -> dict:
    """Full comparison pipeline.

    1. Both models predict on the same texts
    2. LLM judge assigns pseudo-ground-truth labels
    3. McNemar's test for statistical significance
    4. (Optional) Pairwise tiebreaker on ambiguous disagreements

    Returns a dict with all results.
    """
    print(f"\nStep 1/3: Running predictions on {len(texts)} examples...")

    preds_a, confs_a = predict_batch(texts, model_a, tokenizer_a, device_a, id2label_a)
    print("  Model A: predictions complete")

    preds_b, confs_b = predict_batch(texts, model_b, tokenizer_b, device_b, id2label_b)
    print("  Model B: predictions complete")

    # Count agreements
    agreements = sum(a == b for a, b in zip(preds_a, preds_b))
    agree_pct = agreements / len(texts)
    print(f"  Models agree on {agreements}/{len(texts)} ({agree_pct:.1%}) examples")

    print("\nStep 2/3: LLM judge assigning pseudo-ground-truth...")
    judge_labels = []
    ambiguous_flags = []
    verdicts_a = []
    verdicts_b = []

    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(texts)}]", end="\r")

        # Judge scores each prediction independently
        va = judge.score_prediction(text, preds_a[i], confs_a[i])
        vb = judge.score_prediction(text, preds_b[i], confs_b[i])

        # Use judge_label from whichever verdict is more informative
        # (prefer non-None, non-OOD)
        jl = va.judge_label or vb.judge_label
        judge_labels.append(jl)
        ambiguous_flags.append(va.is_ambiguous or vb.is_ambiguous)
        verdicts_a.append(va.verdict)
        verdicts_b.append(vb.verdict)

    print(f"  [{len(texts)}/{len(texts)}] Done.")

    # Build comparison DataFrame
    comparison_df = pd.DataFrame({
        "text": texts,
        "pred_a": preds_a,
        "conf_a": confs_a,
        "verdict_a": verdicts_a,
        "pred_b": preds_b,
        "conf_b": confs_b,
        "verdict_b": verdicts_b,
        "judge_label": judge_labels,
        "is_ambiguous": ambiguous_flags,
    })

    # Compute correctness against judge labels
    a_correct = np.array([
        pred == jl for pred, jl in zip(preds_a, judge_labels)
    ])
    b_correct = np.array([
        pred == jl for pred, jl in zip(preds_b, judge_labels)
    ])

    comparison_df["a_correct"] = a_correct
    comparison_df["b_correct"] = b_correct

    # McNemar's test
    print(f"\nStep 3/3: McNemar's test (alpha={alpha})...")
    mcnemar_result = mcnemar_test(a_correct, b_correct, alpha=alpha)
    print(f"  {mcnemar_result.effect_summary}")

    # Optional: pairwise tiebreaker on ambiguous disagreements
    pairwise_results = []
    if pairwise_on_ambiguous:
        ambig_disagree = comparison_df[
            comparison_df["is_ambiguous"] &
            (comparison_df["pred_a"] != comparison_df["pred_b"])
        ]
        if len(ambig_disagree) > 0:
            n_ambig = len(ambig_disagree)
            print(f"\n  Pairwise tiebreaker on {n_ambig} ambiguous disagreements...")
            for _, row in ambig_disagree.iterrows():
                pw = judge.pairwise_compare(
                    row["text"],
                    row["pred_a"], row["conf_a"],
                    row["pred_b"], row["conf_b"],
                )
                pw["text"] = row["text"]
                pairwise_results.append(pw)

            pw_df = pd.DataFrame(pairwise_results)
            winner_counts = pw_df["winner"].value_counts().to_dict()
            print(f"  Pairwise results: {winner_counts}")

    # Aggregate metrics
    a_accuracy = float(a_correct.mean())
    b_accuracy = float(b_correct.mean())

    result = {
        "model_a_accuracy": round(a_accuracy, 4),
        "model_b_accuracy": round(b_accuracy, 4),
        "accuracy_diff": round(b_accuracy - a_accuracy, 4),
        "agreement_rate": round(agreements / len(texts), 4),
        "mcnemar": mcnemar_result.to_dict(),
        "verdict_summary": {
            "model_a": pd.Series(verdicts_a).value_counts().to_dict(),
            "model_b": pd.Series(verdicts_b).value_counts().to_dict(),
        },
        "pairwise_tiebreaker": (
            pd.DataFrame(pairwise_results)["winner"].value_counts().to_dict()
            if pairwise_results else None
        ),
        "comparison_df": comparison_df,
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two model versions using LLM-as-a-Judge + McNemar's test"
    )
    parser.add_argument("--model_a", required=True,
                        help="Path to model A (LoRA adapter)")
    parser.add_argument("--model_b", default=None,
                        help="Path to model B (LoRA adapter)")
    parser.add_argument("--no_adapter_a", action="store_true",
                        help="Model A is bare base model")
    parser.add_argument("--no_adapter_b", action="store_true",
                        help="Model B is bare base model")
    parser.add_argument("--base_model", default="roberta-base",
                        help="Base model name for both")
    parser.add_argument("--data", required=True,
                        help="CSV with 'text' column (test set or production sample)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N examples from the data (default: all)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for McNemar's test (default: 0.05)")
    parser.add_argument("--provider", default="gemini",
                        choices=["gemini", "openai", "groq"])
    parser.add_argument("--model", default=None,
                        help="LLM model override for the judge")
    parser.add_argument("--no_pairwise", action="store_true",
                        help="Skip pairwise tiebreaker on ambiguous cases")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/comparison/)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load data
    data_df = pd.read_csv(args.data)
    if args.sample and args.sample < len(data_df):
        data_df = data_df.sample(n=args.sample, random_state=42)
    texts = data_df["text"].tolist()
    print(f"Data: {len(texts)} examples from {args.data}")

    # Load label map
    label_map_path = os.path.join(base_dir, "data", "label_map.csv")
    if os.path.exists(label_map_path):
        label_df = pd.read_csv(label_map_path)
        num_labels = len(label_df)
    else:
        num_labels = 10

    # Load models
    print("\nLoading Model A...")
    model_a, tok_a, dev_a, id2l_a = load_model(
        args.model_a, args.base_model, args.no_adapter_a, num_labels
    )
    label_a = "base_model" if args.no_adapter_a else args.model_a
    print(f"  Model A: {label_a}")

    print("Loading Model B...")
    model_b_path = args.model_b or args.model_a
    model_b, tok_b, dev_b, id2l_b = load_model(
        model_b_path, args.base_model, args.no_adapter_b, num_labels
    )
    label_b = "base_model" if args.no_adapter_b else model_b_path
    print(f"  Model B: {label_b}")

    # Initialise judge
    provider = get_provider(args.provider, args.model)
    judge = IntentJudge(provider)
    print(f"LLM Judge: {args.provider} ({provider.model_name})")

    # Run comparison
    result = compare_models(
        texts,
        model_a, tok_a, dev_a, id2l_a,
        model_b, tok_b, dev_b, id2l_b,
        judge,
        alpha=args.alpha,
        pairwise_on_ambiguous=not args.no_pairwise,
    )

    # Print report
    print("\n" + "=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)
    print(f"Model A: {label_a}")
    print(f"Model B: {label_b}")
    print(f"Examples: {len(texts)}")
    print("\nAccuracy (vs judge labels):")
    print(f"  Model A: {result['model_a_accuracy']:.1%}")
    print(f"  Model B: {result['model_b_accuracy']:.1%}")
    print(f"  Diff:    {result['accuracy_diff']:+.1%}")
    print(f"\nAgreement rate: {result['agreement_rate']:.1%}")
    print("\nMcNemar's test:")
    mc = result["mcnemar"]
    print(f"  Statistic:   {mc['statistic']:.4f}")
    print(f"  p-value:     {mc['p_value']:.6f}")
    print(f"  Significant: {mc['significant']} (alpha={mc['alpha']})")
    winner_str = mc["winner"] or "None (no significant difference)"
    print(f"  Winner:      {winner_str}")
    print(f"\n  {mc['effect_summary']}")

    ct = mc["contingency_table"]
    print("\n  Contingency table:")
    print("                   B correct  B wrong")
    print(f"    A correct      {ct['both_correct']:>6d}     {ct['a_correct_b_wrong']:>6d}")
    print(f"    A wrong        {ct['a_wrong_b_correct']:>6d}     {ct['both_wrong']:>6d}")

    if result["pairwise_tiebreaker"]:
        print("\n  Pairwise tiebreaker on ambiguous cases:")
        for winner, count in result["pairwise_tiebreaker"].items():
            print(f"    {winner}: {count}")

    # Save outputs
    output_dir = args.output_dir or os.path.join(base_dir, "results", "comparison")
    os.makedirs(output_dir, exist_ok=True)

    comparison_df = result.pop("comparison_df")
    comparison_df.to_csv(os.path.join(output_dir, "comparison_details.csv"), index=False)

    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved -> {output_dir}/")


if __name__ == "__main__":
    main()
