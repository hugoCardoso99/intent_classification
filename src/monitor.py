"""
Production monitoring for intent classification via LLM-as-a-Judge.

Detects distribution shift, OOD inputs, and confidence-judge divergence
by periodically sampling production predictions and running LLM evaluation.

Designed for periodic batch runs (e.g., daily cron job), not real-time.

Usage:
    # Monitor a production log file
    python src/monitor.py --log production_log.csv

    # Monitor with custom sample rate and alert thresholds
    python src/monitor.py --log production_log.csv \
        --sample_rate 0.1 --ood_threshold 0.05 --wrong_threshold 0.15

    # Append results to an existing monitoring history
    python src/monitor.py --log production_log.csv --append

Production log format (CSV):
    timestamp, text, predicted_intent, confidence
    (no ground-truth labels — that's the point)

Environment:
    GEMINI_API_KEY (or other provider key) — see llm_judge.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

from llm_judge import IntentJudge, get_provider


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

def compute_drift_metrics(judged_df: pd.DataFrame) -> dict:
    """Compute monitoring metrics from judged production samples.

    Returns a dict with:
        - ood_rate: fraction of out-of-domain inputs
        - wrong_rate: fraction of clearly wrong predictions
        - acceptable_rate: fraction of acceptable (ambiguous) predictions
        - correct_rate: fraction of correct predictions
        - confidence_judge_divergence: mean confidence on WRONG predictions
          (high = model is confidently wrong — red flag)
        - intent_distribution: predicted intent frequencies
        - ambiguity_rate: fraction of ambiguous inputs
        - judge_intent_distribution: what the judge thinks the intents are
    """
    total = len(judged_df)
    if total == 0:
        return {"error": "empty dataset"}

    verdict_counts = judged_df["verdict"].value_counts().to_dict()

    # Confidence on wrong predictions (high = dangerous)
    wrong_mask = judged_df["verdict"] == "WRONG"
    conf_on_wrong = (
        judged_df.loc[wrong_mask, "confidence"].mean()
        if wrong_mask.any() else 0.0
    )

    # Confidence on correct predictions (should be high)
    correct_mask = judged_df["verdict"] == "CORRECT"
    conf_on_correct = (
        judged_df.loc[correct_mask, "confidence"].mean()
        if correct_mask.any() else 0.0
    )

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_sampled": total,
        "verdict_counts": verdict_counts,
        "ood_rate": round(verdict_counts.get("OOD", 0) / total, 4),
        "wrong_rate": round(verdict_counts.get("WRONG", 0) / total, 4),
        "acceptable_rate": round(verdict_counts.get("ACCEPTABLE", 0) / total, 4),
        "correct_rate": round(verdict_counts.get("CORRECT", 0) / total, 4),
        "ambiguity_rate": round(
            judged_df["is_ambiguous"].sum() / total, 4
        ) if "is_ambiguous" in judged_df.columns else 0.0,
        "mean_confidence_on_wrong": round(conf_on_wrong, 4),
        "mean_confidence_on_correct": round(conf_on_correct, 4),
        "confidence_judge_divergence": round(conf_on_wrong - conf_on_correct, 4),
        "predicted_intent_distribution": (
            judged_df["predicted_intent"]
            .value_counts(normalize=True)
            .round(4)
            .to_dict()
        ),
    }

    # Judge's view of the distribution (what intents are actually coming in)
    if "judge_label" in judged_df.columns:
        judge_dist = (
            judged_df["judge_label"]
            .dropna()
            .value_counts(normalize=True)
            .round(4)
            .to_dict()
        )
        metrics["judge_intent_distribution"] = judge_dist

    return metrics


def check_alerts(metrics: dict,
                 ood_threshold: float = 0.05,
                 wrong_threshold: float = 0.15,
                 divergence_threshold: float = 0.3) -> list[dict]:
    """Check metrics against thresholds and return alerts.

    Args:
        ood_threshold: Alert if OOD rate exceeds this (default 5%)
        wrong_threshold: Alert if wrong rate exceeds this (default 15%)
        divergence_threshold: Alert if model is confidently wrong (default 0.3)

    Returns:
        List of alert dicts with severity, metric, value, threshold, message
    """
    alerts = []

    if metrics["ood_rate"] > ood_threshold:
        alerts.append({
            "severity": "WARNING" if metrics["ood_rate"] < ood_threshold * 2 else "CRITICAL",
            "metric": "ood_rate",
            "value": metrics["ood_rate"],
            "threshold": ood_threshold,
            "message": (
                f"OOD rate is {metrics['ood_rate']:.1%} (threshold: {ood_threshold:.1%}). "
                f"New types of requests may be appearing that the model wasn't trained on."
            ),
        })

    if metrics["wrong_rate"] > wrong_threshold:
        alerts.append({
            "severity": "WARNING" if metrics["wrong_rate"] < wrong_threshold * 1.5 else "CRITICAL",
            "metric": "wrong_rate",
            "value": metrics["wrong_rate"],
            "threshold": wrong_threshold,
            "message": (
                f"Wrong prediction rate is {metrics['wrong_rate']:.1%} "
                f"(threshold: {wrong_threshold:.1%}). "
                f"Model quality may be degrading — consider retraining."
            ),
        })

    if metrics["mean_confidence_on_wrong"] > divergence_threshold:
        alerts.append({
            "severity": "CRITICAL",
            "metric": "mean_confidence_on_wrong",
            "value": metrics["mean_confidence_on_wrong"],
            "threshold": divergence_threshold,
            "message": (
                f"Model confidence on wrong predictions is {metrics['mean_confidence_on_wrong']:.1%}. "
                f"The model is confidently wrong — this is a serious calibration issue."
            ),
        })

    return alerts


# ---------------------------------------------------------------------------
# Production log handling
# ---------------------------------------------------------------------------

def load_production_log(path: str) -> pd.DataFrame:
    """Load a production log CSV.

    Expected columns: timestamp, text, predicted_intent, confidence
    Optional columns: request_id, user_id
    """
    df = pd.read_csv(path)
    required = {"text", "predicted_intent", "confidence"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Production log missing columns: {missing}")
        print(f"Required: {required}")
        print(f"Found:    {set(df.columns)}")
        sys.exit(1)
    return df


def sample_production_log(df: pd.DataFrame, sample_rate: float = 0.1,
                          min_samples: int = 20,
                          max_samples: int = 200) -> pd.DataFrame:
    """Sample from production log for LLM evaluation.

    Args:
        sample_rate: Fraction of log to sample (default 10%)
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples (to control API costs)
    """
    n = max(min_samples, min(max_samples, int(len(df) * sample_rate)))
    n = min(n, len(df))
    return df.sample(n=n, random_state=int(datetime.now().timestamp()) % 10000)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Production monitoring via LLM-as-a-Judge"
    )
    parser.add_argument("--log", required=True,
                        help="Path to production log CSV")
    parser.add_argument("--provider", default="gemini",
                        choices=["gemini", "openai", "groq"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample_rate", type=float, default=0.1,
                        help="Fraction of log to sample (default: 0.1)")
    parser.add_argument("--min_samples", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--ood_threshold", type=float, default=0.05,
                        help="OOD rate alert threshold (default: 0.05)")
    parser.add_argument("--wrong_threshold", type=float, default=0.15,
                        help="Wrong rate alert threshold (default: 0.15)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/monitoring/)")
    parser.add_argument("--append", action="store_true",
                        help="Append to monitoring history instead of overwriting")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Setup output
    output_dir = args.output_dir or os.path.join(base_dir, "results", "monitoring")
    os.makedirs(output_dir, exist_ok=True)

    # Load and sample production log
    log_df = load_production_log(args.log)
    print(f"Production log: {len(log_df)} entries from {args.log}")

    sample_df = sample_production_log(
        log_df,
        sample_rate=args.sample_rate,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
    )
    print(f"Sampled {len(sample_df)} entries for LLM evaluation")

    # Run judge (no ground truth labels in production)
    provider = get_provider(args.provider, args.model)
    judge = IntentJudge(provider)
    print(f"Using LLM judge: {args.provider} ({provider.model_name})")

    # Build a minimal DataFrame for the judge
    judge_input = sample_df.copy()
    judge_input["correct"] = True  # placeholder — judge will override
    if "intent_name" not in judge_input.columns:
        judge_input["intent_name"] = None  # no ground truth

    verdicts = []
    for i, (_, row) in enumerate(judge_input.iterrows()):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Judging [{i+1}/{len(judge_input)}]", end="\r")

        verdict = judge.score_prediction(
            text=row["text"],
            predicted_intent=row["predicted_intent"],
            confidence=row["confidence"],
            true_intent=None,  # no ground truth in production
        )
        verdicts.append({
            "text": row["text"],
            "predicted_intent": row["predicted_intent"],
            "confidence": row["confidence"],
            "verdict": verdict.verdict,
            "judge_label": verdict.judge_label,
            "is_ambiguous": verdict.is_ambiguous,
            "reasoning": verdict.reasoning,
            "timestamp": row.get("timestamp", ""),
        })

    print(f"  Judging [{len(judge_input)}/{len(judge_input)}] Done.")

    judged_df = pd.DataFrame(verdicts)

    # Compute metrics
    metrics = compute_drift_metrics(judged_df)

    # Check alerts
    alerts = check_alerts(
        metrics,
        ood_threshold=args.ood_threshold,
        wrong_threshold=args.wrong_threshold,
    )

    # Print report
    print("\n" + "=" * 60)
    print("MONITORING REPORT")
    print("=" * 60)
    print(f"Timestamp:    {metrics['timestamp']}")
    print(f"Samples:      {metrics['total_sampled']}")
    print(f"Correct:      {metrics['correct_rate']:.1%}")
    print(f"Acceptable:   {metrics['acceptable_rate']:.1%}")
    print(f"Wrong:        {metrics['wrong_rate']:.1%}")
    print(f"OOD:          {metrics['ood_rate']:.1%}")
    print(f"Ambiguous:    {metrics['ambiguity_rate']:.1%}")
    print(f"Confidence on wrong: {metrics['mean_confidence_on_wrong']:.1%}")

    if alerts:
        print(f"\n{'!' * 60}")
        print(f"  {len(alerts)} ALERT(S) TRIGGERED")
        print(f"{'!' * 60}")
        for alert in alerts:
            print(f"\n  [{alert['severity']}] {alert['metric']}")
            print(f"  {alert['message']}")
    else:
        print("\nNo alerts — all metrics within thresholds.")

    # Save outputs
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    judged_path = os.path.join(output_dir, f"judged_{run_id}.csv")
    judged_df.to_csv(judged_path, index=False)
    print(f"\nJudged samples saved -> {judged_path}")

    metrics_path = os.path.join(output_dir, f"metrics_{run_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")

    if alerts:
        alerts_path = os.path.join(output_dir, f"alerts_{run_id}.json")
        with open(alerts_path, "w") as f:
            json.dump(alerts, f, indent=2)
        print(f"Alerts saved -> {alerts_path}")

    # Append to history
    history_path = os.path.join(output_dir, "monitoring_history.jsonl")
    history_entry = {
        "run_id": run_id,
        "timestamp": metrics["timestamp"],
        "total_sampled": metrics["total_sampled"],
        "correct_rate": metrics["correct_rate"],
        "wrong_rate": metrics["wrong_rate"],
        "ood_rate": metrics["ood_rate"],
        "ambiguity_rate": metrics["ambiguity_rate"],
        "mean_confidence_on_wrong": metrics["mean_confidence_on_wrong"],
        "alerts_count": len(alerts),
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(history_entry) + "\n")
    print(f"History appended -> {history_path}")


if __name__ == "__main__":
    main()
