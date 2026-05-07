"""
LLM-as-a-Judge for intent classification evaluation.

Uses an LLM (default: Gemini Flash free tier) to qualitatively assess
classifier predictions -- catching label ambiguity, OOD inputs, and
errors that metrics alone miss.

Provider-agnostic: swap in any LLM API by subclassing JudgeProvider.

Usage:
    # Offline evaluation of test set predictions
    python src/llm_judge.py --predictions results/with_adapter/predictions.csv

    # Evaluate with a custom provider/model
    python src/llm_judge.py --predictions results/with_adapter/predictions.csv \
        --provider gemini --model gemini-2.0-flash

    # Only judge misclassified examples
    python src/llm_judge.py --predictions results/with_adapter/predictions.csv \
        --errors_only

Environment:
    GEMINI_API_KEY   -- required for Gemini provider (default)
    OPENAI_API_KEY   -- required for OpenAI provider
    GROQ_API_KEY     -- required for Groq provider
"""

import argparse
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd

# Auto-load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


# ---------------------------------------------------------------------------
# Judge verdict data structure
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    """Result of a single LLM judge evaluation."""
    text: str
    true_intent: Optional[str] = None
    predicted_intent: str
    confidence: float
    verdict: str = Field(description="CORRECT, ACCEPTABLE, WRONG, or OOD")
    reasoning: str
    judge_label: Optional[str] = Field(default=None, description="What the judge thinks the correct label is")
    is_ambiguous: bool = False


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class JudgeProvider(ABC):
    """Abstract base for LLM API providers."""

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Raw API call -- implemented by each provider."""
        ...

    def generate(self, prompt: str, max_retries: int = 5) -> str:
        """Send a prompt with automatic retry on rate-limit errors."""
        wait = self._base_wait
        for attempt in range(max_retries):
            try:
                result = self._call_api(prompt)
                time.sleep(self._base_wait)  # proactive throttle
                return result
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in [
                    "resource_exhausted", "rate_limit", "429", "quota",
                    "too many requests", "retry",
                ])
                if is_rate_limit and attempt < max_retries - 1:
                    match = re.search(r"retry in ([\d.]+)s", str(e))
                    if match:
                        wait = float(match.group(1)) + 1.0
                    print(f"  Rate limited -- waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)  # exponential backoff, cap 2min
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")

    @property
    def _base_wait(self) -> float:
        return 4.0


class GeminiProvider(JudgeProvider):
    """Google Gemini API (free tier supported)."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        try:
            import google.generativeai as genai
        except ImportError:
            print("ERROR: pip install google-generativeai")
            sys.exit(1)

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            print("ERROR: Set GEMINI_API_KEY environment variable.")
            print("Get a free key at https://aistudio.google.com/apikey")
            sys.exit(1)

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    @property
    def _base_wait(self) -> float:
        return 5.0  # free tier: ~10-15 RPM

    def _call_api(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class OpenAIProvider(JudgeProvider):
    """OpenAI-compatible API (also works with OpenRouter)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: pip install openai")
            sys.exit(1)

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            print("ERROR: Set OPENAI_API_KEY environment variable.")
            sys.exit(1)

        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model_name = model

    @property
    def _base_wait(self) -> float:
        return 1.0

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content


class GroqProvider(JudgeProvider):
    """Groq API (free tier with Llama/Mixtral)."""

    def __init__(self, model: str = "llama-3.3-70b-versatile",
                 api_key: Optional[str] = None):
        try:
            from groq import Groq
        except ImportError:
            print("ERROR: pip install groq")
            sys.exit(1)

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            print("ERROR: Set GROQ_API_KEY environment variable.")
            sys.exit(1)

        self.client = Groq(api_key=key)
        self.model_name = model

    @property
    def _base_wait(self) -> float:
        return 2.0

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content


def get_provider(name: str = "gemini", model: Optional[str] = None,
                 api_key: Optional[str] = None) -> JudgeProvider:
    """Factory function to create a provider by name."""
    providers = {
        "gemini": (GeminiProvider, "gemini-2.0-flash"),
        "openai": (OpenAIProvider, "gpt-4o-mini"),
        "groq":   (GroqProvider,   "llama-3.3-70b-versatile"),
    }
    if name not in providers:
        print(f"ERROR: Unknown provider '{name}'. Choose from: {list(providers.keys())}")
        sys.exit(1)

    cls, default_model = providers[name]
    return cls(model=model or default_model, api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INTENT_DESCRIPTIONS = {
    "request_berth_booking":  "Requesting to book or reserve a berth/dock for a vessel",
    "ask_vessel_schedule":    "Asking about vessel arrival/departure times or schedules",
    "submit_customs_docs":    "Submitting or asking about customs documentation",
    "report_port_incident":   "Reporting a safety incident, accident, or emergency at the port",
    "ask_tariff_rates":       "Asking about port fees, tariffs, or pricing",
    "modify_berth_booking":   "Changing, updating, or canceling an existing berth booking",
    "track_container":        "Tracking the location or status of a container or shipment",
    "request_pilotage_tug":   "Requesting pilot or tug boat services for vessel navigation",
    "ask_regulations":        "Asking about port rules, regulations, compliance, or policies",
    "declare_cargo_manifest": "Declaring or submitting cargo manifest details",
}

SCORING_PROMPT = """You are evaluating an intent classifier for maritime port logistics operations.

## Available intents and their meanings:
{intent_list}

## Input to evaluate:
User message: "{text}"
Classifier prediction: {predicted_intent}
Classifier confidence: {confidence:.1%}
{true_label_line}

## Your task:
1. Determine what the CORRECT intent should be for this message.
2. Rate the classifier's prediction using one of these verdicts:
   - CORRECT: The prediction matches the correct intent.
   - ACCEPTABLE: The prediction is wrong per the label, but semantically reasonable
     (the message is genuinely ambiguous between the predicted and true intent).
   - WRONG: The prediction is clearly incorrect.
   - OOD: The message doesn't fit any of the 10 intents (out-of-domain).

## Respond in EXACTLY this JSON format (no other text):
{{
    "verdict": "<CORRECT|ACCEPTABLE|WRONG|OOD>",
    "judge_label": "<the intent you think is correct, or null if OOD>",
    "is_ambiguous": <true if the message could reasonably map to 2+ intents>,
    "reasoning": "<one sentence explaining your judgment>"
}}
"""

PAIRWISE_PROMPT = """You are comparing two versions of a maritime intent classifier on the same input.

## Available intents:
{intent_list}

## Input:
User message: "{text}"

## Predictions:
Model A: {pred_a} (confidence: {conf_a:.1%})
Model B: {pred_b} (confidence: {conf_b:.1%})

## Your task:
Both models gave different predictions. Determine which prediction is more appropriate
for this maritime port logistics message, or if neither is correct.

## Respond in EXACTLY this JSON format (no other text):
{{
    "winner": "<A|B|TIE|NEITHER>",
    "correct_intent": "<the intent you think is correct>",
    "reasoning": "<one sentence explaining your judgment>"
}}
"""


def _format_intent_list() -> str:
    """Format intent descriptions for prompt injection."""
    lines = []
    for intent, desc in INTENT_DESCRIPTIONS.items():
        lines.append(f"- {intent}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core judge logic
# ---------------------------------------------------------------------------

class IntentJudge:
    """LLM-as-a-Judge for intent classification."""

    def __init__(self, provider: JudgeProvider):
        self.provider = provider
        self.intent_list = _format_intent_list()

    def score_prediction(self, text: str, predicted_intent: str,
                         confidence: float,
                         true_intent: Optional[str] = None) -> JudgeVerdict:
        """Score a single classifier prediction."""
        true_label_line = ""
        if true_intent:
            true_label_line = f"Ground-truth label: {true_intent}"

        prompt = SCORING_PROMPT.format(
            intent_list=self.intent_list,
            text=text,
            predicted_intent=predicted_intent,
            confidence=confidence,
            true_label_line=true_label_line,
        )

        raw = self.provider.generate(prompt)
        parsed = self._parse_json(raw)

        return JudgeVerdict(
            text=text,
            true_intent=true_intent,
            predicted_intent=predicted_intent,
            confidence=confidence,
            verdict=parsed.get("verdict", "ERROR"),
            reasoning=parsed.get("reasoning", raw),
            judge_label=parsed.get("judge_label"),
            is_ambiguous=parsed.get("is_ambiguous", False),
        )

    def pairwise_compare(self, text: str,
                         pred_a: str, conf_a: float,
                         pred_b: str, conf_b: float) -> dict:
        """Pairwise comparison of two model predictions on the same input."""
        prompt = PAIRWISE_PROMPT.format(
            intent_list=self.intent_list,
            text=text,
            pred_a=pred_a, conf_a=conf_a,
            pred_b=pred_b, conf_b=conf_b,
        )
        raw = self.provider.generate(prompt)
        return self._parse_json(raw)

    def evaluate_predictions(self, predictions_df: pd.DataFrame,
                             errors_only: bool = False,
                             sample_n: Optional[int] = None) -> pd.DataFrame:
        """Batch-evaluate a predictions CSV from evaluate.py.

        Args:
            predictions_df: DataFrame with columns: text, intent_name, predicted_intent,
                           confidence, correct
            errors_only: If True, only judge misclassified examples
            sample_n: If set, randomly sample N examples to judge

        Returns:
            DataFrame with judge verdicts appended
        """
        df = predictions_df.copy()

        if errors_only:
            df = df[~df["correct"]].copy()
            print(f"Judging {len(df)} misclassified examples...")
        elif sample_n and sample_n < len(df):
            df = df.sample(n=sample_n, random_state=42).copy()
            print(f"Judging random sample of {len(df)} examples...")
        else:
            print(f"Judging all {len(df)} examples...")

        verdicts = []
        for i, (_, row) in enumerate(df.iterrows()):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(df)}]", end="\r")

            verdict = self.score_prediction(
                text=row["text"],
                predicted_intent=row["predicted_intent"],
                confidence=row["confidence"],
                true_intent=row.get("intent_name"),
            )
            verdicts.append(verdict.model_dump())

        print(f"  [{len(df)}/{len(df)}] Done.")

        verdict_df = pd.DataFrame(verdicts)
        # drop duplicate columns that already exist in df
        verdict_cols = ["verdict", "reasoning", "judge_label", "is_ambiguous"]
        result = pd.concat([df.reset_index(drop=True),
                            verdict_df[verdict_cols].reset_index(drop=True)], axis=1)
        return result

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"verdict": "ERROR", "reasoning": f"Failed to parse: {text[:200]}"}

    @staticmethod
    def summarize_verdicts(judged_df: pd.DataFrame) -> dict:
        """Aggregate verdict statistics from a judged DataFrame."""
        total = len(judged_df)
        counts = judged_df["verdict"].value_counts().to_dict()
        ambiguous = judged_df["is_ambiguous"].sum() if "is_ambiguous" in judged_df.columns else 0

        summary = {
            "total_judged": total,
            "verdict_counts": counts,
            "verdict_rates": {k: round(v / total, 3) for k, v in counts.items()},
            "ambiguous_count": int(ambiguous),
            "ambiguous_rate": round(ambiguous / total, 3) if total > 0 else 0,
        }

        if "judge_label" in judged_df.columns:
            judge_agrees = (judged_df["predicted_intent"] == judged_df["judge_label"]).sum()
            summary["judge_classifier_agreement"] = round(judge_agrees / total, 3)

        return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge for intent classification evaluation"
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions.csv from evaluate.py")
    parser.add_argument("--provider", default="gemini",
                        choices=["gemini", "openai", "groq"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model", default=None,
                        help="Model name override for the provider")
    parser.add_argument("--errors_only", action="store_true",
                        help="Only judge misclassified examples")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample size to judge (default: all)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: auto-generated)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load predictions
    pred_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(pred_df)} predictions from {args.predictions}")

    # Initialise judge
    provider = get_provider(args.provider, args.model)
    judge = IntentJudge(provider)
    print(f"Using LLM judge: {args.provider} ({provider.model_name})")

    # Run evaluation
    judged_df = judge.evaluate_predictions(
        pred_df,
        errors_only=args.errors_only,
        sample_n=args.sample,
    )

    # Summary
    summary = IntentJudge.summarize_verdicts(judged_df)
    print("\n" + "=" * 60)
    print("JUDGE SUMMARY")
    print("=" * 60)
    print(f"Total judged:    {summary['total_judged']}")
    for verdict, count in summary["verdict_counts"].items():
        rate = summary["verdict_rates"][verdict]
        print(f"  {verdict:12s}:  {count:4d}  ({rate:.1%})")
    print(f"Ambiguous:       {summary['ambiguous_count']}  ({summary['ambiguous_rate']:.1%})")
    if "judge_classifier_agreement" in summary:
        agr = summary["judge_classifier_agreement"]
        print(f"Judge-classifier agreement: {agr:.1%}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        results_dir = os.path.join(base_dir, "results", "judge")
        os.makedirs(results_dir, exist_ok=True)
        suffix = "_errors" if args.errors_only else ""
        output_path = os.path.join(results_dir, f"judged_predictions{suffix}.csv")

    judged_df.to_csv(output_path, index=False)
    print(f"\nJudged predictions saved -> {output_path}")

    # Save summary
    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
