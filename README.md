# Maritime Port Logistics — Intent Classification with PEFT/LoRA

Fine-tuning a pre-trained HuggingFace transformer (RoBERTa-base) for multi-class intent classification in the maritime and port logistics domain, using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).

## Context

Port operations involve a high volume of structured communications — berth requests, vessel schedule queries, customs filings, container tracking, and incident reports. Automating the classification of these intents enables intelligent routing, faster response times, and reduced manual triage in port management systems.

This project trains a lightweight LoRA adapter on top of `roberta-base` to classify user utterances into **10 maritime intents**:

| Intent | Description | Real-world frequency |
|---|---|---|
| `track_container` | Container location, status, or inspection queries | High |
| `ask_vessel_schedule` | Vessel ETA/ETD and voyage schedule inquiries | High |
| `request_berth_booking` | New berth allocation or anchorage requests | Medium-high |
| `declare_cargo_manifest` | Cargo manifest declarations and bulk cargo filings | Medium |
| `request_pilotage_tug` | Pilot boarding, tug assistance, or escort requests | Medium |
| `submit_customs_docs` | Customs declarations, certificates, and SAD documents | Medium-low |
| `modify_berth_booking` | Berth changes, cancellations, or pilot order modifications | Low |
| `ask_tariff_rates` | Wharfage, handling charges, and fee inquiries | Low |
| `ask_regulations` | ISPS, ballast water, port state control, and compliance questions | Low |
| `report_port_incident` | Security alerts, equipment incidents, and safety reports | Rare |

## Approach

**Why PEFT/LoRA instead of full fine-tuning?**

Full fine-tuning updates all ~125M parameters of RoBERTa-base. LoRA instead injects small trainable rank-decomposition matrices into the attention layers (Q and V projections), training less than 1% of total parameters. This means:

- The LoRA adapter is only a few MB (vs ~500MB for a full model copy)
- Multiple task-specific adapters can share a single base model in production
- Training is faster and requires less GPU memory
- Performance is comparable to full fine-tuning for classification tasks

**Training details:**

- Base model: `roberta-base` (125M params)
- LoRA rank: 8, alpha: 16, dropout: 0.1
- Target modules: `query`, `value` (attention projections)
- Classification head: trained fully via `modules_to_save`
- Learning rate: 2e-4 with linear warmup (10%) and weight decay (0.01)
- Class weights: inverse real-world frequency, so rare intents (e.g. `report_port_incident`) receive higher loss penalties
- Early stopping: patience 2 on macro F1
- Training set: ~1,458 balanced synthetic examples
- Test set: ~400 examples with realistic imbalanced distribution

## Results

| Metric | With LoRA Adapter | Base Model (no adapter) |
|---|---|---|
| Overall Accuracy | **93.0%** | 3.0% |
| Macro F1 | **0.928** | 0.006 |
| Weighted F1 | **0.929** | 0.002 |

The base model without the adapter predicts almost everything as a single class (~3% accuracy), confirming that the LoRA adapter is doing all the meaningful work.

**Per-class highlights:** 5 of 10 intents achieve perfect F1 (1.000). The weakest performers are `report_port_incident` (F1 0.762, only 8 test samples) and `declare_cargo_manifest` (F1 0.790, confused with `ask_vessel_schedule`).

## Project Structure

```
intent_detection/
├── data/
│   ├── train.csv              # balanced training set (~1,458 examples)
│   ├── test.csv               # imbalanced test set (~400 examples)
│   ├── label_map.csv          # intent name <-> label ID
│   └── class_weights.json     # inverse-frequency weights for training
├── src/
│   ├── generate_data.py       # synthetic dataset generation
│   ├── train.py               # LoRA fine-tuning with weighted loss
│   ├── evaluate.py            # metrics, confusion matrix, F1 charts
│   └── predict.py             # interactive / batch inference
├── models/
│   └── roberta-intent/        # saved LoRA adapter (~few MB)
├── results/
│   ├── with_adapter/          # evaluation with LoRA
│   └── base_model/            # baseline evaluation (no adapter)
└── requirements.txt
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (if starting fresh)
python src/generate_data.py

# Train the LoRA adapter
python src/train.py

# Evaluate with adapter
python src/evaluate.py

# Evaluate baseline (no adapter) for comparison
python src/evaluate.py --no_adapter

# Interactive inference
python src/predict.py

# Batch inference
python src/predict.py --input utterances.txt
```

**LoRA hyperparameters** can be tuned via command-line arguments:

```bash
python src/train.py --lora_r 16 --lora_alpha 32 --epochs 10 --lr 1e-4
```

## Production Considerations

In production, only the LoRA adapter is stored and deployed — the base `roberta-base` model is loaded once from HuggingFace's cache and shared across all adapters. This allows:

- Serving multiple domain-specific adapters from one base model
- Fast adapter swaps without reloading the full model
- Minimal storage per fine-tuned variant

The `--no_adapter` flag on evaluate and predict scripts allows quick A/B comparison between the fine-tuned model and the untrained baseline at any time.
