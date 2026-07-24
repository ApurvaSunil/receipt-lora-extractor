# Receipt Field Extraction with LoRA Fine-Tuning

Fine-tuning a small open-weight LLM (`Qwen2.5-0.5B-Instruct`) with **LoRA (PEFT)** to extract
structured data (vendor, date, line items, total) from raw receipt/OCR-style text into JSON.

## Why this project

Document understanding — turning messy OCR text into clean structured records — is a common
real-world ML problem (invoice processing, identity verification, expense automation). This
project builds the full pipeline end-to-end:

- **Synthetic data generation** with injected OCR-style noise (missing spaces, character
  confusion, inconsistent currency symbols) rather than a clean toy dataset
- **LoRA fine-tuning** of an open-weight instruction-tuned model
- **A field-level evaluation harness** (not just training loss) — vendor match, date match,
  numeric total match, item-list match, and valid-JSON rate
- **Before/after comparison** against the un-tuned base model on the same held-out test set

## What's in this repo

| File | Purpose |
|---|---|
| `receipt_lora_finetuning.ipynb` | Full pipeline: data generation → baseline eval → LoRA fine-tuning → eval → results |
| `results.json` | Field-level accuracy scores, base vs. fine-tuned (generated after running the notebook) |
| `requirements.txt` | Dependencies |

## How to run

1. Open `receipt_lora_finetuning.ipynb` in **Google Colab**
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells top to bottom (~20–30 minutes total)
4. `results.json` is written at the end with the before/after comparison

## Results

*(Fill in after running — copy the printed comparison from the notebook.)*

| Metric | Base model | Fine-tuned |
|---|---|---|
| Valid JSON rate | | |
| Vendor accuracy | | |
| Date accuracy | | |
| Total accuracy | | |
| Items exact match | | |

## Design choices worth noting

- **Why Qwen2.5-0.5B-Instruct**: small enough to fine-tune on a free Colab GPU in minutes,
  already instruction-tuned, fully open-weight (no gating). This project is about pipeline
  mechanics, not chasing state-of-the-art numbers.
- **Why synthetic data**: full control over noise level and task difficulty, no licensing
  ambiguity, and it's an explicit skill (synthetic data generation is called out as a bonus
  skill in a lot of applied ML/LLM job postings).
- **Why field-level eval, not just loss**: a model can have low training loss while still
  getting the total or date wrong — the metric that matters is whether the *extracted fields*
  are correct.

## Possible extensions

- Increase synthetic noise to test robustness under harder OCR corruption
- Compare LoRA rank (r=4 / 8 / 16) and report the accuracy/training-time tradeoff
- Swap synthetic text for real OCR output (e.g. run Tesseract over receipt images)
- Extend to a Vision-Language Model that takes the receipt image directly, skipping the OCR step
