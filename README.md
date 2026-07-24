# Receipt Field Extraction with LoRA

Fine-tuning `Qwen2.5-0.5B-Instruct` with LoRA (PEFT) to pull structured data :vendor, date,
line items, total : out of raw receipt/OCR-style text and into clean JSON.

## Why I built this

I wanted actual hands-on experience fine-tuning an open-weight model instead of just calling
an API, so I picked something concrete: turning messy OCR-style text into structured data.
This comes up a lot in practice (invoices, receipts, expense reports), so it felt like a
useful problem to build end to end myself:

- Wrote my own synthetic receipt generator, with injected noise (missing spaces, character
  swaps, inconsistent currency symbols) instead of using a clean toy dataset
- Fine-tuned with LoRA
- Built my own field-level evaluation harness : vendor match, date match, numeric total match,
  item-list match, valid-JSON rate — instead of just tracking loss
- Compared the fine-tuned model against the un-tuned base model on the same held-out test set

## What's here

| File | What it is |
|---|---|
| `receipt_lora_finetuning.ipynb` | The full pipeline : data generation, baseline eval, LoRA fine-tuning, eval, results |
| `results.json` | Field-level accuracy, base vs. fine-tuned |
| `requirements.txt` | Dependencies |

## How to run it

1. Open `receipt_lora_finetuning.ipynb` in Google Colab
2. Runtime should Change runtime type to T4 GPU
3. Run all cells top to bottom (~20–30 min)
4. `results.json` gets written at the end

## Results

Evaluated on 30 held-out synthetic receipts, exact match per field:

| Metric | Base model | Fine-tuned |
|---|---|---|
| Valid JSON rate | 1.00 | 1.00 |
| Vendor accuracy | 0.90 | 1.00 |
| Date accuracy | 0.97 | 1.00 |
| Total accuracy | 1.00 | 1.00 |
| **Items exact match** | **0.00** | **1.00** |

The base model was already decent at valid JSON, vendor, date, and total, but got the item
list wrong on every single example : probably a structural mismatch (e.g. using `"name"`
instead of `"item"` as a key) rather than not understanding the receipt at all. Fine-tuning on
400 examples fixed this completely, bringing every field to 100% on the test set, while only
training about 0.22% of the model's parameters (1.08M out of 495M).

Config: LoRA rank 8, alpha 16, targeting `q_proj/k_proj/v_proj/o_proj`, 400 train / 60 val / 60
test synthetic examples, 3 epochs.

## Some notes on the choices I made

- **Qwen2.5-0.5B-Instruct**: small enough to fine-tune on a free Colab GPU quickly, open-weight,
  already instruction-tuned : so I'm adapting its output behavior with LoRA, not teaching it to
  follow instructions from zero.
- **Synthetic data**: full control over noise level and difficulty, no dataset licensing to
  worry about, and building the generator was itself a good exercise.
- **Field-level eval instead of just loss**: a model can have decent training loss and still get
  the total or an item wrong — what actually matters is whether the extracted fields are right.

## Things I'd try next

- Crank up the noise level and see how much it breaks accuracy
- Compare LoRA rank (4 / 8 / 16) and see the accuracy/training-time tradeoff
- Swap the synthetic text for real OCR output (run Tesseract on actual receipt images)
- Try a VLM that takes the receipt image directly, skipping OCR entirely
