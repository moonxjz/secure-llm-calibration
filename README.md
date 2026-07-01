# secure_llm

Code repository for the paper:

> **Trustworthy Agentic Platform: Secure and Calibrated LLM Deployment for Small Business AI Services**  
> *Pragmatic Cybersecurity*, 2025

---

## Overview

This repo implements a layered prompt-injection defence framework evaluated across three LLMs and six datasets. The framework combines a structured guard prompt with a pre-trained injection detector ([GenTel-Shield](https://huggingface.co/GenTelLab/gentelshield-v1)) and evaluates both detection accuracy (EM) and confidence calibration (ECE).

**Four methods are compared:**

| Method | Guard Prompt | GenTel-Shield |
|---|:---:|:---:|
| Pure LLM | — | — |
| Pure LLM + GenTel-Shield | — | ✓ |
| Guard Prompts | ✓ | — |
| **Guard + GTS (ours)** | ✓ | ✓ |

---

## Datasets

>HF-D: "Same underlying corpus as Real-PI; used as an out-of-domain held-out split (200 rows, 100 benign / 100 attack)."

| Abbrev. | Description | Source |
|---|---|---|
| ATS-CS | ATS Customer Support (AI-generated) | Included in `data/` |
| ATS-EC | ATS E-commerce (AI-generated) | Included in `data/` |
| ATS-GK | ATS General Knowledge (AI-generated) | Included in `data/` |
| Real-PI | Real Prompt Injections | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) |
| HF-D | HF Deepset | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) |
| HF-M | HF Multilingual | [Octavio-Santana/prompt-injection-attack-detection-multilingual](https://huggingface.co/datasets/Octavio-Santana/prompt-injection-attack-detection-multilingual) |

Each dataset is balanced to 500 benign / 500 attack samples. Main results use 100 stratified rows per seed over seeds {42, 1337, 2026}.

---

## Models

| Model | Backend |
|---|---|
| GPT-4.1-mini | Azure OpenAI (`--backend azure`) |
| GPT-4.1 | Azure OpenAI (`--backend azure`) |
| Qwen2.5-1.5B-Instruct | Local HuggingFace (`--backend local`) |

---

## Environment Setup

**Requirements:** Python 3.10+, CUDA optional (GenTel-Shield runs on CPU).

```bash
pip install -r requirements.txt
```

Configure Azure credentials in `.env`:

```ini
API_KEY=...
ENDPOINT=...      # https://<resource>.openai.azure.com/openai/v1
API_VERSION=...   # optional
```

---

## Dataset Acquisition

ATS datasets are included in `data/`. Download the public datasets:

```bash
# deepset/prompt-injections → Real-PI and HF-D
# Octavio-Santana multilingual → HF-M
python scripts/download_hf_datasets.py
```

---

## Reproduce Main Results (Table 1)

Full three-seed evaluation across all six datasets. Runs Guard Prompts and Guard + GTS for both GPT models.

```bash
# Guard Prompts only (guard mode, no GenTel-Shield) — seed 42
python scripts/experiment.py --backend azure --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 42 \
  --output-dir runs_guard_seed42 \
  --datasets data/ATS_Customer_Support_500_balanced.csv \
             data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv \
             data/HF_multilingual_400_balanced.csv

# Guard + GTS (guard mode + GenTel-Shield) — seed 42
python scripts/experiment.py --backend azure --mode guard --gentel true \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 42 \
  --output-dir runs_gts_seed42 \
  --datasets data/ATS_Customer_Support_500_balanced.csv \
             data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv \
             data/HF_multilingual_400_balanced.csv
```

Repeat with `--seed 1337` and `--seed 2026`. Aggregate with `scripts/aggregate_results.py`.

For the open-source model, replace `--backend azure` with `--backend local`.

---

## Ablation Study (Table 3)

Ablates each guard prompt component individually. All runs: GPT-4.1 only, Real-PI dataset, 50 rows, seeds {42, 1337, 2026}.

| Variant | Prompt | Rules | ICL | Calibration |
|---|---|:---:|:---:|:---:|
| P1 — Full (ours) | `prompts/GuardPrompt_calibrated.txt` | ✓ | ✓ | ✓ |
| P2 — No Rules | `prompts/GuardPrompt_norules.txt` | ✗ | ✓ | ✓ |
| P3 — No ICL | `prompts/GuardPrompt_v_noexamples.txt` | ✓ | ✗ | ✗ |
| P4 — No Calibration | `prompts/GuardPrompt.txt` | ✓ | ✓ | ✗ |

```bash
# Example: P1 (full, seed 42)
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 50 --seed 42 \
  --output-dir runs_ablation_P1_seed42 \
  --datasets data/Real_PromptInjection_500_balanced.csv
```

Replace `--guard-prompt` and `--output-dir` for each variant and seed.

---

## Hyperparameter Study (Table 4)

Varies the number of ICL examples in the guard prompt. All runs: GPT-4.1 only, Real-PI dataset, 50 rows, seeds {42, 1337, 2026}.

| Variant | Prompt | Examples |
|---|---|:---:|
| H1 — 0 examples | `prompts/GuardPrompt_v_noexamples.txt` | 0 |
| H2 — 5 examples | `prompts/GuardPrompt_calibrated_5ex.txt` | 5 |
| H3 — 10 examples | `prompts/GuardPrompt_calibrated.txt` | 10 |

```bash
# Example: H2 (5 examples, seed 42)
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated_5ex.txt --limit 50 --seed 42 \
  --output-dir runs_hparam_H2_seed42 \
  --datasets data/Real_PromptInjection_500_balanced.csv
```

---

## Latency / Deployment Overhead

Measures sequential (non-concurrent), per-request end-to-end latency for the four
methods above, to quantify the response-time cost of the layered defences and
structured JSON output. Uses 100 stratified rows/seed on ATS-CS, seeds {42, 1337, 2026},
GPT-4.1-mini and GPT-4.1. Also isolates GenTel-Shield's own local inference time
(CPU-only) from the LLM API call.

```bash
python scripts/latency_benchmark.py --output-dir results/latency_raw
```

Key finding: GenTel-Shield's own inference adds ~13 ms/request (CPU-only), under 3% of
typical end-to-end latency; the guard prompt's longer system prompt and structured
output requirement showed no measurable latency penalty relative to the unguarded
baseline, with observed differences within normal API request-to-request variance.

---

## Key Files

| File | Description |
|---|---|
| `scripts/experiment.py` | Main evaluation entrypoint |
| `scripts/latency_benchmark.py` | Sequential per-request latency benchmark for the layered defences |
| `scripts/gentelshield.py` | GenTel-Shield wrapper (returns attack probability for ECE) |
| `scripts/aggregate_results.py` | Compute mean ± std across seeds |
| `scripts/download_hf_datasets.py` | Download HF-D and HF-M datasets |
| `scripts/balance_datasets.py` | Balance datasets to equal class sizes |
| `prompts/GuardPrompt_calibrated.txt` | Primary guard prompt (ICL + calibration guidance) |
| `prompts/GuardPrompt_norules.txt` | Ablation P2 — no classification rules |
| `prompts/GuardPrompt_v_noexamples.txt` | Ablation P3 / Hparam H1 — no ICL examples |
| `prompts/GuardPrompt.txt` | Ablation P4 — no calibration guidance |
| `prompts/GuardPrompt_calibrated_5ex.txt` | Hparam H2 — 5 ICL examples |
| `data/download_real_dataset.py` | Download Real-PI dataset |
| `process/results.md` | Full numerical results for all tables |

---

## Metrics

- **EM (Exact-Match accuracy):** Checks whether the `label` field in the structured JSON output exactly matches the ground truth. Equivalent to accuracy on balanced binary data; reported instead of Precision/Recall/F1 to reduce redundancy.
- **ECE (Expected Calibration Error):** Measures alignment between predicted confidence and empirical accuracy across confidence bins. Lower is better; zero is perfect calibration.

---

## Citation

```bibtex
@article{2025trustworthy,
  title   = {Trustworthy Agentic Platform: Secure and Calibrated LLM Deployment for Small Business AI Services},
  journal = {Pragmatic Cybersecurity},
  year    = {2025}
}
```
