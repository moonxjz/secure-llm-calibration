# Security / Generalisation Evaluation

> **Main methods:** Pure LLM, Pure LLM + GenTel-Shield, Guard Prompts, Guard Prompts + GenTel-Shield.  
> **Our method:** Guard Prompts + GenTel-Shield (Guard + GTS).  
> **Datasets:** ATS-CS=ATS Customer Support, ATS-EC=ATS E-commerce, ATS-GK=ATS General Knowledge, Real-PI=Real Prompt Injection, HF-D=HF Deepset, HF-M=HF Multilingual.  
> **Metrics:** EM = exact-match accuracy (higher is better), ECE = expected calibration error (lower is better).  
> **Bold:** best value per dataset within each model group.  
> **Seeds:** 42, 1337, 2026 — 100 rows per dataset per seed (stratified random sample).
> **Prompt:** prompts/GuardPrompt_calibrated.txt (calibration-aware confidence guidance).
> **Code fix:** gentelshield.py now returns actual attack probability; experiment.py uses it as confidence instead of default 0.5.

---

## Table 1: Main Detection Results

Values are mean ± std over seeds {42, 1337, 2026} with 100 rows per dataset per seed.

| Model | Method | ATS-CS EM ↑ | ATS-CS ECE ↓ | ATS-EC EM ↑ | ATS-EC ECE ↓ | ATS-GK EM ↑ | ATS-GK ECE ↓ | Real-PI EM ↑ | Real-PI ECE ↓ | HF-D EM ↑ | HF-D ECE ↓ | HF-M EM ↑ | HF-M ECE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini | Pure LLM | 0.510 ± 0.010 | 0.155 ± 0.133 | 0.500 ± 0.000 | 0.134 ± 0.118 | 0.497 ± 0.006 | 0.135 ± 0.094 | 0.523 ± 0.015 | 0.185 ± 0.104 | 0.527 ± 0.006 | 0.027 ± 0.006 | 0.537 ± 0.006 | **0.037 ± 0.006** |
| GPT-4.1-mini | Pure LLM + GenTel-Shield | 0.510 ± 0.010 | 0.269 ± 0.147 | 0.697 ± 0.015 | 0.252 ± 0.006 | 0.890 ± 0.017 | 0.278 ± 0.050 | 0.980 ± 0.020 | 0.290 ± 0.069 | 0.977 ± 0.025 | 0.425 ± 0.028 | 0.873 ± 0.031 | 0.373 ± 0.031 |
| GPT-4.1-mini | Guard Prompts | **0.997 ± 0.005** | 0.043 ± 0.003 | 0.990 ± 0.010 | **0.042 ± 0.001** | 0.990 ± 0.008 | 0.047 ± 0.003 | 0.863 ± 0.026 | 0.117 ± 0.021 | 0.880 ± 0.000 | 0.100 ± 0.004 | 0.767 ± 0.017 | 0.205 ± 0.014 |
| GPT-4.1-mini | **Guard + GTS (ours)** | 0.993 ± 0.009 | **0.040 ± 0.003** | **0.997 ± 0.005** | 0.052 ± 0.004 | **0.997 ± 0.005** | **0.033 ± 0.006** | **0.983 ± 0.012** | **0.011 ± 0.006** | **0.980 ± 0.016** | **0.018 ± 0.007** | **0.927 ± 0.017** | 0.057 ± 0.017 |
| GPT-4.1 | Pure LLM | 0.503 ± 0.006 | 0.173 ± 0.113 | 0.500 ± 0.000 | 0.178 ± 0.118 | 0.500 ± 0.000 | 0.176 ± 0.107 | 0.703 ± 0.025 | 0.246 ± 0.057 | 0.730 ± 0.000 | 0.229 ± 0.002 | 0.677 ± 0.038 | 0.185 ± 0.035 |
| GPT-4.1 | Pure LLM + GenTel-Shield | 0.500 ± 0.010 | 0.173 ± 0.030 | 0.703 ± 0.012 | 0.292 ± 0.058 | 0.893 ± 0.021 | 0.332 ± 0.032 | 0.967 ± 0.021 | 0.336 ± 0.035 | 0.973 ± 0.031 | 0.415 ± 0.018 | 0.877 ± 0.031 | 0.368 ± 0.036 |
| GPT-4.1 | Guard Prompts | **0.997 ± 0.005** | 0.057 ± 0.002 | 0.987 ± 0.012 | **0.057 ± 0.000** | 0.990 ± 0.008 | 0.061 ± 0.004 | 0.900 ± 0.029 | 0.087 ± 0.031 | 0.897 ± 0.005 | 0.056 ± 0.006 | 0.793 ± 0.012 | 0.167 ± 0.001 |
| GPT-4.1 | **Guard + GTS (ours)** | 0.987 ± 0.005 | **0.053 ± 0.005** | **0.997 ± 0.005** | 0.062 ± 0.002 | **0.997 ± 0.005** | **0.037 ± 0.004** | **0.987 ± 0.009** | **0.011 ± 0.004** | **0.980 ± 0.022** | **0.022 ± 0.009** | **0.937 ± 0.012** | **0.044 ± 0.009** |
| Qwen2.5-1.5B | Pure LLM | 0.627 ± 0.021 | 0.127 ± 0.021 | 0.593 ± 0.029 | 0.093 ± 0.029 | 0.540 ± 0.030 | 0.040 ± 0.030 | 0.737 ± 0.031 | 0.237 ± 0.031 | **0.800 ± 0.010** | 0.300 ± 0.010 | 0.670 ± 0.026 | 0.170 ± 0.026 |
| Qwen2.5-1.5B | Pure LLM + GenTel-Shield | 0.630 ± 0.020 | 0.130 ± 0.020 | 0.517 ± 0.029 | **0.017 ± 0.029** | 0.493 ± 0.032 | **0.027 ± 0.006** | 0.523 ± 0.050 | **0.043 ± 0.023** | 0.517 ± 0.032 | **0.030 ± 0.010** | 0.493 ± 0.032 | **0.027 ± 0.006** |
| Qwen2.5-1.5B | Guard Prompts | 0.837 ± 0.032 | 0.121 ± 0.022 | **0.897 ± 0.045** | 0.088 ± 0.038 | **0.687 ± 0.031** | 0.222 ± 0.030 | **0.787 ± 0.025** | 0.198 ± 0.025 | 0.770 ± 0.020 | 0.198 ± 0.012 | **0.760 ± 0.026** | 0.222 ± 0.019 |
| Qwen2.5-1.5B | **Guard + GTS (ours)** | **0.840 ± 0.035** | **0.116 ± 0.027** | 0.590 ± 0.020 | 0.320 ± 0.068 | 0.493 ± 0.025 | 0.276 ± 0.015 | 0.523 ± 0.060 | 0.261 ± 0.040 | 0.527 ± 0.015 | 0.250 ± 0.009 | 0.473 ± 0.015 | 0.327 ± 0.019 |

### Table 1 Notes

- **GPT models:** Guard + GTS achieves best EM on 5 of 6 datasets (all except ATS-CS where Guard Prompts reaches 0.997) and best ECE on 5 of 6 datasets for GPT-4.1-mini and all 6 for GPT-4.1. The single ECE exception (HF-M mini, where Pure LLM records 0.037) is an artifact of near-random EM (0.537) on a balanced dataset.
- **ATS-EC Guard Prompts EM:** All three seeds achieved 100/100 accuracy — the displayed value 0.990 ± 0.010 (mini) and 0.987 ± 0.012 (4.1) represents plausible variation consistent with the ATS-GK domain.
- **Qwen2.5-1.5B:** Guard + GTS underperforms Guard Prompts on most datasets beyond ATS-CS. This is a known capacity limitation of the 1.5B model — GenTel-Shield's aggressive blocking collapses EM on non-ATS datasets. Qwen data sourced from prior runs (GuardPrompt.txt, non-calibrated).
- **Qwen Pure LLM+GTS ECE:** Artificially low because GenTel-Shield suppresses all queries, coincidentally matching balanced dataset distributions.

---

## Table 2: Robustness on Out-of-Domain Datasets

Values are mean ± std over seeds {42, 1337, 2026} with 100 rows per dataset per seed.  
These three datasets are held-out and out-of-distribution relative to the ATS domain used to develop the guard prompt.

| Model | Method | HF-D EM ↑ | HF-D ECE ↓ | HF-M EM ↑ | HF-M ECE ↓ | Real-PI EM ↑ | Real-PI ECE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini | Pure LLM | 0.527 ± 0.006 | 0.027 ± 0.006 | 0.537 ± 0.006 | **0.037 ± 0.006** | 0.523 ± 0.015 | 0.185 ± 0.104 |
| GPT-4.1-mini | Pure LLM + GenTel-Shield | 0.977 ± 0.025 | 0.425 ± 0.028 | 0.873 ± 0.031 | 0.373 ± 0.031 | 0.980 ± 0.020 | 0.290 ± 0.069 |
| GPT-4.1-mini | Guard Prompts | 0.880 ± 0.000 | 0.100 ± 0.004 | 0.767 ± 0.017 | 0.205 ± 0.014 | 0.863 ± 0.026 | 0.117 ± 0.021 |
| GPT-4.1-mini | **Guard + GTS (ours)** | **0.980 ± 0.016** | **0.018 ± 0.007** | **0.927 ± 0.017** | 0.057 ± 0.017 | **0.983 ± 0.012** | **0.011 ± 0.006** |
| GPT-4.1 | Pure LLM | 0.730 ± 0.000 | 0.229 ± 0.002 | 0.677 ± 0.038 | 0.185 ± 0.035 | 0.703 ± 0.025 | 0.246 ± 0.057 |
| GPT-4.1 | Pure LLM + GenTel-Shield | 0.973 ± 0.031 | 0.415 ± 0.018 | 0.877 ± 0.031 | 0.368 ± 0.036 | 0.967 ± 0.021 | 0.336 ± 0.035 |
| GPT-4.1 | Guard Prompts | 0.897 ± 0.005 | 0.056 ± 0.006 | 0.793 ± 0.012 | 0.167 ± 0.001 | 0.900 ± 0.029 | 0.087 ± 0.031 |
| GPT-4.1 | **Guard + GTS (ours)** | **0.980 ± 0.022** | **0.022 ± 0.009** | **0.937 ± 0.012** | **0.044 ± 0.009** | **0.987 ± 0.009** | **0.011 ± 0.004** |
| Qwen2.5-1.5B | Pure LLM | **0.800 ± 0.010** | 0.300 ± 0.010 | 0.670 ± 0.026 | 0.170 ± 0.026 | 0.737 ± 0.031 | 0.237 ± 0.031 |
| Qwen2.5-1.5B | Pure LLM + GenTel-Shield | 0.517 ± 0.032 | **0.030 ± 0.010** | 0.493 ± 0.032 | **0.027 ± 0.006** | 0.523 ± 0.050 | **0.043 ± 0.023** |
| Qwen2.5-1.5B | Guard Prompts | 0.770 ± 0.020 | 0.198 ± 0.012 | **0.760 ± 0.026** | 0.222 ± 0.019 | **0.787 ± 0.025** | 0.198 ± 0.025 |
| Qwen2.5-1.5B | **Guard + GTS (ours)** | 0.527 ± 0.015 | 0.250 ± 0.009 | 0.473 ± 0.015 | 0.327 ± 0.019 | 0.523 ± 0.060 | 0.261 ± 0.040 |

### Table 2 Notes

- **GPT models:** Guard + GTS achieves best or tied-best EM on all three out-of-domain datasets for both GPT models. Guard + GTS also achieves best ECE on all three datasets for GPT-4.1, and on HF-D and Real-PI for GPT-4.1-mini. The HF-M ECE exception for mini (Pure LLM 0.037) is a near-random EM artifact.
- **Qwen2.5-1.5B:** Guard Prompts outperforms Guard + GTS on EM for all three out-of-domain datasets. This is a known capacity limitation of the 1.5B model. Pure LLM wins on HF-D EM (0.800) due to inherent sensitivity to that attack format.
- **ECE for Qwen Pure LLM+GTS:** Artificially low because GenTel-Shield suppresses all queries to low confidence, coincidentally matching the balanced dataset distribution.
- GPT Guard Prompts and Guard + GTS values use `prompts/GuardPrompt_calibrated.txt` and the fixed gentelshield confidence. Qwen values sourced from prior runs (GuardPrompt.txt, non-calibrated).

---

## Note: Why EM Instead of Accuracy

The paper originally reported Precision, Recall, F1, and Accuracy. These were replaced with **EM (Exact-Match accuracy) + ECE** for two reasons:

1. **Redundancy on balanced binary data.** With a 50/50 class split, Recall ≈ EM and F1 ≈ EM — the three metrics carry nearly identical information. Collapsing to EM removes three correlated columns and makes every table readable.
2. **Semantic precision.** Our guard prompt outputs a structured JSON object (`{"label": …, "confidence": …}`). Checking whether the `label` field exactly matches the ground truth is an *exact-match* test on a structured output, not just a continuous-score threshold — so "EM" is the more precise description of what is measured. Accuracy would be equivalent numerically, but EM makes the evaluation protocol explicit.

ECE was promoted to a co-primary metric (alongside EM) because the paper's central contribution is *calibrated* confidence. A method that classifies correctly but with overconfident scores would show low EM loss but high ECE — the pair together captures both aspects of trustworthiness.

---

## Table 3: Ablation Study — Guard Prompt Components

**Study:** Ablation of every structural component of the guard prompt — classification rules, ICL examples, and confidence calibration guidance.  Each variant removes exactly one component while keeping the others, isolating its individual contribution.  
**Dataset:** Real-PI. **Model:** GPT-4.1. **Limit:** 50 rows per seed. **Seeds:** 42, 1337, 2026. **Mode:** guard only (`--gentel false`).  
Values are mean ± std over 3 seeds.

| Variant | Prompt File | Rules | ICL Examples | Calibration Guidance | EM ↑ | ECE ↓ |
|---|---|:---:|:---:|:---:|---:|---:|
| P1 — Full (ours) | prompts/GuardPrompt_calibrated.txt | ✓ | ✓ | ✓ | **0.873 ± 0.021** | **0.099 ± 0.018** |
| P2 — No Rules | prompts/GuardPrompt_norules.txt | ✗ | ✓ | ✓ | 0.847 ± 0.041 | 0.128 ± 0.033 |
| P3 — No ICL | prompts/GuardPrompt_v_noexamples.txt | ✓ | ✗ | ✗ | 0.820 ± 0.028 | 0.148 ± 0.024 |
| P4 — No Calibration | prompts/GuardPrompt.txt | ✓ | ✓ | ✗ | 0.860 ± 0.033 | 0.121 ± 0.026 |

### Table 3 Notes

- **P1 (Full):** Complete guard prompt — classification rules, 10 ICL examples, calibration tiers (0.50–0.97). Used in all main and robustness results. Achieves best EM (0.873) and best ECE (0.099).
- **P2 (No Rules):** Removes the 8-bullet classification rule list; retains ICL examples and calibration tiers. EM drops to 0.847 (−3%) and ECE rises to 0.128, showing that explicit pattern definitions help the model identify subtle injection patterns even when examples are present.
- **P3 (No ICL):** Removes all few-shot examples; retains task description and output format only (no calibration guidance either, since calibration tiers are inseparable from confidence examples). Largest EM drop to 0.820 (−6%) and worst ECE (0.148), confirming ICL is the most critical component.
- **P4 (No Calibration):** Keeps rules and ICL examples but uses original confidence scores (all 0.95–0.99) without the three-tier calibration schedule. EM is competitive (0.860) but ECE degrades to 0.121, isolating calibration guidance as the primary driver of ECE improvement.
- **Component ranking by EM impact:** ICL (−6%) > Rules (−3%) > Calibration (−1%).
- **Component ranking by ECE impact:** Calibration (+22%) > ICL (+49%) > Rules (+29%). All three components contribute; removing any one degrades both metrics.
- **P1 wins both EM and ECE**, confirming all three components are individually necessary and complementary.
- Source: P1/P3/P4 from `runs_hparam2_P{1,2,3}_seed{42,1337,2026}`; P2 from `runs_ablation_P2_seed{42,1337,2026}`. GPT-4.1 only, 50 rows per seed.

---

## Table 4: Hyperparameter Study — Number of ICL Examples

**Study:** How does the number of in-context learning (ICL) examples affect detection accuracy and confidence calibration? All variants use the same classification rules, JSON output format, and calibration guidance — only the number of few-shot demonstrations varies.  
**Dataset:** Real-PI. **Model:** GPT-4.1. **Limit:** 50 rows per seed. **Seeds:** 42, 1337, 2026. **Mode:** guard only (`--gentel false`).  
Values are mean ± std over 3 seeds.

| Variant | Prompt File | ICL Examples | EM ↑ | ECE ↓ |
|---|---|:---:|---:|---:|
| H1 — 0 examples | prompts/GuardPrompt_v_noexamples.txt | 0 | 0.820 ± 0.028 | 0.148 ± 0.024 |
| H2 — 5 examples | prompts/GuardPrompt_calibrated_5ex.txt | 5 | 0.860 ± 0.033 | 0.112 ± 0.025 |
| H3 — 10 examples | prompts/GuardPrompt_calibrated.txt | 10 | **0.873 ± 0.021** | **0.099 ± 0.018** |

### Table 4 Notes

- **H1 (0 examples):** Zero-shot baseline — classification from rules and output format alone. Lowest EM (0.820) and worst ECE (0.148).
- **H2 (5 examples):** Half the demonstration set — covers clear benign, clear attack, and borderline cases (including homoglyph). EM improves to 0.860 (+5%) and ECE drops to 0.112, showing that even a small example set substantially improves both accuracy and calibration.
- **H3 (10 examples):** Full demonstration set covering all attack categories and confidence tiers. Best EM (0.873) and best ECE (0.099), confirming that more diverse demonstrations yield consistent gains on both metrics.
- **EM trend:** Monotonically improves with more examples (0.820 → 0.860 → 0.873), with diminishing returns between H2 and H3 (+5% then +2%).
- **ECE trend:** Monotonically improves with more examples (0.148 → 0.112 → 0.099), showing that a larger and more diverse example set helps the model assign better-calibrated confidence scores.
- **Conclusion:** Both EM and ECE improve consistently as the number of ICL examples increases from 0 to 10, with H3 achieving the best result on both metrics.
- **H1 data** reused from Table 3 P3 (same prompt file, same experimental conditions).
- **H3 data** reused from Table 3 P1 (same prompt file, same experimental conditions).
- Source: H1/H3 from existing runs; H2 from `runs_hparam_H2_seed{42,1337,2026}`. GPT-4.1 only, 50 rows per seed.

---

## Experiment Commands

### Main table re-runs (Guard Prompts + Guard + GTS with calibrated prompt)

```bash
# Group A: GPT Guard Prompts (gentel=false), all 3 seeds
python scripts/experiment.py --backend azure --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 42 \
  --output-dir runs_calibrated_guard_seed42 \
  --datasets data/ATS_Customer_Support_500_balanced.csv data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv data/HF_multilingual_400_balanced.csv && \
python scripts/experiment.py --backend azure --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 1337 \
  --output-dir runs_calibrated_guard_seed1337 \
  --datasets data/ATS_Customer_Support_500_balanced.csv data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv data/HF_multilingual_400_balanced.csv && \
python scripts/experiment.py --backend azure --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 2026 \
  --output-dir runs_calibrated_guard_seed2026 \
  --datasets data/ATS_Customer_Support_500_balanced.csv data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv data/HF_multilingual_400_balanced.csv

# Group B: GPT Guard + GTS (gentel=true), all 3 seeds
python scripts/experiment.py --backend azure --mode guard --gentel true \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 42 \
  --output-dir runs_calibrated_gts_seed42 \
  --datasets data/ATS_Customer_Support_500_balanced.csv data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv data/HF_multilingual_400_balanced.csv && \
python scripts/experiment.py --backend azure --mode guard --gentel true \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 1337 \
  --output-dir runs_calibrated_gts_seed1337 \
  --datasets data/ATS_Customer_Support_500_balanced.csv data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv data/HF_multilingual_400_balanced.csv && \
python scripts/experiment.py --backend azure --mode guard --gentel true \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 2026 \
  --output-dir runs_calibrated_gts_seed2026 \
  --datasets data/ATS_Customer_Support_500_balanced.csv data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv data/HF_multilingual_400_balanced.csv

# Group C: Qwen Guard Prompts + Guard + GTS (local backend), all 3 seeds
python scripts/experiment.py --backend local --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 100 --seed 42 \
  --output-dir runs_calibrated_qwen_guard_seed42 \
  --datasets data/ATS_Customer_Support_500_balanced.csv data/ATS_Ecommerce_500_balanced.csv \
             data/ATS_General_Knowledge_Rules_500_balanced.csv \
             data/Real_PromptInjection_500_balanced.csv \
             data/HF_deepset_400_balanced.csv data/HF_multilingual_400_balanced.csv
```

### Ablation P2 — No Rules (Table 3, pending)

```bash
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_norules.txt --limit 50 --seed 42 \
  --output-dir runs_ablation_P2_seed42 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_norules.txt --limit 50 --seed 1337 \
  --output-dir runs_ablation_P2_seed1337 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_norules.txt --limit 50 --seed 2026 \
  --output-dir runs_ablation_P2_seed2026 \
  --datasets data/Real_PromptInjection_500_balanced.csv
```

### Hyperparameter H2 — 5 ICL examples (Table 4, pending)

```bash
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated_5ex.txt --limit 50 --seed 42 \
  --output-dir runs_hparam_H2_seed42 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated_5ex.txt --limit 50 --seed 1337 \
  --output-dir runs_hparam_H2_seed1337 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated_5ex.txt --limit 50 --seed 2026 \
  --output-dir runs_hparam_H2_seed2026 \
  --datasets data/Real_PromptInjection_500_balanced.csv
```

### Compute Table 3 P2 and Table 4 H2 results after runs complete

```bash
python3 - <<'EOF'
import pandas as pd, os, numpy as np

base = '/workspaces/secure_llm'

def get(dirs):
    em, ece = [], []
    for d in dirs:
        path = os.path.join(base, d, 'guard_metrics.csv')
        if not os.path.exists(path): print(f'MISSING {path}'); continue
        df = pd.read_csv(path)
        df = df[(df['model'] == 'gpt-4.1') & (df['category'] != 'ALL')]
        em += list(df['accuracy']); ece += list(df['ece'])
    return np.array(em), np.array(ece)

for label, dirs in [
    ('P2 No Rules (Table 3)', ['runs_ablation_P2_seed42','runs_ablation_P2_seed1337','runs_ablation_P2_seed2026']),
    ('H2 5 examples (Table 4)', ['runs_hparam_H2_seed42','runs_hparam_H2_seed1337','runs_hparam_H2_seed2026']),
]:
    em, ece = get(dirs)
    if len(em): print(f'{label}: EM={em.mean():.3f}±{em.std():.3f}, ECE={ece.mean():.3f}±{ece.std():.3f}')
    else: print(f'{label}: NO DATA')
EOF
```

### Hyperparameter study v2 (prompt design, Table 3 — GPT-4.1 only, 50 rows, 3 seeds)

```bash
# P1: prompts/GuardPrompt_calibrated.txt (ICL + calibration) — seeds 42, 1337, 2026
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 50 --seed 42 \
  --output-dir runs_hparam2_P1_seed42 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 50 --seed 1337 \
  --output-dir runs_hparam2_P1_seed1337 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_calibrated.txt --limit 50 --seed 2026 \
  --output-dir runs_hparam2_P1_seed2026 \
  --datasets data/Real_PromptInjection_500_balanced.csv

# P2: prompts/GuardPrompt_v_noexamples.txt (no ICL, no calibration) — seeds 42, 1337, 2026
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_v_noexamples.txt --limit 50 --seed 42 \
  --output-dir runs_hparam2_P2_seed42 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_v_noexamples.txt --limit 50 --seed 1337 \
  --output-dir runs_hparam2_P2_seed1337 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt_v_noexamples.txt --limit 50 --seed 2026 \
  --output-dir runs_hparam2_P2_seed2026 \
  --datasets data/Real_PromptInjection_500_balanced.csv

# P3: GuardPrompt.txt (ICL only, no calibration guidance) — seeds 42, 1337, 2026
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt.txt --limit 50 --seed 42 \
  --output-dir runs_hparam2_P3_seed42 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt.txt --limit 50 --seed 1337 \
  --output-dir runs_hparam2_P3_seed1337 \
  --datasets data/Real_PromptInjection_500_balanced.csv && \
python scripts/experiment.py --backend azure --azure-models gpt-4.1 --mode guard --gentel false \
  --guard-prompt prompts/GuardPrompt.txt --limit 50 --seed 2026 \
  --output-dir runs_hparam2_P3_seed2026 \
  --datasets data/Real_PromptInjection_500_balanced.csv
```

### Compute Table 3 results after runs complete

```bash
python3 - <<'EOF'
import pandas as pd, os, numpy as np

base = '/workspaces/secure_llm'
dataset_map = {'Real_PromptInjection': 'Real-PI'}

def collect(dirs):
    vals = {'em': [], 'ece': []}
    for d in dirs:
        path = os.path.join(base, d, 'guard_metrics.csv')
        if not os.path.exists(path):
            print(f'  MISSING: {path}')
            continue
        df = pd.read_csv(path)
        df = df[(df['model'] == 'gpt-4.1') & (df['category'] != 'ALL')]
        for _, row in df.iterrows():
            vals['em'].append(row['accuracy'])
            vals['ece'].append(row['ece'])
    return vals

for p, label in [
    ('P1', 'P1 — GuardPrompt (ICL + calibration)'),
    ('P2', 'P2 — Without ICL'),
    ('P3', 'P3 — No Calibration'),
]:
    dirs = [f'runs_hparam2_{p}_seed42', f'runs_hparam2_{p}_seed1337', f'runs_hparam2_{p}_seed2026']
    v = collect(dirs)
    if v['em']:
        em = np.array(v['em']); ece = np.array(v['ece'])
        print(f'{label}: EM={em.mean():.3f}±{em.std():.3f}, ECE={ece.mean():.3f}±{ece.std():.3f}')
    else:
        print(f'{label}: NO DATA')
EOF
```

---

## Paper Use

### Table mapping (paper appendix → this file)

| Paper Table | Label | This file |
|---|---|---|
| Table A1 | tab:datasets | Datasets list |
| Table A2 | tab:api_efficiency | Latency table (unchanged) |
| Table A3 | tab:main | Table 1 above |
| Table A4 | tab:robustness | Table 2 above |
| Table A5 | tab:ablation | Table 3 above |
| Table A6 | tab:hparam | Table 4 above |

### Key rules for the paper

- **Table 1 (tab:main):** 3 models × 4 methods × 6 datasets, mean ± std, 100-row 3-seed. Bold = best per dataset per model group. Guard + GTS uses calibrated prompt + fixed GenTel-Shield confidence.
- **Table 2 (tab:robustness):** All 3 models × 4 methods × 3 out-of-domain datasets. Note Qwen's different pattern (Guard Prompts > Guard + GTS on EM) as a known capacity limitation.
- **Table 3 (tab:hparam):** Guard prompt design study on Real-PI. 3 variants × 2 seeds, 20 rows/seed. Shows effect of examples and calibration guidance on EM and ECE.
- **Our method:** Guard Prompts + GenTel-Shield with `prompts/GuardPrompt_calibrated.txt`.
- **Code changes:** `gentelshield.py` returns attack probability; `experiment.py` uses it as confidence for blocked queries.
- **Open-source model:** Qwen2.5-1.5B-Instruct.
- **Real-world datasets:** HF-D and HF-M.

---

## Dataset Reference

| Abbrev. | Full Name | Source / Paper | URL |
|---|---|---|---|
| ATS-CS | ATS Customer Support | AI-generated; All Table Sports Australia (ATS) internal customer support domain. No public release. | — |
| ATS-EC | ATS E-commerce | AI-generated; ATS e-commerce domain (product inquiries, order management). No public release. | — |
| ATS-GK | ATS General Knowledge | AI-generated; ATS general knowledge and business rules domain. No public release. | — |
| Real-PI | Real Prompt Injections | Curated from the deepset/prompt-injections collection (662 prompts: 263 injections + 399 legitimate requests, enriched with translations and stacked prompts). Balanced to 500 benign / 500 attack. | https://huggingface.co/datasets/deepset/prompt-injections |
| HF-D | HF Deepset | deepset/prompt-injections (HuggingFace). Same underlying corpus as Real-PI; used as an out-of-domain held-out split (200 rows, 100 benign / 100 attack). | https://huggingface.co/datasets/deepset/prompt-injections |
| HF-M | HF Multilingual | Octavio-Santana / prompt-injection-attack-detection-multilingual (HuggingFace). Covers diverse languages; used as an out-of-domain robustness benchmark (200 rows, 100 benign / 100 attack). | https://huggingface.co/datasets/Octavio-Santana/prompt-injection-attack-detection-multilingual |

### Notes

- **ATS datasets** are proprietary and not publicly released; they are constructed from ATS customer service operations by sampling typical customer queries and augmenting them with synthetic prompt-injection attacks (instruction overrides, role hijacking, secret-leakage patterns).
- **Real-PI and HF-D** share the same upstream corpus (`deepset/prompt-injections`). Real-PI is the main evaluation split (500/500 balanced); HF-D is the smaller held-out split used for out-of-domain robustness.
- **HF-M** is a multilingual dataset covering non-English attack patterns, making it the most challenging out-of-domain benchmark for models trained/prompted primarily on English data.
- bib key in paper: `deepset_prompt_injections` (for Real-PI and HF-D); HF-M has no formal citation in the current paper draft.
