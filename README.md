# secure_llm

Code repository for the secure LLM prompt-injection detection experiments used in the paper.

## What this repo runs

- `pure` mode: direct binary attack classification prompt.
- `guard` mode: classification using the safeguard prompt from `GuardPrompt.txt`.
- optional `gentel` pre-check: runs GentelShield first, then automatically reruns without Gentel for paired comparison.

Main entrypoint: `experiment.py`

## Environment setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure Azure/OpenAI-compatible credentials in `.env`:

```ini
API_KEY=...
ENDPOINT=...      # e.g. https://<resource>.openai.azure.com/openai/v1
API_VERSION=...   # optional; used for deployment-scoped endpoints
```

## Dataset acquisition

### Included in repo

- `data/ATS_Customer_Support_500_balanced.csv`
- `data/ATS_Ecommerce_500_balanced.csv`
- `data/ATS_General_Knowledge_Rules_500_balanced.csv`

These are the ATS balanced files used by default.

### External source used in paper

- Hugging Face `deepset/prompt-injections`: https://huggingface.co/datasets/deepset/prompt-injections

Create the balanced file used by the experiments:

```bash
python data/download_real_dataset.py
```

This generates `data/Real_PromptInjection_500_balanced.csv`.

## Reproduce paper run (150 rows per dataset)

Run the main setup used for the 150-row experiments:

```bash
python experiment.py \
   --backend azure \
   --azure-models gpt-4.1-mini gpt-4.1 \
   --datasets \
      data/ATS_Customer_Support_500_balanced.csv \
      data/ATS_Ecommerce_500_balanced.csv \
      data/ATS_General_Knowledge_Rules_500_balanced.csv \
      data/Real_PromptInjection_500_balanced.csv \
   --guard-prompt GuardPrompt.txt \
   --mode both \
   --limit 150 \
   --concurrency 2 \
   --temperature 0.0 \
   --max-tokens 16 \
   --gentel true
```

## Optional ablations from the paper workflow

No examples (A2):

```bash
python experiment.py --backend azure --azure-models gpt-4.1-mini gpt-4.1 --mode guard --limit 150 --concurrency 2 --temperature 0.0 --max-tokens 16 --guard-prompt GuardPrompt_v_noexamples.txt --gentel true
```

No JSON format (A3):

```bash
python experiment.py --backend azure --azure-models gpt-4.1-mini gpt-4.1 --mode guard --limit 150 --concurrency 2 --temperature 0.0 --max-tokens 16 --guard-prompt GuardPrompt_v_noformat.txt --gentel true
```

Temperature 0.1 (H2):

```bash
python experiment.py --backend azure --azure-models gpt-4.1-mini gpt-4.1 --mode guard --limit 150 --concurrency 2 --temperature 0.1 --max-tokens 16 --guard-prompt GuardPrompt.txt --gentel true
```

## Notes

- `--gentel` accepts explicit boolean values: `true` or `false`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
- If you run local inference instead of Azure, use `--backend local --local-models ...`.
