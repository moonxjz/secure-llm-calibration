#!/usr/bin/env python3
"""Run the Secure LLM experiment suite across three dataset sampling seeds."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 1337, 2026]
EVAL_LIMIT = 400


def existing_dataset(preferred: str, fallback: str) -> str:
    return preferred if Path(preferred).exists() else fallback


HF_DEEPSET = existing_dataset("data/HF_deepset_400_balanced.csv", "data/HF_deepset_200_balanced.csv")
HF_MULTILINGUAL = existing_dataset("data/HF_multilingual_400_balanced.csv", "data/HF_multilingual_200_balanced.csv")

DATASETS = {
    "Original ATS": [
        "data/ATS_Customer_Support_500_balanced.csv",
        "data/ATS_Ecommerce_500_balanced.csv",
        "data/ATS_General_Knowledge_Rules_500_balanced.csv",
    ],
    "Real World": ["data/Real_PromptInjection_500_balanced.csv"],
    "Public HF": [HF_DEEPSET, HF_MULTILINGUAL],
    "Robustness": ["data/robustness_evaluation_200.csv"],
}
ALL_DATASETS = [dataset for datasets in DATASETS.values() for dataset in datasets]

GUARD_PROMPTS = {
    "baseline": Path("GuardPrompt.txt"),
    "robust": Path("GuardPrompt_robust.txt"),
    "hf_multilingual": Path("GuardPrompt_hf_multilingual.txt"),
}


def run_experiment(
    datasets: list[str],
    backend: str = "local",
    models: list[str] | None = None,
    guard_prompt: Path | None = None,
    mode: str = "both",
    limit: int | None = EVAL_LIMIT,
    output_dir: str | None = None,
    experiment_name: str = "test",
    seed: int = 42,
) -> dict:
    if models is None:
        models = ["Qwen/Qwen2.5-1.5B-Instruct"]
    if guard_prompt is None:
        guard_prompt = GUARD_PROMPTS["baseline"]
    if output_dir is None:
        output_dir = f"runs_{TIMESTAMP}_seed{seed}_{experiment_name}"

    print(f"\n{'=' * 70}")
    print(f"Experiment: {experiment_name}")
    print(f"Backend: {backend}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Guard Prompt: {guard_prompt.name}")
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Limit per dataset: {limit}")
    print(f"Output Dir: {output_dir}")
    print(f"{'=' * 70}")

    cmd = [
        "python",
        "experiment.py",
        "--backend",
        backend,
        "--guard-prompt",
        str(guard_prompt),
        "--mode",
        mode,
        "--output-dir",
        output_dir,
        "--seed",
        str(seed),
        "--datasets",
        *datasets,
    ]
    if backend == "local":
        cmd.extend(["--local-models", *models])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    cmd.extend(["--gentel", "false"])

    print(f"\nRunning: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\nExperiment completed: {experiment_name} seed={seed}")
        return {"status": "success", "output_dir": output_dir, "seed": seed}
    except subprocess.CalledProcessError as exc:
        print(f"\nExperiment failed: {experiment_name} seed={seed}")
        print(f"Error: {exc}")
        return {"status": "failed", "output_dir": output_dir, "seed": seed, "error": str(exc)}


def main() -> None:
    print(f"""
    Secure LLM - Comprehensive Experiment Suite
    Timestamp: {TIMESTAMP}
    Seeds: {SEEDS}
    Limit: up to {EVAL_LIMIT} rows per dataset per seed
    """)

    results_summary: list[dict] = []
    for seed in SEEDS:
        print(f"\n\n########## RANDOM SEED {seed} ##########")

        jobs = [
            ("Qwen_Baseline_AllDatasets", ALL_DATASETS, "local", ["Qwen/Qwen2.5-1.5B-Instruct"], GUARD_PROMPTS["baseline"], "guard", f"runs_{TIMESTAMP}_seed{seed}_qwen_baseline"),
            ("Robustness_Baseline_Prompt", ["data/robustness_evaluation_200.csv"], "local", ["Qwen/Qwen2.5-1.5B-Instruct"], GUARD_PROMPTS["baseline"], "guard", f"runs_{TIMESTAMP}_seed{seed}_robustness_baseline"),
            ("Robustness_Robust_Prompt", ["data/robustness_evaluation_200.csv"], "local", ["Qwen/Qwen2.5-1.5B-Instruct"], GUARD_PROMPTS["robust"], "guard", f"runs_{TIMESTAMP}_seed{seed}_robustness_robust"),
            ("HF_Deepset_GPT", [HF_DEEPSET], "azure", None, GUARD_PROMPTS["baseline"], "guard", f"runs_{TIMESTAMP}_seed{seed}_hf_deepset"),
            ("HF_Multilingual_GPT_Baseline", [HF_MULTILINGUAL], "azure", None, GUARD_PROMPTS["baseline"], "guard", f"runs_{TIMESTAMP}_seed{seed}_hf_multilingual_baseline"),
            ("HF_Multilingual_GPT_Optimized", [HF_MULTILINGUAL], "azure", None, GUARD_PROMPTS["hf_multilingual"], "guard", f"runs_{TIMESTAMP}_seed{seed}_hf_multilingual_optimized"),
            ("Ablation_HF_Datasets", [HF_DEEPSET, HF_MULTILINGUAL], "azure", None, GUARD_PROMPTS["baseline"], "both", f"runs_{TIMESTAMP}_seed{seed}_ablation_hf"),
        ]

        for name, datasets, backend, models, prompt, mode, output_dir in jobs:
            result = run_experiment(
                datasets=datasets,
                backend=backend,
                models=models,
                guard_prompt=prompt,
                mode=mode,
                limit=EVAL_LIMIT,
                output_dir=output_dir,
                experiment_name=name,
                seed=seed,
            )
            result["name"] = name
            results_summary.append(result)

    passed = sum(1 for item in results_summary if item["status"] == "success")
    total = len(results_summary)
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for item in results_summary:
        marker = "OK" if item["status"] == "success" else "FAIL"
        print(f"{marker} seed={item['seed']} {item['name']}: {item['status']}")
    print(f"\nTotal: {passed}/{total} experiments passed")

    summary_file = RESULTS_DIR / f"experiment_summary_{TIMESTAMP}.json"
    summary_file.write_text(json.dumps({
        "timestamp": TIMESTAMP,
        "seeds": SEEDS,
        "limit_per_dataset": EVAL_LIMIT,
        "experiments": results_summary,
        "passed": passed,
        "total": total,
    }, indent=2))
    print(f"\nSummary saved to: {summary_file}")
    print("Next step: run aggregate_results.py to compile mean/std report")


if __name__ == "__main__":
    main()
