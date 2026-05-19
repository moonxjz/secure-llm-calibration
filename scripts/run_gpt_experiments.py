#!/usr/bin/env python3
"""
Run comprehensive experiments on all datasets using GPT-4.1-mini via Azure.
Focus on validating the core claim: guard prompts + structured output = excellent calibration + high F1
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# All datasets to evaluate
DATASETS = {
    "original_ats": [
        "data/ATS_Customer_Support_500_balanced.csv",
        "data/ATS_Ecommerce_500_balanced.csv",
        "data/ATS_General_Knowledge_Rules_500_balanced.csv",
    ],
    "real_world": [
        "data/Real_PromptInjection_500_balanced.csv",
    ],
    "public_hf": [
        "data/HF_deepset_200_balanced.csv",
        "data/HF_multilingual_200_balanced.csv",
    ],
}

# All datasets combined for unified evaluation
ALL_DATASETS = []
for datasets_list in DATASETS.values():
    ALL_DATASETS.extend(datasets_list)

ROBUSTNESS_DATASET = ["data/robustness_evaluation_200.csv"]

GUARD_PROMPTS = {
    "baseline": Path("GuardPrompt.txt"),
    "robust": Path("GuardPrompt_robust.txt"),
    "multilingual": Path("GuardPrompt_hf_multilingual.txt"),
}

def run_experiment(
    datasets: list,
    guard_prompt: Path,
    mode: str = "both",
    gentel: bool = True,
    output_dir: str = None,
    exp_name: str = "",
) -> dict:
    """Run experiment with GPT-4.1-mini via Azure."""
    
    if output_dir is None:
        output_dir = f"runs_{TIMESTAMP}_{exp_name}"
    
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Datasets: {len(datasets)} files")
    print(f"Guard Prompt: {guard_prompt.name}")
    print(f"Mode: {mode}")
    print(f"GenTel: {gentel}")
    print(f"{'='*70}\n")
    
    cmd = [
        "python", "experiment.py",
        "--backend", "azure",
        "--guard-prompt", str(guard_prompt),
        "--mode", mode,
        "--output-dir", output_dir,
        "--datasets",
    ] + datasets + [
        "--gentel", str(gentel).lower(),
        "--concurrency", "8"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {exp_name} completed\n")
        return {"status": "success", "output_dir": output_dir}
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {exp_name} failed: {e}\n")
        return {"status": "failed", "output_dir": output_dir, "error": str(e)}

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   Secure LLM - Comprehensive GPT-4 Evaluation                   ║
║   (All Public Datasets + Robustness Testing)                    ║
║   {TIMESTAMP}                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    results_summary = []
    
    # ========================================================================
    # PHASE 1: MAIN EVALUATION - Guard + GenTel on all datasets
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Main Evaluation - Guard + GenTel on All Datasets")
    print("="*70)
    print("Reproducing core results on original + new public datasets\n")
    
    result = run_experiment(
        datasets=ALL_DATASETS,
        guard_prompt=GUARD_PROMPTS["baseline"],
        mode="both",
        gentel=True,
        output_dir=f"runs_{TIMESTAMP}_main_guard_gentel",
        exp_name="Main_Guard_GenTel_AllDatasets"
    )
    results_summary.append(("Main_Guard_GenTel", result["status"]))
    
    # ========================================================================
    # PHASE 2: ROBUSTNESS TESTING - Baseline vs Robust Prompt
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: Robustness Testing - Adversarial Mutations")
    print("="*70)
    print("Testing against format variations and obfuscation\n")
    
    # Baseline on robustness
    result_robust_base = run_experiment(
        datasets=ROBUSTNESS_DATASET,
        guard_prompt=GUARD_PROMPTS["baseline"],
        mode="guard",
        gentel=False,
        output_dir=f"runs_{TIMESTAMP}_robustness_baseline",
        exp_name="Robustness_Baseline"
    )
    results_summary.append(("Robustness_Baseline", result_robust_base["status"]))
    
    # Robust variant on robustness
    result_robust_var = run_experiment(
        datasets=ROBUSTNESS_DATASET,
        guard_prompt=GUARD_PROMPTS["robust"],
        mode="guard",
        gentel=False,
        output_dir=f"runs_{TIMESTAMP}_robustness_robust",
        exp_name="Robustness_Robust_Variant"
    )
    results_summary.append(("Robustness_Robust", result_robust_var["status"]))
    
    # ========================================================================
    # PHASE 3: PUBLIC DATASET DEEP DIVE
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 3: Public HuggingFace Datasets - Baseline vs Optimized")
    print("="*70)
    print("Evaluating with prompt variants\n")
    
    # Deepset with baseline
    result_deepset = run_experiment(
        datasets=DATASETS["public_hf"][:1],  # HF_deepset only
        guard_prompt=GUARD_PROMPTS["baseline"],
        mode="both",
        gentel=False,
        output_dir=f"runs_{TIMESTAMP}_hf_deepset_baseline",
        exp_name="HF_Deepset_Baseline"
    )
    results_summary.append(("HF_Deepset_Baseline", result_deepset["status"]))
    
    # Multilingual with baseline
    result_multi_base = run_experiment(
        datasets=DATASETS["public_hf"][1:],  # Multilingual only
        guard_prompt=GUARD_PROMPTS["baseline"],
        mode="both",
        gentel=False,
        output_dir=f"runs_{TIMESTAMP}_hf_multilingual_baseline",
        exp_name="HF_Multilingual_Baseline"
    )
    results_summary.append(("HF_Multilingual_Baseline", result_multi_base["status"]))
    
    # Multilingual with optimized prompt
    result_multi_opt = run_experiment(
        datasets=DATASETS["public_hf"][1:],  # Multilingual only
        guard_prompt=GUARD_PROMPTS["multilingual"],
        mode="guard",
        gentel=False,
        output_dir=f"runs_{TIMESTAMP}_hf_multilingual_optimized",
        exp_name="HF_Multilingual_Optimized"
    )
    results_summary.append(("HF_Multilingual_Optimized", result_multi_opt["status"]))
    
    # ========================================================================
    # PHASE 4: ABLATION STUDY on combined public data
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 4: Ablation Study - Component Importance")
    print("="*70)
    print("Testing format/examples impact on new datasets\n")
    
    result_ablation = run_experiment(
        datasets=DATASETS["public_hf"],  # Both HF datasets
        guard_prompt=GUARD_PROMPTS["baseline"],
        mode="both",
        gentel=False,
        output_dir=f"runs_{TIMESTAMP}_ablation_public",
        exp_name="Ablation_PublicData"
    )
    results_summary.append(("Ablation_Public", result_ablation["status"]))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for exp_name, status in results_summary:
        status_emoji = "✓" if status == "success" else "✗"
        print(f"{status_emoji} {exp_name}: {status}")
    
    passed = sum(1 for _, s in results_summary if s == "success")
    total = len(results_summary)
    print(f"\nTotal: {passed}/{total} experiments passed")
    
    # Save summary
    summary_file = RESULTS_DIR / f"experiment_summary_{TIMESTAMP}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": TIMESTAMP,
            "experiments": [{"name": n, "status": s} for n, s in results_summary],
            "passed": passed,
            "total": total
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("="*70)

if __name__ == "__main__":
    main()
