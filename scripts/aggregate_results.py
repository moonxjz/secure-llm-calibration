#!/usr/bin/env python3
"""Aggregate experiment CSVs into a mean/std evaluation report."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RUN_MAPPINGS = {
    "runs_*_main_guard_gentel": "Main Evaluation (Guard + GenTel)",
    "runs_*_qwen_baseline": "Qwen Baseline All Datasets",
    "runs_*_robustness_baseline": "Robustness - Baseline Prompt",
    "runs_*_robustness_robust": "Robustness - Robust Variant",
    "runs_*_hf_deepset": "HF Deepset - Baseline",
    "runs_*_hf_deepset_baseline": "HF Deepset - Baseline",
    "runs_*_hf_multilingual_baseline": "HF Multilingual - Baseline",
    "runs_*_hf_multilingual_optimized": "HF Multilingual - Optimized",
    "runs_*_ablation_hf": "Ablation - Public Data",
    "runs_*_ablation_public": "Ablation - Public Data",
}
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "ece"]


def infer_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def latest_seed_campaign() -> str | None:
    campaigns = []
    for path in Path(".").glob("runs_*_seed*_*"):
        match = re.match(r"runs_(\d{4}-\d{2}-\d{2}_\d{6})_seed\d+_", path.name)
        if match:
            campaigns.append(match.group(1))
    return max(campaigns) if campaigns else None


def find_run_dirs() -> Dict[str, List[Path]]:
    run_dirs: Dict[str, List[Path]] = {}
    campaign = latest_seed_campaign()
    if campaign:
        print(f"Using latest seeded campaign only: {campaign}")

    for pattern, label in RUN_MAPPINGS.items():
        matches = sorted(Path(".").glob(pattern), key=lambda x: x.stat().st_mtime)
        if campaign:
            matches = [m for m in matches if m.name.startswith(f"runs_{campaign}_seed")]
        if matches:
            run_dirs.setdefault(label, [])
            for match in matches:
                if match not in run_dirs[label]:
                    run_dirs[label].append(match)
    for label, dirs in run_dirs.items():
        print(f"Found {len(dirs)} run(s): {label}")
        for run_dir in dirs:
            print(f"  - {run_dir}")
    return run_dirs


def load_metrics(run_dir: Path, filename: str = "guard_metrics.csv") -> pd.DataFrame:
    metric_file = run_dir / filename
    if not metric_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(metric_file)
    df["run_dir"] = run_dir.name
    df["seed"] = infer_seed(run_dir)
    return df


def collect_metrics(run_dirs: Dict[str, List[Path]]) -> pd.DataFrame:
    frames = []
    for label, dirs in run_dirs.items():
        for run_dir in dirs:
            metrics = load_metrics(run_dir)
            if metrics.empty:
                continue
            metrics["experiment"] = label
            frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = ["experiment", "mode", "model", "category"]
    summary = df.groupby(group_cols, dropna=False)[METRIC_COLS].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    for metric in METRIC_COLS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0.0)
    return summary


def fmt_mean_std(row: pd.Series, metric: str, digits: int = 4) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_std']:.{digits}f}"


def dataset_overview() -> str:
    rows = []
    for path in sorted(Path("data").glob("*_balanced.csv")) + sorted(Path("data").glob("robustness_evaluation_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        label_col = "Answer" if "Answer" in df.columns else "Gold Answer" if "Gold Answer" in df.columns else None
        if label_col:
            counts = df[label_col].astype(int).value_counts().to_dict()
            rows.append((path.as_posix(), len(df), counts.get(0, 0), counts.get(1, 0)))
    md = "## Dataset Overview\n\n"
    md += "| Dataset | Rows | Benign | Attack |\n|---|---:|---:|---:|\n"
    for path, total, benign, attack in rows:
        md += f"| `{path}` | {total} | {benign} | {attack} |\n"
    return md + "\n"


def results_table(summary: pd.DataFrame, category_filter: str | None = None) -> str:
    if summary.empty:
        return "No metrics found.\n"
    work = summary.copy()
    if category_filter is not None:
        work = work[work["category"].astype(str).str.contains(category_filter, case=False, na=False)]
    if work.empty:
        return "No matching metrics found.\n"

    md = "| Experiment | Category | Model | Runs | Accuracy | Precision | Recall | F1 | ECE |\n"
    md += "|---|---|---|---:|---:|---:|---:|---:|---:|\n"
    for _, row in work.sort_values(["experiment", "category", "model"]).iterrows():
        model = str(row["model"]).split("/")[-1]
        runs = int(row["f1_count"])
        md += (
            f"| {row['experiment']} | {row['category']} | {model} | {runs} | "
            f"{fmt_mean_std(row, 'accuracy')} | {fmt_mean_std(row, 'precision')} | "
            f"{fmt_mean_std(row, 'recall')} | {fmt_mean_std(row, 'f1')} | {fmt_mean_std(row, 'ece')} |\n"
        )
    return md + "\n"


def main() -> None:
    print(f"Results aggregation timestamp: {TIMESTAMP}")
    run_dirs = find_run_dirs()
    metrics = collect_metrics(run_dirs)
    if metrics.empty:
        print("No completed metrics found. Run experiments first.")
        return

    summary = summarize_metrics(metrics)
    summary_csv = RESULTS_DIR / f"evaluation_metrics_summary_{TIMESTAMP}.csv"
    summary.to_csv(summary_csv, index=False)

    markdown = f"""# Secure LLM - Comprehensive Evaluation Report
*Generated: {TIMESTAMP}*

This report aggregates all matching experiment runs. When three seeded runs are present, values are reported as mean +/- std across seeds. The evaluation runner samples up to 400 rows per dataset per seed using stratified random sampling.

"""
    markdown += dataset_overview()
    markdown += "## Main Results (Mean +/- Std)\n\n"
    markdown += results_table(summary)
    markdown += "## Robustness Results\n\n"
    markdown += results_table(summary[summary["experiment"].astype(str).str.contains("Robustness", case=False, na=False)])
    markdown += "## Public Dataset Results\n\n"
    markdown += results_table(summary[summary["experiment"].astype(str).str.contains("HF|Ablation", case=False, na=False)])
    markdown += "## Notes\n\n"
    markdown += "ECE is computed as prediction confidence versus correctness. This avoids reporting artificially perfect calibration from class-probability/sign inversions or from first-token JSON logprobs.\n"

    output_file = RESULTS_DIR / f"comprehensive_evaluation_{TIMESTAMP}.md"
    output_file.write_text(markdown)

    json_file = RESULTS_DIR / f"evaluation_metrics_{TIMESTAMP}.json"
    json_file.write_text(json.dumps({
        "timestamp": TIMESTAMP,
        "summary_csv": str(summary_csv),
        "experiments": sorted(metrics["experiment"].unique().tolist()),
        "run_count": int(metrics[["experiment", "run_dir"]].drop_duplicates().shape[0]),
    }, indent=2))

    print(f"Saved markdown: {output_file}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved metrics JSON: {json_file}")


if __name__ == "__main__":
    main()
