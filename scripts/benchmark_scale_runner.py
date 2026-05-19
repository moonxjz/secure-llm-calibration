import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_500_DATASETS = [
    Path("data/ATS_Customer_Support_500_balanced.csv"),
    Path("data/ATS_Ecommerce_500_balanced.csv"),
    Path("data/ATS_General_Knowledge_Rules_500_balanced.csv"),
    Path("data/Real_PromptInjection_500_balanced.csv"),
]


def run_experiment(cmd: List[str]) -> tuple[int, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - start
    return proc.returncode, elapsed


def make_1000_balanced(in_path: Path, out_path: Path) -> None:
    df = pd.read_csv(in_path)
    if "Answer" not in df.columns:
        raise RuntimeError(f"Dataset {in_path} is missing 'Answer' column")

    df["Answer"] = pd.to_numeric(df["Answer"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    benign = df[df["Answer"] == 0]
    attack = df[df["Answer"] == 1]
    if len(benign) == 0 or len(attack) == 0:
        raise RuntimeError(f"Dataset {in_path} must contain both benign(0) and attack(1) rows")

    benign_500 = benign.sample(n=500, replace=len(benign) < 500, random_state=42)
    attack_500 = attack.sample(n=500, replace=len(attack) < 500, random_state=42)
    out_df = pd.concat([benign_500, attack_500], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


def expand_datasets_to_1000(source_datasets: List[Path]) -> List[Path]:
    out_paths: List[Path] = []
    for src in source_datasets:
        if not src.exists():
            raise RuntimeError(f"Missing dataset: {src}")

        name = src.name
        if "_500_" in name:
            out_name = name.replace("_500_", "_1000_")
        elif name.endswith("_500_balanced.csv"):
            out_name = name.replace("_500_balanced.csv", "_1000_balanced.csv")
        else:
            out_name = src.stem + "_1000_balanced.csv"

        out_path = src.with_name(out_name)
        make_1000_balanced(src, out_path)
        out_paths.append(out_path)
        print(f"Created {out_path} (1000 rows target)")
    return out_paths


def build_base_cmd(
    backend: str,
    models: List[str],
    output_dir: Path,
    datasets: List[Path],
    limit: int | None,
    concurrency: int,
    store_raw: bool,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "experiment.py",
        "--backend",
        backend,
        "--mode",
        "both",
        "--output-dir",
        str(output_dir),
        "--concurrency",
        str(concurrency),
        "--datasets",
    ]
    cmd.extend([str(d) for d in datasets])

    if backend == "local":
        cmd.append("--local-models")
        cmd.extend(models)

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if store_raw:
        cmd.append("--store-raw")

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 500-row benchmark first, then automatically expand all datasets to 1000 rows "
            "and rerun if benchmark completes within threshold."
        )
    )
    parser.add_argument("--backend", choices=["local", "azure"], default="local")
    parser.add_argument(
        "--local-models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ],
    )
    parser.add_argument("--threshold-minutes", type=float, default=20.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--store-raw", action="store_true")
    parser.add_argument("--benchmark-output-dir", type=Path, default=Path("data/runs_benchmark_500"))
    parser.add_argument("--expanded-output-dir", type=Path, default=Path("data/runs_benchmark_1000"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: benchmark at 500 rows per dataset using --limit 500.
    args.benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    bench_cmd = build_base_cmd(
        backend=args.backend,
        models=args.local_models,
        output_dir=args.benchmark_output_dir,
        datasets=DEFAULT_500_DATASETS,
        limit=500,
        concurrency=args.concurrency,
        store_raw=args.store_raw,
    )

    print("Running benchmark (500 rows per dataset)...")
    print(" ".join(bench_cmd))
    code, elapsed_sec = run_experiment(bench_cmd)
    elapsed_min = elapsed_sec / 60.0
    print(f"Benchmark exit_code={code}, elapsed={elapsed_sec:.2f}s ({elapsed_min:.2f} min)")

    if code != 0:
        print("Benchmark failed. Expansion run skipped.")
        sys.exit(code)

    if elapsed_sec > args.threshold_minutes * 60.0:
        print(
            f"Benchmark exceeded threshold ({args.threshold_minutes:.2f} min). "
            "Expansion to 1000 rows per dataset skipped."
        )
        return

    # Step 2: expand each dataset to 1000 rows (500 benign + 500 attack).
    print("Benchmark within threshold. Expanding datasets to 1000 rows each...")
    datasets_1000 = expand_datasets_to_1000(DEFAULT_500_DATASETS)

    # Step 3: run full evaluation on 1000-row datasets.
    args.expanded_output_dir.mkdir(parents=True, exist_ok=True)
    expanded_cmd = build_base_cmd(
        backend=args.backend,
        models=args.local_models,
        output_dir=args.expanded_output_dir,
        datasets=datasets_1000,
        limit=None,
        concurrency=args.concurrency,
        store_raw=args.store_raw,
    )

    print("Running expanded benchmark (1000 rows per dataset)...")
    print(" ".join(expanded_cmd))
    code2, elapsed2_sec = run_experiment(expanded_cmd)
    print(f"Expanded exit_code={code2}, elapsed={elapsed2_sec:.2f}s ({elapsed2_sec/60.0:.2f} min)")

    if code2 != 0:
        sys.exit(code2)


if __name__ == "__main__":
    main()
