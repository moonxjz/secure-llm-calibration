#!/usr/bin/env python3
"""
Latency micro-benchmark for the layered prompt-injection defences.

Measures per-request, sequential (non-concurrent) end-to-end latency for four
deployment configurations on the same sampled rows:
  - pure       : Pure LLM classifier prompt only
  - pure_gts   : GenTel-Shield pre-filter + Pure LLM fallback
  - guard      : Guard Prompt (structured JSON output) only
  - guard_gts  : Guard Prompt + GenTel-Shield (the paper's recommended method)

Concurrency is intentionally fixed at 1: the question this benchmark answers is
"how long does a single customer-support request take", not aggregate
throughput, so queueing/parallelism effects are excluded from the timing.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    PURE_SYSTEM_PROMPT,
    build_messages,
    build_openai_client_for_deployment,
    extract_text,
    load_datasets,
    parse_binary_label_with_reason,
    read_guard_prompt,
)

CONFIGS = ["pure", "pure_gts", "guard", "guard_gts"]
DEFAULT_MODELS = ["gpt-4.1-mini", "gpt-4.1"]
DEFAULT_SEEDS = [42, 1337, 2026]


def uses_gentel(config: str) -> bool:
    return config.endswith("_gts")


def system_prompt_for(config: str, guard_prompt_text: str) -> str:
    return guard_prompt_text if config.startswith("guard") else PURE_SYSTEM_PROMPT


def run_one_config(
    model: str,
    config: str,
    dataset: pd.DataFrame,
    guard_prompt_text: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    progress_every: int,
) -> List[Dict[str, Any]]:
    client = build_openai_client_for_deployment(model)
    system_prompt = system_prompt_for(config, guard_prompt_text)
    use_gts = uses_gentel(config)
    gentel_pipeline = None
    if use_gts:
        from gentelshield import pipeline as gentel_pipeline  # noqa: F811 (imported lazily; triggers model load)

    records: List[Dict[str, Any]] = []
    total = len(dataset)
    for i, row in enumerate(dataset.itertuples(index=False), start=1):
        question = str(row.Question)
        t0 = time.perf_counter()

        gts_ms = None
        blocked = False
        if use_gts:
            tg0 = time.perf_counter()
            _, gts_label, _gts_conf = gentel_pipeline(question)
            gts_ms = (time.perf_counter() - tg0) * 1000.0
            blocked = gts_label == 1

        llm_ms = None
        pred = 1 if blocked else 0
        if not blocked:
            tl0 = time.perf_counter()
            messages = build_messages(question, system_prompt)
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            llm_ms = (time.perf_counter() - tl0) * 1000.0
            raw = extract_text(result)
            pred, _reason = parse_binary_label_with_reason(raw)

        total_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "model": model,
                "config": config,
                "seed": seed,
                "id": getattr(row, "id", i),
                "is_attack": int(row.is_attack),
                "blocked_by_gts": blocked,
                "pred": pred,
                "gts_ms": gts_ms,
                "llm_ms": llm_ms,
                "total_ms": total_ms,
            }
        )
        if i % progress_every == 0 or i == total:
            logging.info(
                "progress model=%s config=%s seed=%s %d/%d (last total_ms=%.1f)",
                model, config, seed, i, total, total_ms,
            )
    return records


def summarize(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    rows = []
    for (model, config), group in df.groupby(["model", "config"]):
        totals = group["total_ms"].tolist()
        llm_only = group["llm_ms"].dropna().tolist()
        gts_only = group["gts_ms"].dropna().tolist()
        rows.append(
            {
                "model": model,
                "config": config,
                "n": len(group),
                "blocked_pct": 100.0 * group["blocked_by_gts"].mean(),
                "mean_total_ms": statistics.mean(totals),
                "median_total_ms": statistics.median(totals),
                "p95_total_ms": statistics.quantiles(totals, n=100)[94] if len(totals) >= 20 else max(totals),
                "std_total_ms": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
                "mean_llm_ms": statistics.mean(llm_only) if llm_only else None,
                "mean_gts_ms": statistics.mean(gts_only) if gts_only else None,
            }
        )
    return pd.DataFrame.from_records(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/ATS_Customer_Support_500_balanced.csv"))
    parser.add_argument("--guard-prompt", type=Path, default=Path("prompts/GuardPrompt_calibrated.txt"))
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--configs", type=str, nargs="+", default=CONFIGS, choices=CONFIGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/latency_raw"))
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    guard_prompt_text = read_guard_prompt(args.guard_prompt)

    gts_cold_start_ms = None
    if any(uses_gentel(c) for c in args.configs):
        t0 = time.perf_counter()
        from gentelshield import pipeline as _warmup_pipeline

        _warmup_pipeline("warm-up request to trigger one-time model load")
        gts_cold_start_ms = (time.perf_counter() - t0) * 1000.0
        logging.info("GenTel-Shield cold-start model load: %.1f ms (one-time, excluded from per-request timings)", gts_cold_start_ms)

    all_records: List[Dict[str, Any]] = []
    for seed in args.seeds:
        dataset = load_datasets([args.dataset], args.limit, seed=seed)
        for model in args.models:
            for config in args.configs:
                logging.info("=== starting model=%s config=%s seed=%s (n=%d) ===", model, config, seed, len(dataset))
                records = run_one_config(
                    model=model,
                    config=config,
                    dataset=dataset,
                    guard_prompt_text=guard_prompt_text,
                    seed=seed,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    progress_every=args.progress_every,
                )
                all_records.extend(records)
                # Persist incrementally so partial progress survives interruption.
                pd.DataFrame.from_records(all_records).to_csv(args.output_dir / "latency_raw.csv", index=False)

    raw_path = args.output_dir / "latency_raw.csv"
    pd.DataFrame.from_records(all_records).to_csv(raw_path, index=False)
    logging.info("Saved raw per-request latencies to %s", raw_path)

    summary_df = summarize(all_records)
    summary_path = args.output_dir / "latency_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved summary to %s", summary_path)

    manifest = {
        "dataset": str(args.dataset),
        "guard_prompt": str(args.guard_prompt),
        "models": args.models,
        "configs": args.configs,
        "seeds": args.seeds,
        "limit_per_seed": args.limit,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "gts_cold_start_ms": gts_cold_start_ms,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
