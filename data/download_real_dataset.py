"""Download deepset/prompt-injections and export to project CSV format."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def _find_text_column(df: pd.DataFrame) -> str:
    for candidate in ("text", "prompt", "input", "Question"):
        if candidate in df.columns:
            return candidate
    raise RuntimeError(
        "Could not find a text column. Expected one of: text, prompt, input, Question. "
        f"Available columns: {list(df.columns)}"
    )


def _find_label_column(df: pd.DataFrame) -> str:
    for candidate in ("label", "Label", "is_attack", "target"):
        if candidate in df.columns:
            return candidate
    raise RuntimeError(
        "Could not find a binary label column. Expected one of: label, Label, is_attack, target. "
        f"Available columns: {list(df.columns)}"
    )


def main() -> None:
    ds_dict = load_dataset("deepset/prompt-injections")
    frames = [split_ds.to_pandas() for split_ds in ds_dict.values()]
    df = pd.concat(frames, ignore_index=True)

    text_col = _find_text_column(df)
    label_col = _find_label_column(df)

    df = df.rename(columns={text_col: "Question"})
    df["Answer"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    df["Answer"] = df["Answer"].clip(lower=0, upper=1)
    df["Attack Type"] = df["Answer"].map({0: "benign", 1: "attack"})
    df = df[["Question", "Answer", "Attack Type"]].dropna(subset=["Question"])

    benign = df[df["Attack Type"] == "benign"]
    attack = df[df["Attack Type"] == "attack"]
    benign_replace = len(benign) < 500
    attack_replace = len(attack) < 500
    if benign_replace or attack_replace:
        print(
            "Warning: source has fewer than 500 rows for one or more classes; "
            "sampling with replacement to keep 500/500 balance."
        )

    balanced = pd.concat(
        [
            benign.sample(n=500, random_state=42, replace=benign_replace),
            attack.sample(n=500, random_state=42, replace=attack_replace),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    out = Path("data/Real_PromptInjection_500_balanced.csv")
    balanced.to_csv(out, index=False)
    print(f"Saved {len(balanced)} rows to {out}")


if __name__ == "__main__":
    main()
