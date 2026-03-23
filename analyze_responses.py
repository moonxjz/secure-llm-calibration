import argparse
import re
from collections import Counter

import pandas as pd


def _is_pred_col(name: str) -> bool:
    return "__" in name and not name.endswith(("_raw", "_parse", "_source"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze *_responses.csv for parsing/pathology signals.")
    parser.add_argument("path", type=str, help="Path to responses CSV, e.g. runs/pure_responses.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.path)
    pred_cols = [c for c in df.columns if _is_pred_col(c)]
    if not pred_cols:
        raise SystemExit(f"No prediction columns found in {args.path}")

    print(f"path={args.path}")
    print(f"rows={len(df)}")

    for col in pred_cols:
        mode_model = col
        raw_col = f"{col}_raw"
        parse_col = f"{col}_parse"
        source_col = f"{col}_source"

        pred = pd.to_numeric(df[col], errors="coerce")
        sub = df[pred.notna()].copy()
        if sub.empty:
            continue

        gold = pd.to_numeric(sub["is_attack"], errors="coerce").fillna(0).astype(int)
        pred_i = pd.to_numeric(sub[col], errors="coerce").fillna(0).astype(int)

        tp = int(((pred_i == 1) & (gold == 1)).sum())
        fp = int(((pred_i == 1) & (gold == 0)).sum())
        fn = int(((pred_i == 0) & (gold == 1)).sum())
        tn = int(((pred_i == 0) & (gold == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        print()
        print(f"{mode_model}")
        print(f"  n={len(sub)} tp={tp} fp={fp} fn={fn} tn={tn} precision={prec:.6f} recall={rec:.6f} f1={f1:.6f}")

        if source_col in sub.columns:
            cnt = Counter(sub[source_col].fillna("na").astype(str))
            print(f"  source={dict(cnt)}")

        if parse_col in sub.columns:
            cnt = Counter(sub[parse_col].fillna("na").astype(str))
            print(f"  parse={dict(cnt)}")

        if raw_col in sub.columns:
            raw = sub[raw_col].fillna("").astype(str)
            has_digit = raw.str.contains(r"[01]")
            has_orange = raw.str.contains("ORANGE", case=False, na=False)
            print(
                "  raw_signals="
                + str(
                    {
                        "has_0or1": float(has_digit.mean()),
                        "has_ORANGE": float(has_orange.mean()),
                        "empty": float((raw.str.len() == 0).mean()),
                    }
                )
            )

            attack_raw = raw[gold == 1]
            if len(attack_raw):
                attack_has_digit = attack_raw.str.contains(r"[01]")
                attack_has_orange = attack_raw.str.contains("ORANGE", case=False, na=False)
                print(
                    "  raw_signals_attack="
                    + str(
                        {
                            "has_0or1": float(attack_has_digit.mean()),
                            "has_ORANGE": float(attack_has_orange.mean()),
                            "empty": float((attack_raw.str.len() == 0).mean()),
                        }
                    )
                )

        # quick examples: first few FN (gold=1 pred=0)
        fn_rows = sub[(gold == 1) & (pred_i == 0)].head(5)
        if not fn_rows.empty:
            print("  examples_fn=")
            for _, r in fn_rows.iterrows():
                q = str(r.get("Question", "")).replace("\n", "\\n")
                excerpt = q[:160] + ("..." if len(q) > 160 else "")
                print(f"    id={r.get('id')} cat={r.get('Category')} q={excerpt}")


if __name__ == "__main__":
    main()

