import os
import re
import json
import argparse
from collections import Counter

import pandas as pd

# Try sklearn for stratified split (recommended)
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


LABEL_RE = re.compile(r"\bC([1-8])\b")

SYSTEM_PROMPT = (
    "You are a telecom troubleshooting assistant. "
    "From the 8 root causes C1..C8, output ONLY the final choice in the exact format "
    "\\boxed{C#} with no extra text."
)

def extract_label(text: str) -> str:
    """Return label like 'C2' from various answer formats."""
    if text is None:
        return None
    m = LABEL_RE.search(str(text))
    if not m:
        return None
    return f"C{m.group(1)}"

def to_train_record(question: str, label: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"\\boxed{{{label}}}"},
        ]
    }

def to_test_record(question: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    }

def write_jsonl(path: str, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Path to train.csv (ID,question,answer)")
    ap.add_argument("--test_csv", required=True, help="Path to phase_1_test.csv (ID,question)")
    ap.add_argument("--out_dir", required=True, help="Output directory for jsonl files")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    assert {"ID", "question", "answer"}.issubset(train_df.columns), "train.csv must have ID,question,answer"

    # Extract labels
    train_df["label"] = train_df["answer"].apply(extract_label)
    bad = train_df[train_df["label"].isna()]
    if len(bad) > 0:
        print(f"[WARN] {len(bad)} rows have no valid label C1..C8. They will be dropped.")
        train_df = train_df.dropna(subset=["label"]).copy()

    # Basic stats
    dist = Counter(train_df["label"].tolist())
    print("[INFO] Label distribution:", dict(sorted(dist.items())))

    # Stratified split
    if not SKLEARN_OK:
        raise RuntimeError(
            "scikit-learn not found. Install it: pip install scikit-learn\n"
            "Stratified split is important to avoid label imbalance."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_ratio, random_state=args.seed
    )
    y = train_df["label"].values
    idx = train_df.index.values

    train_idx, val_idx = next(splitter.split(idx, y))
    tr = train_df.iloc[train_idx].reset_index(drop=True)
    va = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"[INFO] Train size: {len(tr)} | Val size: {len(va)}")

    # Build jsonl
    train_records = [to_train_record(q, lab) for q, lab in zip(tr["question"], tr["label"])]
    val_records   = [to_train_record(q, lab) for q, lab in zip(va["question"], va["label"])]

    out_train = os.path.join(args.out_dir, "train.jsonl")
    out_val   = os.path.join(args.out_dir, "val.jsonl")
    write_jsonl(out_train, train_records)
    write_jsonl(out_val, val_records)

    # Build test.jsonl (for inference)
    test_df = pd.read_csv(args.test_csv)
    assert {"ID", "question"}.issubset(test_df.columns), "phase_1_test.csv must have ID,question"
    test_records = [to_test_record(q) for q in test_df["question"].tolist()]
    out_test = os.path.join(args.out_dir, "test.jsonl")
    write_jsonl(out_test, test_records)

    # Save ID order for later submission merging
    test_ids_path = os.path.join(args.out_dir, "test_ids.txt")
    with open(test_ids_path, "w", encoding="utf-8") as f:
        for _id in test_df["ID"].tolist():
            f.write(str(_id) + "\n")

    print("[DONE] Wrote:")
    print(" -", out_train)
    print(" -", out_val)
    print(" -", out_test)
    print(" -", test_ids_path)

if __name__ == "__main__":
    main()
