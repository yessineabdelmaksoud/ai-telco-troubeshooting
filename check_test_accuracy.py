import re
import pandas as pd
from collections import Counter

SUBMISSION_CSV = "submission.csv"
TRUTH_CSV = "data_orig/phase_1_test_truth.csv"

# extract C1..C8 from any text
LABEL_RE = re.compile(r"\bC([1-8])\b")

def extract_label(x: str):
    m = LABEL_RE.search(str(x))
    return f"C{m.group(1)}" if m else None

def base_id(full_id: str) -> str:
    """
    ID_XLWWVM40IW_1  -> ID_XLWWVM40IW
    ID_3WB8KX32W3_4 -> ID_3WB8KX32W3
    """
    s = str(full_id).strip()
    return re.sub(r"_(\d+)$", "", s)

def majority_vote(labels):
    c = Counter(labels)
    return c.most_common(1)[0][0]

def main():
    sub = pd.read_csv(SUBMISSION_CSV)
    truth = pd.read_csv(TRUTH_CSV)

    # submission: parse predicted label from the 1.5B column text
    sub["pred"] = sub["Qwen2.5-1.5B-Instruct"].apply(extract_label)
    sub["ID_base"] = sub["ID"].apply(base_id)

    # truth: use the 1.5B gold column, normalize IDs to base
    truth = truth[["ID", "Qwen2.5-1.5B-Instruct"]].rename(columns={"Qwen2.5-1.5B-Instruct": "gold"})
    truth["gold"] = truth["gold"].astype(str).str.strip()
    truth["ID_base"] = truth["ID"].apply(base_id)

    # aggregate gold by base ID (majority across _1.._4)
    gold_base = truth.groupby("ID_base")["gold"].apply(list).reset_index()
    gold_base["gold_maj"] = gold_base["gold"].apply(majority_vote)

    # join
    merged = sub.merge(gold_base[["ID_base", "gold_maj"]], on="ID_base", how="inner")

    if len(merged) == 0:
        raise SystemExit("❌ No overlap. Check SUBMISSION_CSV path and ID formats.")

    merged["ok"] = merged["pred"] == merged["gold_maj"]
    acc = merged["ok"].mean()

    print(f"Matched base IDs: {len(merged)}")
    print(f"Accuracy (base-ID majority vote): {acc*100:.2f}%")

    # show some errors
    errs = merged[~merged["ok"]].head(15)[["ID", "pred", "gold_maj"]]
    if len(errs):
        print("\nExamples of mistakes:")
        print(errs.to_string(index=False))

    # optional: show distributions
    print("\nPred distribution:", Counter(merged["pred"]))
    print("Gold distribution:", Counter(merged["gold_maj"]))

if __name__ == "__main__":
    main()
