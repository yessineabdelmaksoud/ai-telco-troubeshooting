import os
import re
import json
import argparse
from collections import Counter
from io import StringIO

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

LABEL_RE = re.compile(r"\bC([1-8])\b")

SYSTEM_PROMPT = (
    "You are a telecom troubleshooting assistant. "
    "From the 8 root causes C1..C8, output ONLY the final choice in the exact format "
    "\\boxed{C#} with no extra text."
)

# --- Short fixed header to reduce tokens ---
SHORT_HEADER = """Analyze 5G drive-test + engineering parameters.
Select the most likely root cause among C1..C8 and output \\boxed{C#} only.

Causes:
C1: downtilt too large (weak far coverage)
C2: overshoot distance >1km
C3: neighbor higher throughput
C4: severe overlap from non-colocated co-frequency neighbors
C5: frequent handovers
C6: PCI mod 30 collision causing interference
C7: speed > 40km/h
C8: avg scheduled RB < 160

Beamwidth mapping:
DEFAULT/SCENARIO_1..5 => 6deg ; SCENARIO_6..11 => 12deg ; >=SCENARIO_12 => 25deg
Default electronic downtilt 255 => 6deg; other values are degrees.
"""

# -------------- Utilities --------------

def extract_label(text: str) -> str | None:
    if text is None:
        return None
    m = LABEL_RE.search(str(text))
    return f"C{m.group(1)}" if m else None

def safe_read_pipe_table(table_text: str) -> pd.DataFrame:
    """Read a '|' separated table block into DataFrame."""
    lines = [ln.strip() for ln in table_text.strip().splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    # Ensure header exists
    return pd.read_csv(
        pd.compat.StringIO("\n".join(lines)),
        sep="|",
        engine="python",
        dtype=str
    )

def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find a column whose name contains all keyword substrings (case-insensitive)."""
    cols = list(df.columns)
    for c in cols:
        low = c.lower()
        ok = True
        for kw in keywords:
            if kw.lower() not in low:
                ok = False
                break
        if ok:
            return c
    return None

def to_float_series(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def round_col(df: pd.DataFrame, col: str, nd=1):
    if col in df.columns:
        df[col] = to_float_series(df[col]).round(nd)

def compact_drive_table(question: str, max_rows: int = 10) -> tuple[str, set[str]]:
    """
    Returns compacted drive table text and a set of PCIs to keep for engineering.
    """
    # Split blocks
    if "Engeneering parameters data as follows" in question:
        q_main, _ = question.split("Engeneering parameters data as follows", 1)
    else:
        q_main = question

    # Extract drive-test block: from "Timestamp|" header to before engineering
    m = re.search(r"Timestamp\|.*", q_main)
    if not m:
        return "DriveTest: (not found)\n", set()

    drive_block = q_main[m.start():].strip()

    # Cut at first double newline if any extra text
    # (Sometimes there are multiple blank lines before engineering)
    # We'll keep all lines that look like table rows
    # Parse as pipe table
    try:
        df = pd.read_csv(
            StringIO(drive_block), sep="|", engine="python"
        )
    except Exception:
        # fallback: keep raw limited lines
        raw_lines = drive_block.splitlines()[:(max_rows + 2)]
        return "DriveTest(raw):\n" + "\n".join(raw_lines) + "\n", set()

    # Identify columns
    c_ts   = find_col(df, ["timestamp"]) or "Timestamp"
    c_spd  = find_col(df, ["gps", "speed"])
    c_pci  = find_col(df, ["serving", "pci"])
    c_rsrp = find_col(df, ["ss-rsrp"]) or find_col(df, ["rsrp"])
    c_sinr = find_col(df, ["ss-sinr"]) or find_col(df, ["sinr"])
    c_thr  = find_col(df, ["throughput"])
    c_rb   = find_col(df, ["rb"])

    # Neighbor PCIs + RSRPs (Top1/Top2)
    n1_pci  = find_col(df, ["top 1 pci"]) or find_col(df, ["top 1", "pci"])
    n2_pci  = find_col(df, ["top 2 pci"]) or find_col(df, ["top 2", "pci"])
    n1_rsrp = find_col(df, ["top 1", "brsrp"]) or find_col(df, ["top 1", "rsrp"])
    n2_rsrp = find_col(df, ["top 2", "brsrp"]) or find_col(df, ["top 2", "rsrp"])

    # Keep only useful columns
    keep = [c for c in [c_ts, c_spd, c_pci, c_rsrp, c_sinr, c_thr, c_rb, n1_pci, n1_rsrp, n2_pci, n2_rsrp] if c and c in df.columns]
    df = df[keep].copy()

    # Drop Lon/Lat if present (already not kept)
    # Numeric rounding
    for col, nd in [(c_spd, 0), (c_rsrp, 1), (c_sinr, 1), (c_thr, 1), (c_rb, 0), (n1_rsrp, 1), (n2_rsrp, 1)]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(nd)

    # Choose rows: all THR < 600 plus neighbor context rows
    pcis_to_keep = set()

    if c_thr and c_thr in df.columns:
        thr = pd.to_numeric(df[c_thr], errors="coerce")
        bad_idx = thr[thr < 600].index.tolist()
    else:
        bad_idx = []

    chosen = set()
    for i in bad_idx:
        chosen.add(i)
        if i - 1 in df.index: chosen.add(i - 1)
        if i + 1 in df.index: chosen.add(i + 1)

    if not chosen:
        chosen = set(df.index[:min(max_rows, len(df))].tolist())

    chosen = sorted(list(chosen))[:max_rows]
    df = df.loc[chosen].copy()

    # Rename columns to short names
    rename = {}
    if c_ts:   rename[c_ts] = "TS"
    if c_spd:  rename[c_spd] = "SPD"
    if c_pci:  rename[c_pci] = "PCI"
    if c_rsrp: rename[c_rsrp] = "RSRP"
    if c_sinr: rename[c_sinr] = "SINR"
    if c_thr:  rename[c_thr] = "THR"
    if c_rb:   rename[c_rb] = "RB"
    if n1_pci: rename[n1_pci] = "N1_PCI"
    if n1_rsrp:rename[n1_rsrp] = "N1_RSRP"
    if n2_pci: rename[n2_pci] = "N2_PCI"
    if n2_rsrp:rename[n2_rsrp] = "N2_RSRP"

    df = df.rename(columns=rename)

    # Collect PCIs
    for k in ["PCI", "N1_PCI", "N2_PCI"]:
        if k in df.columns:
            for v in df[k].astype(str).tolist():
                v = v.strip()
                if v != "nan" and v != "-" and v != "":
                    pcis_to_keep.add(v)

    # Convert to a compact pipe table
    table_text = df.to_csv(sep="|", index=False)
    return "DriveTest:\n" + table_text.strip() + "\n", pcis_to_keep


def compact_eng_table(question: str, pcis_to_keep: set[str], max_rows: int = 12) -> str:
    if "Engeneering parameters data as follows" not in question:
        return "Engineering: (not found)\n"

    _, tail = question.split("Engeneering parameters data as follows", 1)
    # Find header line "gNodeB ID|Cell ID|..."
    m = re.search(r"gNodeB ID\|.*", tail)
    if not m:
        return "Engineering: (parse error)\n"
    eng_block = tail[m.start():].strip()

    try:
        df = pd.read_csv(StringIO(eng_block), sep="|", engine="python", dtype=str)
    except Exception:
        raw_lines = eng_block.splitlines()[:(max_rows + 2)]
        return "Engineering(raw):\n" + "\n".join(raw_lines) + "\n"

    # keep useful columns
    # Identify PCI col
    c_pci = find_col(df, ["pci"]) or "PCI"
    keep_cols = []
    for name in ["gNodeB ID", "Cell ID", "Mechanical Azimuth", "Mechanical Downtilt",
                 "Digital Tilt", "Digital Azimuth", "Beam Scenario", "Height", "PCI",
                 "TxRx Mode", "Max Transmit Power", "Antenna Model"]:
        if name in df.columns:
            keep_cols.append(name)

    if keep_cols:
        df = df[keep_cols].copy()

    # Filter by pcis_to_keep (if possible)
    if c_pci in df.columns and pcis_to_keep:
        df = df[df[c_pci].astype(str).isin(list(pcis_to_keep))].copy()

    # Limit rows
    df = df.head(max_rows)

    # Shorten column names
    ren = {}
    for c in df.columns:
        cc = c
        cc = cc.replace("Mechanical ", "M_")
        cc = cc.replace("Digital ", "D_")
        cc = cc.replace("Max Transmit Power", "PWR")
        cc = cc.replace("Beam Scenario", "BEAM")
        cc = cc.replace("Antenna Model", "ANT")
        cc = cc.replace("TxRx Mode", "MIMO")
        ren[c] = cc
    df = df.rename(columns=ren)

    return "Engineering:\n" + df.to_csv(sep="|", index=False).strip() + "\n"


def compress_question(question: str) -> str:
    drive_text, pcis = compact_drive_table(question, max_rows=10)
    eng_text = compact_eng_table(question, pcis_to_keep=pcis, max_rows=12)
    return SHORT_HEADER + "\n" + drive_text + "\n" + eng_text


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
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    assert {"ID", "question", "answer"}.issubset(train_df.columns)

    train_df["label"] = train_df["answer"].apply(extract_label)
    bad = train_df[train_df["label"].isna()]
    if len(bad) > 0:
        print(f"[WARN] {len(bad)} rows missing labels -> dropped.")
        train_df = train_df.dropna(subset=["label"]).copy()

    # Compress questions
    print("[INFO] Compressing train questions...")
    train_df["q_comp"] = train_df["question"].apply(compress_question)

    # Stratified split
    y = train_df["label"].values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    idx = train_df.index.values
    train_idx, val_idx = next(splitter.split(idx, y))

    tr = train_df.iloc[train_idx].reset_index(drop=True)
    va = train_df.iloc[val_idx].reset_index(drop=True)

    print("[INFO] Label distribution:", dict(sorted(Counter(train_df["label"]).items())))
    print(f"[INFO] Train size: {len(tr)} | Val size: {len(va)}")

    train_records = [to_train_record(q, lab) for q, lab in zip(tr["q_comp"], tr["label"])]
    val_records   = [to_train_record(q, lab) for q, lab in zip(va["q_comp"], va["label"])]

    out_train = os.path.join(args.out_dir, "train.jsonl")
    out_val   = os.path.join(args.out_dir, "val.jsonl")
    write_jsonl(out_train, train_records)
    write_jsonl(out_val, val_records)

    # Test
    test_df = pd.read_csv(args.test_csv)
    assert {"ID", "question"}.issubset(test_df.columns)

    print("[INFO] Compressing test questions...")
    test_df["q_comp"] = test_df["question"].apply(compress_question)

    test_records = [to_test_record(q) for q in test_df["q_comp"].tolist()]
    out_test = os.path.join(args.out_dir, "test.jsonl")
    write_jsonl(out_test, test_records)

    # IDs order
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
