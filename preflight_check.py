import json
import os
from transformers import AutoTokenizer

DATA_DIR = "data_jsonl_opt"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 1536

print("🔍 Preflight check...\n")

# 1) Check files
for f in ["train.jsonl", "val.jsonl"]:
    path = os.path.join(DATA_DIR, f)
    if not os.path.exists(path):
        raise SystemExit(f"❌ Missing file: {path}")
    print(f"✅ Found {path}")

# 2) Load one example
sample = json.loads(open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8").readline())
msgs = sample["messages"]

# 3) Check message roles
roles = [m["role"] for m in msgs]
print("Roles:", roles)
assert roles == ["system", "user", "assistant"], "❌ Wrong message structure"

# 4) Check assistant format
assistant = msgs[-1]["content"]
assert "\\boxed{" in assistant, "❌ Assistant answer not boxed"
print("✅ Assistant format OK:", assistant)

# 5) Token length check
tok = AutoTokenizer.from_pretrained(MODEL)
text = tok.apply_chat_template(msgs, tokenize=False)
length = len(tok(text)["input_ids"])
print(f"Token length = {length}")
assert length <= MAX_LEN, "❌ Example exceeds MAX_LEN"

print("\n🎯 All checks PASSED. Safe to train.")
