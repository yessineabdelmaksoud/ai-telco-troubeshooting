import os
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ========= CONFIG =========
MODEL_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR   = "models/qwen1.5b-lora-telco-opt-masked"   # ton modèle finetuné
DATA_DIR   = "data_jsonl_opt"                          # contient test.jsonl + test_ids.txt
OUT_CSV    = "submission.csv"
MAX_NEW_TOKENS = 16
# ==========================

LABEL_RE = re.compile(r"\bC([1-8])\b")

def extract_label(text: str) -> str:
    """Extrait C1..C8 depuis la génération."""
    m = LABEL_RE.search(text)
    return f"C{m.group(1)}" if m else "C3"  # fallback (tu peux mettre la classe la plus fréquente)

def main():
    assert os.path.exists(LORA_DIR), f"Missing LORA_DIR: {LORA_DIR}"
    test_jsonl = os.path.join(DATA_DIR, "test.jsonl")
    test_ids   = os.path.join(DATA_DIR, "test_ids.txt")
    assert os.path.exists(test_jsonl), f"Missing: {test_jsonl}"
    assert os.path.exists(test_ids), f"Missing: {test_ids}"

    # quantization
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # tokenizer (use the finetuned folder is OK; fallback to base if needed)
    tok = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # better for generation batching

    # model + lora
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    # read ids
    ids = [l.strip() for l in open(test_ids, encoding="utf-8").read().splitlines() if l.strip()]

    # read prompts (messages)
    prompts = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["messages"])

    assert len(ids) == len(prompts), f"IDs ({len(ids)}) != prompts ({len(prompts)})"

    preds = []
    final_texts = []

    with torch.no_grad():
        for msgs in prompts:
            # build prompt with generation header
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

            gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            c = extract_label(gen)
            preds.append(c)

            # required column text in submission
            final_texts.append(f"Based on the provided data, the most likely root cause is: \\boxed{{{c}}}")

    # build submission exactly like SampleSubmission
    sub = pd.DataFrame({
        "ID": ids,
        "Qwen3-32B": ["placeholder"] * len(ids),
        "Qwen2.5-7B-Instruct": ["placeholder"] * len(ids),
        "Qwen2.5-1.5B-Instruct": final_texts,
    })

    sub.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} with {len(sub)} rows.")
    print("First rows:\n", sub.head(5))

if __name__ == "__main__":
    main()
