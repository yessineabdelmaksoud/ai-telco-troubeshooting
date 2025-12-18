import json, re, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

LABEL_RE = re.compile(r"\bC([1-8])\b")
def extract_label(s):
    m = LABEL_RE.search(str(s))
    return f"C{m.group(1)}" if m else None

MODEL_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "models/qwen1.5b-lora-telco-opt-masked"
VAL_PATH = "data_jsonl_opt/val.jsonl"

N = 200

def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tok = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    lines = open(VAL_PATH, encoding="utf-8").read().splitlines()
    random.shuffle(lines)
    lines = lines[:N]

    correct = 0
    format_ok = 0

    for line in lines:
        obj = json.loads(line)
        msgs = obj["messages"]
        gold = extract_label(msgs[-1]["content"])

        prompt_msgs = msgs[:-1]
        text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = extract_label(gen)

        if pred is not None:
            format_ok += 1
        if pred == gold:
            correct += 1
        print(f"Real: {gold} | Pred: {pred}")

    print(f"Samples: {N}")
    print(f"Format OK: {format_ok}/{N} = {format_ok/N:.2%}")
    print(f"Accuracy : {correct}/{N} = {correct/N:.2%}")

if __name__ == "__main__":
    main()
