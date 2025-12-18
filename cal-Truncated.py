import json, numpy as np
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
lens=[]
for line in open("data_jsonl_opt/train.jsonl",encoding="utf-8"):
    obj=json.loads(line)
    text=tok.apply_chat_template(obj["messages"], tokenize=False)
    lens.append(len(tok(text)["input_ids"]))
lens=np.array(lens)
MAX_LEN=1536
print("Truncated %:", (lens>MAX_LEN).mean()*100)
print("P95:", np.percentile(lens,95))