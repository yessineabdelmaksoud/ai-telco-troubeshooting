import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "data_jsonl_opt"
OUT_DIR = "models/qwen1.5b-lora-telco-opt"

MAX_LEN = 1536  # couvre 100% (max ~1222)

def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # reduce VRAM
    model.gradient_checkpointing_enable()

    # 4-bit + LoRA
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "val.jsonl"),
    })
    print(f"📊 Train: {len(ds['train'])} | Val: {len(ds['validation'])}")

    def tokenize_batch(batch):
        texts = [
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        out = tok(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        return out

    ds_tok = ds.map(tokenize_batch, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # effective batch = 32
        learning_rate=8e-5,
        num_train_epochs=3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="steps",   
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        weight_decay=0.01,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("✅ Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
