import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "data_jsonl_opt"
OUT_DIR = "models/qwen1.5b-lora-telco-opt-masked"

MAX_LEN = 1536

@dataclass
class DataCollatorForCompletionOnly:
    """Collator qui pad correctement input_ids, attention_mask ET labels."""
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for f in features:
            pad_length = max_length - len(f["input_ids"])
            
            batch["input_ids"].append(
                f["input_ids"] + [self.tokenizer.pad_token_id] * pad_length
            )
            batch["attention_mask"].append(
                f["attention_mask"] + [0] * pad_length
            )
            batch["labels"].append(
                f["labels"] + [-100] * pad_length
            )
        
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }

def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print("🔄 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "val.jsonl"),
    })
    print(f"📊 Train: {len(ds['train'])} | Val: {len(ds['validation'])}")

    def build_example(messages):
        """
        Méthode ROBUSTE recommandée:
        1. Tokenize le prompt (system + user) avec add_generation_prompt=True
        2. Tokenize le texte complet (system + user + assistant)
        3. La différence donne EXACTEMENT où commence la réponse assistant
        
        Cette méthode est GARANTIE de fonctionner car elle ne fait pas 
        de recherche de patterns qui peuvent apparaître ailleurs.
        """
        # Messages sans la réponse assistant
        prompt_messages = messages[:-1]
        
        # 1. Prompt avec le header assistant mais sans la réponse
        prompt_text = tok.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True  # Ajoute "<|im_start|>assistant\n"
        )
        
        # 2. Texte complet avec la réponse
        full_text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 3. Tokenize les deux (MÊME contexte, tokenization cohérente)
        prompt_tokens = tok(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        
        full_tokens = tok(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN,
        )
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        
        # 4. Créer les labels
        # Tout avant prompt_len est masqué (-100)
        # Tout après prompt_len est utilisé pour la loss
        prompt_len = len(prompt_tokens)
        
        # S'assurer que prompt_len ne dépasse pas la longueur après troncature
        prompt_len = min(prompt_len, len(input_ids))
        
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Ajuster si nécessaire (troncature)
        labels = labels[:len(input_ids)]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def map_fn(ex):
        return build_example(ex["messages"])

    print("🔄 Tokenizing with ROBUST loss masking...")
    ds_tok = ds.map(
        map_fn,
        remove_columns=ds["train"].column_names,
        desc="Processing",
        num_proc=1,
    )
    
    # Vérification détaillée
    print("\n🔍 VÉRIFICATION DU MASKING:")
    for i in range(min(3, len(ds_tok["train"]))):
        example = ds_tok["train"][i]
        non_masked = sum(1 for x in example["labels"] if x != -100)
        total = len(example["labels"])
        
        print(f"\nExemple {i+1}:")
        print(f"  Length: {total} tokens")
        print(f"  Loss tokens: {non_masked} ({non_masked/total*100:.1f}%)")
        
        # Décoder la partie non-masquée
        response_tokens = [x if x != -100 else tok.pad_token_id 
                          for x in example["labels"]]
        response_text = tok.decode(response_tokens).strip()
        # Garder seulement le texte non-vide
        response_text = response_text.replace(tok.pad_token, "").strip()
        print(f"  Response learned: {response_text[:100]}")
    
    avg_loss_ratio = sum(
        sum(1 for x in ex["labels"] if x != -100) / len(ex["labels"])
        for ex in ds_tok["train"]
    ) / len(ds_tok["train"])
    
    print(f"\n📊 Average loss ratio: {avg_loss_ratio*100:.2f}%")
    print(f"✅ Expected: 1-3% (only assistant response)\n")
    
    if avg_loss_ratio > 0.05:
        print("⚠️  WARNING: Loss ratio > 5%! Check build_example()\n")

    collator = DataCollatorForCompletionOnly(tokenizer=tok)

    # Training arguments avec LR CONSERVATEUR comme recommandé
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,   # effective batch = 32
        
        # 🎯 LR SAFE comme recommandé par ton ami
        learning_rate=5e-5,  # Conservateur pour éviter overfitting
        num_train_epochs=3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        
        fp16=True,
        bf16=False,
        fp16_full_eval=True,
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        
        max_grad_norm=0.3,
        weight_decay=0.01,
        
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        
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

    print("="*60)
    print("🚀 TRAINING WITH ROBUST LOSS MASKING")
    print("="*60)
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"📊 Method: prompt_len (guaranteed correct)")
    print(f"📉 Learning rate: 5e-5 (conservative)")
    print(f"🎯 MAX_LEN: {MAX_LEN} tokens")
    print("="*60 + "\n")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        trainer.train()
        
        trainer.save_model(OUT_DIR)
        tok.save_pretrained(OUT_DIR)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"📁 Model saved to: {OUT_DIR}")
        print(f"📈 Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print("\n💡 Next steps:")
        print("  - Check eval_loss curve for overfitting")
        print("  - If stable and underfit → try LR=8e-5 next run")
        print("  - If overfitting → reduce num_train_epochs to 2")
        print("="*60)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ OUT OF MEMORY!")
            print("💡 Solutions:")
            print("  1. per_device_train_batch_size=2")
            print("  2. gradient_accumulation_steps=16")
            print("  3. MAX_LEN=1280")
        raise

if __name__ == "__main__":
    main()