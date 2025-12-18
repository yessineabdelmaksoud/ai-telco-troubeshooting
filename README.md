#  AI Telco Troubleshooting – Qwen 1.5B Fine-Tuning

##  Overview
This project addresses the **AI Telco Troubleshooting Challenge**, where the goal is to automatically identify the **root cause (C1–C8)** of throughput degradation in **5G drive-test scenarios**, using both **user plane measurements** and **engineering parameters**.

We fine-tune **Qwen/Qwen2.5-1.5B-Instruct** with **LoRA + 4-bit quantization** to build a **robust, domain-specialized telecom diagnostic model**, capable of reasoning over semi-structured technical data and outputting a precise classification.

---

## Challenge Description
Given:
- 5G drive-test data (RSRP, SINR, throughput, RBs, speed, neighbors)
- gNodeB engineering parameters (tilt, beam scenario, PCI, height, etc.)

The task is to select **one root cause among 8 predefined classes (C1–C8)** explaining why throughput drops below 600 Mbps.

Output format is strictly:
\boxed{C#}

---

##  Main Challenges
1. **Very long inputs** (raw prompts > 2500 tokens initially)
2. **Semi-structured tabular data embedded in text**
3. **High risk of model collapse** (predicting only one class)
4. **Small dataset** (~2,400 samples)
5. **Limited hardware** (RTX 4050 Laptop – 6 GB VRAM)
6. **Multi-ID evaluation** (each scenario split into `_1.._4` samples)

---

##  Actions & Design Choices

### 1️ Data Optimization
- Removed redundant text and columns
- Renamed long feature names (e.g. `SS-RSRP [dBm] → RSRP`)
- Kept only **high-signal features**
- Reduced prompt length from ~2400 → **~900 tokens**

➡️ Result: Faster training, larger effective batch size, better stability.

---

### 2️ Robust Supervised Fine-Tuning (SFT)
We used **completion-only loss masking**, meaning:
- ❌ No loss on system/user prompt
- ✅ Loss applied **only on assistant output** (`\boxed{C#}`)

This avoids:
- Prompt memorization
- Label leakage
- Overfitting to input structure

Loss masking is done by computing:
prompt_len = len(tokenized_prompt)
labels = [-100]*prompt_len + input_ids[prompt_len:]

---

### 3️ Model & Training Setup

**Base model**
Qwen/Qwen2.5-1.5B-Instruct

**Fine-tuning strategy**
- LoRA (Parameter-efficient tuning)
- 4-bit quantization (NF4)
- Gradient checkpointing

**Trainable parameters**
~18.4M / 1.56B (≈ 1.18%)

---

##  Hyperparameters

| Parameter | Value | Rationale |
|---------|------|-----------|
| MAX_LEN | 1536 | Covers 100% of samples |
| Batch size | 4 | Fits GPU memory |
| Grad Accumulation | 8 | Effective batch = 32 |
| Learning rate | **5e-5** | Conservative, avoids overfitting |
| Epochs | 3 | Converges without degradation |
| Optimizer | paged_adamw_8bit | Memory efficient |
| Scheduler | cosine | Smooth decay |
| Warmup | 5% | Stabilizes early steps |
| Precision | FP16 | RTX 40xx optimized |

---

##  Results

###  Offline Evaluation (Base-ID Majority Vote)
- **Accuracy: 53.82%**
- **Random baseline: 12.5%**
- **No class collapse**
- Predictions spread across all C1–C8

**Class distribution**
- Gold: perfectly balanced
- Predictions: realistic bias (C4/C7/C8 slightly dominant)

 This confirms **real learning of telecom patterns**, not memorization.

---

##  Error Analysis
Most errors occur between **conceptually close causes**:

| Confusion | Telecom Interpretation |
|---------|------------------------|
| C4 ↔ C8 | Interference vs low RB scheduling |
| C7 ↔ C8 | High speed impacts scheduling |
| C1 ↔ C8 | Weak coverage reduces effective RBs |

These confusions are **reasonable even for human engineers**.

---

## Conclusion
- ✔️ Stable training
- ✔️ No collapse
- ✔️ Strong signal over noise
- ✔️ Domain adaptation successful
- ✔️ Production-ready inference pipeline

This is a **solid and credible telecom AI solution**, especially given the hardware and data constraints.

---

## Possible Improvements

### Short-term (no retraining)
- Rule-based post-processing (RB < 160 → boost C8)
- Self-consistency (multiple generations + vote)

### With retraining
- Add short natural-language justification before `\boxed{C#}`
- Oversample hard classes (C1/C2)
- Slightly higher LR (6e-5 / 7e-5)
- One extra epoch on misclassified samples

---

## Repository Structure
├── data_orig/
├── data_jsonl_opt/
├── models/
├── train_lora_corrected.py
├── make_submission.py
├── check_submission_accuracy.py
├── submission.csv
└── README.md