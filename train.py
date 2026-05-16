# train_agent_lora.py

import json
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

DATASET_PATH = "dataset_v8.jsonl"

OUTPUT_DIR = "./agent_lora"

MAX_SEQ_LENGTH = 2048

# =====================================================
# TRAINING
# =====================================================

EPOCHS = 3

LEARNING_RATE = 2e-4

BATCH_SIZE = 1

GRAD_ACCUM = 16

WARMUP_RATIO = 0.05

WEIGHT_DECAY = 0.01

LOGGING_STEPS = 10

SAVE_STEPS = 200

# =====================================================
# LORA
# =====================================================

LORA_R = 32

LORA_ALPHA = 64

LORA_DROPOUT = 0.05

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = (
    "You are a coding agent that outputs JSON workflows."
)

# =====================================================
# 4BIT CONFIG
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# =====================================================
# TOKENIZER
# =====================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================================================
# MODEL
# =====================================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()

# =====================================================
# PREPARE FOR QLORA
# =====================================================

model = prepare_model_for_kbit_training(model)

# =====================================================
# TARGET MODULES
# =====================================================

target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# =====================================================
# LORA CONFIG
# =====================================================

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# =====================================================
# APPLY LORA
# =====================================================

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# =====================================================
# FORMAT DATASET
# =====================================================

def format_sample(sample):

    goal = sample.get("goal", "modify codebase")

    assistant_json = json.dumps(
        sample,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    text = (
        f"<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{goal}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{assistant_json}<|im_end|>"
    )

    return {"text": text}

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

samples = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:
            obj = json.loads(line)

            if (
                isinstance(obj, dict)
                and "goal" in obj
                and "m" in obj
                and isinstance(obj["m"], list)
                and len(obj["m"]) >= 3
            ):
                samples.append(obj)

        except Exception as e:
            print("Skipping malformed sample:", e)

print(f"Loaded {len(samples)} samples")

# =====================================================
# FORMAT
# =====================================================

formatted = []

for x in samples:

    text = format_sample(x)["text"]

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    truncated_text = tokenizer.decode(tokens["input_ids"])

    formatted.append({
        "text": truncated_text
    })

dataset = Dataset.from_list(formatted)

# =====================================================
# TRAINING CONFIG
# =====================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LEARNING_RATE,

    lr_scheduler_type="cosine",

    warmup_ratio=WARMUP_RATIO,

    weight_decay=WEIGHT_DECAY,

    logging_steps=LOGGING_STEPS,

    save_steps=SAVE_STEPS,

    bf16=True,

    optim="paged_adamw_8bit",

    report_to="none",

    save_total_limit=2,

    gradient_checkpointing=True,

    max_grad_norm=0.3,

    packing=False,
)

# =====================================================
# TRAINER
# =====================================================

trainer = SFTTrainer(
    model=model,

    train_dataset=dataset,

    args=training_args,

    processing_class=tokenizer,

    formatting_func=lambda x: x["text"],
)

# =====================================================
# TRAIN
# =====================================================

print("Starting training...")

trainer.train()

# =====================================================
# SAVE
# =====================================================

print("Saving adapter...")

trainer.model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")