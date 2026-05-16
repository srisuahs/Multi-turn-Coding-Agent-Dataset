import json
import torch
from typing import Any, Dict, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# =========================================================
# CONFIG
# =========================================================

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_PATH = "./agent_lora"
USE_ADAPTER = False

# =========================================================
# MODEL LOAD
# =========================================================

def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb,
        trust_remote_code=True,
    )

    if USE_ADAPTER:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()

    model.eval()

    print("Model ready ✔")
    return model, tokenizer


# =========================================================
# SYSTEM PROMPT (kept same)
# =========================================================

SYSTEM_PROMPT = """
You are a strict tool-using JSON agent.

Return ONLY valid JSON.

Schema:
{
  "goal": string,
  "m": [
    {
      "t": "read_files | search_files | write_files | modify_files | run_command",
      "a": object,
      "o": string
    }
  ],
  "end": 1
}

Rules:
- Output ONLY JSON
- MUST include 3 to 6 steps
- MUST fully close JSON
"""


# =========================================================
# JSON PARSER (ROBUST)
# =========================================================

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip().replace("```json", "").replace("```", "")

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except:
                    return None
    return None


# =========================================================
# PROMPT
# =========================================================

def build_prompt(task: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]


# =========================================================
# GENERATION (FIXED)
# =========================================================

def generate(model, tokenizer, task):

    messages = build_prompt(task)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,

            #  FIX #1: BIG TOKEN LIMIT (THIS WAS YOUR MAIN BUG)
            max_new_tokens=10000,

            #  FIX #2: stable generation
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,

            #  FIX #3: proper stopping
            eos_token_id=tokenizer.eos_token_id,
        )

    # SAFE decoding (same as your working version)
    generated = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print("\nRAW OUTPUT:\n", generated[:2000])

    return extract_json(generated)


# =========================================================
# RUN
# =========================================================

def run(task):

    model, tokenizer = load_model()

    print("\nITERATION 1")
    print("Generating...")

    result = generate(model, tokenizer, task)

    if not result:
        print("FAILED: JSON TRUNCATED OR INVALID")
        return

    print("SUCCESS")

    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run("Create a small CLI todo app in Python")