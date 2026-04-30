from datasets import load_dataset

DATASET_NAME = "SALT-NLP/SWE-chat"
CONFIG = "conversations"
SAVE_PATH = "./swe_chat_local"

print("Downloading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    CONFIG,
    split="train",
    token=True
)

print("Saving locally...")
dataset.save_to_disk(SAVE_PATH)

print(f"Saved to {SAVE_PATH}")