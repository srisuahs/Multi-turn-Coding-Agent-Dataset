import json
from collections import Counter
from datasets import load_from_disk
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_PATH = "./swe_chat_local"
OUTPUT_FILE = "tool_stats.json"

# =========================
# LOAD
# =========================
print("Loading dataset...")
dataset = load_from_disk(DATA_PATH)
print(f"Total rows: {len(dataset)}\n")

# =========================
# COUNTERS
# =========================
tool_counter = Counter()
command_counter = Counter()

# =========================
# HELPERS
# =========================
def extract_command_prefix(cmd):
    if not isinstance(cmd, str):
        return None
    parts = cmd.strip().split()
    return parts[0] if parts else None


# =========================
# MAIN LOOP
# =========================
print("Scanning tools...\n")

for row in tqdm(dataset, desc="Processing"):
    
    role = row.get("role")
    tool_name = row.get("tool_name")
    command = row.get("command")

    # only care about tool_use
    if role != "tool_use":
        continue

    # primary signal
    if tool_name:
        tool_counter[tool_name] += 1

    # fallback signal
    if command:
        cmd = extract_command_prefix(command)
        if cmd:
            command_counter[cmd] += 1


# =========================
# SORT RESULTS
# =========================
tool_sorted = dict(sorted(tool_counter.items(), key=lambda x: -x[1]))
cmd_sorted = dict(sorted(command_counter.items(), key=lambda x: -x[1]))

# =========================
# PRINT SUMMARY
# =========================
print("\n=== TOOL NAMES ===")
for k, v in list(tool_sorted.items())[:30]:
    print(f"{k}: {v}")

print("\n=== COMMAND PREFIXES ===")
for k, v in list(cmd_sorted.items())[:30]:
    print(f"{k}: {v}")

# =========================
# SAVE
# =========================
output = {
    "tool_names": tool_sorted,
    "command_prefixes": cmd_sorted
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {OUTPUT_FILE}")