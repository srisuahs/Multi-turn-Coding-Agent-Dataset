import json
from datasets import load_from_disk
from tqdm import tqdm
import os
import re

# =========================
# CONFIG
# =========================
DATA_PATH = "./swe_chat_local"
OUTPUT_FILE = "agent_dataset_clean.jsonl"

MAX_STEPS = 12
MAX_OUTPUT_CHARS = 80000
MIN_STEPS = 3


# =========================
# TOOL MAP
# =========================
TOOL_MAP = {
    "Read": "read_files",
    "read": "read_files",

    "Grep": "search_files",
    "grep": "search_files",
    "Glob": "search_files",
    "glob": "search_files",

    "Edit": "modify_files",
    "apply_patch": "modify_files",

    "Write": "write_files",

    "Bash": "run_command",
    "bash": "run_command",

    "TaskUpdate": "manage_task",
    "TaskCreate": "manage_task",
    "Task": "manage_task",
    "TodoWrite": "manage_task",
    "TaskList": "manage_task",
}


# =========================
# UTIL
# =========================

def map_tool(name):
    return TOOL_MAP.get(name, "run_command")


def parse_json_safe(x):
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except:
        return {"raw": x}


def clean_output(text):
    if not isinstance(text, str):
        return text

    # remove system reminders
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)

    # truncate
    return text[:MAX_OUTPUT_CHARS]


# =========================
# SUCCESS DETECTION
# =========================

def is_success(step):
    if step["t"] != "run_command":
        return False

    out = step["o"]

    if not isinstance(out, str):
        return False

    out_lower = out.lower()

    # test success
    if "ok" in out_lower and "fail" not in out_lower:
        return True

    # lint success
    if "0 issues" in out_lower:
        return True

    return False


# =========================
# GOAL EXTRACTION
# =========================

def extract_goal(steps):
    for s in steps[:3]:
        args = s["a"]

        file_path = args.get("file_path") or args.get("path")
        if file_path:
            name = os.path.basename(file_path)

            if "test" in name.lower():
                return f"fix failing test in {name}"
            return f"modify or inspect {name}"

        pattern = args.get("pattern")
        if pattern:
            return f"search for '{pattern}' in codebase"

    return "perform codebase modification"


# =========================
# SESSION ITERATOR
# =========================

def session_iterator(dataset):
    current_sid = None
    buffer = []

    for row in dataset:
        sid = row["session_id"]

        if current_sid is None:
            current_sid = sid

        if sid != current_sid:
            yield buffer
            buffer = []
            current_sid = sid

        buffer.append(row)

    if buffer:
        yield buffer


# =========================
# EXTRACT + FILTER
# =========================

def extract_session(rows):
    steps = []
    i = 0
    n = len(rows)

    while i < n:
        row = rows[i]

        if row.get("role") == "tool_use":
            tool = map_tool(row.get("tool_name"))
            args = parse_json_safe(row.get("tool_input_json"))

            # find tool_result
            j = i + 1
            result = None

            while j < n:
                if rows[j].get("role") == "tool_result":
                    result = rows[j].get("content", "")
                    break
                j += 1

            if result:
                result = clean_output(result)

                steps.append({
                    "t": tool,
                    "a": args,
                    "o": result
                })

                i = j
            else:
                i += 1
        else:
            i += 1

    if len(steps) < MIN_STEPS:
        return None

    # =========================
    # TRIM AFTER SUCCESS
    # =========================
    trimmed = []
    for step in steps:
        trimmed.append(step)
        if is_success(step):
            break

    steps = trimmed

    # =========================
    # REMOVE REDUNDANT run_command
    # =========================
    deduped = []
    last_cmd = None

    for s in steps:
        if s["t"] == "run_command":
            cmd = s["a"].get("command", "")

            if cmd == last_cmd:
                continue

            last_cmd = cmd
        else:
            last_cmd = None

        deduped.append(s)

    steps = deduped

    # =========================
    # CAP LENGTH
    # =========================
    if len(steps) > MAX_STEPS:
        steps = steps[:MAX_STEPS]

    # =========================
    # QUALITY SCORING
    # =========================
    score = score_session(steps)

    if score < 0.4:
        return None

    goal = extract_goal(steps)

    return {
        "goal": goal,
        "m": steps,
        "end": 1
    }


# =========================
# SCORING FUNCTION
# =========================

def score_session(steps):
    score = 0.0

    tools = [s["t"] for s in steps]

    # diversity
    unique_tools = len(set(tools))
    score += min(unique_tools / 4, 1.0) * 0.3

    # has modify
    if "modify_files" in tools:
        score += 0.3

    # has read before modify
    for i in range(1, len(steps)):
        if steps[i]["t"] == "modify_files" and steps[i-1]["t"] == "read_files":
            score += 0.2
            break

    # has validation
    if any(s["t"] == "run_command" for s in steps):
        score += 0.2

    return score


# =========================
# MAIN
# =========================

def main():
    print("Loading dataset...")
    dataset = load_from_disk(DATA_PATH)

    total = 0
    kept = 0

    print("Processing...")

    with open(OUTPUT_FILE, "w") as f:
        for session in tqdm(session_iterator(dataset), desc="Sessions"):
            total += 1

            session = sorted(session, key=lambda x: x["turn_number"])

            data = extract_session(session)

            if data:
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
                kept += 1

    print("\n=== DONE ===")
    print(f"Total sessions: {total}")
    print(f"Kept sessions: {kept}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()