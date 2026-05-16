import json
import re
from tqdm import tqdm
from datasets import load_from_disk

DATA_PATH = "./swe_chat_local"
OUTPUT_PATH = "dataset_v8.jsonl"

MAX_STEPS = 20
MAX_OUTPUT_LINES = 40

KEYWORDS_KEEP = [
    "error",
    "fail",
    "exception",
    "0 issues",
    "ok",
    "passed",
    "success",
    "panic",
    "undefined",
    "warning",
    "traceback",
]

# =========================================================
# TOOL MAP
# keep ONLY the original 5 tools
# =========================================================

TOOL_MAP = {
    "Read": "read_files",
    "read": "read_files",

    "Grep": "search_files",
    "grep": "search_files",
    "Glob": "search_files",

    "Edit": "modify_files",
    "apply_patch": "modify_files",

    "Write": "write_files",

    "Bash": "run_command",
    "bash": "run_command",
}


VALID_TOOLS = {
    "read_files",
    "search_files",
    "modify_files",
    "write_files",
    "run_command",
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_paths(text):
    if not isinstance(text, str):
        return ""

    # normalize /Users/name/
    text = re.sub(
        r"/Users/[^/\s]+",
        "/workspace",
        text
    )

    # normalize /home/name/
    text = re.sub(
        r"/home/[^/\s]+",
        "/workspace",
        text
    )

    # normalize repo paths
    text = re.sub(
        r"/workspace/[^/\s]+",
        "/workspace/repo",
        text
    )

    return text


def clean_output(text):
    if not isinstance(text, str):
        return ""

    text = normalize_paths(text)

    # remove system reminders
    text = re.sub(
        r"<system-reminder>.*?</system-reminder>",
        "",
        text,
        flags=re.DOTALL
    )

    # normalize tool errors
    if "File does not exist" in text:
        return "file not found"

    text = text.strip()

    if len(text) < 15:
        return ""

    # line aware truncation
    lines = text.splitlines()

    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES]

    return "\n".join(lines)


# =========================================================
# CANONICALIZATION
# =========================================================

def canonicalize_modify_output(step):
    if step["t"] not in ["modify_files", "write_files"]:
        return step

    path = step["a"].get("file_path", "unknown_file")

    path = normalize_paths(path)

    step["o"] = f"Modified file: {path}"

    return step


def canonicalize_run_output(step):
    if step["t"] != "run_command":
        return step

    out = step["o"]

    if not out:
        return step

    out_lower = out.lower()

    # successful lint/build
    if "0 issues" in out_lower:
        step["o"] = "lint passed"

    # successful tests/build
    elif (
        re.search(r"\b(ok|passed|success)\b", out_lower)
        and "fail" not in out_lower
        and "error" not in out_lower
    ):
        step["o"] = "tests passed"

    # errors/failures
    elif (
        "error" in out_lower or
        "fail" in out_lower or
        "panic" in out_lower or
        "undefined" in out_lower
    ):
        lines = out.splitlines()

        useful = []

        for line in lines:
            l = line.lower()

            if any([
                "error" in l,
                "fail" in l,
                "undefined" in l,
                "panic" in l,
                "warning" in l,
                "traceback" in l,
                ".go:" in l,
                ".py:" in l,
                ".ts:" in l,
                ".js:" in l,
                ".rs:" in l,
            ]):
                useful.append(line)

        step["o"] = "\n".join(useful[:10])

    else:
        step["o"] = out[:400]

    return step


# =========================================================
# HELPERS
# =========================================================

def parse_json_safe(x):
    if isinstance(x, dict):
        return x

    try:
        return json.loads(x)
    except:
        return {}


def map_tool(name):
    tool = TOOL_MAP.get(name)

    if tool not in VALID_TOOLS:
        return None

    return tool


def has_signal(text):
    text = text.lower()

    return any(k in text for k in KEYWORDS_KEEP)


# =========================================================
# SESSION GROUPING
# =========================================================

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


# =========================================================
# EXTRACTION
# =========================================================

def extract_steps(rows):
    steps = []

    i = 0
    n = len(rows)

    while i < n:
        r = rows[i]

        if r.get("role") == "tool_use":

            tool = map_tool(r.get("tool_name"))

            # skip unknown tools
            if tool is None:
                i += 1
                continue

            args = parse_json_safe(
                r.get("tool_input_json")
            )

            # skip reviewer/subagents
            if args.get("subagent_type"):
                i += 1
                continue

            args = {
                k: normalize_paths(str(v))
                if isinstance(v, str)
                else v
                for k, v in args.items()
            }

            result = ""

            j = i + 1

            while j < n:
                rr = rows[j]

                if rr.get("role") == "tool_result":
                    result = rr.get("content", "")
                    break

                j += 1

            step = {
                "t": tool,
                "a": args,
                "o": clean_output(result)
            }

            step = canonicalize_modify_output(step)
            step = canonicalize_run_output(step)

            steps.append(step)

            i = j

        else:
            i += 1

    return steps


# =========================================================
# COMPRESSION
# =========================================================

def dedupe_run_commands(steps):
    result = []

    last_command = None

    for s in steps:

        if s["t"] == "run_command":

            cmd = s["a"].get("command", "")

            if cmd == last_command:
                continue

            last_command = cmd

        else:
            last_command = None

        result.append(s)

    return result


def dedupe_reads(steps):
    result = []

    seen = set()

    for s in steps:

        if s["t"] == "read_files":

            path = s["a"].get("file_path", "")

            offset = s["a"].get("offset", 0)

            key = (path, offset)

            if key in seen:
                continue

            seen.add(key)

        result.append(s)

    return result


def trim_run_commands(steps):
    result = []

    for s in steps:

        if s["t"] == "run_command":

            cmd = s["a"].get("command", "").lower()

            # always keep verification commands
            if any(x in cmd for x in [
                "test",
                "lint",
                "pytest",
                "go test",
                "cargo test",
                "npm test",
                "build",
            ]):
                result.append(s)
                continue

            # keep only informative outputs
            if not has_signal(s["o"]):
                continue

        result.append(s)

    return result


# =========================================================
# GOAL
# =========================================================

def build_goal(steps):

    for s in steps:

        if s["t"] == "run_command":

            cmd = s["a"].get("command", "").lower()

            out = s["o"].lower()

            if "lint" in cmd or "lint" in out:
                return "fix lint errors"

            if "test" in cmd or "test" in out:
                return "fix failing tests"

            if (
                "error" in out or
                "fail" in out or
                "panic" in out
            ):
                return "fix build errors"

    for s in steps:

        if s["t"] in ["modify_files", "write_files"]:

            path = s["a"].get("file_path", "")

            if path:
                return f"modify {path.split('/')[-1]}"

    return "modify codebase"


# =========================================================
# META
# =========================================================

def build_meta(steps):

    files = set()

    has_verification = False

    for s in steps:

        path = s["a"].get("file_path")

        if path:
            files.add(path)

        if s["t"] == "run_command":

            cmd = s["a"].get("command", "").lower()

            if any(x in cmd for x in [
                "test",
                "lint",
                "pytest",
                "go test",
                "cargo test",
                "npm test",
                "build",
            ]):
                has_verification = True

    return {
        "tool_count": len(steps),
        "has_verification": has_verification,
        "multi_file": len(files) > 1,
    }


# =========================================================
# SCORING
# =========================================================

def score_session(steps):

    tools = [s["t"] for s in steps]

    score = 0

    if "read_files" in tools:
        score += 1

    if "modify_files" in tools:
        score += 2

    if "run_command" in tools:
        score += 1

    if "search_files" in tools:
        score += 1

    if len(set(tools)) >= 3:
        score += 1

    run_ratio = tools.count("run_command") / len(tools)

    if run_ratio > 0.5:
        score -= 1

    return score


# =========================================================
# PIPELINE
# =========================================================

def process_session(rows):

    steps = extract_steps(rows)

    if len(steps) < 3:
        return None

    steps = dedupe_run_commands(steps)

    steps = dedupe_reads(steps)

    steps = trim_run_commands(steps)

    if len(steps) < 3:
        return None

    if len(steps) > MAX_STEPS:
        steps = steps[:MAX_STEPS]

    if not any(
        s["t"] in ["modify_files", "write_files"]
        for s in steps
    ):
        return None

    if score_session(steps) < 3:
        return None

    return {
        "goal": build_goal(steps),
        "meta": build_meta(steps),
        "m": steps,
        "end": 1
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading dataset...")

    dataset = load_from_disk(DATA_PATH)

    if hasattr(dataset, "keys"):
        dataset = dataset[list(dataset.keys())[0]]

    total = 0
    kept = 0

    with open(OUTPUT_PATH, "w") as out:

        for rows in tqdm(
            session_iterator(dataset),
            desc="Processing"
        ):

            total += 1

            rows = sorted(
                rows,
                key=lambda x: x["turn_number"]
            )

            processed = process_session(rows)

            if processed:

                kept += 1

                out.write(
                    json.dumps(processed) + "\n"
                )

    print("\n=== DONE ===")
    print("Total:", total)
    print("Kept:", kept)
    print("Dropped:", total - kept)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()