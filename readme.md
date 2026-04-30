# Dataset for Multi-Turn Coding Agent

## Overview

This repository contains a curated dataset and processing pipeline designed for training **tool-using coding agents**, with a specific focus on **multi-turn reasoning and tool interaction**.

The goal is not just to fine-tune a model to *call tools*, but to embed **structured tool usage patterns directly into model behavior**, enabling more reliable and autonomous coding agents—especially at smaller model scales (e.g., 3B parameters).

---

## Motivation

Most existing tool-call datasets fall short in one or more of the following ways:

* **ToolBench / Hermes-style datasets**

  * Often synthetic or templated
  * Lack realistic iteration patterns
  * Weak signal for debugging workflows

* **Pure synthetic datasets**

  * Clean but not representative of real-world usage
  * Miss failure → iteration → validation loops

* **Prompt-based tool usage**

  * Relies heavily on system prompts
  * Does not generalize well to autonomous agents

This project addresses these gaps by leveraging **real-world developer-agent interaction logs** and refining them into a **high-signal training dataset**.

---

## Dataset Source

We use the **SWE-chat dataset**, which consists of real conversations between developers and AI systems (e.g., Claude, Gemini).

### Why SWE-chat?

* Contains **authentic debugging workflows**
* Naturally includes:

  * File reads
  * Code edits
  * Shell execution
  * Iterative fixes
* Exhibits realistic patterns:

  * Failure → inspection → modification → validation

However, it is:

* Extremely large and noisy
* Redundant (many repeated tool calls)
* Not directly usable for training

---

## Core Challenges

During multiple iterations of dataset generation, we observed:

1. **Excessive trajectory length**

   * Sessions often contain hundreds of steps

2. **Redundant tool calls**

   * Repeated `run_command` or `read_files` with no added value

3. **Low-signal outputs**

   * Status updates, logs, or trivial outputs

4. **Poor goal definition**

   * Missing or vague task descriptions

5. **Imbalanced trajectories**

   * Over-representation of shell commands vs reasoning steps

---

## Key Design Principles

To address these issues, the dataset pipeline enforces:

### 1. Tool Normalization

All tools are mapped to a minimal, consistent schema:

* `read_files`
* `search_files`
* `modify_files`
* `write_files`
* `run_command`

Non-essential tools (e.g., task tracking) are removed.

---

### 2. Session Reconstruction

Raw logs are converted into structured trajectories:

* Grouped by `session_id`
* Ordered by `turn_number`
* Converted into `(tool, args, output)` triples

---

### 3. Hard Filtering

Sessions are discarded if they:

* Contain fewer than 3 tool calls
* Lack code modification (`modify_files` / `write_files`)
* Are dominated by `run_command`
* Have low tool diversity

---

### 4. Trajectory Compression

This is the most critical step.

#### Techniques:

* Collapse repeated tool calls
* Keep:

  * First failure
  * Final success
* Remove:

  * Low-signal outputs (`< 20 chars`, status logs)
* Filter `run_command` outputs:

  * Keep only meaningful signals:

    * `error`, `fail`, `exception`, `0 issues`, `ok`

---

### 5. Goal Reconstruction

Each session is assigned a clear objective:

Priority:

1. First failing command → infer error type
2. First file modification → infer intent
3. First file read → fallback context

Examples:

* `fix lint errors`
* `fix failing test in xyz.ts`
* `modify config file`

---

### 6. Scoring System

Each session is scored based on:

* Tool diversity
* Presence of:

  * read → modify → run pattern
* Successful resolution
* Iterative behavior

Penalties:

* Too many steps
* Excessive shell usage

Only high-quality sessions are retained.

---

## Final Dataset Format

Each sample:

```json
{
  "goal": "fix lint errors",
  "m": [
    {"t": "run_command", "a": {...}, "o": "..."},
    {"t": "read_files", "a": {...}, "o": "..."},
    {"t": "modify_files", "a": {...}, "o": "..."}
  ],
  "end": 1
}
```

---

## Why This Matters

This dataset enables:

* Training models to:

  * Understand when to use tools
  * Sequence tool calls effectively
  * Iterate toward solutions

* Moving beyond:

  * Prompt-engineered agents
  * Static tool APIs

Toward:

> **Embedded tool reasoning inside the model weights**

---

## Research Direction

This project explores a key hypothesis:

> A sufficiently clean, high-signal dataset of tool interactions can allow small models (e.g., 3B) to behave like capable coding agents without heavy prompt scaffolding.

Key areas of investigation:

* Tool usage as a learned behavior vs prompted behavior
* Compression vs information retention trade-offs
* Minimum dataset size for generalization
* Effect of trajectory structure on reasoning quality

---

## Current Status

* Dataset pipeline: **functional and improving**
* Multiple filtering strategies tested
* High-quality trajectories identified
* Remaining issue:

  * Eliminating redundancy while preserving reasoning depth

---

## Next Steps

* Finalize optimal filtering strategy
* Generate full dataset
* Fine-tune a 3B model (LoRA / full fine-tune)
* Evaluate:

  * Tool usage accuracy
  * Task completion rate
  * Iterative reasoning capability

---

## Contribution

Contributions are welcome in:

* Filtering strategies
* Scoring heuristics
* Evaluation benchmarks
* Model training experiments

---

## Summary

This repository is not just about dataset creation.

It is about answering a deeper question:

> Can tool use be learned as a native capability of language models, rather than injected at runtime?

This dataset is a step toward that direction.
