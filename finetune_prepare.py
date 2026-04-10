"""
Data preparation for Qwen3.5-0.8B fine-tuning on slot extraction task.

Reads train.jsonl / val.jsonl, formats each sample as a ChatML conversation
with <think> CoT in the assistant turn, tokenizes, applies label masking
(input tokens → -100, only the assistant response is trained on), and saves
tokenized tensors to a cache directory.

Usage:
    python finetune_prepare.py
"""

import json
import os
import pickle

import torch

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration — edit these; no CLI flags
# ---------------------------------------------------------------------------

MODEL_DIR          = "D:/git_repository/Qwen3.5-0.8B"
DATA_DIR           = "dataset/data/labeled/20260409_222413"
CACHE_DIR          = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "finetune")
MAX_SEQ_LEN        = 1536   # ~95% coverage after adding system prompt (~855 tokens)
SYSTEM_PROMPT_PATH = "dataset/doc/intent_keywords_system_prompt.md"

# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def _load_system_prompt(path: str) -> str:
    """
    Load the slot-extraction system prompt.
    The file ends with '## 当前输入' as a template placeholder — strip it so
    the tokenizer's chat template places the user message after the system turn.
    """
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        content = f.read().rstrip()
    # Remove trailing section header that marks where user input begins
    content = content.rsplit("## 当前输入", 1)[0].rstrip()
    return content


_SYSTEM_PROMPT: str = ""   # populated in main() and tokenize_split()


def format_messages(sample: dict) -> list[dict]:
    """
    Convert a JSONL sample to ChatML messages.

    Includes the task system prompt so training matches inference conditions:
    the model learns to follow the slot-extraction rules, not just memorise
    query→output mappings — which would overfit and lose generalisation.
    """
    user_content = sample["input"].strip()
    asst_content = f"<think>\n{sample['cot'].strip()}\n</think>\n\n{sample['output'].strip()}"
    messages = []
    if _SYSTEM_PROMPT:
        messages.append({"role": "system", "content": _SYSTEM_PROMPT})
    messages.append({"role": "user",      "content": user_content})
    messages.append({"role": "assistant", "content": asst_content})
    return messages


def build_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """
    Mask out the prompt prefix with -100.
    Only the assistant response tokens contribute to the loss.
    """
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return labels


def tokenize_sample(tokenizer, sample: dict, max_len: int):
    """
    Returns (input_ids, labels) tensors or None if the sample is too long
    and truncation would cut into the response.
    """
    messages = format_messages(sample)

    # Full conversation text
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Prompt-only text (up to and including "<|im_start|>assistant\n")
    # Keep all turns except the final assistant response so the template
    # always ends with a valid user message (required by Qwen's chat template).
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_ids  = tokenizer.encode(full_text,   add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    prompt_len = len(prompt_ids)

    # Truncate if needed — keep the response intact by truncating the input
    if len(full_ids) > max_len:
        # Check whether response alone fits within max_len
        response_len = len(full_ids) - prompt_len
        if response_len >= max_len:
            return None   # response alone exceeds budget — skip
        # Shrink prompt to fit
        allowed_prompt = max_len - response_len
        full_ids  = full_ids[:allowed_prompt] + full_ids[len(full_ids) - response_len:]
        prompt_len = allowed_prompt

    labels = build_labels(full_ids, prompt_len)
    assert len(full_ids) == len(labels)

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels,   dtype=torch.long),
    )

# ---------------------------------------------------------------------------
# Dataset saving / loading
# ---------------------------------------------------------------------------

def tokenize_split(tokenizer, split_path: str, max_len: int, split_name: str):
    """Tokenize all samples in a JSONL file, return list of (input_ids, labels)."""
    global _SYSTEM_PROMPT
    if not _SYSTEM_PROMPT:
        _SYSTEM_PROMPT = _load_system_prompt(SYSTEM_PROMPT_PATH)
        if _SYSTEM_PROMPT:
            sp_tokens = len(tokenizer.encode(_SYSTEM_PROMPT))
            print(f"  System prompt: {len(_SYSTEM_PROMPT)} chars, {sp_tokens} tokens")
        else:
            print("  [warn] System prompt not found — training without task instructions")

    if not os.path.exists(split_path):
        print(f"  {split_name}: file not found, skipping.")
        return []

    with open(split_path, encoding="utf-8") as f:
        lines = f.readlines()

    samples = []
    skipped = 0
    for line in lines:
        sample = json.loads(line)
        result = tokenize_sample(tokenizer, sample, max_len)
        if result is None:
            skipped += 1
            continue
        samples.append(result)

    print(f"  {split_name}: {len(samples)} samples "
          f"({skipped} skipped as too long), "
          f"max_len={max_len}")
    return samples


def save_cache(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    print(f"  Saved {len(samples)} samples → {path}")


def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(samples, name: str):
    if not samples:
        return
    lengths = [s[0].size(0) for s in samples]
    response_lengths = [(s[1] != -100).sum().item() for s in samples]
    print(f"  {name}: n={len(lengths)}, "
          f"seq_len p50={sorted(lengths)[len(lengths)//2]}, "
          f"seq_len max={max(lengths)}, "
          f"resp_len p50={sorted(response_lengths)[len(response_lengths)//2]}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _SYSTEM_PROMPT
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(CACHE_DIR, f"train_{MAX_SEQ_LEN}.pkl")
    val_cache   = os.path.join(CACHE_DIR, f"val_{MAX_SEQ_LEN}.pkl")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"Cache already exists at {CACHE_DIR}")
        train = load_cache(train_cache)
        val   = load_cache(val_cache)
        print_stats(train, "train")
        print_stats(val,   "val  ")
        return

    _SYSTEM_PROMPT = _load_system_prompt(SYSTEM_PROMPT_PATH)
    if _SYSTEM_PROMPT:
        print(f"System prompt loaded: {SYSTEM_PROMPT_PATH}")
    else:
        print(f"[warn] System prompt not found at {SYSTEM_PROMPT_PATH}")

    print(f"Loading tokenizer from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    print(f"Tokenizing (max_seq_len={MAX_SEQ_LEN}) ...")
    train = tokenize_split(tokenizer, os.path.join(DATA_DIR, "train.jsonl"), MAX_SEQ_LEN, "train")
    val   = tokenize_split(tokenizer, os.path.join(DATA_DIR, "val.jsonl"),   MAX_SEQ_LEN, "val  ")

    print("Statistics:")
    print_stats(train, "train")
    print_stats(val,   "val  ")

    print("Saving cache ...")
    save_cache(train, train_cache)
    save_cache(val,   val_cache)

    print(f"\nDone. Ready to fine-tune (run finetune.py).")


if __name__ == "__main__":
    main()
