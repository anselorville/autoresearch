"""
Fine-tuning Qwen3.5-0.8B on the RAG slot-extraction task.
Single-file, CPU-compatible, follows the autoresearch pattern.

Two training modes:
  "head_only"  — train only the lm_head; base model runs under torch.no_grad().
                 CPU-feasible: ~14s/step at 64 tokens.  Use for CPU verification.
  "partial"    — train the last TRAINABLE_LAYERS + lm_head with full backprop.
                 Needs GPU for practical speed (backward through DeltaNet is slow on CPU).

Run finetune_prepare.py once first, then:
    python finetune.py
"""

import gc
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_prepare import (
    CACHE_DIR, MAX_SEQ_LEN, MODEL_DIR,
    load_cache,
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly, no CLI flags needed
# ---------------------------------------------------------------------------

# Where to save checkpoints and the final model
CKPT_DIR = "finetune_checkpoints"

# Training mode:
#   "head_only" — CPU-feasible, trains lm_head only (~254M params).
#   "partial"   — trains last TRAINABLE_LAYERS + lm_head (needs GPU).
TRAINING_MODE = "head_only"

# Smoke-test: run 2 optimizer steps on a tiny subset and exit.
# Set to True to quickly verify the pipeline runs end-to-end on CPU (~2 min).
SMOKE_TEST = False

# Optimization
EPOCHS           = 3       # full passes over the training set
BATCH_SIZE       = 1       # per-step batch size (1 works fine; no padding needed)
GRAD_ACCUM_STEPS = 8       # effective batch size = BATCH_SIZE × GRAD_ACCUM_STEPS
LEARNING_RATE    = 3e-5    # peak learning rate
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.05    # fraction of total optimizer steps for LR warm-up

# "partial" mode only: number of transformer layers to unfreeze (from the end)
TRAINABLE_LAYERS = 8       # last 8 of 24 layers

# Time budget — training stops after this many hours (like train.py's 5-min budget).
# Set to a large value (e.g. 999) to disable and let EPOCHS control run length.
TIME_BUDGET_HOURS = 2.0

# Evaluation / checkpointing
EVAL_EVERY_STEPS = 100     # run validation loss every N optimizer steps
SAVE_EVERY_STEPS = 200     # save checkpoint every N optimizer steps

# Reproducibility
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    mode: str
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    trainable_layers: int
    max_seq_len: int


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> tuple[str, torch.dtype]:
    """
    Auto-detect available hardware and return (device_map, dtype).
    GPU  : device_map="auto", dtype=bfloat16
    CPU  : device_map="cpu",  dtype=float32
           Exception — head_only mode loads base in bfloat16 (inference-only
           under no_grad) then casts lm_head to float32 for training.
           This halves base-model memory with no training-stability impact.
    """
    if torch.cuda.is_available():
        name    = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [device] GPU: {name}  VRAM={vram_gb:.1f}GB → bfloat16 + device_map=auto")
        return "auto", torch.bfloat16
    print("  [device] No GPU found → float32 + device_map=cpu")
    return "cpu", torch.float32


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    print(f"Loading tokenizer from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    device_map, dtype = detect_device()

    if device_map == "cpu" and TRAINING_MODE == "partial":
        print("  [warn] TRAINING_MODE=partial on CPU is very slow (DeltaNet backward). "
              "Consider switching to head_only for CPU runs.")

    # head_only + CPU: load base in bfloat16 (no_grad inference → half memory),
    # then cast only lm_head to float32 so gradients stay numerically stable.
    load_dtype = (torch.bfloat16
                  if device_map == "cpu" and TRAINING_MODE == "head_only"
                  else dtype)

    print(f"Loading model from {MODEL_DIR} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=load_dtype,
        device_map=device_map,
    )

    if device_map == "cpu" and TRAINING_MODE == "head_only":
        model.lm_head.to(torch.float32)
        mem_base = sum(p.numel() * 2 for p in model.model.parameters()) / 1e9
        mem_head = sum(p.numel() * 4 for p in model.lm_head.parameters()) / 1e9
        print(f"  Mixed precision: base=bfloat16 ({mem_base:.1f}GB)  "
              f"lm_head=float32 ({mem_head:.2f}GB)")

    print(f"  Loaded {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params "
          f"in {time.time() - t0:.1f}s")
    return model, tokenizer


def configure_trainable_params(model, mode: str, trainable_layers: int):
    """Freeze/unfreeze parameters according to training mode."""
    if mode == "head_only":
        # Freeze everything; only lm_head (which is tied to embed_tokens)
        # keeps requires_grad=True.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Mode=head_only: {n_trainable / 1e6:.1f}M trainable params "
              f"(lm_head only)")

    elif mode == "partial":
        for p in model.parameters():
            p.requires_grad = False
        # Last N transformer layers
        layers = model.model.layers
        total  = len(layers)
        start  = max(0, total - trainable_layers)
        for layer in layers[start:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Final norm + lm_head
        for p in model.model.norm.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())
        print(f"  Mode=partial: layers {start}–{total - 1} + norm + lm_head  "
              f"({n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M, "
              f"{100 * n_trainable / n_total:.1f}%)")
    else:
        raise ValueError(f"Unknown TRAINING_MODE: {mode!r}")


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def compute_loss(model, ids: torch.Tensor, labels: torch.Tensor,
                 mode: str) -> torch.Tensor:
    """
    Compute SFT cross-entropy loss (labels with -100 masked out).

    head_only: base model runs under torch.no_grad(); gradient only through lm_head.
    partial  : standard forward with gradient through all unfrozen layers.
    """
    if mode == "head_only":
        # Frozen forward: no gradients through the DeltaNet layers
        with torch.no_grad():
            h = model.model(ids).last_hidden_state  # (B, T, D), detached
        logits = model.lm_head(h)                   # grad flows only here

    else:  # "partial"
        logits = model(ids).logits

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def make_dataloader(samples: list, batch_size: int, shuffle: bool):
    """Yield (input_ids, labels) tensors, one batch at a time."""
    indices = list(range(len(samples)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        if batch_size == 1:
            ids, labels = samples[batch_idx[0]]
            yield ids.unsqueeze(0), labels.unsqueeze(0)
        else:
            max_len = max(samples[j][0].size(0) for j in batch_idx)
            b_ids    = torch.zeros(len(batch_idx), max_len, dtype=torch.long)
            b_labels = torch.full((len(batch_idx), max_len), -100, dtype=torch.long)
            for k, j in enumerate(batch_idx):
                ids, labels = samples[j]
                L = ids.size(0)
                b_ids[k, :L]    = ids
                b_labels[k, :L] = labels
            yield b_ids, b_labels


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup_steps: int,
           base_lr: float, min_lr_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_samples: list, mode: str) -> float:
    """Average cross-entropy loss on the validation set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for ids, labels in make_dataloader(val_samples, batch_size=1, shuffle=False):
        # Always use head_only for eval (faster, same result for head_only training)
        h = model.model(ids).last_hidden_state
        logits = model.lm_head(h)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += (shift_labels >= 0).sum().item()
    model.train()
    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# Inference (greedy decode)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_prediction(model, tokenizer, prompt: str, max_new: int = 150) -> str:
    """Greedy decode to check model output format."""
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt")
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    model.train()
    return tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step: int, val_loss: float, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    path = os.path.join(ckpt_dir, f"step_{step:06d}_val{val_loss:.4f}.pt")
    torch.save({
        "step": step,
        "val_loss": val_loss,
        "model_state_dict": {k: v for k, v in model.state_dict().items()
                             if k in trainable_names},
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  Checkpoint saved → {path}")


def save_best_model(model, tokenizer, ckpt_dir: str):
    """Save the full model in HuggingFace format for easy reloading."""
    out_dir = os.path.join(ckpt_dir, "best_model")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  Best model saved → {out_dir}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, tokenizer, train_samples: list, val_samples: list,
          cfg: TrainConfig, time_budget_hours: float = 999.0) -> float:
    n_batches    = math.ceil(len(train_samples) / cfg.batch_size)
    total_micro  = cfg.epochs * n_batches
    grad_steps   = math.ceil(total_micro / cfg.grad_accum_steps)
    warmup_steps = max(1, int(grad_steps * cfg.warmup_ratio))
    budget_secs  = time_budget_hours * 3600

    print(f"Training: {len(train_samples)} samples × {cfg.epochs} epochs "
          f"= {total_micro} micro-batches → {grad_steps} optimizer steps")
    print(f"  mode={cfg.mode}  LR={cfg.learning_rate:.2e}  "
          f"warmup={warmup_steps}  grad_accum={cfg.grad_accum_steps}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    model.train()
    step        = 0
    micro_step  = 0
    accum_loss  = 0.0
    smooth_loss = None
    t_start     = time.time()
    best_val    = float("inf")

    timed_out = False
    for epoch in range(1, cfg.epochs + 1):
        if timed_out:
            break
        for ids, labels in make_dataloader(train_samples, cfg.batch_size, shuffle=True):
            loss = compute_loss(model, ids, labels, cfg.mode)
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % cfg.grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )

            lr = get_lr(step, grad_steps, warmup_steps, cfg.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss    = accum_loss / cfg.grad_accum_steps
            accum_loss  = 0.0
            smooth_loss = (avg_loss if smooth_loss is None
                           else 0.9 * smooth_loss + 0.1 * avg_loss)
            elapsed     = time.time() - t_start

            print(
                f"\repoch {epoch}/{cfg.epochs}  "
                f"step {step + 1:04d}/{grad_steps}  "
                f"loss {smooth_loss:.4f}  "
                f"lr {lr:.2e}  "
                f"elapsed {elapsed:.0f}s   ",
                end="", flush=True,
            )
            step += 1

            if time.time() - t_start > budget_secs:
                print(f"\n[timeout] TIME_BUDGET_HOURS={time_budget_hours}h reached at step {step}, stopping early.")
                timed_out = True
                break

            if step % EVAL_EVERY_STEPS == 0:
                print()
                val_loss = evaluate(model, val_samples, cfg.mode)
                print(f"  [eval] step {step}  val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_best_model(model, tokenizer, CKPT_DIR)

            if step % SAVE_EVERY_STEPS == 0:
                val_loss = evaluate(model, val_samples, cfg.mode)
                save_checkpoint(model, optimizer, step, val_loss, CKPT_DIR)

    print()  # final newline after \r log

    val_loss = evaluate(model, val_samples, cfg.mode)
    print(f"Final val_loss={val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        save_best_model(model, tokenizer, CKPT_DIR)

    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(SEED)
    t_start = time.time()

    # --- Load tokenized data ---
    train_cache = os.path.join(CACHE_DIR, f"train_{MAX_SEQ_LEN}.pkl")
    val_cache   = os.path.join(CACHE_DIR, f"val_{MAX_SEQ_LEN}.pkl")
    if not os.path.exists(train_cache):
        raise RuntimeError(
            "Cache not found. Run `python finetune_prepare.py` first."
        )

    print("Loading tokenized data ...")
    train_samples = load_cache(train_cache)
    val_samples   = load_cache(val_cache)
    print(f"  train={len(train_samples)}  val={len(val_samples)}")

    # Smoke-test: minimal subset of short samples for a quick CPU sanity check
    epochs           = EPOCHS
    grad_accum_steps = GRAD_ACCUM_STEPS
    if SMOKE_TEST:
        print("\n[SMOKE TEST] Using minimal dataset for CPU sanity check ...")
        # Take 4 shortest samples (min seq_len in dataset is ~125 tokens)
        by_len        = sorted(train_samples, key=lambda s: s[0].size(0))
        val_by_len    = sorted(val_samples,   key=lambda s: s[0].size(0))
        train_samples = by_len[:4]
        val_samples   = val_by_len[:2]
        print(f"  {len(train_samples)} train / {len(val_samples)} val samples  "
              f"(lengths: {[s[0].size(0) for s in train_samples]})")
        epochs           = 1
        grad_accum_steps = 2

    # --- Model ---
    model, tokenizer = load_model_and_tokenizer()
    configure_trainable_params(model, TRAINING_MODE, TRAINABLE_LAYERS)

    # --- Config ---
    cfg = TrainConfig(
        mode            = TRAINING_MODE,
        epochs          = epochs,
        batch_size      = BATCH_SIZE,
        grad_accum_steps= grad_accum_steps,
        learning_rate   = LEARNING_RATE,
        weight_decay    = WEIGHT_DECAY,
        warmup_ratio    = WARMUP_RATIO,
        trainable_layers= TRAINABLE_LAYERS,
        max_seq_len     = MAX_SEQ_LEN,
    )
    print(f"\nConfig: {asdict(cfg)}")

    # --- Baseline eval ---
    n_eval = 2 if SMOKE_TEST else 20
    print(f"\nBaseline on {n_eval} val samples:")
    gc.collect()
    baseline = evaluate(model, val_samples[:n_eval], TRAINING_MODE)
    print(f"  val_loss = {baseline:.4f}")

    # --- Train ---
    print()
    best_val = train(model, tokenizer, train_samples, val_samples, cfg,
                     time_budget_hours=TIME_BUDGET_HOURS)

    # --- Inference demo ---
    print("\nInference demo (first val sample):")
    raw_val = "dataset/data/labeled/20260409_222413/val.jsonl"
    if os.path.exists(raw_val):
        with open(raw_val, encoding="utf-8") as fv:
            first = json.loads(fv.readline())
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": first["input"].strip()}],
            tokenize=False, add_generation_prompt=True,
        )
        pred = sample_prediction(model, tokenizer, prompt, max_new=150)
        print(f"  Input    : {first['input'].strip()[:100]}")
        print(f"  Expected : {first['output']}")
        print(f"  Model    : {pred}")

    # --- Summary (machine-readable, parsed by pipeline.py) ---
    total_time = time.time() - t_start
    print("\n---")
    print(f"best_val_loss:  {best_val:.6f}")
    print(f"total_seconds:  {total_time:.1f}")
    print(f"ckpt_dir:       {CKPT_DIR}")
    print(f"training_mode:  {TRAINING_MODE}")
