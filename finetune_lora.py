"""
LoRA fine-tuning of Qwen3.5-0.8B on the RAG slot-extraction task.
Single-file, CPU-compatible, follows the autoresearch pattern.

Uses PEFT (Parameter-Efficient Fine-Tuning) LoRA adapters — only adapter
weights are trained, leaving the base model frozen. This gives:
  - Tiny trainable parameter count (~1–4M vs 800M total)
  - CPU-feasible: no full backprop through the frozen base

Run finetune_prepare.py once first, then:
    python finetune_lora.py

Autoresearch search space (constants Claude can modify):
  LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
  LEARNING_RATE, GRAD_ACCUM_STEPS, WARMUP_RATIO, WEIGHT_DECAY, EPOCHS
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from finetune_prepare import (
    CACHE_DIR, MAX_SEQ_LEN, MODEL_DIR,
    load_cache,
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly (autoresearch search space)
# ---------------------------------------------------------------------------

# Where to save checkpoints and the final adapter
CKPT_DIR = "finetune_lora_checkpoints"

# Smoke-test: run 2 optimizer steps on a tiny subset and exit.
SMOKE_TEST = True

# LoRA adapter configuration
LORA_R              = 8      # adapter rank (higher = more capacity, more memory)
LORA_ALPHA          = 16     # scaling factor (effective scale = alpha / r)
LORA_DROPOUT        = 0.05   # dropout applied to LoRA layers
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # modules to add LoRA to
                              # options: q_proj, k_proj, v_proj, o_proj,
                              #          gate_proj, up_proj, down_proj

# Optimization
LEARNING_RATE    = 5e-5    # LoRA adapters benefit from higher LR than full SFT
WEIGHT_DECAY     = 0.01
GRAD_ACCUM_STEPS = 8       # effective batch size = 1 × GRAD_ACCUM_STEPS
WARMUP_RATIO     = 0.05
EPOCHS           = 999     # TIME_BUDGET_HOURS controls stopping; large value = train until timeout

# Time budget — training stops after this many hours.
# Set to 999 to disable and let EPOCHS control run length.
TIME_BUDGET_HOURS = 2.0

# Evaluation / checkpointing
EVAL_EVERY_STEPS = 100
SAVE_EVERY_STEPS = 200

# Reproducibility
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class LoraTrainConfig:
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list
    learning_rate: float
    weight_decay: float
    grad_accum_steps: int
    warmup_ratio: float
    epochs: int
    max_seq_len: int


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> tuple[str, torch.dtype]:
    """
    Auto-detect available hardware and return (device_map, dtype).
    GPU: device_map="auto", dtype=bfloat16 (fast, half memory vs float32)
    CPU: device_map="cpu",  dtype=float32  (bfloat16 has no CPU training kernels)
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

    print(f"Loading model from {MODEL_DIR} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map=device_map,
    )
    print(f"  Loaded {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params "
          f"in {time.time() - t0:.1f}s")
    return model, tokenizer


def apply_lora(model, cfg: LoraTrainConfig):
    """Wrap the model with LoRA adapters via PEFT."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied: r={cfg.lora_r}  alpha={cfg.lora_alpha}  "
          f"target={cfg.lora_target_modules}")
    print(f"  Trainable: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M "
          f"({100 * n_trainable / n_total:.2f}%)")
    return model


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def compute_loss(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute SFT cross-entropy loss through LoRA-wrapped model.
    Only LoRA adapter weights receive gradients; base is frozen by PEFT.
    """
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
    indices = list(range(len(samples)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        if batch_size == 1:
            ids, labels = samples[batch_idx[0]]
            yield ids.unsqueeze(0), labels.unsqueeze(0)
        else:
            max_len  = max(samples[j][0].size(0) for j in batch_idx)
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
def evaluate(model, val_samples: list) -> float:
    """Average cross-entropy loss on the validation set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for ids, labels in make_dataloader(val_samples, batch_size=1, shuffle=False):
        logits = model(ids).logits
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

def save_adapter_checkpoint(model, step: int, val_loss: float, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:06d}_val{val_loss:.4f}")
    model.save_pretrained(path)
    print(f"  Adapter checkpoint saved → {path}")


def save_best_adapter(model, tokenizer, ckpt_dir: str):
    """Save the best LoRA adapter in PEFT format for easy reloading."""
    out_dir = os.path.join(ckpt_dir, "best_adapter")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  Best adapter saved → {out_dir}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, tokenizer, train_samples: list, val_samples: list,
          cfg: LoraTrainConfig, time_budget_hours: float = 999.0) -> float:
    n_batches    = math.ceil(len(train_samples) / 1)  # batch_size=1
    total_micro  = cfg.epochs * n_batches
    grad_steps   = math.ceil(total_micro / cfg.grad_accum_steps)
    warmup_steps = max(1, int(grad_steps * cfg.warmup_ratio))
    budget_secs  = time_budget_hours * 3600

    print(f"Training: {len(train_samples)} samples × {cfg.epochs} epochs "
          f"= {total_micro} micro-batches → {grad_steps} optimizer steps")
    print(f"  LR={cfg.learning_rate:.2e}  warmup={warmup_steps}  "
          f"grad_accum={cfg.grad_accum_steps}  budget={time_budget_hours}h")

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
        for ids, labels in make_dataloader(train_samples, batch_size=1, shuffle=True):
            loss = compute_loss(model, ids, labels)
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
                print(f"\n[timeout] TIME_BUDGET_HOURS={time_budget_hours}h reached "
                      f"at step {step}, stopping early.")
                timed_out = True
                break

            if step % EVAL_EVERY_STEPS == 0:
                print()
                val_loss = evaluate(model, val_samples)
                print(f"  [eval] step {step}  val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_best_adapter(model, tokenizer, CKPT_DIR)

            if step % SAVE_EVERY_STEPS == 0:
                val_loss = evaluate(model, val_samples)
                save_adapter_checkpoint(model, step, val_loss, CKPT_DIR)

    print()

    val_loss = evaluate(model, val_samples)
    print(f"Final val_loss={val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        save_best_adapter(model, tokenizer, CKPT_DIR)

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

    # Smoke-test: minimal subset
    epochs           = EPOCHS
    grad_accum_steps = GRAD_ACCUM_STEPS
    if SMOKE_TEST:
        print("\n[SMOKE TEST] Using minimal dataset ...")
        by_len        = sorted(train_samples, key=lambda s: s[0].size(0))
        val_by_len    = sorted(val_samples,   key=lambda s: s[0].size(0))
        train_samples = by_len[:4]
        val_samples   = val_by_len[:2]
        print(f"  {len(train_samples)} train / {len(val_samples)} val  "
              f"(lengths: {[s[0].size(0) for s in train_samples]})")
        epochs           = 1
        grad_accum_steps = 2

    # --- Model + LoRA ---
    model, tokenizer = load_model_and_tokenizer()
    cfg = LoraTrainConfig(
        lora_r              = LORA_R,
        lora_alpha          = LORA_ALPHA,
        lora_dropout        = LORA_DROPOUT,
        lora_target_modules = LORA_TARGET_MODULES,
        learning_rate       = LEARNING_RATE,
        weight_decay        = WEIGHT_DECAY,
        grad_accum_steps    = grad_accum_steps,
        warmup_ratio        = WARMUP_RATIO,
        epochs              = epochs,
        max_seq_len         = MAX_SEQ_LEN,
    )
    model = apply_lora(model, cfg)
    print(f"\nConfig: {asdict(cfg)}")

    # --- Baseline eval ---
    n_eval = 2 if SMOKE_TEST else 20
    print(f"\nBaseline on {n_eval} val samples:")
    baseline = evaluate(model, val_samples[:n_eval])
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
    print(f"lora_r:         {LORA_R}")
    print(f"lora_alpha:     {LORA_ALPHA}")
