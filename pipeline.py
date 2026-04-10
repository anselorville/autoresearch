"""
Autoresearch pipeline — automated experiment loop for Qwen3.5-0.8B fine-tuning.

Supports two training backends via the 'mode' config field:
  "sft"  — Qwen3.5-0.8B SFT fine-tuning  (finetune.py,      CPU-compatible)
  "lora" — Qwen3.5-0.8B LoRA fine-tuning (finetune_lora.py, CPU-compatible)

Uses service/claude_client.py as the LLM node to iteratively modify the
target training script, run experiments, and log results.

Usage:
    python pipeline.py                           # use pipeline_config.yml
    python pipeline.py --config my_config.yml    # custom config
    python pipeline.py --dry-run                 # skip actual training (mock metrics)
    python pipeline.py --n-loops 5               # override loop count
    python pipeline.py --mode lora               # override mode
    python pipeline.py --smoke-test --n-loops 1 --mode sft   # ~2 min real-data check
    python pipeline.py --smoke-test --n-loops 1 --mode lora

Workflow:
  1. Setup   — load config, init results.tsv
  2. Baseline — run target script as-is
  3. Loop N times:
       a. Ask Claude for a modification (JSON: description + find/replace changes)
       b. Apply the change to the target file and git commit
       c. Run training; parse metric from --- summary block in log
       d. Keep commit if metric improved, else git reset --hard HEAD~1
       e. Record in results.tsv
  4. Analysis — print summary via analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import yaml
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------

_SFT_SEARCH_SPACE = """\
You are optimizing finetune.py — an SFT (supervised fine-tuning) script for
Qwen3.5-0.8B on a RAG slot-extraction task.

Tunable constants (your search space):
  TRAINING_MODE    : "head_only" (CPU-feasible, lm_head only) or
                     "partial"   (last TRAINABLE_LAYERS + lm_head, needs GPU)
  LEARNING_RATE    : float  — peak LR for AdamW (e.g. 1e-5 to 5e-5)
  WEIGHT_DECAY     : float  — AdamW weight decay (e.g. 0.001 to 0.1)
  GRAD_ACCUM_STEPS : int    — gradient accumulation (effective batch size)
  WARMUP_RATIO     : float  — fraction of steps for LR warmup (e.g. 0.03–0.1)
  TRAINABLE_LAYERS : int    — (partial only) number of layers to unfreeze from end
  EPOCHS           : int    — passes over the training set

Rules:
- Do NOT modify TIME_BUDGET_HOURS, CKPT_DIR, SMOKE_TEST, SEED, or evaluation code.
- Do NOT add new imports or dependencies.
- Keep changes minimal — one conceptual change per experiment."""

_LORA_SEARCH_SPACE = """\
You are optimizing finetune_lora.py — a LoRA adapter fine-tuning script for
Qwen3.5-0.8B on a RAG slot-extraction task.

Tunable constants (your search space):
  LORA_R              : int   — LoRA rank (e.g. 4, 8, 16, 32, 64)
  LORA_ALPHA          : int   — LoRA scaling alpha (commonly 2×r or equal to r)
  LORA_DROPOUT        : float — dropout on LoRA layers (e.g. 0.0 to 0.1)
  LORA_TARGET_MODULES : list  — which projection layers to add LoRA to:
                         options: "q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"
                         Example: ["q_proj", "v_proj", "k_proj"]
  LEARNING_RATE       : float — peak LR for AdamW (e.g. 5e-5 to 5e-4)
  WEIGHT_DECAY        : float — AdamW weight decay
  GRAD_ACCUM_STEPS    : int   — gradient accumulation steps
  WARMUP_RATIO        : float — LR warmup fraction
  EPOCHS              : int   — passes over training set

Rules:
- Do NOT modify TIME_BUDGET_HOURS, CKPT_DIR, SMOKE_TEST, SEED, or evaluation code.
- Do NOT add new imports or dependencies.
- Keep changes minimal — one conceptual change per experiment."""

MODE_CONFIG: dict[str, dict] = {
    "sft": {
        "target_file": "finetune.py",
        "run_cmd":     f"{sys.executable} finetune.py",
        "metric_key":  "best_val_loss",
        "search_space": _SFT_SEARCH_SPACE,
    },
    "lora": {
        "target_file": "finetune_lora.py",
        "run_cmd":     f"{sys.executable} finetune_lora.py",
        "metric_key":  "best_val_loss",
        "search_space": _LORA_SEARCH_SPACE,
    },
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "mode":               "sft",
    "n_loops":            3,
    "dry_run":            False,
    "time_budget_hours":  2.0,
    "time_budget_sec":    300,
    "results_file":       "results.tsv",
    "log_dir":            "logs",
    "train_timeout_sec":  14400,
    "llm": {
        "api_url":     "",
        "api_model":   "",
        "temperature": 0.5,
        "max_tokens":  4096,
    },
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    base = {k: v for k, v in DEFAULT_CONFIG.items()}
    base["llm"] = dict(DEFAULT_CONFIG["llm"])
    try:
        with open(path, encoding="utf-8") as f:
            if path.endswith((".yml", ".yaml")):
                user = yaml.safe_load(f) or {}
            else:
                user = json.load(f)
        for k, v in (user or {}).items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            if k == "llm" and isinstance(v, dict):
                base["llm"].update(v)
            else:
                base[k] = v
    except FileNotFoundError:
        print(f"[pipeline] Config file {path!r} not found, using defaults.")
    return base


# ---------------------------------------------------------------------------
# Claude client setup
# ---------------------------------------------------------------------------

def build_llm(cfg: dict):
    root = Path(__file__).parent
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "service"))

    import claude_client as cc  # type: ignore

    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("api_url"):
        cc.API_URL = llm_cfg["api_url"]
    if llm_cfg.get("api_model"):
        cc.API_MODEL = llm_cfg["api_model"]
    if llm_cfg.get("temperature"):
        cc.TEMPERATURE = float(llm_cfg["temperature"])
    if llm_cfg.get("max_tokens"):
        cc.MAX_TOKENS = int(llm_cfg["max_tokens"])

    return cc


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_file(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"<file not found: {path}>"


def write_file(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# results.tsv helpers
# ---------------------------------------------------------------------------

TSV_HEADER = "commit\tmetric\tmemory_gb\tstatus\tdescription\n"


def init_results_tsv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(TSV_HEADER)
        print(f"[pipeline] Initialized {path}")
    else:
        print(f"[pipeline] Appending to existing {path}")


def append_result(path: str, commit: str, metric: float,
                  memory_gb: float, status: str, description: str) -> None:
    row = f"{commit}\t{metric:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(row)
    print(f"[pipeline] Logged: {status:8s} metric={metric:.6f}  {description[:60]}")


def read_results_tsv(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return TSV_HEADER


def _read_best_metric(tsv_path: str) -> float:
    best = float("inf")
    try:
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[3] not in ("status", "crash"):
                    try:
                        m = float(parts[1])
                        if m > 0:
                            best = min(best, m)
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return best if best != float("inf") else float("inf")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(args: list[str], cwd: str = ".") -> tuple[int, str]:
    result = subprocess.run(
        ["git"] + args, cwd=cwd, capture_output=True, text=True
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def git_short_hash(cwd: str = ".") -> str:
    rc, out = git(["rev-parse", "--short", "HEAD"], cwd=cwd)
    return out.strip() if rc == 0 else "0000000"


def git_commit_target(target_file: str, message: str, cwd: str = ".") -> str:
    git(["add", target_file], cwd=cwd)
    rc, out = git(["commit", "-m", message], cwd=cwd)
    if rc != 0:
        print(f"  [git] commit failed: {out}")
    return git_short_hash(cwd=cwd)


def git_revert_last_commit(cwd: str = ".") -> None:
    """Undo the last commit (program.md: 'git reset back to where you started')."""
    rc, out = git(["reset", "--hard", "HEAD~1"], cwd=cwd)
    if rc != 0:
        print(f"  [git] reset failed: {out}")


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def _apply_patches(content: str, mode: str, cfg: dict, smoke_test: bool) -> str:
    """Apply all in-memory patches to the training script content."""
    # Time budget
    budget_hours = cfg.get("time_budget_hours", 2.0)
    content = re.sub(r'^(TIME_BUDGET_HOURS\s*=\s*)[\d.]+',
                     f'\\g<1>{budget_hours}', content, flags=re.MULTILINE)
    # Smoke test
    if smoke_test:
        content = re.sub(r'^(SMOKE_TEST\s*=\s*)False',
                         r'\g<1>True', content, flags=re.MULTILINE)
    return content


def run_training(mode: str, target_file: str, cfg: dict,
                 log_path: str, dry_run: bool = False,
                 smoke_test: bool = False) -> tuple[float, float]:
    """Run training script. Returns (metric_value, memory_gb)."""
    if dry_run:
        base  = getattr(run_training, "_last_metric", 1.0)
        delta = random.gauss(0.0, 0.005)
        value = max(0.05, base + delta)
        run_training._last_metric = value
        print(f"  [dry-run] Mocked metric={value:.6f}")
        time.sleep(1)
        return value, 0.0

    mc         = MODE_CONFIG[mode]
    run_cmd    = mc["run_cmd"]
    timeout    = cfg.get("train_timeout_sec", 14400)
    patch_file = target_file

    # Read original, apply all patches in memory, write once
    original = read_file(patch_file)
    patched  = _apply_patches(original, mode, cfg, smoke_test)
    if patched != original:
        write_file(patch_file, patched)
        print(f"  [pipeline] Patched {patch_file} "
              f"TIME_BUDGET_HOURS → {cfg.get('time_budget_hours', 2.0)}h"
              + ("  SMOKE_TEST → True" if smoke_test else ""))

    try:
        print(f"  [train] Running: {run_cmd}  (timeout={timeout}s)")
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                run_cmd, shell=True,
                stdout=log_f, stderr=log_f,
                timeout=timeout,
            )
        if proc.returncode != 0:
            print(f"  [train] Process exited with code {proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [train] TIMEOUT after {timeout}s — treating as crash")
        return 0.0, 0.0
    except Exception as e:
        print(f"  [train] Unexpected error: {e}")
        return 0.0, 0.0
    finally:
        if patched != original:
            write_file(patch_file, original)
            print(f"  [pipeline] Restored {patch_file}")

    return _parse_log(log_path, mc["metric_key"])


def _parse_log(log_path: str, metric_key: str) -> tuple[float, float]:
    try:
        content = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return 0.0, 0.0

    metric    = 0.0
    memory_gb = 0.0

    m = re.search(rf'^{re.escape(metric_key)}:\s*([\d.]+)', content, re.MULTILINE)
    if m:
        metric = float(m.group(1))

    m = re.search(r'^peak_vram_mb:\s*([\d.]+)', content, re.MULTILINE)
    if m:
        memory_gb = float(m.group(1)) / 1024.0

    if metric == 0.0:
        lines = content.splitlines()
        print(f"  [train] {metric_key} not found — likely crashed. Last lines:")
        for ln in lines[-20:]:
            print(f"    {ln}")

    return metric, memory_gb


# ---------------------------------------------------------------------------
# Claude prompt construction + parsing
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are an AI researcher running automated experiments to minimize a metric \
(lower is always better) on a machine learning training script.

{search_space}

Output ONLY valid JSON — no commentary, no markdown fences — in this exact schema:
{{
  "description": "One-sentence description of this experiment",
  "changes": [
    {{"find": "exact string to find in the script", "replace": "exact replacement string"}}
  ]
}}

Rules for changes:
- Each "find" must be an exact verbatim substring present in the current script.
- Keep changes minimal and focused — one conceptual change per experiment.
- To change a constant value (e.g. LORA_R = 8 → LORA_R = 16), use the full line.
- Your first suggestion should be a modest, high-confidence improvement.
"""


def build_prompt(target_content: str, results_tsv: str,
                 loop_idx: int, n_loops: int, mode: str) -> str:
    mc = MODE_CONFIG[mode]
    system = _BASE_SYSTEM_PROMPT.format(search_space=mc["search_space"])
    return (
        f"{system}\n\n"
        f"--- Loop {loop_idx}/{n_loops} ---\n\n"
        f"Optimization target: {mc['metric_key']} (lower is better)\n\n"
        f"Current results.tsv:\n{results_tsv}\n\n"
        f"Current {mc['target_file']}:\n"
        f"```python\n{target_content}\n```\n\n"
        f"Suggest exactly ONE modification. Output ONLY the JSON object."
    )


def parse_llm_response(response: str) -> dict:
    clean = re.sub(r'```(?:json)?\s*', '', response).strip()
    clean = re.sub(r'```\s*$', '', clean).strip()
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        clean = m.group(0)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response not valid JSON: {e}\nResponse:\n{response[:500]}"
        ) from e
    if not isinstance(parsed.get("description"), str):
        raise ValueError("Missing 'description'")
    if not isinstance(parsed.get("changes"), list):
        raise ValueError("Missing 'changes'")
    for ch in parsed["changes"]:
        if not isinstance(ch.get("find"), str) or not isinstance(ch.get("replace"), str):
            raise ValueError(f"Invalid change entry: {ch}")
    return parsed


def apply_changes(content: str, changes: list[dict]) -> tuple[str, list[str]]:
    result   = content
    warnings = []
    for ch in changes:
        if ch["find"] not in result:
            warnings.append(f"  [apply] 'find' not found: {ch['find'][:80]!r}")
        else:
            result = result.replace(ch["find"], ch["replace"], 1)
    return result, warnings


# ---------------------------------------------------------------------------
# Mock LLM changes for dry-run testing
# ---------------------------------------------------------------------------

_MOCK_CHANGES: dict[str, list[dict]] = {
    "sft": [
        {"description": "Increase learning rate from 2e-5 to 3e-5",
         "changes": [{"find": "LEARNING_RATE    = 2e-5",
                      "replace": "LEARNING_RATE    = 3e-5"}]},
        {"description": "Increase grad accumulation steps from 8 to 16",
         "changes": [{"find": "GRAD_ACCUM_STEPS = 8",
                      "replace": "GRAD_ACCUM_STEPS = 16"}]},
        {"description": "Reduce warmup ratio from 0.05 to 0.03",
         "changes": [{"find": "WARMUP_RATIO     = 0.05",
                      "replace": "WARMUP_RATIO     = 0.03"}]},
    ],
    "lora": [
        {"description": "Increase LoRA rank from 8 to 16",
         "changes": [{"find": "LORA_R              = 8",
                      "replace": "LORA_R              = 16"}]},
        {"description": "Add k_proj to LoRA target modules",
         "changes": [{"find": 'LORA_TARGET_MODULES = ["q_proj", "v_proj"]',
                      "replace": 'LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj"]'}]},
        {"description": "Increase learning rate from 1e-4 to 2e-4",
         "changes": [{"find": "LEARNING_RATE    = 1e-4",
                      "replace": "LEARNING_RATE    = 2e-4"}]},
    ],
}


def _mock_llm_change(loop_idx: int, mode: str, target_content: str) -> dict:
    candidates = _MOCK_CHANGES.get(mode, [])
    idea = candidates[(loop_idx - 1) % max(1, len(candidates))]
    if idea["changes"] and idea["changes"][0]["find"] not in target_content:
        return {"description": f"loop {loop_idx}: mock (find not found)", "changes": []}
    return idea


# ---------------------------------------------------------------------------
# One experiment iteration
# ---------------------------------------------------------------------------

def _log_path(log_dir: str, label: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"run_{label}_{ts}.log")


def run_experiment(loop_idx: int, n_loops: int, mode: str,
                   cfg: dict, cc, results_tsv_path: str,
                   dry_run: bool, smoke_test: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"[pipeline] Loop {loop_idx}/{n_loops}  mode={mode}")
    print(f"{'='*60}")

    target_file    = MODE_CONFIG[mode]["target_file"]
    target_content = read_file(target_file)
    results_str    = read_results_tsv(results_tsv_path)

    # ── Ask Claude ──────────────────────────────────────────────────────────
    print(f"  [llm] Asking Claude for modification idea ...")
    t0 = time.time()
    try:
        if dry_run:
            proposal = _mock_llm_change(loop_idx, mode, target_content)
            print(f"  [dry-run] Mocked: {proposal['description']}")
        else:
            prompt   = build_prompt(target_content, results_str, loop_idx, n_loops, mode)
            response = cc.ask(prompt)
            print(f"  [llm] Response in {time.time()-t0:.1f}s ({len(response)} chars)")
            proposal = parse_llm_response(response)
    except (ValueError, Exception) as e:
        print(f"  [llm] Error: {e}")
        append_result(results_tsv_path, git_short_hash(), 0.0, 0.0, "crash",
                      f"loop {loop_idx}: LLM failed — {str(e)[:60]}")
        return

    description = proposal["description"]
    print(f"  [idea] {description}")

    # ── Apply changes ───────────────────────────────────────────────────────
    modified, warnings = apply_changes(target_content, proposal["changes"])
    for w in warnings:
        print(w)
    if modified == target_content:
        print("  [apply] No changes applied. Skipping.")
        append_result(results_tsv_path, git_short_hash(), 0.0, 0.0, "crash",
                      f"loop {loop_idx}: no changes — {description}")
        return
    write_file(target_file, modified)
    print(f"  [apply] Applied {len(proposal['changes'])} change(s) to {target_file}")

    # ── Git commit ──────────────────────────────────────────────────────────
    commit_hash = git_commit_target(
        target_file, f"autoresearch loop {loop_idx}: {description}"
    )
    print(f"  [git] Committed: {commit_hash}")

    # ── Run training ────────────────────────────────────────────────────────
    log_path   = _log_path(cfg["log_dir"], loop_idx)
    metric_val, memory_gb = run_training(mode, target_file, cfg, log_path,
                                         dry_run, smoke_test)

    # ── Decide keep / discard ───────────────────────────────────────────────
    best_metric = _read_best_metric(results_tsv_path)
    crashed     = (metric_val == 0.0)

    if crashed:
        status = "crash"
        print(f"  [result] CRASH — reverting {target_file}")
        git_revert_last_commit()
    elif metric_val < best_metric:
        status = "keep"
        print(f"  [result] KEEP  metric={metric_val:.6f} < best={best_metric:.6f} "
              f"(Δ={best_metric - metric_val:.6f})")
    else:
        status = "discard"
        print(f"  [result] DISCARD metric={metric_val:.6f} >= best={best_metric:.6f}")
        git_revert_last_commit()

    append_result(results_tsv_path, commit_hash, metric_val, memory_gb,
                  status, description)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch pipeline")
    parser.add_argument("--config",  default="pipeline_config.yml",
                        help="Path to YAML or JSON config file")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Skip actual training (mock metrics)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Patch SMOKE_TEST=True in training script: "
                             "4 samples, 1 epoch, ~2 min on CPU")
    parser.add_argument("--n-loops", type=int, default=None,
                        help="Override n_loops from config")
    parser.add_argument("--mode",    choices=["sft", "lora"], default=None,
                        help="Override mode from config (sft/lora)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.smoke_test:
        cfg["smoke_test"] = True
    if args.n_loops is not None:
        cfg["n_loops"] = args.n_loops
    if args.mode is not None:
        cfg["mode"] = args.mode

    mode       = cfg["mode"]
    dry_run    = cfg["dry_run"]
    n_loops    = cfg["n_loops"]
    smoke_test = cfg.get("smoke_test", False)

    if mode not in MODE_CONFIG:
        print(f"[pipeline] Unknown mode: {mode!r}. Must be one of {list(MODE_CONFIG)}")
        sys.exit(1)

    mc = MODE_CONFIG[mode]
    print(f"[pipeline] Starting autoresearch loop")
    print(f"  config:      {args.config}")
    print(f"  mode:        {mode}  ({mc['target_file']})")
    print(f"  metric:      {mc['metric_key']} (lower is better)")
    print(f"  n_loops:     {n_loops}")
    print(f"  dry_run:     {dry_run}")
    print(f"  budget:      {cfg['time_budget_hours']}h per run")
    if smoke_test:
        print(f"  smoke_test:  True  (4 samples, 1 epoch, ~2 min)")

    cc = build_llm(cfg)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    results_path = cfg["results_file"]
    init_results_tsv(results_path)

    # ── Baseline run ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] Baseline run  ({mc['target_file']})")
    print(f"{'='*60}")

    baseline_log = _log_path(cfg["log_dir"], "baseline")
    metric_val, memory_gb = run_training(mode, mc["target_file"], cfg,
                                         baseline_log, dry_run, smoke_test)
    baseline_hash = git_short_hash()

    if metric_val > 0:
        append_result(results_path, baseline_hash, metric_val, memory_gb,
                      "keep", "baseline")
        print(f"  [baseline] {mc['metric_key']}={metric_val:.6f}  "
              f"memory={memory_gb:.1f}GB")
    else:
        print("  [baseline] Baseline run crashed — cannot proceed.")
        sys.exit(1)

    # ── Experiment loop ───────────────────────────────────────────────────
    for i in range(1, n_loops + 1):
        run_experiment(
            loop_idx=i, n_loops=n_loops, mode=mode,
            cfg=cfg, cc=cc,
            results_tsv_path=results_path,
            dry_run=dry_run,
            smoke_test=smoke_test,
        )

    # ── Final analysis ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] All {n_loops} loops done. Running analysis ...")
    print(f"{'='*60}")
    try:
        subprocess.run(
            [sys.executable, "analysis.py", "--results", results_path],
            check=False,
        )
    except Exception as e:
        print(f"  [analysis] Could not run analysis.py: {e}")

    print(f"\n[pipeline] Done. Results in {results_path}")


if __name__ == "__main__":
    main()
