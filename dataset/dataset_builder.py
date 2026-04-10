"""
金融查询槽位提取数据集构建 Pipeline
=====================================
Phase 1：原始 Query → NER 增强 → Teacher（Claude）打标 → JSONL 训练集

使用方式：
    # 读取 raw/ 下所有 raw_questions*.txt，全量生成
    python dataset/dataset_builder.py

    # 随机采样 10 条（快速验证）
    python dataset/dataset_builder.py --sample

    # 随机采样 N 条
    python dataset/dataset_builder.py --sample 20

    # 指定 raw 目录 / 限流
    python dataset/dataset_builder.py --raw-dir dataset/data/raw --ner-rpm 20 --llm-rpm 5

    # 强制重跑（不跳过已处理条目）
    python dataset/dataset_builder.py --no-resume

    # 1. 构建原始数据集
    python dataset/dataset_builder.py

    # 2. 用 Teacher 模型评估并修正
    python dataset/dataset_eval.py --batch-dir dataset/data/labeled/20260409_222413

    # 3. 从 fix.jsonl 重新切分 train/val
    python dataset/dataset_builder.py --from-fix dataset/data/labeled/20260409_222413

    # 自定义验证集比例
    python dataset/dataset_builder.py --from-fix dataset/data/labeled/20260409_222413 --val-ratio 0.15

输出（每次运行生成独立时间戳目录，不覆盖历史结果）：
    dataset/data/labeled/{YYYYMMDD_HHMMSS}/all.jsonl    全量，用于增量续跑
    dataset/data/labeled/{YYYYMMDD_HHMMSS}/done.txt     已处理 query
    dataset/data/labeled/{YYYYMMDD_HHMMSS}/train.jsonl
    dataset/data/labeled/{YYYYMMDD_HHMMSS}/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# 兼容 `python dataset/dataset_builder.py` 和 `python -m dataset.dataset_builder`
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from service.ner_client import call_ner
from service.ner_processor import main as process_ner
from service.claude_client import ask

log = logging.getLogger(__name__)

# ── 路径常量 ──────────────────────────────────────────────────────────────────
_PROMPT_PATH     = Path(__file__).parent / "prompts" / "slot_extraction.md"
_DEFAULT_RAW_DIR = Path(__file__).parent / "data" / "raw"
_DEFAULT_LABELED = Path(__file__).parent / "data" / "labeled"


def _load_all_raw_queries(raw_dir: Path) -> list[str]:
    """
    读取 raw_dir 下所有 raw_questions*.txt 文件，合并去重后返回 query 列表。
    文件名匹配规则：raw_questions.txt 和 raw_questions_*.txt（时间戳后缀）。
    """
    # 精确匹配：raw_questions.txt 和 raw_questions_{timestamp}.txt（下划线分隔）
    files = sorted(
        [raw_dir / "raw_questions.txt"]
        + list(raw_dir.glob("raw_questions_*.txt"))
    )
    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"raw 目录下未找到任何 raw_questions*.txt：{raw_dir}")

    seen:    set[str]  = set()
    queries: list[str] = []
    for f in files:
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        added = 0
        for q in lines:
            if q not in seen:
                seen.add(q)
                queries.append(q)
                added += 1
        log.info("加载 %s：%d 条（新增 %d 条）", f.name, len(lines), added)

    log.info("全部文件合并后共 %d 条唯一 query", len(queries))
    return queries


# ── RPM 限流器 ────────────────────────────────────────────────────────────────
class RateLimiter:
    """
    简单令牌桶：每次调用前等待到下一个可用时间槽。
    rpm=0 表示不限流（仅用于测试）。
    """

    def __init__(self, rpm: int) -> None:
        self._interval: float = (60.0 / rpm) if rpm > 0 else 0.0
        self._last: float = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        gap = self._interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


# ── Prompt 加载 ───────────────────────────────────────────────────────────────
def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Teacher prompt 文件不存在: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8")


# ── 输入格式构造 ──────────────────────────────────────────────────────────────
def _build_model_input(query: str, ner_ctx: dict) -> str:
    """
    将 query + NER 处理结果拼装为模型输入文本。

    ENT 格式：全称|WindCode...|entity_type（括号内为用户原文，供模型理解措辞）
    ner_ctx 各字段现在直接为 list（ner_processor.main 已返回结构化数据）。
    """
    enterprises: list[dict] = ner_ctx.get("ner_enterprise", [])
    times:       list[dict] = ner_ctx.get("ner_time",       [])
    persons:     list[dict] = ner_ctx.get("ner_person",     [])
    date: str               = ner_ctx.get("current_date", "")

    ent_parts: list[str] = []
    for e in enterprises:
        segs: list[str] = [e["name"]]
        segs.extend(e.get("codes") or [])
        if e.get("entity_type"):
            segs.append(e["entity_type"])
        token = "|".join(segs)
        # 若 NER 原文（用户实际写法）与标准全称不同，附注在括号内
        raw_text = e.get("entity", "")
        if raw_text and raw_text != e["name"]:
            token += f"({raw_text})"
        ent_parts.append(token)

    lines: list[str] = [f"[QUERY] {query}", f"[DATE]  {date}"]
    if ent_parts:
        lines.append("[ENT]   " + "  ".join(ent_parts))
    if times:
        lines.append("[TIME]  " + " ".join(t["raw"] for t in times))
    if persons:
        lines.append("[PERSON] " + " ".join(p["name"] for p in persons))

    return "\n".join(lines)


# ── 输出解析 ──────────────────────────────────────────────────────────────────
def _strip_code_fence(text: str) -> str:
    """移除 Teacher 有时添加的 markdown 代码块包裹（``` 或 ```text 等）。"""
    import re
    text = text.strip()
    # 匹配整体被 ``` 包裹的情况：开头 ```[可选语言标识]\n ... 结尾 ```
    text = re.sub(r"^```[^\n]*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_teacher_output(raw: str) -> tuple[str, str]:
    """
    从 Teacher 响应中分离 <think>…</think> 推理链和最终槽位输出。
    若无 think 标签则整体视为 label。
    自动剥离 label 外层的 markdown 代码块包裹。
    """
    raw = raw.strip()
    cot, label = "", raw
    if "<think>" in raw and "</think>" in raw:
        s = raw.index("<think>") + len("<think>")
        e = raw.index("</think>")
        cot   = raw[s:e].strip()
        label = raw[e + len("</think>"):].strip()
    label = _strip_code_fence(label)
    return cot, label


def _validate_label(label: str) -> bool:
    """最小有效性校验：label 必须包含 type 字段。"""
    return any(line.strip().startswith("type:") for line in label.splitlines())


def _extract_quality_from_label(label: str) -> tuple[str, str | None]:
    """
    从 label 文本中提取并移除 `quality:` 行。

    返回 (去掉 quality 行的 label, quality 值或 None)。
    quality 行不属于推理输出目标，必须从 output 中剥离，
    仅作为样本元数据保存在 all.jsonl 顶层字段。
    """
    kept: list[str] = []
    quality: str | None = None
    for line in label.splitlines():
        stripped = line.strip()
        if stripped.startswith("quality:"):
            quality = stripped.split(":", 1)[1].strip()
        else:
            kept.append(line)
    clean_label = "\n".join(kept).strip()
    return clean_label, quality


def _assess_quality(query: str, label: str) -> str:
    """
    根据 query 特征和 label 输出内容启发式评估样本质量。

    fine  — 语义完整、置信度全高、有效字段 ≥ 3 个
    hard  — query 过短/语义明显残缺，或 label 存在 low 置信度
    normal— 其余情况
    """
    # 解析 label 字段
    fields: dict[str, str] = {}
    for line in label.splitlines():
        line = line.strip()
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()

    # 统计已填充的业务字段数
    biz_keys   = {"type", "pub", "by", "when", "about"}
    field_count = sum(1 for k in fields if k in biz_keys)

    # 扫描 when 段中的置信度标记
    has_low  = False
    has_high = False
    when_val = fields.get("when", "")
    for segment in when_val.split(";"):
        parts = segment.strip().split("|")
        if len(parts) >= 3:
            conf = parts[2].strip().lower()
            if conf == "low":
                has_low = True
            elif conf == "high":
                has_high = True

    query_len = len(query.strip())

    if has_low or field_count <= 1 or query_len < 8:
        return "hard"
    if has_high and field_count >= 3 and not has_low:
        return "fine"
    return "normal"


def _stratified_split(
    samples:   list[dict],
    val_ratio: float,
) -> tuple[list[dict], list[dict]]:
    """
    按 quality 分层后各自按 val_ratio 切分，再合并 shuffle，
    保证 train / val 两个集合的质量分布基本一致。
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[s.get("quality", "normal")].append(s)

    train_all: list[dict] = []
    val_all:   list[dict] = []

    for quality in ("fine", "normal", "hard"):
        group = buckets.get(quality, [])
        if not group:
            continue
        random.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        val_all.extend(group[:n_val])
        train_all.extend(group[n_val:])
        log.info(
            "quality=%-6s  共 %3d 条  →  train %d / val %d",
            quality, len(group), len(group) - n_val, n_val,
        )

    random.shuffle(train_all)
    random.shuffle(val_all)
    return train_all, val_all


# ── 单条样本构建 ──────────────────────────────────────────────────────────────
def build_sample(
    query:      str,
    prompt:     str,
    ner_rl:     RateLimiter,
    llm_rl: RateLimiter,
) -> dict | None:
    """
    对单条 query 执行完整 pipeline：
      1. NER（带限流）
      2. Teacher 标注（带限流）
      3. 解析 + 校验

    返回 {"input": str, "cot": str, "output": str}，失败返回 None。
    """
    try:
        # Stage 1: NER 增强
        ner_rl.wait()
        ner_raw = call_ner(query)
        ner_ctx = process_ner(ner_raw, query)

        model_input = _build_model_input(query, ner_ctx)

        # 将当前输入拼到 prompt 末尾（prompt 末行已预留 "## 当前输入"）
        full_prompt = prompt + "\n\n" + model_input

        # Stage 2: Teacher 打标
        llm_rl.wait()
        response      = ask(full_prompt)
        cot, label    = _parse_teacher_output(response)

        if not _validate_label(label):
            log.warning(
                "teacher 输出无效（缺少 type 字段）| query=%.50s | preview=%.80s",
                query, label,
            )
            return None

        # 提取 quality 值（all.jsonl 的 output 保留 quality 行，供复查；
        # 写 train/val 时再剥离，使学生模型不学习输出 quality）
        _, quality_from_label = _extract_quality_from_label(label)
        quality = quality_from_label or _assess_quality(query, label)
        return {"input": model_input, "cot": cot, "output": label, "quality": quality}

    except Exception as exc:
        log.error("样本构建失败 | query=%.50s | err=%s", query, exc)
        return None


# ── stat.txt 摘要写入 ────────────────────────────────────────────────────────
def _write_stat_summary(stat_path: Path, info: dict) -> None:
    """
    将批次统计摘要写到 stat.txt 最顶部；原有日志内容（运行日志）追加在摘要之后。
    """
    start: datetime = info["start"]
    end:   datetime = info["end"]
    dur = int((end - start).total_seconds())
    m, s = divmod(dur, 60)

    W = 52  # 框宽
    sep = "─" * W

    def row(label: str, val: str) -> str:
        content = f"  {label:<12} {val}"
        return content

    lines = [
        "=" * W,
        "  批次生成统计摘要",
        sep,
        row("开始时间", start.strftime("%Y-%m-%d %H:%M:%S")),
        row("结束时间", end.strftime("%Y-%m-%d %H:%M:%S")),
        row("耗时", f"{m} 分 {s} 秒"),
        sep,
    ]

    raw_files: int = info.get("raw_files", 0)
    raw_total: int = info.get("raw_total", 0)
    sample_n:  int | None = info.get("sample_n")

    if raw_files:
        lines.append(row("原始文件数", f"{raw_files} 个"))
    if raw_total:
        lines.append(row("合并去重", f"{raw_total} 条"))
    if sample_n is not None:
        lines.append(row("运行模式", f"随机采样 {sample_n} 条"))
    else:
        lines.append(row("运行模式", "全量"))

    lines += [
        sep,
        row("本轮处理", f"{info['total']} 条"),
        row("  ├─ 成功", f"{info['new_ok']} 条"),
        row("  └─ 失败", f"{info['fail']} 条"),
        row("累计有效", f"{info['samples']} 条"),
        sep,
        row("输出目录", str(info["out_dir"])),
        "=" * W,
        "",
        sep,
        "  运行日志",
        sep,
        "",
    ]

    header = "\n".join(lines) + "\n"
    existing = stat_path.read_text(encoding="utf-8") if stat_path.exists() else ""
    stat_path.write_text(header + existing, encoding="utf-8")


# ── 批量构建 + 写盘 ───────────────────────────────────────────────────────────
def build_dataset(
    queries:     list[str],
    out_dir:     str | Path = _DEFAULT_LABELED,
    val_ratio:   float      = 0.1,
    ner_rpm:     int        = 30,
    llm_rpm:     int        = 10,
    resume:      bool       = True,
    stat_meta:   dict | None = None,
) -> None:
    """
    主流程：
    - resume=True 时跳过 done.txt 中已处理的 query，支持中断续跑
    - 边处理边追加写 all.jsonl 和 done.txt，避免中断丢失进度
    - 完成后按 val_ratio 切分写入 train.jsonl / val.jsonl
    - 完成后在 stat.txt 顶部写入批次统计摘要
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── stat.txt 实时日志 handler ─────────────────────────────────────────────
    stat_path = out / "stat.txt"
    _fh = logging.FileHandler(stat_path, encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(_fh)
    start_time = datetime.now()

    prompt  = _load_prompt()
    ner_rl  = RateLimiter(rpm=ner_rpm)
    llm_rl = RateLimiter(rpm=llm_rpm)

    # 加载已处理集合（resume 模式）
    done_path = out / "done.txt"
    all_path  = out / "all.jsonl"
    done: set[str] = set()
    samples: list[dict] = []

    if resume:
        if done_path.exists():
            done = set(done_path.read_text(encoding="utf-8").splitlines())
            log.info("恢复模式：跳过已处理 %d 条", len(done))
        if all_path.exists():
            with open(all_path, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
            log.info("已加载历史样本 %d 条", len(samples))

    pending = [q for q in queries if q not in done]
    total   = len(pending)
    log.info("待处理 %d 条（已跳过 %d 条）| 输出目录: %s", total, len(done), out)

    # resume 模式下记录历史基数，用于末尾汇总"本轮新增"
    _history_ok = len(samples)
    ok_count    = len(samples)   # 累计成功（含历史）
    fail_count  = 0              # 本轮失败

    # 增量写盘（每条处理后立即刷入，防止中断丢失）
    with (
        open(all_path,  "a", encoding="utf-8") as f_all,
        open(done_path, "a", encoding="utf-8") as f_done,
    ):
        for idx, query in enumerate(pending):
            sample = build_sample(query, prompt, ner_rl, llm_rl)
            if sample:
                ok_count += 1
                samples.append(sample)
                f_all.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_all.flush()
                flag = "✓"
            else:
                fail_count += 1
                flag = "✗"

            # 无论成功失败都记录为已处理，避免反复重试稳定失败的条目
            f_done.write(query + "\n")
            f_done.flush()

            log.info(
                "[%d/%d] %s | 成功 %d  失败 %d | %s",
                idx + 1, total, flag, ok_count, fail_count, query[:50],
            )

    new_ok = ok_count - _history_ok
    log.info(
        "处理完成 | 本轮: 成功 %d / 失败 %d / 共 %d | 累计有效样本 %d 条",
        new_ok, fail_count, total, len(samples),
    )

    # 分层切分并写入 train / val（按 quality 保持分布）
    # output 字段在写盘前剥离 quality 行，使学生模型不学习输出 quality
    train_data, val_data = _stratified_split(samples, val_ratio)
    for name, data in [("train", train_data), ("val", val_data)]:
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                clean_output, _ = _extract_quality_from_label(s["output"])
                record = {**s, "output": clean_output}
                record.pop("quality", None)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("写入 %s：%d 条", path, len(data))

    # ── 关闭日志 handler，写入 stat.txt 顶部统计摘要 ─────────────────────────
    end_time = datetime.now()
    logging.getLogger().removeHandler(_fh)
    _fh.close()

    _write_stat_summary(stat_path, {
        "start":   start_time,
        "end":     end_time,
        "out_dir": out,
        "total":   total,
        "new_ok":  new_ok,
        "fail":    fail_count,
        "samples": len(samples),
        **(stat_meta or {}),
    })
    log.info("统计摘要已写入 %s", stat_path)


# ── 从 fix.jsonl 重新切分 train/val ──────────────────────────────────────────
def resplit_from_fix(
    batch_dir: str | Path,
    val_ratio: float = 0.1,
) -> None:
    """
    读取 batch_dir/fix.jsonl，按 quality 分层重新切分，
    覆盖写入同目录下的 train.jsonl / val.jsonl。

    fix.jsonl 由 dataset_eval.py 生成，output 字段仍含 quality 行；
    写入 train/val 时与 build_dataset 保持一致：剥离 quality 行，
    并清除 eval_verdict / original_cot / original_output 等评估元字段，
    确保学生模型只看到 input / cot / output 三字段。
    """
    batch = Path(batch_dir)
    fix_path = batch / "fix.jsonl"
    if not fix_path.exists():
        raise FileNotFoundError(f"找不到 fix.jsonl: {fix_path}")

    with open(fix_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    log.info("从 fix.jsonl 加载 %d 条样本，目录: %s", len(samples), batch)

    if not samples:
        log.warning("fix.jsonl 为空，无内容可切分")
        return

    train_data, val_data = _stratified_split(samples, val_ratio)

    _STRIP_KEYS = {"quality", "eval_verdict", "original_cot", "original_output"}

    for name, data in [("train", train_data), ("val", val_data)]:
        path = batch / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                clean_output, _ = _extract_quality_from_label(s["output"])
                record = {k: v for k, v in s.items() if k not in _STRIP_KEYS}
                record["output"] = clean_output
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("写入 %s：%d 条", path, len(data))

    log.info(
        "resplit 完成 | fix=%d  train=%d  val=%d | 目录: %s",
        len(samples), len(train_data), len(val_data), batch,
    )


# ── CLI 入口 ──────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="金融查询槽位提取数据集构建 Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--from-fix", metavar="BATCH_DIR",
        help=(
            "跳过构建流程，直接读取指定批次目录的 fix.jsonl，"
            "按 --val-ratio 重新切分并覆盖写入 train.jsonl / val.jsonl"
        ),
    )
    p.add_argument(
        "--raw-dir", default=str(_DEFAULT_RAW_DIR),
        help="raw query 目录，读取其中所有 raw_questions*.txt",
    )
    p.add_argument(
        "--out-base", default=str(_DEFAULT_LABELED),
        help="输出根目录，每次运行在其下创建时间戳子目录",
    )
    p.add_argument(
        "--sample", nargs="?", const=10, type=int, metavar="N",
        help="随机采样 N 条运行（不指定 N 时默认 10 条），用于快速验证",
    )
    p.add_argument("--val-ratio",    type=float, default=0.1,  help="验证集比例")
    p.add_argument("--ner-rpm",      type=int,   default=20,   help="NER 服务 RPM 上限")
    p.add_argument("--llm-rpm",      type=int,   default=20,   help="Teacher LLM RPM 上限")
    p.add_argument(
        "--no-resume", action="store_true",
        help="禁用断点续跑（从头重新处理所有 query）",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # ── --from-fix 模式：直接从 fix.jsonl 重切分，不走 NER/LLM ──────────────
    if args.from_fix:
        batch_path = Path(args.from_fix)
        if not batch_path.is_dir():
            log.error("批次目录不存在: %s", batch_path)
            sys.exit(1)
        resplit_from_fix(batch_dir=batch_path, val_ratio=args.val_ratio)
        sys.exit(0)

    # ── 常规构建模式 ──────────────────────────────────────────────────────────
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        log.error("raw 目录不存在: %s", raw_dir)
        sys.exit(1)

    try:
        queries = _load_all_raw_queries(raw_dir)
    except FileNotFoundError as e:
        log.error("%s", e)
        sys.exit(1)

    raw_total = len(queries)
    raw_files = len(sorted(
        [raw_dir / "raw_questions.txt"] + list(raw_dir.glob("raw_questions_*.txt"))
    ))

    # ── 随机采样（--sample 模式）─────────────────────────────────────────────
    sample_n: int | None = None
    if args.sample is not None:
        n = args.sample
        if n <= 0:
            log.error("--sample 必须为正整数")
            sys.exit(1)
        if n > len(queries):
            log.warning("--sample %d 超过总量 %d，使用全量", n, len(queries))
        else:
            queries = random.sample(queries, n)
            sample_n = n
            log.info("--sample 模式：随机采样 %d 条", len(queries))

    # ── 每次运行创建独立时间戳输出目录 ────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_base) / ts
    log.info("输出目录: %s", out_dir)

    build_dataset(
        queries     = queries,
        out_dir     = out_dir,
        val_ratio   = args.val_ratio,
        ner_rpm     = args.ner_rpm,
        llm_rpm     = args.llm_rpm,
        resume      = not args.no_resume,
        stat_meta   = {
            "raw_files": raw_files,
            "raw_total": raw_total,
            "sample_n":  sample_n,
        },
    )
