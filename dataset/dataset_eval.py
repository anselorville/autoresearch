"""
金融查询槽位提取数据集质量评估脚本
====================================
针对给定批次目录（如 dataset/data/labeled/20260409_222413），
逐条核验 all.jsonl 中的每个样本，使用 Teacher 模型（Claude）判断是否正确：

- 输出 verdict: yes    → 样本通过
- 输出 verdict: false  → 样本有误，附带 correction 修正内容

最终将通过的样本（原始或修正后）保存到 fix.jsonl。

使用方式：
    # 针对指定批次目录评估
    python dataset/dataset_eval.py --batch-dir dataset/data/labeled/20260409_222413

    # 随机采样 10 条快速验证
    python dataset/dataset_eval.py --batch-dir dataset/data/labeled/20260409_222413 --sample 10

    # 指定 RPM 限流
    python dataset/dataset_eval.py --batch-dir dataset/data/labeled/20260409_222413 --llm-rpm 5

    # 强制重跑（不跳过已处理条目）
    python dataset/dataset_eval.py --batch-dir dataset/data/labeled/20260409_222413 --no-resume

输出（写入原批次目录）：
    {batch_dir}/fix.jsonl      通过 + 修正后的合格样本（output 已剥离 quality 行）
    {batch_dir}/eval_done.txt  已评估条目标识（用于断点续跑）
    {batch_dir}/eval_stat.txt  评估统计摘要
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

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from service.claude_client import ask

log = logging.getLogger(__name__)

_EVAL_PROMPT_PATH = Path(__file__).parent / "prompts" / "eval_quality.md"


# ── RPM 限流器（复用 dataset_builder 相同策略）────────────────────────────────
class RateLimiter:
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


# ── Prompt 加载 ────────────────────────────────────────────────────────────────
def _load_eval_prompt() -> str:
    if not _EVAL_PROMPT_PATH.exists():
        raise FileNotFoundError(f"评估 prompt 文件不存在: {_EVAL_PROMPT_PATH}")
    return _EVAL_PROMPT_PATH.read_text(encoding="utf-8")


# ── 构建评估输入 ───────────────────────────────────────────────────────────────
def _build_eval_input(sample: dict) -> str:
    """
    将样本的 input / cot / output 三段组装为供 Teacher 审核的文本。
    """
    model_input  = sample.get("input",  "").strip()
    model_cot    = sample.get("cot",    "").strip()
    model_output = sample.get("output", "").strip()
    return f"[INPUT]\n{model_input}\n\n[COT]\n{model_cot}\n\n[OUTPUT]\n{model_output}"


# ── 解析评估响应 ───────────────────────────────────────────────────────────────
def _parse_eval_response(raw: str) -> tuple[str, dict | None]:
    """
    解析 Teacher 审核响应，提取 verdict 和可选的 correction 字典。

    返回 (verdict, correction)：
      - verdict: "yes" 或 "false"
      - correction: {"cot": str, "output": str}，仅 verdict=false 时有值

    correction 块格式：
        correction:
        cot:
        （修正后的推理链，多行）
        output:
        （修正后的槽位结果，多行）
    """
    raw = raw.strip()
    lines = raw.splitlines()

    # 提取 verdict
    verdict = "unknown"
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("verdict:"):
            val = stripped.split(":", 1)[1].strip()
            if val in ("yes", "false"):
                verdict = val
            break

    if verdict == "yes":
        return "yes", None

    if verdict == "false":
        # 定位 correction: 行
        corr_body_start: int | None = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("correction:"):
                corr_body_start = i + 1
                break

        if corr_body_start is None:
            log.warning("verdict=false 但未找到 correction 块，响应片段: %.200s", raw)
            return "false", None

        corr_lines = lines[corr_body_start:]

        # 在 correction 块中定位 cot: 和 output: 段标记
        cot_start = output_start = None
        for i, line in enumerate(corr_lines):
            s = line.strip().lower()
            if s == "cot:" and cot_start is None:
                cot_start = i + 1
            elif s == "output:" and output_start is None:
                output_start = i + 1

        corr_cot: str = ""
        corr_output: str = ""

        if cot_start is not None and output_start is not None:
            # cot 内容在 cot: 之后、output: 之前
            cot_end = output_start - 1  # output: 行本身
            corr_cot    = "\n".join(corr_lines[cot_start:cot_end]).strip()
            corr_output = "\n".join(corr_lines[output_start:]).strip()
        elif output_start is not None:
            # 只有 output: 段
            corr_output = "\n".join(corr_lines[output_start:]).strip()
        elif cot_start is not None:
            # 只有 cot: 段（罕见，兜底）
            corr_cot = "\n".join(corr_lines[cot_start:]).strip()
        else:
            # 降级：整个 correction 块视为 output
            corr_output = "\n".join(corr_lines).strip()

        if not corr_cot and not corr_output:
            log.warning("correction 块解析为空，响应片段: %.200s", raw)
            return "false", None

        return "false", {"cot": corr_cot, "output": corr_output}

    # 无法解析：降级视为 yes
    log.warning("无法解析 verdict，响应片段: %.200s", raw[:200])
    return "yes", None


# ── 生成样本唯一标识（基于 input 内容，与 done.txt 逻辑类似）────────────────
def _sample_key(sample: dict) -> str:
    """用 input 的前 200 字符作为样本唯一键（避免重复评估）。"""
    return sample.get("input", "")[:200]


# ── 单条样本评估 ───────────────────────────────────────────────────────────────
def eval_sample(
    sample: dict,
    eval_prompt: str,
    llm_rl: RateLimiter,
) -> dict | None:
    """
    对单条样本执行 Teacher 审核。

    返回已处理的样本字典（含 eval_verdict / correction 字段），
    失败时返回 None。
    """
    try:
        eval_input = _build_eval_input(sample)
        full_prompt = eval_prompt + "\n\n" + eval_input

        llm_rl.wait()
        response = ask(full_prompt)

        verdict, correction = _parse_eval_response(response)

        result = dict(sample)
        result["eval_verdict"] = verdict
        if verdict == "false" and correction:
            # correction 为 {"cot": str, "output": str}
            result["correction"] = correction
        return result

    except Exception as exc:
        log.error("样本评估失败 | input_preview=%.50s | err=%s",
                  sample.get("input", "")[:50], exc)
        return None


# ── 统计摘要写入 ───────────────────────────────────────────────────────────────
def _write_eval_stat(stat_path: Path, info: dict) -> None:
    start: datetime = info["start"]
    end:   datetime = info["end"]
    dur = int((end - start).total_seconds())
    m, s = divmod(dur, 60)

    W = 52
    sep = "─" * W

    def row(label: str, val: str) -> str:
        return f"  {label:<14} {val}"

    lines = [
        "=" * W,
        "  数据集评估统计摘要",
        sep,
        row("开始时间", start.strftime("%Y-%m-%d %H:%M:%S")),
        row("结束时间", end.strftime("%Y-%m-%d %H:%M:%S")),
        row("耗时", f"{m} 分 {s} 秒"),
        sep,
        row("批次目录", str(info["batch_dir"])),
        row("输入样本总数", f"{info['total_input']} 条"),
        sep,
        row("本轮评估", f"{info['evaluated']} 条"),
        row("  ├─ 通过(yes)", f"{info['pass_yes']} 条"),
        row("  ├─ 修正(false)", f"{info['pass_corrected']} 条"),
        row("  └─ 评估失败", f"{info['fail']} 条"),
        sep,
        row("写入 fix.jsonl", f"{info['written']} 条"),
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


# ── 主评估流程 ─────────────────────────────────────────────────────────────────
def eval_dataset(
    batch_dir: str | Path,
    llm_rpm:   int  = 10,
    resume:    bool = True,
    sample_n:  int | None = None,
) -> None:
    """
    主流程：
    - 读取 batch_dir/all.jsonl
    - 逐条调用 Teacher 审核（verdict: yes / false + correction）
    - 通过的样本（原始或修正后）写入 fix.jsonl
    - 支持断点续跑（eval_done.txt 记录已处理条目）
    """
    batch = Path(batch_dir)
    if not batch.is_dir():
        raise FileNotFoundError(f"批次目录不存在: {batch}")

    all_path = batch / "all.jsonl"
    if not all_path.exists():
        raise FileNotFoundError(f"找不到 all.jsonl: {all_path}")

    fix_path       = batch / "fix.jsonl"
    done_path      = batch / "eval_done.txt"
    stat_path      = batch / "eval_stat.txt"

    # ── stat.txt 实时日志 handler ─────────────────────────────────────────────
    _fh = logging.FileHandler(stat_path, encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(_fh)
    start_time = datetime.now()

    eval_prompt = _load_eval_prompt()
    llm_rl = RateLimiter(rpm=llm_rpm)

    # ── 加载 all.jsonl ─────────────────────────────────────────────────────────
    with open(all_path, encoding="utf-8") as f:
        all_samples: list[dict] = [json.loads(line) for line in f if line.strip()]
    log.info("加载 all.jsonl：%d 条样本", len(all_samples))

    # ── 断点续跑：读取已处理键集合 ────────────────────────────────────────────
    done_keys: set[str] = set()
    existing_fix: list[dict] = []

    if resume:
        if done_path.exists():
            done_keys = set(done_path.read_text(encoding="utf-8").splitlines())
            log.info("恢复模式：已处理 %d 条，跳过", len(done_keys))
        if fix_path.exists():
            with open(fix_path, encoding="utf-8") as f:
                existing_fix = [json.loads(line) for line in f if line.strip()]
            log.info("已有 fix.jsonl：%d 条", len(existing_fix))

    # ── 筛选待处理样本 ─────────────────────────────────────────────────────────
    pending = [s for s in all_samples if _sample_key(s) not in done_keys]

    if sample_n is not None:
        if sample_n > len(pending):
            log.warning("--sample %d 超过待处理量 %d，使用全量", sample_n, len(pending))
        else:
            pending = random.sample(pending, sample_n)
            log.info("--sample 模式：随机采样 %d 条", len(pending))

    total_pending = len(pending)
    log.info("待评估 %d 条（已跳过 %d 条）| 批次目录: %s",
             total_pending, len(done_keys), batch)

    pass_yes       = 0
    pass_corrected = 0
    fail_count     = 0

    with (
        open(fix_path,  "a", encoding="utf-8") as f_fix,
        open(done_path, "a", encoding="utf-8") as f_done,
    ):
        for idx, sample in enumerate(pending):
            key = _sample_key(sample)
            result = eval_sample(sample, eval_prompt, llm_rl)

            if result is None:
                fail_count += 1
                flag = "✗"
            else:
                verdict = result.get("eval_verdict", "yes")

                corr: dict | None = result.get("correction")

                if verdict == "false" and corr:
                    # 用修正后的 cot/output 还原完整样本；
                    # 若 correction 中某段为空则回退到原始值
                    corr_cot    = corr.get("cot",    "").strip() or result.get("cot",    "")
                    corr_output = corr.get("output", "").strip() or result.get("output", "")
                    fix_record = {
                        "input":           result["input"],
                        "cot":             corr_cot,
                        "output":          corr_output,
                        "quality":         result.get("quality", "normal"),
                        "eval_verdict":    "corrected",
                        "original_cot":    result.get("cot",    ""),
                        "original_output": result.get("output", ""),
                    }
                    pass_corrected += 1
                    flag = "~"
                elif verdict == "false" and not corr:
                    # verdict=false 但 correction 解析失败 → 丢弃
                    fail_count += 1
                    flag = "✗"
                    f_done.write(key + "\n")
                    f_done.flush()
                    log.info(
                        "[%d/%d] %s | yes %d  corrected %d  fail %d | %s",
                        idx + 1, total_pending, flag,
                        pass_yes, pass_corrected, fail_count,
                        sample.get("input", "")[:50],
                    )
                    continue
                else:
                    # verdict=yes 或解析降级 → 原样写入
                    fix_record = {
                        "input":        result["input"],
                        "cot":          result.get("cot",    ""),
                        "output":       result.get("output", ""),
                        "quality":      result.get("quality", "normal"),
                        "eval_verdict": "yes",
                    }
                    pass_yes += 1
                    flag = "✓"

                f_fix.write(json.dumps(fix_record, ensure_ascii=False) + "\n")
                f_fix.flush()

            f_done.write(key + "\n")
            f_done.flush()

            log.info(
                "[%d/%d] %s | yes %d  corrected %d  fail %d | %s",
                idx + 1, total_pending, flag,
                pass_yes, pass_corrected, fail_count,
                sample.get("input", "")[:50],
            )

    total_written = pass_yes + pass_corrected + len(existing_fix)
    log.info(
        "评估完成 | 通过 %d / 修正 %d / 失败 %d / 共 %d | fix.jsonl 累计 %d 条",
        pass_yes, pass_corrected, fail_count, total_pending, total_written,
    )

    end_time = datetime.now()
    logging.getLogger().removeHandler(_fh)
    _fh.close()

    _write_eval_stat(stat_path, {
        "start":         start_time,
        "end":           end_time,
        "batch_dir":     batch,
        "total_input":   len(all_samples),
        "evaluated":     total_pending,
        "pass_yes":      pass_yes,
        "pass_corrected": pass_corrected,
        "fail":          fail_count,
        "written":       total_written,
    })
    log.info("评估摘要已写入 %s", stat_path)


# ── CLI 入口 ───────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="金融查询槽位提取数据集质量评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--batch-dir", required=True,
        help="批次目录路径，如 dataset/data/labeled/20260409_222413",
    )
    p.add_argument(
        "--sample", nargs="?", const=10, type=int, metavar="N",
        help="随机采样 N 条评估（不指定 N 时默认 10 条），用于快速验证",
    )
    p.add_argument("--llm-rpm",   type=int,  default=30,   help="Teacher LLM RPM 上限")
    p.add_argument(
        "--no-resume", action="store_true",
        help="禁用断点续跑（从头重新评估所有样本）",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        log.error("批次目录不存在: %s", batch_dir)
        sys.exit(1)

    eval_dataset(
        batch_dir = batch_dir,
        llm_rpm   = args.llm_rpm,
        resume    = not args.no_resume,
        sample_n  = args.sample,
    )
