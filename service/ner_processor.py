"""
ner_processor.py
======================================
NER 结果处理节点。

基于 tests/vector_recommend_tests/node_1_ner_processor.py，保留新版特性：
  - entity_type 字段（stockCN/stockHK/...），用于 sub_query_list 标签映射
  - sub_query_list 输出字段，供公告栏目多路向量检索使用

与 src/nodes/node_1_ner_processor.py 的差异：
  - 增加 entity_type 字段
  - 增加 sub_query_list 输出
  - 函数签名：main(ner_result, query="") 增加 query 参数

sub_query_list 标签映射规则：
  stockCN  → A股上市公司
  stockHK  → 港股上市公司
  stockUS  → 美股上市公司
  stockNTB → 北交所挂牌公司
  stockTW  → 台股上市公司
  stockFN  → 境外上市公司
  entity_type 为 enterprise 或未知值 → 不生成 sub_query（栏目不支持无市场区分的企业类型）
"""
import json
import re
from datetime import datetime

# ── 证券实体类型 → 公告栏目检索标签 ──────────────────────────────────────
_ENTITY_TYPE_LABEL: dict[str, str] = {
    "stockCN":    "A股上市公司",
    "stockHK":    "港股上市公司",
    "stockUS":    "美股上市公司",
    "stockNTB":   "北交所挂牌公司",
    "stockTW":    "台股上市公司",
    "stockFN":    "境外上市公司",
}

# ── 报告期类型关键词（按长度降序，优先匹配更完整的形式）────────────────────
_PERIOD_KEYWORDS: list[str] = [
    "第一季度", "第二季度", "第三季度", "第四季度",
    "一季度",   "二季度",   "三季度",   "四季度",
    "季度",
    "半年度",   "半年",
    "中期",
    "年度",
]


def _extract_period_keyword(raw: str) -> str:
    """若时间实体中包含报告期类型关键词，返回从该关键词起的子串；否则返回空字符串。

    示例：
      '2024年第三季度' → '第三季度'
      '2023年半年'     → '半年'
      '2024年中期'     → '中期'
      '2024年年度'     → '年度'
      '2024年'         → ''
      '近三年'         → ''
    """
    for kw in _PERIOD_KEYWORDS:
        idx = raw.find(kw)
        if idx != -1:
            return raw[idx:]
    return ""


def build_sub_query(
    query: str,
    ner_enterprise_list: list,
    ner_time_list: list,
    ner_person_list: list,
) -> str:
    """从原始问句中移除时间/人名/企业实体，将企业实体替换为市场分类标签，
    生成适合公告栏目向量检索的子查询。

    处理顺序：
      1. 企业名 → 替换为市场分类标签（同类型去重，后续出现直接删除）
      2. 时间实体 → 条件性删除（含报告期关键词则保留报告期部分）
      3. 人名实体 → 删除
      4. 清理多余连接词 / 空白
    """
    if not query:
        return query

    text = query

    # ── 1. 企业名 → 市场分类标签 ───────────────────────────────────────────
    seen_labels: set[str] = set()
    for ent in sorted(ner_enterprise_list, key=lambda x: len(x.get("entity") or x["name"]), reverse=True):
        match_text = ent.get("entity") or ent["name"]
        label = _ENTITY_TYPE_LABEL.get(ent.get("entity_type", ""))
        if match_text not in text:
            continue
        if label is None or label in text or label in seen_labels:
            text = text.replace(match_text, "", 1)
        else:
            text = text.replace(match_text, label, 1)
            seen_labels.add(label)

    # ── 2. 时间实体 → 条件性删除 ──────────────────────────────────────────
    for t in sorted(ner_time_list, key=lambda x: len(x["raw"]), reverse=True):
        raw = t["raw"]
        if raw not in text:
            continue
        period = _extract_period_keyword(raw)
        if period and period not in text.replace(raw, "", 1):
            text = text.replace(raw, period, 1)
        else:
            text = text.replace(raw, "", 1)

    # ── 3. 人名实体 → 删除 ─────────────────────────────────────────────────
    for p in sorted(ner_person_list, key=lambda x: len(x["name"]), reverse=True):
        pname = p["name"]
        if pname in text:
            text = text.replace(pname, "", 1)

    # ── 4. 清理 ────────────────────────────────────────────────────────────
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"^[的和与及、，,。？?！!：:]+", "", text)
    text = re.sub(r"[的和与及、，,。？?！!：:]+$", "", text)

    return text.strip()


def build_sub_query_list(
    query: str,
    ner_enterprise_list: list,
    ner_time_list: list,
    ner_person_list: list,
) -> list:
    """为每个唯一市场分类标签生成一条检索子问句，供 node_2 多路向量检索使用。

    每个唯一标签（如“港股上市公司”、“美股上市公司”）生成一条子问句：
      1. 将当前标签对应企业的原始实体文本替换为标签（首次出现），后续出现删除
      2. 删除其他标签的企业名称
      3. 时间实体按 build_sub_query 规则条件性删除
      4. 删除人名实体并清理多余连接词

    相同内容的子问句去重输出。

    Returns:
        list of {"label": str, "query": str}，无企业实体或 query 为空时返回 []
    """
    if not query or not ner_enterprise_list:
        return []

    # 按标签分组（无市场分类的 enterprise 类型跳过，不生成 sub_query）
    label_to_ents: dict = {}
    for ent in ner_enterprise_list:
        lbl = _ENTITY_TYPE_LABEL.get(ent.get("entity_type", ""))
        if lbl is None:
            continue
        if lbl not in label_to_ents:
            label_to_ents[lbl] = []
        label_to_ents[lbl].append(ent)

    results: list = []
    seen_queries: set = set()

    for label, label_ents in label_to_ents.items():
        text = query

        # ── Step 1: 当前标签的企业 → 第一个替换为标签，其余删除 ─────────────
        replaced = False
        for ent in sorted(label_ents, key=lambda x: len(x.get("entity") or x["name"]), reverse=True):
            match = ent.get("entity") or ent["name"]
            if match not in text:
                continue
            if not replaced:
                text = text.replace(match, label, 1)
                replaced = True
            else:
                text = text.replace(match, "", 1)

        # ── Step 2: 其他标签的企业（含无市场分类的 enterprise 类型）→ 全部删除 ─────
        other_ents = [
            e for e in ner_enterprise_list
            if _ENTITY_TYPE_LABEL.get(e.get("entity_type", "")) != label
        ]
        for ent in sorted(other_ents, key=lambda x: len(x.get("entity") or x["name"]), reverse=True):
            match = ent.get("entity") or ent["name"]
            if match in text:
                text = text.replace(match, "", 1)

        # ── Step 3: 时间实体 → 条件性删除（同 build_sub_query）────────────────────
        for t in sorted(ner_time_list, key=lambda x: len(x["raw"]), reverse=True):
            raw = t["raw"]
            if raw not in text:
                continue
            period = _extract_period_keyword(raw)
            if period and period not in text.replace(raw, "", 1):
                text = text.replace(raw, period, 1)
            else:
                text = text.replace(raw, "", 1)

        # ── Step 4: 人名实体 → 删除 ──────────────────────────────────────────────
        for p in sorted(ner_person_list, key=lambda x: len(x["name"]), reverse=True):
            if p["name"] in text:
                text = text.replace(p["name"], "", 1)

        # ── Step 5: 清理 ──────────────────────────────────────────────────────────
        text = re.sub(r"[\s\u3000]+", "", text)
        text = re.sub(r"^[的和与及、，,。？?!!：:]+", "", text)
        text = re.sub(r"[的和与及、，,。？?!!：:]+$", "", text)
        text = text.strip()

        if text and text not in seen_queries:
            seen_queries.add(text)
            results.append({"label": label, "query": text})

    return results


def main(ner_result: str, query: str = "") -> dict:
    """NER 结果处理入口。

    Args:
        ner_result: NER 服务返回的 JSON 字符串
        query:      用户原始问句（用于生成 sub_query_list）

    Returns:
        dict，各字段类型如下：
          current_date   (str)  : 当前日期，格式 YYYY-MM-DD
          ner_enterprise (list) : 企业实体列表，每项 {"name", "entity", "codes", "entity_type"}
          ner_time       (list) : 时间实体列表，每项 {"raw"}
          ner_person     (list) : 人名实体列表，每项 {"name"}
          reference      (list) : 其他命名实体原始文本列表
          sub_query_list (list) : 多路检索子问句列表，每项 {"label", "query"}

        地点与 excluded_types 企业（如 financeEnterprise）写入 reference，不再单独输出 location。
    """
    # ── BOM / 编码预处理 ───────────────────────────────────────────────────
    if isinstance(ner_result, (bytes, bytearray)):
        try:
            ner_result = ner_result.decode("utf-8-sig")
        except UnicodeDecodeError:
            ner_result = ner_result.decode("gbk", errors="replace")
    elif isinstance(ner_result, str) and ner_result.startswith("\ufeff"):
        ner_result = ner_result.lstrip("\ufeff")

    payload = json.loads(ner_result) if ner_result else {}
    data_list = payload.get("data", [])

    ner_enterprise_list: list = []
    ner_time_list:       list = []
    ner_person_list:     list = []

    ner_enterprise_set: list = []
    ner_time_set:       list = []
    ner_person_set:     list = []

    reference_set: set = set()

    excluded_ner_types = {"post", "code", "index", "product"}
    excluded_types = {
        "bond", "commodity", "bankWealthManage", "financeEnterprise", "bondIssuer",
        "insurance", "options", "nz", "module", "sector",
    }
    # 企业候选命中 excluded_types 时仅此类写入 reference（如城商行），非全部 excluded_types
    reference_excluded_enterprise_types = {"financeEnterprise"}

    for group in data_list:
        items = group if isinstance(group, list) else [group]
        for item in items:
            if not isinstance(item, dict):
                continue

            ner_type    = item.get("nerType")
            entity_type = item.get("type", "")
            entity_id   = item.get("id", "")
            entity_name = item.get("entity")

            # ── 企业实体 ──────────────────────────────────────────────────
            if ner_type == "enterprise" or entity_type in (
                "stockCN", "stockHK", "stockUS", "stockNTB", "stockTW", "stockFN", "enterprise"
            ):
                # 若存在 candidateEntities，遍历候选实体（多地上市情况）；否则以当前对象作为单元素列表处理
                candidate_entities = item.get("candidateEntities")
                candidates = candidate_entities if candidate_entities else [item]
                for candidate in candidates:
                    c_entity_type = candidate.get("type", "")
                    c_entity_id   = candidate.get("id", "")
                    c_entity_name = candidate.get("entity")
                    c_full_name   = candidate.get("fullName")
                    if not c_entity_name:
                        continue
                    if c_entity_type in excluded_types:
                        if c_entity_type in reference_excluded_enterprise_types:
                            if c_entity_name and not c_entity_name.isdigit():
                                reference_set.add(c_entity_name)
                            if c_full_name and str(c_full_name).strip() and not str(c_full_name).strip().isdigit():
                                reference_set.add(str(c_full_name).strip())
                        continue
                    codes = [c_entity_id]
                    if not c_entity_id or c_entity_id.isdigit():
                        codes = []
                    enterprise_info = {
                        "name":        c_full_name if c_full_name else c_entity_name,
                        "entity":      entity_name,     # ← 统一使用外层 entity（用户问句中的原始文本）
                        "codes":       codes,
                        "entity_type": c_entity_type,   # ← 供 sub_query 标签映射使用
                    }
                    # 去重键为 (name, entity_type)：同名但不同市场（如 A+H 双重上市）保留独立条目
                    _dedup_key = (enterprise_info["name"], enterprise_info["entity_type"])
                    if _dedup_key in ner_enterprise_set:
                        for existing in ner_enterprise_list:
                            if existing["name"] == enterprise_info["name"] and existing["entity_type"] == enterprise_info["entity_type"]:
                                existing["codes"].extend(enterprise_info["codes"])
                                existing["codes"] = list(set(existing["codes"]))
                                break
                    else:
                        ner_enterprise_set.append(_dedup_key)
                        ner_enterprise_list.append(enterprise_info)

            # ── 时间实体 ──────────────────────────────────────────────────
            elif ner_type == "time":
                if entity_name and entity_name not in ner_time_set:
                    ner_time_list.append({"raw": entity_name})
                    ner_time_set.append(entity_name)

            # ── 地点：不再单独建列表，写入 reference ─────────────────────
            elif ner_type == "location":
                if entity_name and not entity_name.isdigit():
                    reference_set.add(entity_name)
                if entity_id and not str(entity_id).isdigit():
                    reference_set.add(str(entity_id).strip())

            # ── 人名实体 ──────────────────────────────────────────────────
            elif ner_type == "person":
                if entity_name and entity_name not in ner_person_set:
                    ner_person_list.append({"name": entity_name})
                    ner_person_set.append(entity_name)

            # ── 排除项 ────────────────────────────────────────────────────
            elif ner_type in excluded_ner_types:
                continue
            elif entity_type in excluded_types:
                continue

            # ── 其他：加入 reference ──────────────────────────────────────
            else:
                if entity_name and not entity_name.isdigit():
                    reference_set.add(entity_name)
                if entity_id and not entity_id.isdigit():
                    reference_set.add(entity_id)

    reference = list(reference_set)

    # ── 生成多路检索列表 ─────────────────────────────────────────────────────────────
    sub_query_list_data = build_sub_query_list(query, ner_enterprise_list, ner_time_list, ner_person_list)

    return {
        "current_date":   datetime.now().strftime("%Y-%m-%d"),
        "ner_enterprise": ner_enterprise_list,
        "ner_time":       ner_time_list,
        "ner_person":     ner_person_list,
        "reference":      reference,
        "sub_query_list": sub_query_list_data,
    }

if __name__ == "__main__":
    from ner_client import call_ner
    query = "最近5年阿里巴巴的年报中关于战略的描述"
    ner_result = call_ner(query)
    output = main(ner_result, query)
    print(json.dumps(output, ensure_ascii=False, indent=2))
