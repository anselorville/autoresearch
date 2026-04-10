"""
NER 服务客户端。

调用内部 NER 服务进行命名实体识别，返回原始 JSON 文本。
"""
from __future__ import annotations

import json
import logging
import os

import requests

NER_ENDPOINT = "http://xxxx.xxx.xxx/ner"
NER_TIMEOUT = 60


log = logging.getLogger(__name__)


def call_ner(query: str, timeout: int | None = None) -> str:
    """调用 NER 服务，返回命名实体识别结果的原始 JSON 文本。

    新接口地址: windIpNer，请求体结构:
        {"body": {"text": ..., "outType": "tuple"},
         "sessionID": ..., "source": "xxxxxxxxxxxx"}
    响应外层包含 message / status / xxxxxxxxx，
    本函数会解包 xxxxxxxxx 并返回其 JSON 文本，
    使下游解析器仍可直接读取 data 字段。

    Args:
        query:   用户问句。
        timeout: 请求超时秒数，None 时使用配置中的默认值。

    Returns:
        xxxxxxxxxx 内层的 JSON 文本（包含 data / succeed / cost_time 等字段）。

    Raises:
        requests.RequestException: 网络或连接异常。
        requests.HTTPError:        HTTP 状态码非 200。
        ValueError:                响应中缺少 windNerPlugInfo 字段。
    """
    endpoint = NER_ENDPOINT
    t = timeout if timeout is not None else NER_TIMEOUT
    headers = {"Content-Type": "application/json"}
    payload = {
        "body": {
            "text": query,
            "outType": "tuple",
        },
        "sessionID": "xxxxxxxxxxxxxxxxxxxx",
        "source": "xx.xxxx.xxx",
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=t)
    response.raise_for_status()

    # 解包外层 xxxxxx，使下游保持兼容
    outer = response.json()
    ner_info = outer.get("xxxxxxxxx")
    if ner_info is None:
        log.warning("NER 响应缺少 windNerPlugInfo，返回完整响应: %s", response.text[:200])
        return response.text.strip()
    return json.dumps(ner_info, ensure_ascii=False)

if __name__ == "__main__":
    query = "中信证券近期发布的茅台研报"
    print(call_ner(query))
