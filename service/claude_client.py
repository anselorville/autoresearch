"""
Minimal Claude gateway client used by local analysis and automation.

This client talks to an internal gateway endpoint. Historically that gateway
usually returned Anthropic-style SSE events, but in practice it may also return:
- OpenAI-style SSE chunks
- a normal JSON completion payload

The parser below accepts all of those formats so callers do not need to care.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
import urllib.error
import urllib.request
from http.client import HTTPResponse
from typing import Any

log = logging.getLogger(__name__)

API_URL = "http://xx.xxx.xx.xxx/chat/completions"
API_MODEL = "xxxxxxx"
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
SOURCE = "xxxxxxx"

MAX_TOKENS = 8192
TEMPERATURE = 0.7
TIMEOUT_SEC = 300


class APIError(RuntimeError):
    """HTTP or gateway-level error."""


class ParseError(RuntimeError):
    """Response payload could not be parsed into text."""


def _extract_text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif item:
                parts.append(str(item))
        return "".join(parts)
    return ""


def _extract_text_from_payload(payload: dict[str, Any]) -> str:
    # Check gateway-level error status (status="0" means success)
    status = payload.get("status")
    if status is not None and str(status) not in ("0", "ok", "200", ""):
        msg = payload.get("message") or payload.get("msg") or json.dumps(payload, ensure_ascii=False)[:300]
        raise APIError(f"Gateway returned error: status={status}, detail={msg}")

    # Custom gateway wraps Anthropic response under a "body" key (may be dict or JSON-encoded string)
    body_inner = payload.get("body")
    if isinstance(body_inner, dict):
        text = _extract_text_from_payload(body_inner)
        if text:
            return text
    elif isinstance(body_inner, str) and body_inner.strip().startswith("{"):
        try:
            parsed_body = json.loads(body_inner)
            text = _extract_text_from_payload(parsed_body)
            if text:
                return text
        except json.JSONDecodeError:
            pass

    if (
        payload.get("type") == "content_block_delta"
        and isinstance(payload.get("delta"), dict)
        and payload["delta"].get("type") == "text_delta"
    ):
        return str(payload["delta"].get("text", "") or "")

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = first_choice.get("delta", {})
        if isinstance(delta, dict):
            text = _extract_text_from_content(delta.get("content"))
            if text:
                return text
        message = first_choice.get("message", {})
        if isinstance(message, dict):
            text = _extract_text_from_content(message.get("content"))
            if text:
                return text

    message = payload.get("message")
    if isinstance(message, dict):
        text = _extract_text_from_content(message.get("content"))
        if text:
            return text

    content = payload.get("content")
    if content is not None:
        text = _extract_text_from_content(content)
        if text:
            return text

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or json.dumps(error, ensure_ascii=False)
        raise APIError(f"Claude gateway returned error payload: {message}")

    return ""


def _parse_json_body(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Could not parse JSON response: {exc}\nbody={text[:300]!r}") from exc
    if not isinstance(payload, dict):
        raise ParseError(f"JSON response is not an object: {text[:200]!r}")
    extracted = _extract_text_from_payload(payload)
    if not extracted:
        snippet = text[:400].replace("\n", " ")
        log.error("[claude_client] Failed to extract text from payload: %s", snippet)
        raise ParseError(f"JSON response did not contain any text content\nbody={snippet!r}")
    return extracted


def _parse_sse_stream(response: HTTPResponse) -> str:
    text_buf = ""
    raw_buf = b""

    try:
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            raw_buf += chunk

            while True:
                idx_lf = raw_buf.find(b"\n\n")
                idx_crlf = raw_buf.find(b"\r\n\r\n")
                idx, sep = -1, 2
                if idx_lf != -1:
                    idx, sep = idx_lf, 2
                if idx_crlf != -1 and (idx == -1 or idx_crlf < idx):
                    idx, sep = idx_crlf, 4
                if idx == -1:
                    break

                block = raw_buf[:idx].decode("utf-8", errors="ignore")
                raw_buf = raw_buf[idx + sep :]

                data_lines = [
                    line[5:].lstrip()
                    for raw_line in block.strip().splitlines()
                    for line in [raw_line.strip()]
                    if line.startswith("data:")
                ]
                if not data_lines:
                    continue

                data_str = "\n".join(data_lines).strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    text_buf += _extract_text_from_payload(payload)

        if raw_buf.strip():
            remainder = raw_buf.decode("utf-8", errors="ignore").strip()
            if remainder.startswith("{") and remainder.endswith("}"):
                return text_buf + _parse_json_body(remainder)
            data_lines = [
                line[5:].lstrip()
                for raw_line in remainder.splitlines()
                for line in [raw_line.strip()]
                if line.startswith("data:")
            ]
            for data_str in data_lines:
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    text_buf += _extract_text_from_payload(payload)
    except (APIError, ParseError):
        raise
    except Exception as exc:
        raise ParseError(f"Unexpected error while parsing response stream: {exc}") from exc

    if not text_buf:
        raise ParseError(
            "SSE stream finished but no text content was extracted; response may be empty or event format mismatched"
        )
    return text_buf


def _parse_response(response: HTTPResponse) -> str:
    headers = getattr(response, "headers", {}) or {}
    content_type = ""
    if hasattr(headers, "get"):
        content_type = str(headers.get("Content-Type", "") or "")

    if "application/json" in content_type.lower():
        body = response.read().decode("utf-8", errors="ignore")
        return _parse_json_body(body)

    return _parse_sse_stream(response)


def ask(question: str) -> str:
    if not API_URL or not API_MODEL or not API_KEY:
        raise ValueError("Claude client configuration is incomplete")

    payload = json.dumps(
        {
            "body": {
                "model": API_MODEL,
                "maxTokens": MAX_TOKENS,
                "stream": False,
                "temperature": TEMPERATURE,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    }
                ],
            },
            "source": SOURCE,
            "PKey": API_KEY,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        url=API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as response:
            if response.status != 200:
                body = response.read().decode("utf-8", errors="ignore")
                raise APIError(f"Request failed: HTTP {response.status}\n{body}")
            return _parse_response(response)
    except APIError:
        raise
    except ParseError:
        raise
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise APIError(f"HTTP error {exc.code} {exc.reason}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise APIError(f"Network request failed: {exc.reason}") from exc


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def create(self, model: str = "", messages=None, **kwargs) -> _Completion:
        question = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                    break
        if not question:
            raise ValueError("messages does not contain a user message")
        content = ask(question)
        return _Completion(content)


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class ClaudeClientWrapper:
    """OpenAI-compatible wrapper around ask()."""

    def __init__(self, **kwargs) -> None:
        self.chat = _Chat()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "请用一句话介绍你自己。"
    print(f"[Q] {question}\n")
    try:
        answer = ask(question)
        print(f"[A]\n{answer}")
    except Exception:
        print("\n[ERROR] Claude call failed:\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
