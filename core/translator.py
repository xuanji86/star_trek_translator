from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable
import os
import backoff
from openai import OpenAI

@dataclass
class TranslationResult:
    text: str
    meta: Dict[str, Any]


def _make_client(api_key: Optional[str]) -> OpenAI:
    """优先使用 GUI 传入的 api_key；否则回退到环境变量 OPENAI_API_KEY。"""
    if api_key and api_key.strip():
        return OpenAI(api_key=api_key.strip())
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI()
    raise RuntimeError("未提供 API Key。请在 GUI 侧边栏输入或设置环境变量 OPENAI_API_KEY。")


class Translator:
    """Adapter for OpenAI Chat/Responses API 与 Batch JSONL 组装。
    - translate_once(): 单次调用（非流式）
    - prepare_batch_items(): 生成 Batch API JSONL 行
    """

    def __init__(self, model: str, temperature: float = 1.0, max_output_tokens: int | None = None,
                 response_format: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None):
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = max_output_tokens
        self.response_format = response_format  # e.g., {"type":"json_object"}
        self._client = _make_client(api_key)

    # 指数退避重试：429/5xx
    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def _chat(self, system_prompt: str, user_text: str):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
        # 仅当温度 != 1 时才传，避免某些模型只支持默认温度
        if abs(self.temperature - 1.0) > 1e-6:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens:
            # 兼容新版参数名
            kwargs["max_completion_tokens"] = int(self.max_output_tokens)
        if self.response_format:
            kwargs["response_format"] = self.response_format
        return self._client.chat.completions.create(**kwargs)

    def translate_once(self, system_prompt: str, user_text: str) -> TranslationResult:
        resp = self._chat(system_prompt, user_text)
        choice = resp.choices[0]
        content = choice.message.content or ""
        meta = {
            "model": self.model,
            "finish_reason": choice.finish_reason,
            "usage": getattr(resp, "usage", None) and resp.usage.model_dump() or {},
        }
        return TranslationResult(text=content, meta=meta)

    def prepare_batch_items(self, items: Iterable[dict]) -> list[dict]:
        """将单条请求转换为 Batch API JSONL 行（无需 API Key）。"""
        rows = []
        for it in items:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": it.get("system_prompt", "")},
                    {"role": "user", "content": it.get("user_text", "")},
                ],
            }
            if abs(self.temperature - 1.0) > 1e-6:
                body["temperature"] = self.temperature
            if self.max_output_tokens:
                body["max_completion_tokens"] = int(self.max_output_tokens)
            if self.response_format:
                body["response_format"] = self.response_format

            rows.append({
                "custom_id": it.get("id"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })
        return rows