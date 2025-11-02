from __future__ import annotations
import math, re
from typing import Optional

try:
    import tiktoken
except Exception:
    tiktoken = None

MODEL_ENCODINGS = {
    "gpt-5": "o200k_base",
    "gpt-5-mini": "o200k_base",
}


def estimate_tokens(text: str, model: str) -> int:
    """Prefer tiktoken; fallback to heuristic: ascii/4 + non-ascii."""
    if tiktoken is not None:
        enc_name = MODEL_ENCODINGS.get(model, "o200k_base")
        try:
            enc = tiktoken.get_encoding(enc_name)
            return len(enc.encode(text))
        except Exception:
            pass
    ascii_chars = len(re.findall(r"[ -~]", text))
    non_ascii = len(text) - ascii_chars
    return math.ceil(ascii_chars / 4) + non_ascii