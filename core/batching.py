from __future__ import annotations
from pathlib import Path
from typing import Iterable
from .io_utils import write_jsonl


def chapters_from_folder(folder: str | Path, suffix: str = ".txt") -> list[dict]:
    folder = Path(folder)
    items = []
    for p in sorted(folder.glob(f"*{suffix}")):
        items.append({"id": p.stem, "text": p.read_text(encoding="utf-8")})
    return items


def build_batch_jsonl(rows: Iterable[dict], out_path: str | Path) -> None:
    write_jsonl(rows, out_path)