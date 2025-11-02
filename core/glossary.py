from __future__ import annotations
import regex as re
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class GlossaryHit:
    term: str
    span: tuple[int, int]
    target: str
    type: str | None = None


def load_glossary_csv(path: str | None, fallback: pd.DataFrame) -> pd.DataFrame:
    if path:
        try:
            return pd.read_csv(path)
        except Exception:
            return fallback.copy()
    return fallback.copy()


def build_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["source", "target"]).copy()
    df["pattern"] = df["source"].apply(lambda s: re.compile(rf"(?<!\\w){re.escape(s)}(?!\\w)", flags=re.IGNORECASE))
    return df


def find_hits(text: str, gdf: pd.DataFrame) -> List[GlossaryHit]:
    hits: List[GlossaryHit] = []
    for _, row in gdf.iterrows():
        for m in row["pattern"].finditer(text):
            hits.append(GlossaryHit(
                term=row["source"],
                span=(m.start(), m.end()),
                target=row["target"],
                type=row.get("type", None)
            ))
    return hits

NAME_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

def detect_names(text: str, glossary_df: pd.DataFrame) -> list[str]:
    glossary_sources = set(str(x).lower() for x in glossary_df["source"].tolist())
    cand: set[str] = set()
    for m in NAME_PATTERN.finditer(text):
        s = m.group(0).strip()
        if s.lower() not in glossary_sources and len(s.split()) <= 3:
            cand.add(s)
    return sorted(cand)