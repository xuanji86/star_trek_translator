# core/glossary.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Optional
import pandas as pd
import regex  # pip install regex

# 供外部复用的必需列
REQUIRED_COLS = ["source", "target", "type", "note"]

# --------------------------
# 数据类：内存中的编译项与命中项
# --------------------------

@dataclass
class GlossaryPattern:
    term: str              # 原文术语（source）
    target: str            # 译名（target）
    type: str              # 类别（ship/species/role/rank/item/org/place/tech…）
    note: str              # 备注
    corpus: str            # 语料库名（来源）
    pattern: object        # 编译好的 regex.Pattern（仅内存使用，不写回 df）

@dataclass
class Hit:
    term: str
    target: str
    type: str
    corpus: str
    start: int
    end: int
    match: str

# --------------------------
# 基础工具
# --------------------------

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """确保 df 至少包含 source/target/type/note 列；去掉编译列；全转字符串并 strip。"""
    df = df.copy()
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""
    if "pattern" in df.columns:
        # 非序列化列，避免与 pyarrow 冲突
        df = df.drop(columns=["pattern"])
    for c in REQUIRED_COLS:
        df[c] = df[c].astype(str).fillna("").map(lambda x: x.strip())
    # 允许保留其他列（如 corpus），但放到后面
    ordered = REQUIRED_COLS + [c for c in df.columns if c not in REQUIRED_COLS]
    return df[ordered]

# --------------------------
# 对外：加载/规范化/合并
# --------------------------

def load_glossary_csv(uploaded_file, fallback_df: pd.DataFrame) -> pd.DataFrame:
    """优先使用上传文件，否则回退到 fallback；并做列规范化与清理。"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = fallback_df.copy()
    else:
        df = fallback_df.copy()
    df = _ensure_cols(df)
    # 兼容性：若存在 corpus 列，统一为字符串；若没有，外部可通过 normalize_glossary_df 补齐
    if "corpus" in df.columns:
        df["corpus"] = df["corpus"].astype(str).fillna("").map(lambda x: x.strip() or "base")
    return df

def normalize_glossary_df(df: pd.DataFrame, corpus_name: str = "base") -> pd.DataFrame:
    """
    规范化单个语料库 DataFrame：
      - 确保包含 source/target/type/note 列
      - 去掉不可序列化的 pattern 列
      - 统一为字符串并去除首尾空格
      - 增加 'corpus' 列记录来源库名
    """
    df = _ensure_cols(df)
    df["corpus"] = str(corpus_name)
    return df

def merge_corpora(corpora: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    合并多个语料库（dict[name] = df）：
      - 先规范化每个 df
      - 纵向合并，保留 'corpus' 来源
      - 对 (source,target,type,corpus) 进行去重
    """
    if not corpora:
        return pd.DataFrame(columns=REQUIRED_COLS + ["corpus"])
    frames: List[pd.DataFrame] = []
    for name, gdf in corpora.items():
        frames.append(normalize_glossary_df(gdf, corpus_name=name))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["source", "target", "type", "corpus"])
    return merged

# --------------------------
# 构建编译正则与命中查找
# --------------------------

def _make_token_pattern(term: str) -> regex.Pattern:
    """
    为术语生成稳健的正则：
      - 使用 (?<!\\w) 与 (?!\\w) 作为“词界”以避免误匹配子串；
      - 对术语本身做 regex.escape 以安全处理点号/连字符/空格等；
      - 大小写不敏感匹配（I），必要时可根据 type 决定是否区分大小写。
    """
    escaped = regex.escape(term)
    pat = rf"(?<!\w){escaped}(?!\w)"
    return regex.compile(pat, flags=regex.I | regex.V0)

def build_patterns(df: pd.DataFrame) -> List[GlossaryPattern]:
    """
    基于术语表构建内存中的编译项列表。
    注意：不往 df 写入 pattern 列，以避免后续导出/显示时报错。
    """
    df = _ensure_cols(df)
    # 若没有 corpus 列则默认 base
    corpus_col = df["corpus"] if "corpus" in df.columns else ["base"] * len(df)

    compiled: List[GlossaryPattern] = []
    for (src, tgt, tp, note, corpus) in zip(
        df["source"], df["target"], df["type"], df["note"], corpus_col
    ):
        src = (src or "").strip()
        if not src:
            continue
        try:
            pat = _make_token_pattern(src)
        except Exception:
            # 极端情况下编译失败则跳过该条
            continue
        compiled.append(
            GlossaryPattern(
                term=src,
                target=(tgt or "").strip(),
                type=(tp or "").strip(),
                note=(note or "").strip(),
                corpus=str(corpus or "base"),
                pattern=pat,
            )
        )
    return compiled

def find_hits(text: str, patterns: Iterable[GlossaryPattern]) -> List[Hit]:
    """
    在给定文本中查找所有术语命中，返回可序列化的 Hit 列表。
    """
    hits: List[Hit] = []
    if not text:
        return hits
    for gp in patterns:
        # 对每个术语找所有匹配
        for m in gp.pattern.finditer(text):
            hits.append(
                Hit(
                    term=gp.term,
                    target=gp.target,
                    type=gp.type,
                    corpus=gp.corpus,
                    start=m.start(),
                    end=m.end(),
                    match=m.group(0),
                )
            )
    # 可按 start 排序，便于显示
    hits.sort(key=lambda h: (h.start, h.end))
    return hits
