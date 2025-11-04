import os
import io
import re
import math
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI
from typing import Optional, Dict, List, Tuple  # 添加 Tuple

from core.tokenizer import estimate_tokens
from core.pricing import DEFAULT_PRICING, estimate_cost
from core.glossary import (
    load_glossary_csv,
    build_patterns,
    find_hits,
    normalize_glossary_df,
    merge_corpora,
)
from core.rules import parse_rules_yaml
from core.prompts import build_system_prompt
from core.qc import quick_qc
from core.translator import Translator
from core.batching import chapters_from_folder, build_batch_jsonl

# =============================
# 分段翻译辅助函数
# =============================

def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _safe_paragraphs(text: str) -> List[str]:
    t = _normalize_newlines(text)
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p and p.strip()]
    if paras:
        return paras
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s and s.strip()]
    if sents:
        return sents
    CHUNK = 1200
    t = t.strip()
    if not t:
        return []
    return [t[i : i + CHUNK] for i in range(0, len(t), CHUNK)]

def split_text_by_tokens(text: str, model: str, max_input_tokens: int = 6000):
    if not isinstance(text, str) or not text.strip():
        return
    budget = max(1000, int(max_input_tokens or 6000))
    units = _safe_paragraphs(text)
    buf: List[str] = []
    buf_tokens = 0
    for u in units:
        t = estimate_tokens(u, model)
        if t >= budget:
            if buf:
                yield "\n\n".join(buf)
                buf, buf_tokens = [], 0
            yield u
            continue
        if buf and buf_tokens + t > budget:
            yield "\n\n".join(buf)
            buf, buf_tokens = [u], t
        else:
            buf.append(u)
            buf_tokens += t
    if buf:
        yield "\n\n".join(buf)

def translate_full_text(adapter: Translator, system_prompt: str, text: str, model: str,
                        max_input_tokens: int = 6000) -> str:
    chunks = list(split_text_by_tokens(text, model, max_input_tokens=max_input_tokens))
    if not chunks:
        return ""
    outputs: List[str] = []
    for i, ck in enumerate(chunks, 1):
        st.info(f"翻译分段 {i}/{len(chunks)}…")
        res = adapter.translate_once(system_prompt=system_prompt, user_text=ck)
        outputs.append((res.text or "").strip())
    return "\n\n".join([o for o in outputs if o]).strip()

# =============================
# Batch API 辅助函数
# =============================

def _make_client_from_gui(api_key: Optional[str]) -> OpenAI:
    if api_key and api_key.strip():
        return OpenAI(api_key=api_key.strip())
    return OpenAI()

def _save_results_jsonl_bytes(client: OpenAI, file_id: str, out_path: Path) -> None:
    content: bytes = client.files.content(file_id).content
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)

def _parse_results_jsonl(jsonl_path: Path) -> List[dict]:
    results: List[dict] = []
    if not jsonl_path.exists():
        return results
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = __import__("json").loads(line)
        except Exception:
            continue
        cid = obj.get("custom_id")
        body = obj.get("response", {}).get("body", {})
        try:
            text = body["choices"][0]["message"]["content"]
        except Exception:
            text = ""
        results.append({"custom_id": cid, "text": text, "raw": body})
    return results

def _write_txt_outputs(results: List[dict], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for r in results:
        cid = (r.get("custom_id") or "unknown").replace("/", "_")
        p = out_dir / f"{cid}.txt"
        p.write_text(r.get("text") or "", encoding="utf-8")
        paths.append(p)
    return paths

def _zip_paths(paths: List[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.read()

# =============================
# 质检与修复辅助函数
# =============================

def _merge_corpora_for_check(corpora: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    try:
        return merge_corpora(corpora)
    except Exception:
        frames = []
        for name, df in (corpora or {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                tmp = df.copy()
                if "corpus" not in tmp.columns:
                    tmp["corpus"] = name
                frames.append(tmp)
        if not frames:
            return pd.DataFrame(columns=["source", "target", "type", "note", "corpus"])
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["source", "target", "type"], keep="first")
        return out

def _collect_rank_words(merged: pd.DataFrame) -> List[str]:
    if merged is None or merged.empty:
        return ["上校", "中校", "少校", "上尉", "中尉", "少尉", "舰长"]
    # 只取 type == rank 的 target
    try:
        ranks = merged[merged["type"] == "rank"]["target"].dropna().astype(str).tolist()
    except Exception:
        ranks = []
    ranks = sorted({r.strip() for r in ranks if r.strip()}, key=len, reverse=True)
    return ranks or ["上校", "中校", "少校", "上尉", "中尉", "少尉", "舰长"]

def _normalize_paragraphs(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    t = txt.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    lines = [ln.rstrip() for ln in t.split("\n")]
    fixed: List[str] = []
    for i, ln in enumerate(lines):
        fixed.append(ln)
        if ln.strip() and i + 1 < len(lines) and lines[i + 1].strip():
            fixed.append("")
    t = "\n".join(fixed)
    t = t.strip("\n")
    t = re.sub(r"(\n)\s*(\n)", "\n\n", t)
    return t

def _find_rank_order_issues(translated: str, rank_words: List[str]) -> List[Tuple[str, str, int, int]]:
    """检测到 “军衔 在前、名字 在后”的违例，返回 (rank, name, start, end)。"""
    if not translated:
        return []
    name_en = r"[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2}"
    rank_regex = r"(?:" + "|".join(map(re.escape, rank_words)) + r")"
    pat_wrong = re.compile(rf"({rank_regex})\s*({name_en})")
    issues: List[Tuple[str, str, int, int]] = []
    for m in pat_wrong.finditer(translated):
        rank, name = m.group(1), m.group(2)
        issues.append((rank, name, m.start(), m.end()))
    return issues

def _auto_fix_rank_order(translated: str, rank_words: List[str]) -> str:
    if not translated:
        return translated
    name_en = r"[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2}"
    rank_regex = r"(?:" + "|".join(map(re.escape, rank_words)) + r")"
    pat_wrong = re.compile(rf"\b({rank_regex})\s+({name_en})\b")
    return pat_wrong.sub(r"\2 \1", translated)

def _glossary_coverage(english: str, translated: str, merged: pd.DataFrame) -> pd.DataFrame:
    """
    基于英文原文命中项，检查：
    - hit_in_english: 该 source 是否出现在英文原文中
    - found_target_in_zh: 任一 target 是否出现在译文中
    - found_source_in_zh: 英文 source 本身是否仍残留在译文中（用于自动修复）
    """
    if not english or merged is None or merged.empty:
        return pd.DataFrame(columns=[
            "source", "target", "type",
            "hit_in_english", "found_target_in_zh", "found_source_in_zh",
        ])

    # 构建英文命中
    gdf = build_patterns(merged.drop(columns=["pattern"], errors="ignore"))
    hits = find_hits(english, gdf) or []
    dedup = {}
    for h in hits:
        if h.term not in dedup:
            dedup[h.term] = h

    records = []
    zh = translated or ""
    for term, _h in dedup.items():
        rows = merged[merged["source"] == term]
        if rows.empty:
            continue

        targets = []
        for _, r in rows.iterrows():
            tgt = str(r.get("target") or "").strip()
            if tgt:
                targets.append(tgt)

        # 译文中是否已经有任意一个 target
        any_target = False
        for tgt in targets:
            if tgt in zh:
                any_target = True
                break

        records.append({
            "source": term,
            "target": "; ".join(targets),
            "type": rows.iloc[0].get("type"),
            "hit_in_english": True,
            "found_target_in_zh": any_target,
            "found_source_in_zh": term in zh,
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(
            ["found_target_in_zh", "type", "source"],
            ascending=[True, True, True]
        )
    return df

def _apply_glossary_repairs(translated: str, cov_df: pd.DataFrame) -> str:
    """
    根据覆盖报告自动修复：
    - 条件：hit_in_english=True 且 found_target_in_zh=False 且 found_source_in_zh=True
    - 操作：把译文中的英文 source 替换为语料库中第一个 target
    """
    if translated is None:
        return ""
    if cov_df is None or cov_df.empty:
        return translated

    text = translated
    for _, row in cov_df.iterrows():
        try:
            src = str(row.get("source") or "").strip()
            targets = [t.strip() for t in str(row.get("target") or "").split(";") if t.strip()]
            if not src or not targets:
                continue

            # 只对「英文原文中出现过、译文中还残留 source、且没有 target」的条目做替换
            if not row.get("hit_in_english"):
                continue
            if row.get("found_target_in_zh"):
                continue
            if not row.get("found_source_in_zh"):
                continue

            first_target = targets[0]
            # 简单按子串替换；如需更严格可改成带词边界的正则
            pattern = re.compile(re.escape(src))
            text = pattern.sub(first_target, text)
        except Exception:
            # 防御性兜底，单条失败不影响整体
            continue

    return text
# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Star Trek 翻译助手 · GUI & Batch", layout="wide")

# Sidebar
st.sidebar.header("设置")
model = st.sidebar.selectbox("选择模型", ["gpt-5-mini", "gpt-5"], index=0)
batch_mode = st.sidebar.checkbox("Batch API（约 -20% 成本）", value=True)
batch_discount = 0.20 if batch_mode else 0.0
api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="仅当前会话使用；留空则尝试使用环境变量 OPENAI_API_KEY"
)

pricing: Dict[str, Dict[str, float]] = {}
for m in ["gpt-5-mini", "gpt-5"]:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pi = st.number_input(f"{m} 输入($/1M)", value=float(DEFAULT_PRICING[m]["input"]), min_value=0.0, step=0.05, key=f"pi_{m}")
    with col2:
        po = st.number_input(f"{m} 输出($/1M)", value=float(DEFAULT_PRICING[m]["output"]), min_value=0.0, step=0.10, key=f"po_{m}")
    pricing[m] = {"input": pi, "output": po}

out_multiplier = st.sidebar.slider("输出token/输入比例", 1.05, 1.30, 1.10, 0.01)

st.title("🖖 星际迷航翻译助手：单章 + 批量 + 质检")

# Tabs
T1, T2, T3, T4, T5 = st.tabs([
    "① 粘贴整章估价",
    "② 术语/多语料库管理",
    "③ 规则与系统提示",
    "④ 批处理/JSONL 生成",
    "⑤ 成品质检/修复",
])

# Tab1: 估价
with T1:
    st.subheader("整章英文内容")
    chapter_text = st.text_area("粘贴英文原文：", height=300, placeholder="Paste chapter text here…")
    if st.button("计算成本", type="primary"):
        if not chapter_text.strip():
            st.warning("请先粘贴文本。")
        else:
            in_tokens = estimate_tokens(chapter_text, model)
            out_tokens = math.ceil(in_tokens * out_multiplier)
            cb = estimate_cost(model, in_tokens, out_tokens, pricing, batch_discount)
            c1, c2, c3 = st.columns(3)
            c1.metric("输入 tokens", f"{cb.input_tokens:,}")
            c2.metric("输出 tokens(估)", f"{cb.output_tokens:,}")
            c3.metric("预计成本(USD)", f"{cb.total_cost:.2f}")
            st.caption(f"模型：{model} · Batch：{'ON' if batch_mode else 'OFF'} · 输出倍率：{out_multiplier:.2f}")

# Tab2: 多语料库管理
with T2:
    st.subheader("多语料库（可扩展）：舰船/物种/职位/物品…")

    if "corpora" not in st.session_state:
        sample_path = Path("data/glossary_sample.csv")
        fallback_df = (
            pd.read_csv(sample_path)
            if sample_path.exists()
            else pd.DataFrame([
                {"source": "U.S.S. Enterprise", "target": "联邦星舰企业号", "type": "ship", "note": "舰名翻译"},
                {"source": "Enterprise", "target": "企业号", "type": "ship", "note": "舰名翻译"},
                {"source": "U.S.S. Titan", "target": "联邦星舰泰坦号", "type": "ship", "note": "舰名翻译"},
                {"source": "Titan", "target": "泰坦号", "type": "ship", "note": "舰名翻译"},
                {"source": "U.S.S. Aventine", "target": "联邦星舰安文婷号", "type": "ship", "note": "舰名翻译"},
                {"source": "Aventine", "target": "安文婷号", "type": "ship", "note": "舰名翻译"},
                {"source": "Borg", "target": "博格", "type": "species", "note": "物种"},
                {"source": "Borg drone", "target": "博格个体", "type": "species", "note": "个体称谓（按物种归类）"},
                {"source": "Starfleet", "target": "星际舰队", "type": "org", "note": ""},
                {"source": "Captain", "target": "上校", "type": "rank", "note": ""},
                {"source": "Commander", "target": "中校", "type": "rank", "note": ""},
                {"source": "Lieutenant Commander", "target": "少校", "type": "rank", "note": ""},
                {"source": "Operations manager", "target": "操作官", "type": "role", "note": ""},
                {"source": "Security chief", "target": "安全官", "type": "role", "note": ""},
                {"source": "Flight controller", "target": "舵手", "type": "role", "note": ""},
                {"source": "Number One", "target": "大副", "type": "role", "note": ""},
                {"source": "Chief engineer", "target": "轮机长", "type": "role", "note": ""},
                {"source": "turbolift", "target": "涡轮电梯", "type": "item", "note": ""},
            ])
        )
        st.session_state["corpora"] = {"base": normalize_glossary_df(fallback_df, corpus_name="base")}

    corpora = st.session_state["corpora"]

    st.markdown("**从目录批量导入 CSV**（每个 CSV 视为一个语料库，文件名为语料库名）")
    colA, colB = st.columns([2, 1])
    with colA:
        corpora_dir = st.text_input("语料库目录", value=str(Path.cwd() / "data/corpora"))
    with colB:
        if st.button("扫描并导入目录"):
            p = Path(corpora_dir)
            if p.exists() and p.is_dir():
                count = 0
                for f in sorted(p.glob("*.csv")):
                    try:
                        df = pd.read_csv(f)
                        df = normalize_glossary_df(df, corpus_name=f.stem)
                        corpora[f.stem] = df
                        count += 1
                    except Exception as e:
                        st.warning(f"跳过 {f.name}: {e}")
                st.success(f"已导入 {count} 个语料库。")
            else:
                st.error("目录不存在。")

    st.markdown("**上传 CSV 新增语料库**（字段: source,target,type,note）")
    up_files = st.file_uploader("选择一个或多个 CSV", type=["csv"], accept_multiple_files=True)
    if up_files:
        for uf in up_files:
            try:
                df = pd.read_csv(uf)
                df = normalize_glossary_df(df, corpus_name=Path(uf.name).stem)
                corpora[Path(uf.name).stem] = df
            except Exception as e:
                st.warning(f"跳过 {uf.name}: {e}")
        st.success(f"已添加 {len(up_files)} 个语料库到会话。")

    with st.expander("➕ 新建空白语料库", expanded=False):
        new_name = st.text_input("语料库名称", placeholder="例如 ships/species/roles/items 或任意自定义")
        if st.button("创建空白语料库"):
            if not new_name.strip():
                st.warning("请输入名称。")
            elif new_name in corpora:
                st.warning("该名称已存在。")
            else:
                corpora[new_name] = normalize_glossary_df(
                    pd.DataFrame(columns=["source", "target", "type", "note"]),
                    corpus_name=new_name,
                )
                st.success(f"已创建：{new_name}")

    st.markdown("**编辑语料库**")
    corpus_names = sorted(corpora.keys())
    sel = st.selectbox("选择语料库", corpus_names, index=corpus_names.index("base") if "base" in corpus_names else 0)
    cur_df = corpora[sel].copy().drop(columns=["pattern"], errors="ignore")
    st.dataframe(cur_df, use_container_width=True, height=240)

    with st.expander("✍️ 就地编辑并保存", expanded=False):
        edited = st.data_editor(
            cur_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300,
            column_config={"source": "英文", "target": "中文", "type": "类别", "note": "备注", "corpus": "语料库"},
        )
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("保存到当前语料库"):
            edited = normalize_glossary_df(edited, corpus_name=sel)
            corpora[sel] = edited
            st.success("已保存到会话。")
        if c2.button("下载当前语料库 CSV"):
            st.download_button("点此下载", edited.to_csv(index=False).encode("utf-8"), file_name=f"{sel}.csv")
        if c3.button("删除该语料库"):
            if sel == "base":
                st.warning("基础语料库 base 不建议删除。")
            else:
                del corpora[sel]
                st.experimental_rerun()
        with c4:
            st.caption("type 常见: ship/species/role/rank/item/org/place/tech…")

    merged = merge_corpora(corpora)
    st.markdown("**合并总览（仅展示，不含编译列）**")
    st.dataframe(merged.drop(columns=["pattern"], errors="ignore"), use_container_width=True, height=260)

    if not chapter_text or not chapter_text.strip():
        st.info("Tab1 粘贴文本后，这里可做术语命中分析。")
    else:
        gdf = build_patterns(merged.drop(columns=["pattern"], errors="ignore"))
        hits = find_hits(chapter_text, gdf)
        if hits:
            hit_df = pd.DataFrame([h.__dict__ for h in hits])
            st.dataframe(hit_df, use_container_width=True, height=240)
            st.download_button("下载命中报告 CSV", hit_df.to_csv(index=False).encode("utf-8"), "glossary_hits.csv")
        else:
            st.caption("未检测到术语命中。")

# Tab3: 规则与系统提示 + 调用
with T3:
    st.subheader("翻译规则 (YAML)")
    rules_path = Path("data/rules_sample.yaml")
    default_yaml = rules_path.read_text(encoding="utf-8") if rules_path.exists() else None
    rules_text = st.text_area("编辑/粘贴规则 YAML：", value=default_yaml, height=240)
    rules = parse_rules_yaml(rules_text)
    st.caption("这些规则将注入系统提示，强制人名不翻、舰名/军衔/职位/物种/物品等统一译名。")

    if chapter_text and chapter_text.strip():
        corpora = st.session_state.get("corpora", {})
        merged = merge_corpora(corpora)
        gdf = build_patterns(merged.drop(columns=["pattern"], errors="ignore"))
        hits = find_hits(chapter_text, gdf)
        hit_terms = sorted({h.term for h in hits})
        glossary_subset = merged[merged["source"].isin(hit_terms)].copy() if hit_terms else merged.head(50).copy()
        names: List[str] = []  # 人名由提示词自检测
        sys_prompt = build_system_prompt(rules, glossary_subset.drop(columns=["pattern"], errors="ignore"), names)

        st.markdown("**系统提示（用于调用翻译）**")
        st.code(sys_prompt, language="json")

        rep = quick_qc(len(hits))
        with st.expander("快速质量检查"):
            st.json({"glossary_hits": rep.glossary_hits, "violations": rep.violations})

        temp = 1.0
        st.caption("temperature 已固定为 1（该模型仅支持默认值）。")
        max_toks = st.number_input("max_output_tokens(可选)", value=0, min_value=0, step=50, help="0 表示不限制")
        resp_fmt = st.selectbox("response_format", ["text", "json"], index=0, help="若选 json，将发送 JSON schema(简化示例)")
        response_format = None
        if resp_fmt == "json":
            response_format = {"type": "json_object"}

        disabled = not (api_key or os.getenv("OPENAI_API_KEY"))
        if st.button("试运行翻译", type="secondary", disabled=disabled):
            adapter = Translator(model, temperature=temp, max_output_tokens=(None if max_toks == 0 else max_toks), response_format=response_format, api_key=api_key)
            try:
                result = adapter.translate_once(system_prompt=sys_prompt, user_text=chapter_text[:4000])
                st.text_area("返回示例", value=result.text, height=220)
                fname = st.text_input("导出文件名", value="translation.txt")
                st.download_button("下载TXT", result.text.encode("utf-8"), file_name=fname, mime="text/plain")
                if result.meta.get("usage"):
                    st.caption(f"usage: {result.meta['usage']}")
            except Exception as e:
                st.error(f"调用失败：{e}")
        elif disabled:
            st.info("请在左侧输入 OpenAI API Key 或设置环境变量后再试。")

        st.markdown("---")
        st.subheader("整章翻译（自动分段）")
        st.caption("按 token 预算自动切块，逐段调用并合并为全文；适用于长章节/整章。")
        max_in_budget = st.number_input("每段最大输入 tokens（为系统提示与输出留余量）", value=6000, min_value=2000, max_value=24000, step=500)
        full_disabled = disabled or (not chapter_text.strip())
        if st.button("开始整章翻译", type="primary", disabled=full_disabled):
            adapter = Translator(model, temperature=1.0, max_output_tokens=(None if max_toks == 0 else max_toks), response_format=response_format, api_key=api_key)
            try:
                with st.spinner("整章翻译进行中…"):
                    full_text = translate_full_text(adapter, sys_prompt, chapter_text, model, max_input_tokens=int(max_in_budget))
                st.success("整章翻译完成 ✅")
                st.text_area("全文译文（预览）", value=full_text, height=320)
                fname_full = st.text_input("导出文件名（全文）", value="chapter_translation_full.txt", key="full_fn")
                st.download_button("下载全文 TXT", full_text.encode("utf-8"), file_name=fname_full, mime="text/plain")
            except Exception as e:
                st.error(f"整章翻译失败：{e}")

# Tab4: 批处理 JSONL + Batch 流程
with T4:
    st.subheader("批处理：从文件夹读取章节，生成 Batch JSONL")
    colA, colB = st.columns([2, 1])
    with colA:
        folder = st.text_input("章节文件夹路径（读取 *.txt）", value=str(Path.cwd() / "chapters"))
    with colB:
        out_jsonl = st.text_input("输出 JSONL 路径", value=str(Path.cwd() / "batch" / "requests.jsonl"))

    st.caption("流程：读取每个章节 → 合并语料库 → 生成系统提示（按命中子集；人名由提示词自检测）→ 组装为 /v1/chat/completions 的 JSONL 批处理文件。")

    if st.button("生成 JSONL"):
        chs = chapters_from_folder(folder)
        if not chs:
            st.error("未在该路径发现 .txt 章节文件。")
        else:
            corpora = st.session_state.get("corpora", {})
            merged = merge_corpora(corpora)
            gdf = build_patterns(merged.drop(columns=["pattern"], errors="ignore"))
            rows: List[dict] = []
            for it in chs:
                text = it["text"]
                hits = find_hits(text, gdf)
                terms = sorted({h.term for h in hits})
                subset = merged[merged["source"].isin(terms)].copy() if terms else merged.head(50).copy()
                names: List[str] = []  # 人名由提示词自检测
                system_prompt = build_system_prompt(rules, subset.drop(columns=["pattern"], errors="ignore"), names)
                rows.append({"id": it["id"], "system_prompt": system_prompt, "user_text": text})
            adapter = Translator(model, api_key=api_key)
            jsonl_rows = adapter.prepare_batch_items(rows)
            build_batch_jsonl(jsonl_rows, out_jsonl)
            st.success(f"已生成：{out_jsonl}")
            p = Path(out_jsonl)
            if p.exists():
                st.code(p.read_text(encoding="utf-8")[:1200] + "\n...", language="json")

    st.markdown("---")
    st.subheader("Batch API 运行：上传 → 创建任务 → 查询 → 下载结果")

    if "batch_state" not in st.session_state:
        st.session_state["batch_state"] = {"input_file_id": None, "batch_id": None, "output_file_id": None}
    bstate = st.session_state["batch_state"]

    disabled_batch = not (api_key or os.getenv("OPENAI_API_KEY"))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("① 上传 JSONL 并创建 Batch 任务", type="primary", disabled=disabled_batch):
            try:
                client = _make_client_from_gui(api_key)
                up = client.files.create(file=open(out_jsonl, "rb"), purpose="batch")
                bstate["input_file_id"] = up.id
                job = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
                bstate["batch_id"] = job.id
                st.success(f"已创建 Batch：{job.id}")
                st.json({"batch_id": job.id, "status": job.status, "input_file_id": up.id})
            except Exception as e:
                st.error(f"创建失败：{e}")
    with col2:
        if st.button("② 查询/刷新状态", disabled=disabled_batch or not bstate.get("batch_id")):
            try:
                client = _make_client_from_gui(api_key)
                job = client.batches.retrieve(bstate["batch_id"])
                st.info(f"状态：{job.status}")
                if getattr(job, "output_file_id", None):
                    bstate["output_file_id"] = job.output_file_id
                st.json({
                    "status": job.status,
                    "request_counts": getattr(job, "request_counts", None) and job.request_counts.model_dump() or {},
                    "created_at": getattr(job, "created_at", None),
                    "output_file_id": getattr(job, "output_file_id", None),
                })
            except Exception as e:
                st.error(f"查询失败：{e}")

    st.caption("当状态为 completed 时，可下载结果。结果可能乱序，请使用 custom_id 回绑章节。")

    col3, col4 = st.columns(2)
    with col3:
        default_results_path = str(Path.cwd() / "batch" / "results.jsonl")
        res_path = st.text_input("保存结果 JSONL 到：", value=default_results_path)
        if st.button("③ 下载结果 JSONL", disabled=disabled_batch or not bstate.get("output_file_id")):
            try:
                client = _make_client_from_gui(api_key)
                _save_results_jsonl_bytes(client, bstate["output_file_id"], Path(res_path))
                st.success(f"已保存：{res_path}")
                st.code(Path(res_path).read_text(encoding="utf-8")[:800] + "\n...", language="json")
            except Exception as e:
                st.error(f"下载失败：{e}")
    with col4:
        out_dir = st.text_input("按 custom_id 输出到目录：", value=str(Path.cwd() / "batch_outputs"))
        if st.button("④ 解析并导出 TXT", disabled=not Path(res_path).exists()):
            try:
                results = _parse_results_jsonl(Path(res_path))
                paths = _write_txt_outputs(results, Path(out_dir))
                zip_bytes = _zip_paths(paths)
                st.success(f"已写入 {len(paths)} 个章节到 {out_dir}")
                st.download_button("下载所有章节（ZIP）", data=zip_bytes, file_name="batch_outputs.zip")
            except Exception as e:
                st.error(f"解析失败：{e}")

# Tab5: 成品质检/修复
with T5:
    st.subheader("对已翻译 TXT 做一致性质检与快速修复")
    st.caption("检查点：① 术语覆盖（英文原文命中 → 译文是否包含 target）；② 军衔顺序（应为“名字 在前，军衔 在后”）；③ 段落空行（段落间至少一个空行）。")

    # 上传：使用 getvalue()，避免被 .read() 消耗
    colL, colR = st.columns(2)
    with colL:
        en_up = st.file_uploader("上传英文原文 TXT（用于术语命中）", type=["txt"], key="qc_en")
    with colR:
        zh_up = st.file_uploader("上传中文译文 TXT（待质检/修复）", type=["txt"], key="qc_zh")

    # 语料库合并 & 军衔词表
    corpora = st.session_state.get("corpora", {})
    merged = _merge_corpora_for_check(corpora)
    ranks = _collect_rank_words(merged)

    # 把上传内容缓存到 session_state
    if en_up:
        st.session_state["qc_en_text"] = en_up.getvalue().decode("utf-8", errors="ignore")
    if zh_up:
        st.session_state["qc_zh_text"] = zh_up.getvalue().decode("utf-8", errors="ignore")

    en_text = st.session_state.get("qc_en_text", "")
    zh_text = st.session_state.get("qc_zh_text", "")

    # 开始质检：写入 session_state 方便后续操作
    if st.button("开始质检", disabled=not (en_text and zh_text)):
        cov_df = _glossary_coverage(en_text, zh_text, merged)
        issues = _find_rank_order_issues(zh_text, ranks)
        fixed_para = _normalize_paragraphs(zh_text)

        st.session_state["qc_cov_df"] = cov_df
        st.session_state["qc_issues"] = issues
        st.session_state["qc_fixed_para"] = fixed_para
        st.session_state["qc_ranks"] = ranks

        st.success("质检完成（结果已缓存，可在下方查看/修复）。")

    # 从 session_state 取结果
    cov_df = st.session_state.get("qc_cov_df", None)
    issues = st.session_state.get("qc_issues", None)
    fixed_para = st.session_state.get("qc_fixed_para", None)
    ranks_cached = st.session_state.get("qc_ranks", ranks)

    # —— 术语覆盖报告 —— 
    st.markdown("### 术语覆盖报告（含自动修复信息）")
    if cov_df is None:
        st.info("请先点击“开始质检”。")
    else:
        if cov_df.empty:
            st.info("未识别到任何语料库命中，或语料库为空。")
        else:
            # 标出“需要修复”的条目：英文命中 + 译文无 target + 译文仍残留 source
            need_fix_mask = (
                (cov_df["hit_in_english"] == True) &
                (cov_df["found_target_in_zh"] == False) &
                (cov_df["found_source_in_zh"] == True)
            )
            cov_df_show = cov_df.copy()
            cov_df_show["need_auto_fix"] = need_fix_mask

            st.dataframe(cov_df_show, use_container_width=True, height=260)
            st.download_button(
                "下载覆盖报告 CSV",
                cov_df_show.to_csv(index=False).encode("utf-8"),
                file_name="coverage_report.csv",
            )

            n_need_fix = int(need_fix_mask.sum())
            if n_need_fix > 0:
                st.warning(f"有 {n_need_fix} 条术语在英文中出现，但译文本身仍留英文且未使用语料库 target，将在一键修复中自动替换。")
            else:
                st.success("未发现需要自动修复的术语（或无法判断）。")

    # —— 军衔顺序检查 —— 
    st.markdown("### 军衔顺序检查（应为“名字 在前，军衔 在后”）")
    if issues is None:
        st.info("请先点击“开始质检”。")
    else:
        if not issues:
            st.success("未发现军衔在前、名字在后的违例。")
        else:
            preview_rows = []
            for rk, nm, a, b in issues[:200]:
                snippet = zh_text[max(0, a - 20):min(len(zh_text), b + 20)].replace("\n", " ")
                preview_rows.append({"rank": rk, "name": nm, "context": snippet})
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, height=220)
            st.caption(f"共发现 {len(issues)} 处。")

    # —— 段落空行 —— 
    st.markdown("### 段落空行")
    if fixed_para is None:
        st.info("请先点击“开始质检”。")
    else:
        if fixed_para != zh_text:
            st.info("检测到段落空行问题，已生成修复版本（将在一键修复中使用）。")
        else:
            st.success("段落空行看起来正常。")

    st.markdown("---")
    st.subheader("一键修复（语料库术语 + 军衔顺序 + 段落空行）")

    st.checkbox("自动把 ‘军衔 名字’ 互换为 ‘名字 军衔’（仅英文名字场景）", key="qc_do_swap", value=False)
    fix_filename = st.text_input("保存文件名", value="translated_fixed.txt", key="qc_fix_filename")

    apply_disabled = fixed_para is None
    if st.button("应用修复并下载 TXT", disabled=apply_disabled):
        cov_df = st.session_state.get("qc_cov_df", None)
        fixed_para = st.session_state.get("qc_fixed_para", zh_text)
        issues = st.session_state.get("qc_issues", [])
        ranks_cached = st.session_state.get("qc_ranks", ranks)

        # 1) 先用语料库做术语修复（英文 source → 中文 target）
        out_txt = _apply_glossary_repairs(fixed_para or zh_text, cov_df)

        # 2) 可选：军衔顺序修复
        if st.session_state.get("qc_do_swap", False) and issues:
            out_txt = _auto_fix_rank_order(out_txt, ranks_cached)

        # 3) 下载
        st.download_button(
            "下载修复后 TXT",
            data=out_txt.encode("utf-8"),
            file_name=st.session_state.get("qc_fix_filename", "translated_fixed.txt"),
            mime="text/plain",
        )