import os
import math
from pathlib import Path

import pandas as pd
import streamlit as st
import re

from core.tokenizer import estimate_tokens
from core.pricing import DEFAULT_PRICING, estimate_cost
from core.glossary import (
    load_glossary_csv, build_patterns, find_hits,
    normalize_glossary_df, merge_corpora,
)
from core.rules import parse_rules_yaml
from core.prompts import build_system_prompt
from core.qc import quick_qc
from core.translator import Translator
from core.batching import chapters_from_folder, build_batch_jsonl

# ===== 分段翻译辅助函数 =====
def _normalize_newlines(text: str) -> str:
    # 统一换行，避免 Windows/Mac 不同换行导致分割异常
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _safe_paragraphs(text: str) -> list[str]:
    """
    优先按“≥2个换行”切段；若切不动，再退化为按句号/问号/感叹号切句；
    再不行，最后按固定长度兜底，避免任何空分隔符错误。
    """
    t = _normalize_newlines(text)
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p and p.strip()]
    if paras:
        return paras

    # 没有空行就按句子切
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s and s.strip()]
    if sents:
        return sents

    # 仍然切不出来（极端长段），按固定字符宽度兜底
    CHUNK = 1200
    t = t.strip()
    if not t:
        return []
    return [t[i:i+CHUNK] for i in range(0, len(t), CHUNK)]

def split_text_by_tokens(text: str, model: str, max_input_tokens: int = 6000):
    """
    近似按 tokens 切分：以“段/句/定长兜底”为单位累积，超过预算就切块。
    - max_input_tokens 要为系统提示与输出留余量；<1000 时自动抬到 1000 以防极端设置。
    """
    if not isinstance(text, str) or not text.strip():
        return
    budget = max(1000, int(max_input_tokens or 6000))

    units = _safe_paragraphs(text)
    buf, buf_tokens = [], 0
    for u in units:
        t = estimate_tokens(u, model)
        # 单个 unit 超过预算：直接作为独立块（避免死循环）
        if t >= budget:
            if buf:
                yield "\n\n".join(buf)
                buf, buf_tokens = [], 0
            yield u
            continue

        # 正常累积
        if buf and buf_tokens + t > budget:
            yield "\n\n".join(buf)
            buf, buf_tokens = [u], t
        else:
            buf.append(u)
            buf_tokens += t

    if buf:
        yield "\n\n".join(buf)

def translate_full_text(adapter, system_prompt: str, text: str, model: str,
                        max_input_tokens: int = 6000) -> str:
    """
    分段翻译整章并拼接；失败时抛异常，由上层 UI 捕获。
    """
    chunks = list(split_text_by_tokens(text, model, max_input_tokens=max_input_tokens))
    if not chunks:
        return ""
    outputs = []
    for i, ck in enumerate(chunks, 1):
        st.info(f"翻译分段 {i}/{len(chunks)}…")
        res = adapter.translate_once(system_prompt=system_prompt, user_text=ck)
        outputs.append((res.text or "").strip())
    return "\n\n".join([o for o in outputs if o]).strip()

st.set_page_config(page_title="Star Trek 翻译助手 · GUI & Batch", layout="wide")

# ===== Sidebar: 基本设置 =====
st.sidebar.header("设置")
model = st.sidebar.selectbox("选择模型", ["gpt-5-mini", "gpt-5"], index=0)
batch_mode = st.sidebar.checkbox("Batch API（约 -20% 成本）", value=True)
batch_discount = 0.20 if batch_mode else 0.0

# 仅会话内存储，不落盘
api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="仅当前会话使用；留空则尝试使用环境变量 OPENAI_API_KEY"
)

# 计价设置
pricing = {}
for m in ["gpt-5-mini", "gpt-5"]:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pi = st.number_input(
            f"{m} 输入($/1M)",
            value=float(DEFAULT_PRICING[m]["input"]),
            min_value=0.0,
            step=0.05,
            key=f"pi_{m}",
        )
    with col2:
        po = st.number_input(
            f"{m} 输出($/1M)",
            value=float(DEFAULT_PRICING[m]["output"]),
            min_value=0.0,
            step=0.10,
            key=f"po_{m}",
        )
    pricing[m] = {"input": pi, "output": po}

out_multiplier = st.sidebar.slider("输出token/输入比例", 1.05, 1.30, 1.10, 0.01)

st.title("🖖 星际迷航翻译助手：单章 GUI + 批量脚手架（多语料库版）")

# ===== Tabs =====
T1, T2, T3, T4 = st.tabs([
    "① 粘贴整章估价",
    "② 术语/多语料库管理",
    "③ 规则与系统提示",
    "④ 批处理/JSONL 生成",
])

# --- Tab1: 估价 ---
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

# --- Tab2: 多语料库管理 ---
with T2:
    st.subheader("多语料库（可扩展）：舰船/物种/职位/物品…")

    # 初始化会话态：{'base': df, 'ships': df, ...}
    if "corpora" not in st.session_state:
        sample_path = Path("data/glossary_sample.csv")
        fallback_df = (
            pd.read_csv(sample_path)
            if sample_path.exists()
            else pd.DataFrame([
                {"source":"U.S.S. Enterprise","target":"联邦星舰企业号","type":"ship","note":"舰名翻译"},
                {"source":"Enterprise","target":"企业号","type":"ship","note":"舰名翻译"},
                {"source":"U.S.S. Titan","target":"联邦星舰泰坦号","type":"ship","note":"舰名翻译"},
                {"source":"Titan","target":"泰坦号","type":"ship","note":"舰名翻译"},
                {"source":"U.S.S. Aventine","target":"联邦星舰安文婷号","type":"ship","note":"舰名翻译"},
                {"source":"Aventine","target":"安文婷号","type":"ship","note":"舰名翻译"},
                {"source":"Borg","target":"博格","type":"species","note":"物种"},
                {"source":"Borg drone","target":"博格个体","type":"species","note":"个体称谓（按物种归类）"},
                {"source":"Starfleet","target":"星际舰队","type":"org","note":""},
                {"source":"Captain","target":"上校","type":"rank","note":""},
                {"source":"Commander","target":"中校","type":"rank","note":""},
                {"source":"Lieutenant Commander","target":"少校","type":"rank","note":""},
                {"source":"Operations manager","target":"操作官","type":"role","note":""},
                {"source":"Security chief","target":"安全官","type":"role","note":""},
                {"source":"Flight controller","target":"舵手","type":"role","note":""},
                {"source":"Number One","target":"大副","type":"role","note":""},
                {"source":"Chief engineer","target":"轮机长","type":"role","note":""},
                {"source":"turbolift","target":"涡轮电梯","type":"item","note":""},
            ])
        )
        st.session_state["corpora"] = {
            "base": normalize_glossary_df(fallback_df, corpus_name="base")
        }

    corpora = st.session_state["corpora"]

    # 从目录批量导入 *.csv 为多个语料库
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

    # 单文件新增语料库（可多次上传）
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

    # 新建空白语料库
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

    # 选择并编辑某个语料库
    st.markdown("**编辑语料库**")
    corpus_names = sorted(corpora.keys())
    sel = st.selectbox("选择语料库", corpus_names, index=corpus_names.index("base") if "base" in corpus_names else 0)
    cur_df = corpora[sel].copy()
    cur_df = cur_df.drop(columns=["pattern"], errors="ignore")  # 展示时去掉编译列
    st.dataframe(cur_df, use_container_width=True, height=240)

    with st.expander("✍️ 就地编辑并保存", expanded=False):
        edited = st.data_editor(
            cur_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300,
            column_config={"source":"英文", "target":"中文", "type":"类别", "note":"备注", "corpus":"语料库"},
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

    # 合并视图
    merged = merge_corpora(corpora)
    st.markdown("**合并总览（仅展示，不含编译列）**")
    st.dataframe(merged.drop(columns=["pattern"], errors="ignore"), use_container_width=True, height=260)

    # 命中分析（基于合并语料库）
    if 'chapter_text' not in locals() or not chapter_text.strip():
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

# --- Tab3: 规则与系统提示 + 单次调用 ---
with T3:
    st.subheader("翻译规则 (YAML)")
    rules_path = Path("data/rules_sample.yaml")
    default_yaml = rules_path.read_text(encoding="utf-8") if rules_path.exists() else None
    rules_text = st.text_area("编辑/粘贴规则 YAML：", value=default_yaml, height=240)
    rules = parse_rules_yaml(rules_text)
    st.caption("这些规则将注入系统提示，强制人名不翻、舰名/军衔/职位/物种/物品等统一译名。")

    if 'chapter_text' in locals() and chapter_text.strip():
        corpora = st.session_state.get('corpora', {})
        merged = merge_corpora(corpora)
        gdf = build_patterns(merged.drop(columns=["pattern"], errors="ignore"))
        hits = find_hits(chapter_text, gdf)
        hit_terms = sorted({h.term for h in hits})
        glossary_subset = merged[merged['source'].isin(hit_terms)].copy() if hit_terms else merged.head(50).copy()

        # 人名由提示词自检测，不再传名单
        names = []

        sys_prompt = build_system_prompt(rules, glossary_subset.drop(columns=["pattern"], errors='ignore'), names)
        st.markdown("**系统提示（用于调用翻译）**")
        st.code(sys_prompt, language="json")

        # 轻量 QC
        rep = quick_qc(len(hits), len(names))
        with st.expander("快速质量检查"):
            st.json({"glossary_hits": rep.glossary_hits, "names_detected": rep.names_detected, "violations": rep.violations})

        # 单次调用（实际调用）
        temp = 1.0
        st.caption("temperature 已固定为 1（该模型仅支持默认值）。")
        max_toks = st.number_input("max_output_tokens(可选)", value=0, min_value=0, step=50, help="0 表示不限制")
        resp_fmt = st.selectbox("response_format", ["text", "json"], index=0, help="若选 json，将发送 JSON schema(简化示例)")
        response_format = None
        if resp_fmt == "json":
            response_format = {"type": "json_object"}

        disabled = not (api_key or os.getenv("OPENAI_API_KEY"))
        if st.button("试运行翻译", type="secondary", disabled=disabled):
            adapter = Translator(model, temperature=temp, max_output_tokens=(None if max_toks==0 else max_toks), response_format=response_format, api_key=api_key)
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

        # —— 整章翻译（自动分段） ——
        st.markdown("---")
        st.subheader("整章翻译（自动分段）")
        st.caption("按 token 预算自动切块，逐段调用并合并为全文；适用于长章节/整章。")
        max_in_budget = st.number_input(
            "每段最大输入 tokens（为系统提示与输出留余量）",
            value=6000, min_value=2000, max_value=24000, step=500
        )
        full_disabled = disabled or (not chapter_text.strip())
        if st.button("开始整章翻译", type="primary", disabled=full_disabled):
            adapter = Translator(
                model,
                temperature=1.0,
                max_output_tokens=(None if max_toks == 0 else max_toks),
                response_format=response_format,
                api_key=api_key,
            )
            try:
                with st.spinner("整章翻译进行中…"):
                    full_text = translate_full_text(
                        adapter, sys_prompt, chapter_text, model,
                        max_input_tokens=int(max_in_budget)
                    )
                st.success("整章翻译完成 ✅")
                st.text_area("全文译文（预览）", value=full_text, height=320)
                fname_full = st.text_input("导出文件名（全文）", value="chapter_translation_full.txt", key="full_fn")
                st.download_button("下载全文 TXT", full_text.encode("utf-8"), file_name=fname_full, mime="text/plain")
            except Exception as e:
                st.error(f"整章翻译失败：{e}")

# --- Tab4: 批处理 JSONL ---
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
            corpora = st.session_state.get('corpora', {})
            merged = merge_corpora(corpora)
            gdf = build_patterns(merged.drop(columns=["pattern"], errors="ignore"))
            rows = []
            for it in chs:
                text = it["text"]
                hits = find_hits(text, gdf)
                terms = sorted({h.term for h in hits})
                subset = merged[merged['source'].isin(terms)].copy() if terms else merged.head(50).copy()
                names = []  # 人名由提示词自检测

                system_prompt = build_system_prompt(rules, subset.drop(columns=["pattern"], errors="ignore"), names)
                rows.append({
                    "id": it["id"],
                    "system_prompt": system_prompt,
                    "user_text": text,
                })
            adapter = Translator(model, api_key=api_key)
            jsonl_rows = adapter.prepare_batch_items(rows)
            build_batch_jsonl(jsonl_rows, out_jsonl)
            st.success(f"已生成：{out_jsonl}")
            p = Path(out_jsonl)
            if p.exists():
                st.code(p.read_text(encoding='utf-8')[:1200] + "...", language="json")