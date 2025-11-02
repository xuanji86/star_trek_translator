import os
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from core.tokenizer import estimate_tokens
from core.pricing import DEFAULT_PRICING, estimate_cost
from core.glossary import load_glossary_csv, build_patterns, find_hits
from core.rules import parse_rules_yaml
from core.prompts import build_system_prompt
from core.qc import quick_qc
from core.translator import Translator
from core.batching import chapters_from_folder, build_batch_jsonl

st.set_page_config(page_title="Star Trek 翻译助手 · GUI & Batch", layout="wide")

# ===== Sidebar: 基本设置 =====
st.sidebar.header("设置")
model = st.sidebar.selectbox("选择模型", ["gpt-5-mini", "gpt-5"], index=0)
batch_mode = st.sidebar.checkbox("Batch API（约 -20% 成本）", value=True)
batch_discount = 0.20 if batch_mode else 0.0

# 仅会话内存储，不落盘
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="仅当前会话使用；留空则尝试使用环境变量 OPENAI_API_KEY")



# 计价设置
pricing = {}
for m in ["gpt-5-mini", "gpt-5"]:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pi = st.number_input(f"{m} 输入($/1M)", value=float(DEFAULT_PRICING[m]["input"]), min_value=0.0, step=0.05, key=f"pi_{m}")
    with col2:
        po = st.number_input(f"{m} 输出($/1M)", value=float(DEFAULT_PRICING[m]["output"]), min_value=0.0, step=0.10, key=f"po_{m}")
    pricing[m] = {"input": pi, "output": po}

out_multiplier = st.sidebar.slider("输出token/输入比例", 1.05, 1.30, 1.10, 0.01)

st.title("🖖 星际迷航翻译助手：单章 + 批量")

# ===== Tabs =====
T1, T2, T3, T4 = st.tabs(["① 粘贴整章估价", "② 术语与人名", "③ 规则与系统提示", "④ 批处理/JSONL 生成"])

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

# --- Tab2: 术语 & 人名（含编辑） ---
with T2:
    st.subheader("术语表 (CSV，列: source,target,type,note)")
    up = st.file_uploader("上传术语表 CSV（可选）", type=["csv"], key="csv_up")
    sample_path = Path("data/glossary_sample.csv")
    fallback_df = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame([
        {"source":"U.S.S. Enterprise","target":"联邦星舰企业号","type":"ship","note":"舰名翻译"},
        {"source":"Enterprise","target":"企业号","type":"ship","note":"舰名翻译"},
        {"source":"U.S.S. Titan","target":"联邦星舰泰坦号","type":"ship","note":"舰名翻译"},
        {"source":"Titan","target":"泰坦号","type":"ship","note":"舰名翻译"},
        {"source":"U.S.S. Aventine","target":"联邦星舰安文婷号","type":"ship","note":"舰名翻译"},
        {"source":"Aventine","target":"安文婷号","type":"ship","note":"舰名翻译"},
        {"source":"Borg","target":"博格","type":"species","note":"物种"},
        {"source":"Borg drone","target":"博格个体","type":"ship","note":""},
        {"source":"Starfleet","target":"星际舰队","type":"org","note":""},
        {"source":"Captain","target":"上校","type":"rank","note":""},
        {"source":"Commander","target":"中校","type":"rank","note":""},
        {"source":"Lieutenant Commander","target":"少校","type":"rank","note":""},
        {"source":"Operations manager","target":"操作官","type":"role","note":""},
        {"source":"Security chief","target":"安全官","type":"role","note":""},
        {"source":"Flight controller","target":"舵手","type":"role","note":""},
        {"source":"Number One","target":"大副","type":"role","note":""},
        {"source":"Chief engineer","target":"轮机长","type":"role","note":""},
        {"source":"turbolift ","target":"涡轮电梯","type":"item","note":""},
    ])
    glossary_df = load_glossary_csv(up, fallback_df)
    # 若用户已编辑，优先使用会话版本，并去掉不可序列化列
    if "glossary_df" in st.session_state:
        glossary_df = st.session_state["glossary_df"].copy()
    # 🧹 防御性清理：去掉编译后的正则列，避免 PyArrow 报错
    if "pattern" in glossary_df.columns:
        glossary_df = glossary_df.drop(columns=["pattern"]) 

    st.dataframe(glossary_df, use_container_width=True, height=220)

    # ✍️ 术语表编辑
    with st.expander("✍️ 编辑术语表（可增删改）", expanded=False):
        edited_gloss = st.data_editor(
            glossary_df,
            num_rows="dynamic",
            use_container_width=True,
            height=280,
            column_config={"source": "英文原文", "target": "中文译名", "type": "类别", "note": "备注"},
        )
        c1, c2, c3 = st.columns(3)
        if c1.button("保存术语表更改到会话"):
            # 保存前去掉编译后的正则列
            if "pattern" in edited_gloss.columns:
                edited_gloss = edited_gloss.drop(columns=["pattern"]) 
            st.session_state["glossary_df"] = edited_gloss
            st.success("已保存到会话。后续 Tabs 将使用更新后的术语表。")
        c2.download_button("下载当前术语表 CSV", edited_gloss.to_csv(index=False).encode("utf-8"), "glossary_edited.csv")
        c3.caption("建议字段：source,target,type,note；type 常见值：ship/org/rank/role/species/place/tech")

    if 'chapter_text' not in locals() or not chapter_text.strip():
        st.info("请先在 Tab1 粘贴文本并估价，以便进行命中分析。")
    else:
        glossary_current = st.session_state.get("glossary_df", glossary_df)
        # 构建匹配表时使用无 pattern 的副本
        glossary_slim = glossary_current.drop(columns=["pattern"], errors='ignore')
        gdf = build_patterns(glossary_slim)
        hits = find_hits(chapter_text, gdf)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**术语命中**")
            if hits:
                hit_df = pd.DataFrame([h.__dict__ for h in hits])
                st.dataframe(hit_df, use_container_width=True, height=260)
                st.download_button("下载命中报告 CSV", hit_df.to_csv(index=False).encode("utf-8"), "glossary_hits.csv")
            else:
                st.caption("未检测到术语命中。")
        with c2:
            st.write("**不翻译人名（提示词自检测）**")
            st.caption("本模式不再展示/编辑人名清单，模型将依据系统提示自动识别人名并保留英文原样。")

# --- Tab3: 规则与系统提示 + 单次调用 ---
with T3:
    st.subheader("翻译规则 (YAML)")
    rules_path = Path("data/rules_sample.yaml")
    default_yaml = rules_path.read_text(encoding="utf-8") if rules_path.exists() else None
    rules_text = st.text_area("编辑/粘贴规则 YAML：", value=default_yaml, height=240)
    rules = parse_rules_yaml(rules_text)
    st.caption("这些规则将注入系统提示，强制人名不翻、舰名/军衔映射、术语一致等。")

    if 'chapter_text' in locals() and chapter_text.strip():
        try:
            gdf
        except NameError:
            glossary_current = st.session_state.get("glossary_df", glossary_df)
            glossary_slim = glossary_current.drop(columns=["pattern"], errors='ignore')
            gdf = build_patterns(glossary_slim)
        hits = find_hits(chapter_text, gdf)
        hit_terms = sorted({h.term for h in hits})
        glossary_current = st.session_state.get("glossary_df", glossary_df)
        glossary_slim = glossary_current.drop(columns=["pattern"], errors='ignore')
        glossary_subset = glossary_slim[glossary_slim['source'].isin(hit_terms)].copy() if hit_terms else glossary_slim.head(20).copy()

        # 人名列表：提示词自检测，不再传名单
        names = []

        sys_prompt = build_system_prompt(rules, glossary_subset, names)
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

# --- Tab4: 批处理 JSONL ---
with T4:
    st.subheader("批处理：从文件夹读取章节，生成 Batch JSONL")
    colA, colB = st.columns([2,1])
    with colA:
        folder = st.text_input("章节文件夹路径（读取 *.txt）", value=str(Path.cwd() / "chapters"))
    with colB:
        out_jsonl = st.text_input("输出 JSONL 路径", value=str(Path.cwd() / "batch" / "requests.jsonl"))

    st.caption("流程：读取每个章节 → 生成系统提示（基于各自命中的术语子集；人名由提示词自检测）→ 组装为 /v1/chat/completions 的 JSONL 批处理文件。")

    if st.button("生成 JSONL"):
        chs = chapters_from_folder(folder)
        if not chs:
            st.error("未在该路径发现 .txt 章节文件。")
        else:
            glossary_current = st.session_state.get("glossary_df", glossary_df)
            glossary_slim = glossary_current.drop(columns=["pattern"], errors='ignore')
            gdf = build_patterns(glossary_slim)
            rows = []
            for it in chs:
                text = it["text"]
                hits = find_hits(text, gdf)
                terms = sorted({h.term for h in hits})
                subset = glossary_slim[glossary_slim['source'].isin(terms)].copy() if terms else glossary_slim.head(20).copy()
                # 人名由提示词自检测
                names = []

                system_prompt = build_system_prompt(rules, subset, names)
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