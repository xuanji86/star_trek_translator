# Star Trek Translator – GUI + Batch + QC Tool

# 星际迷航翻译助手 – 图形界面 + 批处理 + 质检工具

A Streamlit-based translation and terminology consistency framework for large-scale Star Trek novel translation. Supports interactive single-chapter translation, batch processing via OpenAI Batch API, and comprehensive QC (quality control) tools.
一个基于 Streamlit 的大型文本翻译与术语一致性控制框架，用于《星际迷航》小说的高质量中译。支持单章翻译、批量处理（OpenAI Batch API）与成品质检（QC），可自定义多语料库（舰船、物种、军衔、职位等）。

---

## 🚀 Features / 功能简介

### 1. Interactive Chapter Translation / 单章交互式翻译

* Paste any chapter into GUI to estimate tokens and cost.
  在 GUI 界面粘贴整章英文内容，自动估算 tokens 与成本。
* Adjustable pricing and batch discount.
  可自定义输入/输出单价和批量折扣。
* Auto segmentation by token budget and full-text merge.
  按 token 限额自动分段翻译并拼接全文。

### 2. Glossary & Multi-Corpus Management / 术语表与多语料库管理

* Manage multiple CSV-based corpora (ships, ranks, species, roles…).
  可加载多个 CSV 语料库（舰船、军衔、物种、职位等）。
* Edit inline, merge dynamically, export anytime.
  支持在线编辑、动态合并与导出。
* Auto term matching and hit report.
  自动比对英文原文中的术语命中并生成报告。

### 3. Rules & Prompt Control / 翻译规则与系统提示

* YAML-based rules for rank, name handling, and tone control.
  使用 YAML 文件定义规则（军衔、人名、风格等）。
* Generates structured system prompt for stable translation.
  自动生成一致性系统提示，确保翻译风格统一。

### 4. Batch Translation (OpenAI Batch API) / 批量翻译

* Convert chapters to JSONL and upload via API.
  从文件夹读取章节 TXT，生成 JSONL 并上传至 Batch API。
* Query batch status, download, and auto-split outputs.
  查询任务状态、下载结果并按章节自动输出。

### 5. QC & Auto Repair / 成品质检与修复

* **Glossary coverage:** Ensure all glossary targets appear in translation.
  检查译文中是否包含英文原文命中的所有术语。
* **Rank order:** Ensure rank follows name (e.g. “Picard 上校”).
  确保军衔在姓名之后（如“Picard上校”）。
* **Paragraph spacing:** Auto insert blank lines between paragraphs.
  自动修正段落空行格式。
* One-click fix and export corrected TXT.
  一键修复并导出修正版 TXT。

---

## 🧩 Folder Structure / 文件结构

```
core/
 ├── translator.py        # OpenAI API adapter / OpenAI 调用模块
 ├── glossary.py          # Glossary processing / 术语匹配模块
 ├── tokenizer.py         # Token estimator / token 数估算
 ├── pricing.py           # Cost calculator / 成本估算
 ├── rules.py             # YAML rule parser / 规则解析
 ├── prompts.py           # System prompt builder / 系统提示生成
 ├── batching.py          # Batch JSONL builder / 批处理 JSONL 构建
 ├── qc.py                # Quality control / 质检逻辑

app.py                   # Streamlit main app / 主程序入口
data/
 ├── glossary_sample.csv  # Sample glossary / 示例术语表
 ├── rules_sample.yaml    # Sample rules / 示例规则文件
```

---

## 💻 Installation / 安装步骤

```bash
# 1. Clone project / 克隆项目
$ git clone https://github.com/yourname/star-trek-translator.git
$ cd star-trek-translator

# 2. Install dependencies / 安装依赖
$ pip install -r requirements.txt

# 3. Run Streamlit app / 启动 GUI 应用
$ python3 -m streamlit run app.py
```

---

## 🔑 Usage / 使用方法

1. Input your **OpenAI API Key** in the sidebar or via environment variable `OPENAI_API_KEY`.
   在侧边栏输入 API Key 或通过环境变量设置。
2. Use the corresponding tabs for operations:
   使用不同标签页执行操作：

   * **Tab1**: Cost estimation & input text / 估价与文本输入
   * **Tab2**: Glossary & corpus management / 术语与语料库管理
   * **Tab3**: Rules & prompt setup / 规则与系统提示
   * **Tab4**: Batch translation / 批量翻译
   * **Tab5**: Quality check & auto fix / 成品质检与修复

---

## 🧠 QC Logic Summary / 质检逻辑说明

* **_glossary_coverage()** → Detect if all glossary terms from English text appear in translation.
  检测英文命中词条是否在译文中出现对应译名。
* **_find_rank_order_issues()** → Regex finds patterns like “上校 Picard”.
  通过正则检测“军衔在前”的违例。
* **_normalize_paragraphs()** → Ensure single blank line between paragraphs.
  保证每个段落间至少一个空行。
* **_auto_fix_rank_order()** → Swap “rank name” → “name rank” automatically.
  自动将“军衔 名字”替换为“名字 军衔”。


---

## 🧾 License / 许可证

GPL v3 License – Free to use, modify, and distribute under the same license.
GPL v3 开源许可 – 允许自由使用、修改与再分发，但需保持相同许可条款。


---

**Author / 作者：** Anji Xu
**Project / 项目：** Star Trek Novel Translation Tool
《星际迷航》小说翻译工具
