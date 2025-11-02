from __future__ import annotations
import json
import pandas as pd
from typing import Dict, Any

def build_system_prompt(rules: dict,
                        glossary_subset: pd.DataFrame,
                        names_do_not_translate: list[str] | None = None) -> str:
    names_do_not_translate = names_do_not_translate or []
    gmap = {row["source"]: row["target"] for _, row in glossary_subset.iterrows()}

    payload: Dict[str, Any] = {
        "role": "你是《星际迷航》领域的专业译者。你的首要目标是忠实、统一术语，并严格遵守以下硬性规则。",
        "hard_rules": {
            # 关键：让模型自行检测 & 不翻译人名
            "people_names_do_not_translate": True,
            "self_detect_people_names": True,
            "detection_guidelines": [
                "人名包括单名/多词姓名/连字符姓名（如 Jean-Luc Picard）。",
                "排除：舰船名(配有 U.S.S./‘号’后缀)、组织名(Starfleet/Federation)、星球/地名、物种名(Borg 等)、军衔/职务(Captain/Commander/Operations manager)、型号与代号。",
                "若不确定是否为人名或专有名，一律保留英文原样，不得臆造译名。"
            ],
            "ships_translate": rules.get("ships_translate", True),
            "ranks": rules.get("ranks", {}),
            "name_rank_position": rules.get("name_rank_position", "after"),
            "keep_inline_markup": rules.get("output_style", {}).get("keep_inline_markup", True),
            "on_unknown_term": rules.get("on_unknown_term", "保留英文并报告")
        },
        "glossary_active_subset": gmap,
        "names_do_not_translate_detected": names_do_not_translate,
        "style": rules.get("output_style", {}),
        "output_contract": {
            "format": "text-only",
            "no_explanation": True,
            "violations_policy": "若不慎翻译了应保留英文的人名/专名，应立即改回英文原样，不得输出说明。"
        }
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)