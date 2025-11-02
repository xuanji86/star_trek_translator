from __future__ import annotations
import yaml
from typing import Any, Dict

DEFAULT_RULES_YAML = {
    "people_names_do_not_translate": True,
    "ships_translate": False,
    "ranks": {
        "Captain": "上校",
        "Commander": "中校",
        "Lieutenant Commander": "少校",
        "Lieutenant": "上尉",
        "lieutenant junior grade": "中尉",
        "lieutenant(jg)": "中尉",
        "Ensign": "少尉",
    },
    "name_rank_position": "after",
    "first_deck_term": "一号甲板",
    "chen_trryssa_rank_override": "中尉",
    "output_style": {
        "formality": "正式、克制，书面语气",
        "keep_inline_markup": True,
        "dialogue_style": "保持原文口吻与情绪",
    },
    "on_unknown_term": "保留英文并在JSON violations中报告候选译名",
}


def parse_rules_yaml(yaml_text: str | None) -> Dict[str, Any]:
    if not yaml_text:
        return DEFAULT_RULES_YAML.copy()
    try:
        return yaml.safe_load(yaml_text) or DEFAULT_RULES_YAML.copy()
    except Exception:
        return DEFAULT_RULES_YAML.copy()