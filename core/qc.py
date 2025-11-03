from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class QCReport:
    glossary_hits: int
    violations: List[str]


def quick_qc(glossary_hits: int,) -> QCReport:
    violations: List[str] = []
    if glossary_hits == 0:
        violations.append("No glossary hits; check glossary coverage or casing.")
    return QCReport(glossary_hits=glossary_hits, violations=violations)