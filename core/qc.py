from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class QCReport:
    glossary_hits: int
    names_detected: int
    violations: List[str]


def quick_qc(glossary_hits: int, names_detected: int) -> QCReport:
    violations: List[str] = []
    if glossary_hits == 0:
        violations.append("No glossary hits; check glossary coverage or casing.")
    if names_detected == 0:
        violations.append("No English names detected; verify name detection or text type.")
    return QCReport(glossary_hits=glossary_hits, names_detected=names_detected, violations=violations)