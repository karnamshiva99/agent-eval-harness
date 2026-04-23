"""Dataset loading for eval test cases.

Test cases are JSONL files with one case per line. Required fields:
- input: the prompt or question
- expected: reference answer (for LLM-as-judge scoring)

Optional fields:
- id: stable identifier
- tags: list of strings for filtering
- lang: ISO language code (en, es, fr, de, ja, ...)
- metadata: arbitrary dict
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class TestCase:
    __test__ = False  # pytest: don't collect this dataclass as a test class
    input: str
    expected: str
    id: str = ""
    tags: list[str] = field(default_factory=list)
    lang: str = "en"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.sha256(self.input.encode()).hexdigest()[:12]


@dataclass
class Dataset:
    cases: list[TestCase]
    path: str = ""

    def __iter__(self) -> Iterator[TestCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def hash(self) -> str:
        """Stable hash of the dataset content — used in run serialization."""
        payload = "|".join(c.id for c in self.cases).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def filter(self, tag: str | None = None, lang: str | None = None) -> "Dataset":
        filtered = self.cases
        if tag:
            filtered = [c for c in filtered if tag in c.tags]
        if lang:
            filtered = [c for c in filtered if c.lang == lang]
        return Dataset(cases=filtered, path=self.path)


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    cases: list[TestCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        raw = json.loads(line)
        cases.append(TestCase(**raw))
    return Dataset(cases=cases, path=str(p))
