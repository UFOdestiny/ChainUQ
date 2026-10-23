"""Small JSONL helpers shared by all stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_uid(records: Iterable[dict]) -> dict:
    return {r["uid"]: r for r in records}
