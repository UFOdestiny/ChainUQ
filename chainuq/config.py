"""Runtime configuration: paths and global constants.

Paths come from environment variables, which may be set in a ``.env`` file at
the repository root (see ``.env.example``). Variables already present in the
environment take precedence over the file, so a single command can be pointed
at another artifact root with ``CHAINUQ_ARTIFACTS=... python -m ...``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Export ``KEY=VALUE`` lines of ``path`` that are not already set (``${VAR}`` is expanded)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = os.path.expandvars(value)


def _path(name: str, default: str) -> Path:
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


load_env()

ARTIFACTS = _path("CHAINUQ_ARTIFACTS", "artifacts")  # generations, judge labels, features, checkpoints, results
DATASETS_ROOT = _path("CHAINUQ_DATASETS", "datasets")  # prepared benchmark splits (pipeline/prepare_data.py)
MODELS_ROOT = _path("CHAINUQ_MODELS", "models")  # local model directories, one per model name
PAPER_DIR = _path("CHAINUQ_PAPER_DIR", "latex")  # LaTeX project that receives generated tables and figures

SEED = 2026
HEAD_SEEDS = list(range(2026, 2036))  # every trained head is reported as mean/std over these 10 seeds
