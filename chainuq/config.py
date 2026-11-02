"""Runtime configuration: paths and global constants.

Paths come from environment variables, which may be set in a ``.env`` file at
the repository root (see ``.env.example``). Variables already present in the
environment take precedence over the file, so a single command can be pointed
at another artifact root with ``CHAINUQ_ARTIFACTS=... python -m ...``. Relative
paths resolve against the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_KEYS: set[str] = set()


def load_env(path: Path | None = None) -> None:
    """Load dotenv values without executing shell expressions or overriding exports.

    Supports export prefixes, quotes, comments and ${NAME} interpolation. Relative
    CHAINUQ_ENV_FILE values are resolved from the repository root.
    """
    import re

    if path is None:
        path = Path(os.environ.get("CHAINUQ_ENV_FILE", REPO_ROOT / ".env")).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = match.groups()
        _DOTENV_KEYS.add(key)
        if key in os.environ:
            continue
        if value.startswith(("'", '"')):
            quote = value[0]
            close = value.find(quote, 1)
            if close < 0 or (value[close + 1 :].strip() and not value[close + 1 :].strip().startswith("#")):
                raise ValueError(f"{path}:{number}: invalid quoted value")
            value = value[1:close]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m[1], ""), value)
        os.environ[key] = value


def _path(name: str, default: str | Path) -> Path:
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


load_env()

ARTIFACTS = _path("CHAINUQ_ARTIFACTS", "artifacts")  # generations, judge labels, features, checkpoints, results
POOL_ROOT = _path("CHAINUQ_POOL_ARTIFACTS", ARTIFACTS / "pools")  # artifact tree of the 6,000-question unlabeled pools
DATASETS_ROOT = _path("CHAINUQ_DATASETS", "datasets")  # prepared benchmark splits (pipeline/prepare_data.py)
MODELS_ROOT = _path("CHAINUQ_MODELS", "models")  # local model directories, one per model name
ANALYSIS_ROOT = _path("CHAINUQ_ANALYSIS", ARTIFACTS / "analysis")  # evidence cache and Stage-2 analysis outputs
PAPER_DIR = _path("CHAINUQ_PAPER_DIR", "latex")  # LaTeX project that receives generated tables and figures
LOGS_ROOT = _path("CHAINUQ_LOGS", ARTIFACTS / "logs")  # logs of the driver scripts
JUDGE_PORT = int(os.environ.get("CHAINUQ_JUDGE_PORT", "8100"))

SEED = 2026
HEAD_SEEDS = list(range(2026, 2036))  # every trained head is reported as mean/std over these 10 seeds


def shell_exports() -> str:
    """Emit safely quoted exports for Bash drivers using the same configuration."""
    import shlex

    values = {key: os.environ[key] for key in _DOTENV_KEYS if key in os.environ}
    values.update(
        CHAINUQ_ARTIFACTS=str(ARTIFACTS),
        CHAINUQ_POOL_ARTIFACTS=str(POOL_ROOT),
        CHAINUQ_DATASETS=str(DATASETS_ROOT),
        CHAINUQ_MODELS=str(MODELS_ROOT),
        CHAINUQ_ANALYSIS=str(ANALYSIS_ROOT),
        CHAINUQ_PAPER_DIR=str(PAPER_DIR),
        CHAINUQ_LOGS=str(LOGS_ROOT),
    )
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(values.items()))


if __name__ == "__main__":
    print(shell_exports())
