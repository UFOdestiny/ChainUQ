import json
from pathlib import Path
from typing import Any

import torch


def _tmp_path(path: Path) -> Path:
    if path.suffix:
        return path.with_suffix(f"{path.suffix}.tmp")
    return Path(f"{path}.tmp")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    tmp_path = _tmp_path(path)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)
    tmp_path.rename(path)


def save_torch_atomic(payload: Any, path: Path) -> None:
    tmp_path = _tmp_path(path)
    torch.save(payload, tmp_path)
    tmp_path.rename(path)


def load_torch_with_tmp_recovery(path: Path, *, logger=None, delete_corrupt: bool = True):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        tmp_path = _tmp_path(path)
        if tmp_path.exists():
            try:
                data = torch.load(tmp_path, map_location="cpu", weights_only=False)
                tmp_path.rename(path)
                if logger is not None:
                    logger.warning("%s corrupt, recovered from .tmp file.", path.name)
                return data
            except Exception:
                pass

        if logger is not None:
            logger.warning("%s corrupt (%s).", path.name, exc)
        if delete_corrupt:
            path.unlink(missing_ok=True)
            tmp_path.unlink(missing_ok=True)
        return None
