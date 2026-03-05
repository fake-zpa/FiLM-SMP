from pathlib import Path
from typing import Dict


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_path_str(p: str) -> str:
    return p.replace("\\", "/")


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, object]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
