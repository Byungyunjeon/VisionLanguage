import json
from pathlib import Path
from typing import Any, Dict

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_dist(dist: Dict[str, float], keys) -> Dict[str, float]:
    # Fill missing, clamp, renormalize.
    out = {k: float(dist.get(k, 0.0)) for k in keys}
    s = sum(max(v, 0.0) for v in out.values())
    if s <= 1e-12:
        # uniform fallback
        u = 1.0 / max(1, len(keys))
        return {k: u for k in keys}
    return {k: max(v, 0.0) / s for k, v in out.items()}
