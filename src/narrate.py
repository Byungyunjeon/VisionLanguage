from typing import Dict, Any, Tuple, List
import numpy as np

def aggregate_function_scores(weights_vid: Dict[str,float],
                              weights_aud: Dict[str,float],
                              prior: Dict[str,float] = None,
                              alpha_v: float = 0.55, alpha_a: float = 0.35, alpha_t: float = 0.10) -> Dict[str,float]:
    # combine into function distribution
    keys=set(weights_vid.keys()) | set(weights_aud.keys()) | (set(prior.keys()) if prior else set())
    out={}
    for k in keys:
        out[k] = alpha_v*weights_vid.get(k,0.0) + alpha_a*weights_aud.get(k,0.0) + alpha_t*(prior.get(k,0.0) if prior else 0.0)
    s=sum(out.values())
    if s<=1e-12:
        u=1.0/max(1,len(keys))
        return {k:u for k in keys}
    return {k:v/s for k,v in out.items()}

def pick_top_function(func_scores: Dict[str,float]) -> Tuple[str,float]:
    k=max(func_scores, key=lambda x: func_scores[x])
    return k, float(func_scores[k])

def render_parent_narration(func: str, prob: float, proto_parent: str, extras: List[str]) -> str:
    # short, actionable
    base = f"Likely need: {func.replace('_',' ')} (confidence {prob:.2f})."
    if proto_parent:
        base += " " + proto_parent.strip()
    if extras:
        base += " " + " ".join(extras[:2])
    return base
