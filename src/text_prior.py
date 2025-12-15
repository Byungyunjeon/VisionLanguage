from typing import Dict, Any, Optional
import numpy as np

def best_text_prior(user_text: str, text_priors: Dict[str, Any]) -> Optional[Dict[str,float]]:
    if not user_text:
        return None
    entries = text_priors.get("priors") or text_priors.get("entries") or []
    if not entries:
        return None
    # simple token overlap score (no external embeddings)
    ut = set(user_text.lower().split())
    best=None
    best_score=-1
    for e in entries:
        cand = e.get("nt_description","").lower()
        ct = set(cand.split())
        score = len(ut & ct)
        if score > best_score:
            best_score=score
            best=e
    if best is None:
        return None
    return best.get("function_prior")
