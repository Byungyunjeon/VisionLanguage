#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.utils import load_json  # :contentReference[oaicite:6]{index=6}
from src.library import load_library  # :contentReference[oaicite:7]{index=7}
from src.video_emotion import extract_video_emotions_per_sec  # :contentReference[oaicite:8]{index=8}
from src.audio_emotion import extract_audio_states_per_sec  # :contentReference[oaicite:9]{index=9}
from src.dtw_align import energies_against_library  # :contentReference[oaicite:10]{index=10}
from src.text_prior import best_text_prior  # :contentReference[oaicite:11]{index=11}
from src.narrate import aggregate_function_scores, pick_top_function, render_parent_narration  # :contentReference[oaicite:12]{index=12}

import threading
import time

def _ts():
    return time.strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)

def run_with_heartbeat(fn, label="working", every=2.0, *args, **kwargs):
    done = False
    result = None
    exc = None

    def worker():
        nonlocal done, result, exc
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            exc = e
        done = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while not done:
        log(f"{label} ...")
        time.sleep(every)

    if exc is not None:
        raise exc
    return result

def softmax_neg_energy(E: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Baseline1: we still need a way to convert prototype DTW energies -> weights,
    but we DO NOT use src.physics.boltzmann_weights (that's your physics layer).
    So we use a plain softmax over -E/tau.
    """
    tau = float(max(1e-6, tau))
    x = -(E / tau)
    x = x - float(np.max(x))
    ex = np.exp(x)
    s = float(np.sum(ex))
    if s <= 0:
        return np.ones_like(ex) / max(1, ex.size)
    return ex / s


def prototype_weights_to_function_scores(
    lib, proto_ids: List[str], proto_w: np.ndarray
) -> Dict[str, float]:
    """
    Convert weights over prototypes -> weights over functions.

    Supports two prototype metadata styles:
    - meta["mapped_functions"] = {func: weight, ...}  (preferred)
    - otherwise fallback: meta["pattern"] or prototype.pattern is already a function name
    """
    # Build quick index from id -> Prototype object
    id2p = {p.id: p for p in lib.prototypes}

    scores: Dict[str, float] = {}
    for pid, w in zip(proto_ids, proto_w.tolist()):
        p = id2p.get(pid)
        if p is None:
            continue

        mf = None
        # many of your JSON prototypes carry everything into meta except steps (see load_library) :contentReference[oaicite:13]{index=13}
        if isinstance(p.meta, dict):
            mf = p.meta.get("mapped_functions", None)

        if isinstance(mf, dict) and len(mf) > 0:
            for f, fw in mf.items():
                scores[f] = scores.get(f, 0.0) + float(w) * float(fw)
        else:
            # fallback: treat the pattern as the function label
            f = p.pattern or (p.meta.get("pattern") if isinstance(p.meta, dict) else None) or "unknown"
            scores[f] = scores.get(f, 0.0) + float(w)

    # normalize
    s = sum(scores.values())
    if s <= 1e-12:
        return scores
    return {k: v / s for k, v in scores.items()}


def topk(d: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--vision", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text_priors", required=True)
    ap.add_argument("--out", default="outputs/step1_baseline1_raw.txt")

    # knobs
    ap.add_argument("--tau_v", type=float, default=1.0, help="softmax temperature for video prototype weights")
    ap.add_argument("--tau_a", type=float, default=1.0, help="softmax temperature for audio prototype weights")
    ap.add_argument("--alpha_v", type=float, default=0.55)
    ap.add_argument("--alpha_a", type=float, default=0.35)
    ap.add_argument("--alpha_t", type=float, default=0.10)

    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1) Extract emotion/state sequences (same perception as Step1)
#    v_seq = extract_video_emotions_per_sec(args.video, dt=1.0)  # :contentReference[oaicite:14]{index=14}
    v_seq = run_with_heartbeat(extract_video_emotions_per_sec, "Extracting video emotions (still running)", 2.0, args.video, 1.0)

    a_seq = extract_audio_states_per_sec(args.video, dt=1.0)    # :contentReference[oaicite:15]{index=15}

    # 2) Load prototype libraries (JSON -> object -> PrototypeLibrary)
    Vobj = load_json(args.vision)  # :contentReference[oaicite:16]{index=16}
    Aobj = load_json(args.audio)
    Vlib = load_library(Vobj, name="video")  # :contentReference[oaicite:17]{index=17}
    Alib = load_library(Aobj, name="audio")

    # 3) DTW energies vs prototypes (sequence preserved)
    Ev, v_ids = energies_against_library(v_seq, Vlib, dt=1.0)  # :contentReference[oaicite:18]{index=18}
    Ea, a_ids = energies_against_library(a_seq, Alib, dt=1.0)

    # 4) Convert energies -> prototype weights (NO physics.py)
    wv = softmax_neg_energy(Ev, tau=args.tau_v)
    wa = softmax_neg_energy(Ea, tau=args.tau_a)

    # 5) Convert prototype weights -> function scores
    scores_v = prototype_weights_to_function_scores(Vlib, v_ids, wv)
    scores_a = prototype_weights_to_function_scores(Alib, a_ids, wa)

    # 6) Text prior (optional)
    Tobj = load_json(args.text_priors)
    prior = best_text_prior(args.text.strip(), Tobj) if args.text.strip() else None  # :contentReference[oaicite:19]{index=19}

    # 7) Linear fusion (still NOT physics)
    func_scores = aggregate_function_scores(
        scores_v, scores_a,
        prior=prior,
        alpha_v=args.alpha_v, alpha_a=args.alpha_a, alpha_t=args.alpha_t
    )  # :contentReference[oaicite:20]{index=20}

    top_func, top_prob = pick_top_function(func_scores)  # :contentReference[oaicite:21]{index=21}
    best_video_proto = v_ids[int(np.argmax(wv))] if len(v_ids) else "unknown"

    extras = []
    if args.text.strip():
        extras.append(f"context: {args.text.strip()}")

    # baseline narration (you can pass this into your Ollama rewriter)
    proto_parent = ""
    # try to attach parent narration if present in meta (common in your JSON->meta mapping) :contentReference[oaicite:22]{index=22}
    for p in Vlib.prototypes:
        if p.id == best_video_proto and isinstance(p.meta, dict):
            proto_parent = p.meta.get("narration_parent", "") or p.meta.get("narration", "") or ""
            break

    narration = render_parent_narration(top_func, top_prob, proto_parent, extras)  # :contentReference[oaicite:23]{index=23}

    # Output in Step1-like style
    lines = []
    lines.append("=== STEP1 BASELINE1 RESULT (sequence, no physics) ===")
    lines.append(f"Top function: {top_func} {top_prob}")
    lines.append(f"Top-3 functions: {topk(func_scores, 3)}")
    lines.append(f"Best video proto: {best_video_proto}")
    lines.append(f"tau_v tau_a: {args.tau_v} {args.tau_a}")
    lines.append("")
    lines.append("Narration: " + narration)

    out_text = "\n".join(lines)
    print(out_text)
    Path(args.out).write_text(out_text, encoding="utf-8")


if __name__ == "__main__":
    main()

