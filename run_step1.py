import argparse
import numpy as np
from src.utils import load_json
from src.library import load_library
from src.dtw_align import energies_against_library
from src.physics import boltzmann_weights, glauber_flip_prob, landau_barrier
from src.video_emotion import extract_video_emotions_per_sec
from src.audio_emotion import extract_audio_states_per_sec
from src.text_prior import best_text_prior
from src.narrate import aggregate_function_scores, pick_top_function, render_parent_narration
import time
import sys
import threading

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

def proto_function_votes(lib, proto_id: str):
    # try to read mapped_functions, else vote by pattern name
    for p in lib.prototypes:
        if p.id == proto_id:
            mf = p.meta.get("mapped_functions")
            if isinstance(mf, dict) and len(mf)>0:
                return mf
            # fallback: treat pattern as function
            return {p.pattern: 1.0}
    return {}

def weighted_function_from_proto_weights(lib, proto_weights, proto_ids):
    out={}
    for w,pid in zip(proto_weights, proto_ids):
        votes = proto_function_votes(lib, pid)
        for k,v in votes.items():
            out[k] = out.get(k,0.0) + float(w)*float(v)
    s=sum(out.values())
    if s<=1e-12:
        return {}
    return {k:v/s for k,v in out.items()}
def _ts():
    return time.strftime("%H:%M:%S")

def log(msg: str):
    # Always flush so progress appears live even in buffered environments
    print(f"[{_ts()}] {msg}", flush=True)

class Stage:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(self.label + " ...")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            log(self.label + f" done in {dt:.2f}s")
        else:
            log(self.label + f" FAILED after {dt:.2f}s: {exc_type.__name__}: {exc}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--vision", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text_priors", required=True)
    ap.add_argument("--narr", required=False, default="")
    ap.add_argument("--calib", required=False, default="")
    args = ap.parse_args()

#    V = load_library(load_json(args.vision), "vision")
#    A = load_library(load_json(args.audio), "audio")
#    TP = load_json(args.text_priors)
    with Stage("[1/7] Loading libraries / priors"):
        V = load_library(load_json(args.vision), "vision")
        A = load_library(load_json(args.audio), "audio")
        TP = load_json(args.text_priors)

    # 1) extract observed sequences
#    video_seq = extract_video_emotions_per_sec(args.video, dt=1.0)
#    audio_seq = extract_audio_states_per_sec(args.video, dt=1.0)
    with Stage("[2/7] Extract video emotions (per sec)"):
    #    video_seq = extract_video_emotions_per_sec(args.video, dt=1.0)
    #    video_seq = run_with_heartbeat(
    #        extract_video_emotions_per_sec,
    #        label="[2/7] Extracting video emotions (still running)",
    #        every=2.0,
    #        args=args.video, dt=1.0
    #    )
        video_seq = run_with_heartbeat(extract_video_emotions_per_sec, "[2/7] Extracting video emotions (still running)", 2.0, args.video, 1.0)

    log(f"video_seq length = {len(video_seq)}")

    with Stage("[3/7] Extract audio states (per sec)"):
        audio_seq = extract_audio_states_per_sec(args.video, dt=1.0)
        log(f"audio_seq length = {len(audio_seq)}")

    # 2) energies vs libraries
#    EV, v_ids = energies_against_library(video_seq, V, dt=1.0)
#    EA, a_ids = energies_against_library(audio_seq, A, dt=1.0)
    with Stage("[4/7] Compute energies vs vision library"):
        EV, v_ids = energies_against_library(video_seq, V, dt=1.0)

    with Stage("[5/7] Compute energies vs audio library"):
        EA, a_ids = energies_against_library(audio_seq, A, dt=1.0)

    # 3) temperatures
    Tv=0.2; Ta=0.2
    alpha_v=0.55; alpha_a=0.35; alpha_t=0.10
    if args.calib:
        with Stage("[6/7] Load calibration"):
            c = load_json(args.calib)
            Tv = float(c.get("temperatures",{}).get("video", Tv))
            Ta = float(c.get("temperatures",{}).get("audio", Ta))
            fusion = c.get("fusion",{})
            alpha_v=float(fusion.get("alpha_v", alpha_v))
            alpha_a=float(fusion.get("alpha_a", alpha_a))
            alpha_t=float(fusion.get("alpha_t", alpha_t))

    # 4) Boltzmann weights over prototypes
    wV = boltzmann_weights(EV, temperature=Tv)
    wA = boltzmann_weights(EA, temperature=Ta)

    # 5) Convert proto-weights -> function distribution (Step 1: voting)
    fv = weighted_function_from_proto_weights(V, wV, v_ids)
    fa = weighted_function_from_proto_weights(A, wA, a_ids)

    prior = best_text_prior(args.text, TP) if args.text else None
    prior = prior or {}
    log("[7/7] Scoring + narration ...")

    func_scores = aggregate_function_scores(fv, fa, prior, alpha_v=alpha_v, alpha_a=alpha_a, alpha_t=alpha_t)
    top_func, top_p = pick_top_function(func_scores)

    # 6) Early-20c-ish spice: barrier + glauber gate for meltdown-like warning
    # Treat a set of functions as "high-risk"; compute a soft warning score.
    risk_funcs = ["delayed_meltdown","mask_snap","sensory_overload","frozen_distress","ocd_loop"]
    risk_prob = sum(func_scores.get(k,0.0) for k in risk_funcs)
    # Convert to a pseudo-energy barrier and a flip probability
    barrier = landau_barrier(risk_prob, a=1.5, b=1.0)
    p_flip = glauber_flip_prob(deltaE=barrier, temperature=0.25)

    # 7) choose best-matching video prototype for narration sentence
    best_vid_idx = int(np.argmax(wV))
    best_vid_id = v_ids[best_vid_idx]
    proto_parent = ""
    for p in V.prototypes:
        if p.id == best_vid_id:
            proto_parent = p.meta.get("narration_parent","")

    extras=[]
    extras.append(f"risk(meltdown-ish)≈{risk_prob:.2f}, flip≈{p_flip:.2f}")
    if args.text:
        extras.append("context: " + args.text)

    narration = render_parent_narration(top_func, top_p, proto_parent, extras)

    print("\n=== STEP1 RESULT ===")
    print("Top function:", top_func, top_p)
    print("Narration:", narration)
    print("Top-3 functions:", sorted(func_scores.items(), key=lambda kv: kv[1], reverse=True)[:3])
    print("Best video proto:", best_vid_id)
    print("Temps:", Tv, Ta)

if __name__ == "__main__":
    main()
