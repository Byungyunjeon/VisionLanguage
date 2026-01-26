#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
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

# Reuse your Step1 feature extractors
from src.video_emotion import extract_video_emotions_per_sec
try:
    from src.audio_emotion import extract_audio_states_per_sec
except Exception:
    extract_audio_states_per_sec = None


def mean_dicts(seq: List[Dict[str, float]]) -> Dict[str, float]:
    if not seq:
        return {}
    keys = sorted({k for d in seq for k in d.keys()})
    out = {k: 0.0 for k in keys}
    for d in seq:
        for k in keys:
            out[k] += float(d.get(k, 0.0))
    T = max(1, len(seq))
    return {k: out[k] / T for k in keys}


def topk(d: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]


def normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        n = max(1, len(d))
        return {k: 1.0 / n for k in d}
    return {k: v / s for k, v in d.items()}


def baseline0_caption(emotion: str, conf: float, ctx: str, audio_hint: str = "") -> str:
    # Neutral supportive parent coach, but simple and generic.
    # No autism-specific inference, no function inference, no physics.
    lines = []
    if emotion:
        lines.append(f"Main visible emotion: {emotion} (confidence ~{conf:.2f}).")
    else:
        lines.append("I could not detect a clear facial emotion from this clip.")

    if ctx:
        lines.append(f"Context note: {ctx}")

    if audio_hint:
        lines.append(audio_hint)

    lines.append("")
    lines.append("What you can try (neutral, practical):")
    lines.append("- Speak briefly and calmly.")
    lines.append("- Reduce stimulation if possible (noise/light/requests).")
    lines.append("- Offer one clear next step and wait.")
    return "\n".join(lines)


def audio_hint_from_features(audio_seq: List[Dict[str, float]]) -> str:
    # Very light heuristic: not ASR, not emotion sequence.
    if not audio_seq:
        return ""
    rms = sum(float(d.get("rms", 0.0)) for d in audio_seq) / max(1, len(audio_seq))
    voiced = sum(float(d.get("voiced_prob", 0.0)) for d in audio_seq) / max(1, len(audio_seq))
    if rms > 0.06 and voiced > 0.4:
        return "Audio suggests higher arousal or active vocalization in this clip."
    if rms < 0.02 and voiced < 0.2:
        return "Audio suggests low arousal or quietness in this clip."
    return "Audio is mixed; no strong cue."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", default="", help="Optional parent context (free-form)")
    ap.add_argument("--out", default="outputs/step1_baseline0_raw.txt")
    ap.add_argument("--use_audio", action="store_true", help="Also extract simple audio features")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1) Extract per-second facial emotion distributions
#    video_seq = extract_video_emotions_per_sec(args.video, dt=1.0)
    video_seq = run_with_heartbeat(extract_video_emotions_per_sec, "Extracting video emotions (still running)", 2.0, args.video, 1.0)
    pooled = mean_dicts(video_seq)
    pooled = normalize(pooled)

    # 2) Pick top emotion
    if pooled:
        emo, conf = max(pooled.items(), key=lambda x: x[1])
    else:
        emo, conf = "", 0.0

    # 3) Optional: audio features (still no sequence; just hints)
    audio_seq = []
    hint = ""
    if args.use_audio and extract_audio_states_per_sec is not None:
        try:
            audio_seq = extract_audio_states_per_sec(args.video, dt=1.0)
            hint = audio_hint_from_features(audio_seq)
        except Exception:
            hint = ""

    # 4) Produce baseline narration (generic)
    narration = baseline0_caption(emo, conf, args.text.strip(), hint)

    # 5) Print + save in a Step1-like format (so your rewriter can parse it)
    result = []
    result.append("=== STEP1 BASELINE0 RESULT (no sequence, no physics) ===")
    result.append(f"Top emotion: {emo} {conf}")
    result.append("Top-3 emotions: " + json.dumps(topk(pooled, 3)))
    if args.text.strip():
        result.append(f"Context: {args.text.strip()}")
    result.append("")
    result.append("Narration:")
    result.append(narration)

    output_text = "\n".join(result)
    print(output_text)

    Path(args.out).write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()

