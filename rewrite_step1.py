import re
from pathlib import Path

RAW_PATH = Path("outputs/step1_raw.txt")
OUT_PATH = Path("outputs/step1_natural.txt")

def extract_fields(raw: str):
    # Pull key fields from the Step1 printout
    top_func = re.search(r"Top function:\s*([a-zA-Z0-9_]+)\s*([0-9.]+)?", raw)
    narration = re.search(r"Narration:\s*(.*)", raw)
    top3 = re.search(r"Top-3 functions:\s*(\[.*\])", raw)
    best_proto = re.search(r"Best video proto:\s*([a-zA-Z0-9_]+)", raw)
    temps = re.search(r"Temps:\s*([0-9.]+)\s*([0-9.]+)", raw)

    return {
        "top_func": top_func.group(1) if top_func else None,
        "conf": float(top_func.group(2)) if (top_func and top_func.group(2)) else None,
        "raw_narr": narration.group(1).strip() if narration else None,
        "top3": top3.group(1) if top3 else None,
        "best_proto": best_proto.group(1) if best_proto else None,
        "temps": (float(temps.group(1)), float(temps.group(2))) if temps else None,
    }

def humanize(top_func: str, conf: float | None, raw_narr: str | None) -> str:
    # Neutral supportive parent-coach rewrite WITHOUT using an LLM.
    # (If you later want LLM, we can swap this function.)
    if top_func is None:
        return "I couldn't parse the Step 1 output. Please check outputs/step1_raw.txt."

    conf_note = ""
    if conf is not None:
        if conf < 0.20:
            conf_note = "This is a low-confidence guess, so several explanations may still fit."
        elif conf < 0.40:
            conf_note = "This is a moderate-confidence guess."
        else:
            conf_note = "This looks fairly consistent with the signals."

    # Convert function id to parent-friendly phrase
    mapping = {
        "repair_integrity": "wanting something to feel 'fixed' or complete",
        "share_control": "wanting to start together or keep control shared",
        "avoid_overload": "trying to reduce sensory or emotional overload",
        "need_predictability": "needing the situation to feel predictable",
        "keep_object": "needing to keep a specific object or arrangement stable",
        "delayed_meltdown": "building stress that may show up later",
        "frozen_distress": "shutting down or holding distress inside",
    }
    phrase = mapping.get(top_func, top_func.replace("_", " "))

    # Use the raw narration as content but smooth the tone
    base = raw_narr or ""
    base = base.replace("Likely need:", "Likely need:")  # keep meaning
    base = re.sub(r"\brisk\(meltdown-ish\)≈[0-9.]+", "there may be some risk of escalation", base)
    base = re.sub(r"\bflip≈[0-9.]+", "and the state may shift quickly", base)

    out = []
    out.append(f"What the system is picking up most is **{phrase}**.")
    if conf_note:
        out.append(conf_note)
    out.append("")
    out.append("How to respond (neutral, practical):")
    out.append("- Acknowledge what seems 'off' for him (even if you’re not sure what it is).")
    out.append("- Offer a small repair step (restore, redo, or clearly mark what changed).")
    out.append("- Keep your wording short and predictable; avoid adding extra demands.")
    out.append("")
    if base:
        out.append("Signal summary (same meaning, softer wording):")
        out.append(f"- {base}")
    out.append("")
    out.append("If this repeats, it can help to note what was changed, removed, interrupted, or left unfinished right before the reaction.")

    return "\n".join(out)

def main():
    raw = RAW_PATH.read_text(encoding="utf-8")
    fields = extract_fields(raw)

    natural = humanize(fields["top_func"], fields["conf"], fields["raw_narr"])
    OUT_PATH.write_text(natural, encoding="utf-8")

    print("=== Natural parent-facing narration ===")
    print(natural)
    print(f"\nSaved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
