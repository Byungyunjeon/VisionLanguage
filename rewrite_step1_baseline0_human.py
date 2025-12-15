import subprocess
from pathlib import Path

RAW_PATH = Path("outputs/step1_baseline0_raw.txt")
OUT_PATH = Path("outputs/step1_baseline0_human.txt")
#RAW_PATH = Path("outputs/step1_natural.txt")
#OUT_PATH = Path("outputs/step1_human.txt")

SYSTEM_PROMPT = """
You are a calm, neutral, supportive parent coach.

Rewrite the text for a parent of a neurodivergent child.
Rules:
- Keep meaning exactly the same
- Do not add diagnosis or medical advice
- Be gentle, clear, and human
- Short paragraphs
- Acknowledge uncertainty
- No emojis, no hype
"""

def rewrite_with_ollama(raw_text: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

System output:
{raw_text}

Rewritten explanation:"""

    proc = subprocess.run(
        ["ollama", "run", "qwen2.5:7b-instruct"],
        input=prompt,
        text=True,
        capture_output=True,
    )
    return proc.stdout.strip()

def main():
    raw = RAW_PATH.read_text(encoding="utf-8")
    rewritten = rewrite_with_ollama(raw)

    OUT_PATH.write_text(rewritten, encoding="utf-8")

    print("=== Human parent-facing narration ===\n")
    print(rewritten)
    print(f"\nSaved to {OUT_PATH}")

if __name__ == "__main__":
    main()

