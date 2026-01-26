import subprocess
import time
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
""".strip()


def rewrite_with_ollama(raw_text: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

System output:
{raw_text}

Rewritten explanation:"""

    model = "qwen2.5:7b-instruct"

    print(f"[{time.strftime('%H:%M:%S')}] [ollama] model={model} start", flush=True)
    t0 = time.time()

    # Stream stdout live so you can see progress while waiting.
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout so you see everything
        text=True,
        bufsize=1,                # line-buffered
    )

    assert proc.stdin is not None
    assert proc.stdout is not None

    # Send the prompt then close stdin so ollama starts generating immediately.
    proc.stdin.write(prompt)
    proc.stdin.close()

    chunks = []
    for line in proc.stdout:
        now = time.strftime("%H:%M:%S")
        # Print live with a timestamp, but keep the model's text exactly as-is.
        print(f"[{now}] {line}", end="", flush=True)
        chunks.append(line)

    rc = proc.wait()
    dt = time.time() - t0

    if rc != 0:
        raise RuntimeError(f"ollama exited with code {rc} after {dt:.1f}s")

    print(f"\n[{time.strftime('%H:%M:%S')}] [ollama] done in {dt:.1f}s", flush=True)
    return "".join(chunks).strip()


def main():
    raw = RAW_PATH.read_text(encoding="utf-8")

    print(f"[{time.strftime('%H:%M:%S')}] Reading: {RAW_PATH}", flush=True)
    rewritten = rewrite_with_ollama(raw)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(rewritten, encoding="utf-8")

    print("\n=== Human parent-facing narration ===\n")
    print(rewritten)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

