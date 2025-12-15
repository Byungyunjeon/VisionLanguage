#!/usr/bin/env python3
"""Step-1 calibration (privacy-safe, no training videos needed).

Learns fusion weights (alpha_v, alpha_a, alpha_t) and temperature scalars
(T_video, T_audio) for Step-1 prototype-based inference, using ONLY:
  - vision_lib.json
  - audio_lib.json
  - text_priors.json

Output format matches run_step1.py expectations.

This is NOT clinical calibration; it only stabilizes fusion hyperparameters.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-8


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_prototypes(lib: Any) -> List[Dict[str, Any]]:
    if isinstance(lib, dict):
        return lib.get("prototypes", []) or []
    if isinstance(lib, list):
        return lib
    raise TypeError(f"Unexpected lib type: {type(lib)}")


def get_text_entries(text_lib: Any) -> List[Dict[str, Any]]:
    if not isinstance(text_lib, dict):
        return []
    return text_lib.get("priors") or text_lib.get("entries") or []


def proto_vote_distribution(p: Dict[str, Any]) -> Dict[str, float]:
    mf = p.get("mapped_functions")
    if isinstance(mf, dict) and len(mf) > 0:
        return {str(k): float(v) for k, v in mf.items()}
    pat = p.get("pattern") or p.get("audio_pattern") or p.get("video_pattern") or "unknown"
    return {str(pat): 1.0}


def normalize(d: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    out = {k: float(d.get(k, 0.0)) for k in keys}
    s = sum(max(v, 0.0) for v in out.values())
    if s <= 1e-12:
        u = 1.0 / max(1, len(keys))
        return {k: u for k in keys}
    return {k: max(v, 0.0) / s for k, v in out.items()}


def union_functions(vision_lib: Any, audio_lib: Any, text_lib: Any) -> List[str]:
    C = set()
    for p in get_prototypes(vision_lib):
        C.update(proto_vote_distribution(p).keys())
    for p in get_prototypes(audio_lib):
        C.update(proto_vote_distribution(p).keys())
    for e in get_text_entries(text_lib):
        fp = e.get("function_prior") or {}
        if isinstance(fp, dict):
            C.update(fp.keys())
    return sorted(map(str, C))


def sample_label_from_mixture(mix: Dict[str, float]) -> str:
    keys = list(mix.keys())
    w = [max(0.0, float(mix[k])) for k in keys]
    if sum(w) <= 0:
        return random.choice(keys)
    return random.choices(keys, weights=w, k=1)[0]


def make_synth_example(
    vision_lib: Any,
    audio_lib: Any,
    text_lib: Any,
    C: List[str],
    noise: float = 0.08,
    use_text: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], str]:
    pv = random.choice(get_prototypes(vision_lib))
    pa = random.choice(get_prototypes(audio_lib))

    SV = normalize(proto_vote_distribution(pv), C)
    SA = normalize(proto_vote_distribution(pa), C)

    for c in C:
        SV[c] = max(EPS, SV[c] + random.uniform(-noise, noise))
        SA[c] = max(EPS, SA[c] + random.uniform(-noise, noise))
    SV = normalize(SV, C)
    SA = normalize(SA, C)

    entries = get_text_entries(text_lib)
    if use_text and entries:
        e = random.choice(entries)
        fp = e.get("function_prior") or {}
        pT = normalize({c: float(fp.get(c, 0.0)) for c in C}, C)
    else:
        u = 1.0 / max(1, len(C))
        pT = {c: u for c in C}

    mix = {c: 0.5 * SV[c] + 0.5 * SA[c] for c in C}
    y = sample_label_from_mixture(mix)
    return SV, SA, pT, y


def train_calibration(
    vision_lib: Any,
    audio_lib: Any,
    text_lib: Any,
    steps: int = 800,
    batch_size: int = 128,
    lr: float = 0.05,
    seed: int = 0,
    use_text: bool = True,
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)

    C = union_functions(vision_lib, audio_lib, text_lib)
    idx = {c: i for i, c in enumerate(C)}

    raw_Tv = torch.tensor(0.2, requires_grad=True)
    raw_Ta = torch.tensor(0.2, requires_grad=True)
    raw_alpha = torch.tensor([0.6, 0.3, 0.1], requires_grad=True)

    opt = torch.optim.Adam([raw_Tv, raw_Ta, raw_alpha], lr=lr)

    def pos(x):
        return F.softplus(x) + 1e-3

    for step in range(steps):
        batch = [
            make_synth_example(vision_lib, audio_lib, text_lib, C, noise=0.08, use_text=use_text)
            for _ in range(batch_size)
        ]

        Tv = pos(raw_Tv)
        Ta = pos(raw_Ta)
        alpha = torch.softmax(raw_alpha, dim=0)

        losses = []
        correct = 0
        for SVd, SAd, pTd, y in batch:
            SV = torch.tensor([SVd[c] for c in C], dtype=torch.float32)
            SA = torch.tensor([SAd[c] for c in C], dtype=torch.float32)
            pT = torch.tensor([pTd[c] for c in C], dtype=torch.float32)

            logits = (
                alpha[0] * (torch.log(SV + EPS) / Tv)
                + alpha[1] * (torch.log(SA + EPS) / Ta)
                + alpha[2] * (torch.log(pT + EPS))
            )

            P = torch.softmax(logits, dim=-1)
            y_i = idx[y]
            nll = -torch.log(P[y_i] + EPS)
            losses.append(nll)
            if int(torch.argmax(P).item()) == y_i:
                correct += 1

        L = torch.stack(losses).mean()
        opt.zero_grad()
        L.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            Tv_v = float(pos(raw_Tv).item())
            Ta_v = float(pos(raw_Ta).item())
            a = torch.softmax(raw_alpha, dim=0).detach().cpu().numpy().tolist()
            print(
                f"[{step+1:04d}/{steps}] nll={float(L.item()):.3f} acc={correct/len(batch):.3f} "
                f"Tv={Tv_v:.3f} Ta={Ta_v:.3f} alpha_v={a[0]:.3f} alpha_a={a[1]:.3f} alpha_t={a[2]:.3f}"
            )

    out = {
        "temperatures": {"video": float(pos(raw_Tv).item()), "audio": float(pos(raw_Ta).item())},
        "fusion": {
            "alpha_v": float(torch.softmax(raw_alpha, dim=0)[0].item()),
            "alpha_a": float(torch.softmax(raw_alpha, dim=0)[1].item()),
            "alpha_t": float(torch.softmax(raw_alpha, dim=0)[2].item()),
        },
        "functions": C,
        "notes": [
            "Calibrated on synthetic draws from prototype libraries + text priors.",
            "Not clinical; only stabilizes fusion hyperparameters.",
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text_priors", required=True)
    ap.add_argument("--out", default="data/calib.json")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_text", action="store_true")
    args = ap.parse_args()

    V = load_json(args.vision)
    A = load_json(args.audio)
    T = load_json(args.text_priors)

    calib = train_calibration(
        V, A, T,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_text=(not args.no_text),
    )
    save_json(calib, args.out)
    print("\nSaved calibration:", args.out)
    print(json.dumps(calib, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
