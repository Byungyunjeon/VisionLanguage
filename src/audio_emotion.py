from typing import List, Dict, Tuple
import numpy as np
import subprocess, os

AUDIO_STATES_DEFAULT = [
    "silence","quiet","speech","tense_speech","whine","cry","scream","breathy","laugh","staccato","mutter","giggle","sob","sniffle"
]

def extract_wav_from_video(video_path: str, wav_out: str, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg","-y","-i",video_path,
        "-vn","-ac","1","-ar",str(sr),
        wav_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / max(e.sum(), 1e-12)

def audio_features_per_sec(wav_path: str, dt: float = 1.0, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa
    y, sr2 = librosa.load(wav_path, sr=sr, mono=True)
    hop = int(sr*dt)
    n = max(1, int(np.ceil(len(y)/hop)))
    rms = np.zeros(n)
    zcr = np.zeros(n)
    f0 = np.zeros(n)
    for i in range(n):
        seg = y[i*hop:(i+1)*hop]
        if len(seg) == 0:
            continue
        rms[i] = float(np.sqrt(np.mean(seg**2) + 1e-12))
        zcr[i] = float(np.mean(librosa.feature.zero_crossing_rate(seg, frame_length=min(2048,len(seg)), hop_length=max(256, min(512,len(seg)//4)))[0]))
        # pitch (cheap): pyin can be slow; use yin with fallback
        try:
            f = librosa.yin(seg, fmin=60, fmax=600, sr=sr)
            f0[i] = float(np.nanmedian(f))
        except Exception:
            f0[i] = 0.0
    return rms, zcr, f0

def map_audio_to_state_dist(rms: float, zcr: float, f0: float) -> Dict[str,float]:
    # Very simple heuristic mapping for Step 1.
    # You can replace with a real SER model later.
    # Output is a distribution over AUDIO_STATES_DEFAULT.
    # Scale features
    # "silence/quiet" vs "speech" by rms
    score = {k: 0.0 for k in AUDIO_STATES_DEFAULT}
    if rms < 0.005:
        score["silence"] += 2.0
        score["quiet"] += 1.0
        score["breathy"] += 0.5
    elif rms < 0.02:
        score["quiet"] += 2.0
        score["speech"] += 0.5
        score["mutter"] += 0.5
        score["breathy"] += 0.3
    else:
        score["speech"] += 1.0
        score["tense_speech"] += 0.5
        # high pitch + high zcr can mean scream/cry/whine
        if f0 > 350:
            score["scream"] += 1.2
            score["cry"] += 0.7
        elif f0 > 220:
            score["whine"] += 1.0
            score["cry"] += 0.6
        else:
            score["tense_speech"] += 0.6
    if zcr > 0.15:
        score["staccato"] += 0.6
        score["scream"] += 0.4
    # convert to dist
    keys=list(score.keys())
    x=np.array([score[k] for k in keys], dtype=np.float64)
    p=softmax(x)
    return {k: float(p[i]) for i,k in enumerate(keys)}

def extract_audio_states_per_sec(video_path: str, dt: float = 1.0) -> List[Dict[str,float]]:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "tmp.wav")
        extract_wav_from_video(video_path, wav, sr=16000)
        rms, zcr, f0 = audio_features_per_sec(wav, dt=dt, sr=16000)
        seq=[]
        for i in range(len(rms)):
            seq.append(map_audio_to_state_dist(float(rms[i]), float(zcr[i]), float(f0[i])))
        return seq
