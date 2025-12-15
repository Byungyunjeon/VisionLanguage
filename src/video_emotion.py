from typing import List, Dict
import numpy as np

# Map DeepFace labels -> our prototype labels (coarse).
DF_TO_PROTO = {
    "neutral": "calm",
    "happy": "happy",
    "sad": "sad",
    "fear": "fear",
    "surprise": "worry",
    "angry": "worry",
    "disgust": "worry",
}

def extract_video_emotions_per_sec(video_path: str, dt: float = 1.0) -> List[Dict[str,float]]:
    """Return per-second emotion distribution using DeepFace on sampled frames.
    We *coarsely* map DeepFace's 7-class emotions to our Step1 prototype labels.
    """
    import cv2
    from deepface import DeepFace

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps*dt)))
    idx=0
    seq=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            try:
                r = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(r, list):
                    r=r[0]
                emo = r.get('emotion', {})
                total = sum(max(float(v),0.0) for v in emo.values())
                if total <= 1e-9:
                    seq.append({'calm': 1.0})
                else:
                    # aggregate
                    agg={}
                    for k,v in emo.items():
                        kk = DF_TO_PROTO.get(k, None)
                        if kk is None: 
                            continue
                        agg[kk] = agg.get(kk,0.0) + float(v)/total
                    # normalize
                    s=sum(agg.values())
                    if s<=1e-9:
                        agg={'calm':1.0}
                    else:
                        agg={k:v/s for k,v in agg.items()}
                    seq.append(agg)
            except Exception:
                seq.append({'calm': 1.0})
        idx += 1
    cap.release()
    return seq
