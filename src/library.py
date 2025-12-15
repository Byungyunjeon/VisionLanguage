from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from .utils import normalize_dist

@dataclass
class Step:
    duration_sec: int
    dist: Dict[str, float]
    note: str = ""

@dataclass
class Prototype:
    id: str
    pattern: str
    steps: List[Step]
    meta: Dict[str, Any]

@dataclass
class PrototypeLibrary:
    name: str
    states: List[str]
    prototypes: List[Prototype]
    meta: Dict[str, Any]

def load_library(obj: Any, name: str) -> PrototypeLibrary:
    # Accept either already-normalized dict or older formats.
    if isinstance(obj, list):
        # legacy: list of {sequence_id, steps[{duration, emotion}]}
        # treat as vision library
        states = sorted({k for p in obj for st in p.get("steps", []) for k in st.get("emotion", {}).keys()})
        protos = []
        for p in obj:
            pid = p.get("sequence_id") or p.get("id")
            steps=[]
            for st in p.get("steps", []):
                d = st.get("emotion", st.get("dist", {}))
                steps.append(Step(int(st.get("duration", st.get("duration_sec", 1))), normalize_dist(d, states)))
            protos.append(Prototype(str(pid), p.get("pattern","unknown"), steps,
                                    {"narration_parent":p.get("narration_parent",""),
                                     "narration_technical":p.get("narration_technical","")}))
        return PrototypeLibrary(name=name, states=states, prototypes=protos, meta={"legacy": True})
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported library format for {name}: {type(obj)}")

    states = obj.get("states") or obj.get("audio_states") or obj.get("video_states")
    if not states:
        # infer
        states = sorted({k for p in obj.get("prototypes", []) for st in p.get("steps", []) for k in st.get("dist", st.get("state", {})).keys()})

    protos=[]
    for p in obj.get("prototypes", []):
        pid = p.get("id") or p.get("audio_id") or p.get("sequence_id")
        if pid is None:
            raise KeyError("Each prototype must have an 'id' field (or audio_id/sequence_id).")
        steps=[]
        for st in p.get("steps", []):
            dist = st.get("dist", st.get("state", {}))
            steps.append(Step(int(st.get("duration_sec", st.get("duration", 1))), normalize_dist(dist, states), st.get("note","")))
        meta = {k:v for k,v in p.items() if k not in ("steps",)}
        protos.append(Prototype(str(pid), p.get("pattern","unknown"), steps, meta))
    return PrototypeLibrary(name=name, states=list(states), prototypes=protos, meta=obj.get("meta", {}))

def expand_steps_to_frames(steps: List[Step], states: List[str], dt: float = 1.0) -> List[Dict[str,float]]:
    seq=[]
    for st in steps:
        n = max(1, int(round(st.duration_sec / dt)))
        for _ in range(n):
            seq.append(normalize_dist(st.dist, states))
    return seq
