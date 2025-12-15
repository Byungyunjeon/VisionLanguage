from typing import List, Dict, Tuple
import numpy as np

def sym_kl(p: Dict[str,float], q: Dict[str,float], keys) -> float:
    eps=1e-8
    pv = np.array([max(p.get(k,0.0), eps) for k in keys], dtype=np.float64)
    qv = np.array([max(q.get(k,0.0), eps) for k in keys], dtype=np.float64)
    pv /= pv.sum(); qv /= qv.sum()
    kl_pq = np.sum(pv * (np.log(pv) - np.log(qv)))
    kl_qp = np.sum(qv * (np.log(qv) - np.log(pv)))
    return float(0.5*(kl_pq+kl_qp))

def dtw_energy(obs: List[Dict[str,float]], proto: List[Dict[str,float]], keys, band: int = 10) -> float:
    # Classic DTW with Sakoe-Chiba band.
    n=len(obs); m=len(proto)
    INF=1e18
    D = np.full((n+1, m+1), INF, dtype=np.float64)
    D[0,0]=0.0
    for i in range(1,n+1):
        j0=max(1, i-band)
        j1=min(m, i+band)
        for j in range(j0, j1+1):
            c = sym_kl(obs[i-1], proto[j-1], keys)
            D[i,j] = c + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m] / max(1, n+m))

def energies_against_library(obs: List[Dict[str,float]], lib, dt: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    keys = lib.states
    ids=[]
    energies=[]
    from .library import expand_steps_to_frames
    for p in lib.prototypes:
        proto_seq = expand_steps_to_frames(p.steps, keys, dt=dt)
        e = dtw_energy(obs, proto_seq, keys, band=10)
        energies.append(e); ids.append(p.id)
    return np.array(energies, dtype=np.float64), ids
