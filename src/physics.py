import numpy as np
from typing import Dict, List, Tuple

def boltzmann_weights(energies: np.ndarray, temperature: float = 0.2) -> np.ndarray:
    # w_i ∝ exp(-E_i / T)
    T = max(1e-6, float(temperature))
    e = energies - energies.min()
    w = np.exp(-e / T)
    s = w.sum()
    return w / (s if s>0 else 1.0)

def glauber_flip_prob(deltaE: float, temperature: float = 0.2) -> float:
    # early-20c-ish "single spin" update rule; used as a nonlinearity for "meltdown" gate.
    T=max(1e-6,float(temperature))
    return float(1.0 / (1.0 + np.exp(deltaE / T)))

def landau_barrier(prob: float, a: float = 1.0, b: float = 1.0) -> float:
    # simple double-well potential proxy: V(p)=a p^4 - b p^2
    p = np.clip(prob, 0.0, 1.0)
    return float(a*(p**4) - b*(p**2))
