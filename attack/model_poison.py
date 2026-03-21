from __future__ import annotations

from typing import List

import numpy as np

from .config import ModelPoisonConfig
from .utils import flatten_ndarrays, l2_norm, orthogonal_noise, unflatten_like

NDArrays = List[np.ndarray]


def poison_delta(delta: NDArrays, cfg: ModelPoisonConfig, seed: int | None = None) -> NDArrays:
    """Poison a client update (delta arrays) in a deterministic way."""
    if not cfg.enabled:
        return delta

    mode = str(cfg.mode).lower().strip()
    strength = float(cfg.strength)
    if strength <= 0:
        return delta

    rng = np.random.default_rng(int(seed if seed is not None else cfg.seed))

    if mode in {"signflip", "scale"}:
        out = []
        for d in delta:
            d = np.asarray(d)
            if mode == "signflip":
                out.append(np.asarray((-strength) * d))
            else:
                out.append(np.asarray((strength) * d))
        return out

    # stealth: preserve norm and keep cosine high w.r.t original delta
    v = flatten_ndarrays(delta)
    nrm = l2_norm(v) + 1e-12
    direction = (v / nrm).astype(np.float64, copy=False)

    beta = float(cfg.stealth_beta)
    beta = max(0.0, min(0.99, beta))
    # Choose alpha so that ||alpha*dir + beta*noise|| = 1 (noise is unit and orthogonal).
    alpha = float(np.sqrt(max(1e-12, 1.0 - beta * beta)))
    noise = orthogonal_noise(direction, rng)
    pv = nrm * (alpha * direction + beta * noise)

    # Optional scaling
    pv = strength * pv

    # Clip to avoid exploding weights
    max_scale = float(cfg.max_scale) if float(cfg.max_scale) > 0 else 2.0
    clip = max_scale * nrm
    pv = np.clip(pv, -clip, clip)

    return unflatten_like(pv, delta)

