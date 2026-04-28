from __future__ import annotations
import numpy as np


def step_on_generator(*, ng: int, t0: float, gen_idx: int, dPeq: float):
    """
    발전기 기준 step 외란.
    return: disturbance(t) -> (ng,) numpy array
      - t >= t0 이면 u[gen_idx] = dPeq
      - 그 외 0
    dPeq 단위는 Peq와 동일 (pu 기준을 쓰는 게 가장 안전)
    """
    ng = int(ng)
    t0 = float(t0)
    gen_idx = int(gen_idx)
    dPeq = float(dPeq)

    if not (0 <= gen_idx < ng):
        raise ValueError(f"gen_idx out of range: {gen_idx} (ng={ng})")

    u = np.zeros(ng, dtype=float)
    u[gen_idx] = dPeq

    def disturbance(t: float):
        return u if float(t) >= t0 else 0.0

    return disturbance
