from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass(frozen=True)
class LineMap:
    raw_to_local: Dict[int, int]

def make_linemap(branch_raw_idx: np.ndarray) -> LineMap:
    branch_raw_idx = np.asarray(branch_raw_idx, dtype=int).reshape(-1)
    return LineMap(raw_to_local={int(r): i for i, r in enumerate(branch_raw_idx.tolist())})

def resolve_line_local_idx(linemap: LineMap, line_raw_idx: int) -> int:
    i = linemap.raw_to_local.get(int(line_raw_idx))
    if i is None:
        raise KeyError(f"line_raw_idx={line_raw_idx} is not present after island filtering")
    return int(i)
