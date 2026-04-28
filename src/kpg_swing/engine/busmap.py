from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class BusMap:
    """
    bus id(1-based, MATPOWER) <-> row index(0-based, 현재 bus 배열) 변환을 한 곳에서만 책임진다.
    - bus_ids: (nb,) bus[:,0] 값
    - id_to_idx: bus id -> row index
    """
    bus_ids: np.ndarray
    id_to_idx: Dict[int, int]

    def idx_of(self, bus_id: int) -> int:
        """bus id를 현재 bus 배열 row index로 변환. 없으면 KeyError."""
        return self.id_to_idx[int(bus_id)]

    def maybe_idx_of(self, bus_id: int) -> int | None:
        """bus id를 row index로 변환. 없으면 None."""
        return self.id_to_idx.get(int(bus_id))

    def ids_to_indices(self, bus_ids: np.ndarray) -> np.ndarray:
        """(n,) bus id 배열을 (n,) row index 배열로 변환. 하나라도 없으면 ValueError."""
        bus_ids = np.asarray(bus_ids, dtype=int).reshape(-1)
        out = np.empty(bus_ids.shape[0], dtype=int)
        for k, bid in enumerate(bus_ids.tolist()):
            idx = self.id_to_idx.get(int(bid))
            if idx is None:
                raise ValueError(f"bus id {bid} is not present in current bus table")
            out[k] = idx
        return out

    def in_table_mask(self, bus_ids: np.ndarray) -> np.ndarray:
        """(n,) bus id 배열이 현재 bus table에 존재하는지 마스크."""
        bus_ids = np.asarray(bus_ids, dtype=int).reshape(-1)
        return np.array([int(b) in self.id_to_idx for b in bus_ids.tolist()], dtype=bool)


def make_busmap(bus: np.ndarray) -> BusMap:
    """
    bus[:,0]을 기준으로 BusMap 생성.
    """
    bus = np.asarray(bus)
    if bus.ndim != 2 or bus.shape[1] < 1:
        raise ValueError("bus must be a 2D array with at least 1 column (bus id).")

    bus_ids = bus[:, 0].astype(int)
    # 중복은 허용하지 않음 (MATPOWER bus id는 고유해야 정상)
    if np.unique(bus_ids).shape[0] != bus_ids.shape[0]:
        raise ValueError("duplicate bus ids found in bus table")

    id_to_idx: Dict[int, int] = {int(b): i for i, b in enumerate(bus_ids.tolist())}
    return BusMap(bus_ids=bus_ids, id_to_idx=id_to_idx)


def filter_branch_by_bus_ids(branch: np.ndarray, bus_id_set: set[int]) -> np.ndarray:
    """
    branch의 (fbus,tbus)가 bus_id_set에 모두 포함되는 row만 남김.
    branch는 MATPOWER branch table (1-based bus id가 들어있음)을 가정.
    """
    branch = np.asarray(branch, dtype=float)
    f = branch[:, 0].astype(int)
    t = branch[:, 1].astype(int)
    keep = np.array([(int(f[k]) in bus_id_set) and (int(t[k]) in bus_id_set) for k in range(branch.shape[0])], dtype=bool)
    return branch[keep, :]
