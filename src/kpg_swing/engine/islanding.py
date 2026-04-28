from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def pick_slack_bus_id(bus: np.ndarray) -> int:
    """
    MATPOWER bus table에서 slack(타입=3) 버스 ID를 찾고,
    없으면 첫 bus ID를 slack으로 사용.
    """
    bus_ids = bus[:, 0].astype(int)
    bus_type = bus[:, 1].astype(int)
    idx = np.where(bus_type == 3)[0]
    if idx.size > 0:
        return int(bus_ids[int(idx[0])])
    return int(bus_ids[0])


def _build_adjacency(bus_ids: np.ndarray, branch: np.ndarray) -> List[List[int]]:
    """
    branch status=1인 라인만으로 무방향 그래프 인접 리스트 구성.
    bus_ids: (nb,) MATPOWER bus id (1-based)
    branch: (nl, ?) MATPOWER branch table
    """
    nb = bus_ids.shape[0]
    bus_id_to_idx: Dict[int, int] = {int(b): i for i, b in enumerate(bus_ids)}

    # MATPOWER branch col: fbus(0), tbus(1), ..., status(10)
    f_ids = branch[:, 0].astype(int)
    t_ids = branch[:, 1].astype(int)
    status = (branch[:, 10].astype(float) > 0.5)

    adj: List[List[int]] = [[] for _ in range(nb)]
    for k in range(branch.shape[0]):
        if not status[k]:
            continue
        fi = bus_id_to_idx.get(int(f_ids[k]))
        ti = bus_id_to_idx.get(int(t_ids[k]))
        if fi is None or ti is None:
            # case 파일 내부 불일치가 있으면 그냥 무시(보수적으로는 raise도 가능)
            continue
        if fi == ti:
            continue
        adj[fi].append(ti)
        adj[ti].append(fi)
    return adj


def connected_components(bus: np.ndarray, branch: np.ndarray) -> List[np.ndarray]:
    """
    현재 status=1 라인으로 연결 성분들을 반환.
    각 성분은 bus row index(0-based) 배열.
    """
    bus_ids = bus[:, 0].astype(int)
    adj = _build_adjacency(bus_ids, branch)

    nb = bus.shape[0]
    visited = np.zeros(nb, dtype=bool)
    comps: List[np.ndarray] = []

    for s in range(nb):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(np.array(comp, dtype=int))

    # 큰 성분부터 정렬해두면 디버그가 편함
    comps.sort(key=lambda a: a.size, reverse=True)
    return comps



@dataclass(frozen=True)
class IslandFilterResult:
    bus: np.ndarray
    branch: np.ndarray
    kept_bus_idx: np.ndarray
    old_to_new_bus_idx: np.ndarray
    gen_keep_mask: Optional[np.ndarray]
    slack_bus_id: int
    component_sizes: List[int]
    branch_raw_idx: np.ndarray     # (nl_kept,) 원본 branch row index (0-based)



def filter_to_main_component(
    bus: np.ndarray,
    branch: np.ndarray,
    *,
    slack_bus_id: Optional[int] = None,
    gen_bus_ids: Optional[np.ndarray] = None,  # (ng,) MATPOWER bus id (1-based)
) -> IslandFilterResult:
    """
    slack이 속한 연결 성분(메인 성분)만 남기고 나머지 섬은 제거한다.
    gen_bus_ids가 주어지면, 메인 성분 밖 발전기는 gen_keep_mask로 걸러낼 수 있게 한다.
    """
    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)

    bus_ids = bus[:, 0].astype(int)
    if slack_bus_id is None:
        slack_bus_id = pick_slack_bus_id(bus)
    slack_bus_id = int(slack_bus_id)

    if slack_bus_id not in set(bus_ids.tolist()):
        raise ValueError(f"slack_bus_id={slack_bus_id} not found in bus table")

    comps = connected_components(bus, branch)
    sizes = [int(c.size) for c in comps]

    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}
    slack_idx = bus_id_to_idx[slack_bus_id]

    # slack이 속한 성분 찾기
    main_comp = None
    for c in comps:
        if np.any(c == slack_idx):
            main_comp = c
            break
    if main_comp is None:
        raise RuntimeError("Failed to find slack component (unexpected)")

    kept_bus_idx = np.sort(main_comp)  # 정렬해두면 row slicing이 안정적
    nb_old = bus.shape[0]
    old_to_new = np.full(nb_old, -1, dtype=int)
    old_to_new[kept_bus_idx] = np.arange(kept_bus_idx.size, dtype=int)

    # bus 필터
    bus_f = bus[kept_bus_idx, :]

    # branch 필터: 양 끝 bus가 모두 kept에 포함된 라인만 남김
    kept_bus_id_set = set(bus_ids[kept_bus_idx].tolist())
    f_ids = branch[:, 0].astype(int)
    t_ids = branch[:, 1].astype(int)
    keep_branch = np.array(
        [(int(f_ids[k]) in kept_bus_id_set) and (int(t_ids[k]) in kept_bus_id_set)
         for k in range(branch.shape[0])],
        dtype=bool
    )
    branch_raw_idx = np.where(keep_branch)[0].astype(int)
    branch_f = branch[keep_branch, :]

    gen_keep_mask = None
    if gen_bus_ids is not None:
        gen_bus_ids = np.asarray(gen_bus_ids, dtype=int).reshape(-1)
        gen_keep_mask = np.array([int(b) in kept_bus_id_set for b in gen_bus_ids], dtype=bool)

    return IslandFilterResult(
        bus=bus_f,
        branch=branch_f,
        kept_bus_idx=kept_bus_idx,
        old_to_new_bus_idx=old_to_new,
        gen_keep_mask=gen_keep_mask,
        slack_bus_id=slack_bus_id,
        component_sizes=sizes,
        branch_raw_idx=branch_raw_idx,
    )
