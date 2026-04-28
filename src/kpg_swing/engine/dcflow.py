from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _read_mpc_matrix(lines: list[str], start_key: str) -> list[list[float]]:
    """
    start_key 예: "mpc.bus", "mpc.branch"
    mpc.bus = [
      ...;
    ];
    형태에서 숫자 행렬만 추출.
    """
    out: list[list[float]] = []
    sec = False
    for ln in lines:
        s = ln.strip()
        if s.startswith(start_key):
            sec = True
            continue
        if sec:
            if s.endswith("];"):
                break
            if (not s) or s.startswith("%") or "[" in s:
                continue
            nums = [float(x) for x in re.findall(_NUM, s)]
            if nums:
                out.append(nums)
    return out


def parse_mfile(mfile: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return:
      bus: (nb, ?)  MATPOWER bus table
      branch: (nl, ?) MATPOWER branch table
      baseMVA: float
    """
    mfile = Path(mfile)
    txt = mfile.read_text(encoding="utf-8", errors="ignore").splitlines()

    baseMVA = 100.0
    for ln in txt:
        s = ln.strip()
        if s.startswith("mpc.baseMVA"):
            m = re.search(_NUM, s)
            if m:
                baseMVA = float(m.group())
            break

    bus_rows = _read_mpc_matrix(txt, "mpc.bus")
    branch_rows = _read_mpc_matrix(txt, "mpc.branch")

    if not bus_rows:
        raise ValueError(f"mpc.bus를 파싱하지 못했습니다: {mfile}")
    if not branch_rows:
        raise ValueError(f"mpc.branch를 파싱하지 못했습니다: {mfile}")

    bus = np.array(bus_rows, dtype=float)
    branch = np.array(branch_rows, dtype=float)
    return bus, branch, float(baseMVA)


@dataclass(frozen=True)
class DCBranchMeta:
    """
    라인 메타(원하면 outage 샘플링, 시각화 등에 사용)
    인덱스는 0-based bus index 기준.
    """
    f: np.ndarray
    t: np.ndarray
    x: np.ndarray
    ratio: np.ndarray
    status: np.ndarray

    


def build_B_and_meta(bus: np.ndarray, branch: np.ndarray) -> tuple[np.ndarray, DCBranchMeta]:
    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)

    nb = bus.shape[0]

    bus_ids = bus[:, 0].astype(int)
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}

    fbus_id = branch[:, 0].astype(int)
    tbus_id = branch[:, 1].astype(int)
    x = branch[:, 3].astype(float)

    if np.any(x == 0):
        raise ValueError("branch x=0이 존재합니다. DC B 구성 불가")

    ratio = branch[:, 8].astype(float).copy()
    ratio[ratio == 0] = 1.0

    status = (branch[:, 10].astype(float) > 0.5)

    B = np.zeros((nb, nb), dtype=float)
    bij = 1.0 / x

    for k in range(branch.shape[0]):
        if not status[k]:
            continue

        fi = bus_id_to_idx.get(int(fbus_id[k]))
        ti = bus_id_to_idx.get(int(tbus_id[k]))
        if fi is None or ti is None:
            continue

        i = fi
        j = ti

        b = float(bij[k])
        tau = float(ratio[k])

        B[i, i] += b / tau
        B[j, j] += b / tau
        B[i, j] += -b / tau
        B[j, i] += -b / tau


    meta = DCBranchMeta(f=fbus_id, t=tbus_id, x=x, ratio=ratio, status=status)
    return B, meta
