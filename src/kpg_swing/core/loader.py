from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kpg_swing.engine.islanding import filter_to_main_component, pick_slack_bus_id
from kpg_swing.engine.dcflow import parse_mfile
from kpg_swing.engine.internal_kron import compute_K_and_Peq_from_arrays


@dataclass(frozen=True)
class LoadedSystem:
    # case
    bus: np.ndarray
    branch: np.ndarray
    baseMVA: float
    branch_raw_idx: np.ndarray   # (nl,) 원본 branch row index (0-based)

    # bus-level
    bus_ids: np.ndarray       # (nb,) MATPOWER bus id (1-based)
    Pd_pu: np.ndarray         # (nb,)
    Pbus_pu: np.ndarray       # (nb,)
    slack_bus_id: int

    # generator-level (no aggregation)
    gen_ids: np.ndarray       # (ng,) (int)
    gen_bus_ids: np.ndarray   # (ng,) MATPOWER bus id (1-based), 중복 허용
    Pg_pu: np.ndarray         # (ng,)
    M: np.ndarray             # (ng,)
    D: np.ndarray             # (ng,)
    xd_prime_pu: np.ndarray   # (ng,)

    # reduced
    K: np.ndarray             # (ng, ng)
    Peq: np.ndarray           # (ng,)
    delta_guess: np.ndarray   # (ng,)

    @property
    def ng(self) -> int:
        return int(self.gen_ids.shape[0])

    @property
    def nb(self) -> int:
        return int(self.bus.shape[0])
    




def load_system(
    case_mfile: str | Path,
    dyn_params_csv: str | Path,
    *,
    xdp0_pu_on_machine_base: float = 0.3,
    balance_on_slack: bool = True,
    slack_bus_id: int | None = None,
    drop_slack_gens: bool = True,
) -> LoadedSystem:
    """
    리빌드 헌법 기준 로더:
      - dyn_params.csv를 발전기 단위로 그대로 사용
      - Pbus = sum(Pg@bus) - Pd
      - x'_d는 (권장) xdp0 * (baseMVA / S_base_MVA) 로 시스템 base 변환
      - (K, Peq, delta_guess) 계산
    """
    bus, branch, baseMVA = parse_mfile(Path(case_mfile))
    baseMVA = float(baseMVA)

    bus_ids = bus[:, 0].astype(int)
    nb = bus.shape[0]


    dyn = pd.read_csv(dyn_params_csv)

    required = ["gen_id", "bus", "Pg", "S_base_MVA", "M", "D"]
    for c in required:
        if c not in dyn.columns:
            raise ValueError(f"dyn_params.csv에 컬럼 {c}가 없습니다. 현재: {list(dyn.columns)}")

    gen_ids = dyn["gen_id"].astype(int).to_numpy()
    gen_bus_ids = dyn["bus"].astype(int).to_numpy()  # MATPOWER bus id (1-based)
    Pg_pu = dyn["Pg"].astype(float).to_numpy()
    M = dyn["M"].astype(float).to_numpy()
    D = dyn["D"].astype(float).to_numpy()

    S_base = dyn["S_base_MVA"].astype(float).to_numpy()
    # 시스템 base로 변환한 xd'
    xd_prime_pu = (float(xdp0_pu_on_machine_base) * (baseMVA / S_base)).astype(float)

    # [ISLANDING] keep only the component that contains the slack bus
    if slack_bus_id is None:
        slack_bus_id = pick_slack_bus_id(bus)
    slack_bus_id = int(slack_bus_id)

    iso = filter_to_main_component(
        bus=bus,
        branch=branch,
        slack_bus_id=slack_bus_id,
        gen_bus_ids=gen_bus_ids,
    )

    # update case arrays
    bus = iso.bus
    branch = iso.branch
    branch_raw_idx = iso.branch_raw_idx
    bus_ids = bus[:, 0].astype(int)
    nb = bus.shape[0]

    # drop generators outside the main component
    if iso.gen_keep_mask is not None:
        m = iso.gen_keep_mask
        gen_ids = gen_ids[m]
        gen_bus_ids = gen_bus_ids[m]
        Pg_pu = Pg_pu[m]
        M = M[m]
        D = D[m]
        xd_prime_pu = xd_prime_pu[m]

    # [CHECKS]
    if slack_bus_id not in set(bus_ids.tolist()):
        raise RuntimeError("slack bus is not in the kept component (unexpected)")
    if not set(gen_bus_ids.tolist()).issubset(set(bus_ids.tolist())):
        raise RuntimeError("some generators are mapped to buses that are not in the filtered bus table")


    # MATPOWER bus PD는 3번째 컬럼(0-based 2)
    Pd_MW = bus[:, 2].astype(float)
    Pd_pu = Pd_MW / baseMVA


    # bus별 Pg 합산
    Pg_bus_pu = np.zeros(nb, dtype=float)
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}
    for g in range(gen_bus_ids.shape[0]):
        bi = bus_id_to_idx.get(int(gen_bus_ids[g]))
        if bi is None:
            raise ValueError(f"dyn_params의 발전기 bus={gen_bus_ids[g]}가 case bus에 없습니다")
        Pg_bus_pu[bi] += float(Pg_pu[g])

    Pbus_pu = Pg_bus_pu - Pd_pu

    # [OPTION] drop generators attached to slack bus
    # 이유: 현재 v2 Kron 축소에서 slack-bus gen internal nodes가 K 상에서 고립(components size 1)으로 남는 케이스가 있음.
    # 경향성/상관관계 분석 목적이면 해당 발전기들을 제외하고 진행 가능.
    if drop_slack_gens:
        keep_g = (gen_bus_ids.astype(int) != int(slack_bus_id))
        # slack에 붙은 발전기가 실제로 있으면 제거
        if not np.all(keep_g):
            gen_ids = gen_ids[keep_g]
            gen_bus_ids = gen_bus_ids[keep_g]
            Pg_pu = Pg_pu[keep_g]
            M = M[keep_g]
            D = D[keep_g]
            xd_prime_pu = xd_prime_pu[keep_g]


    # (K, Peq)
    K, delta_guess, Peq, slack_id = compute_K_and_Peq_from_arrays(
        bus=bus,
        branch=branch,
        gen_bus_ids=gen_bus_ids,
        P_bus_pu=Pbus_pu,
        xd_prime_pu=xd_prime_pu,
        slack_bus_id=slack_bus_id,
        balance_on_slack=balance_on_slack,
    )

    return LoadedSystem(
        bus=bus,
        branch=branch,
        baseMVA=baseMVA,
        bus_ids=bus_ids,
        branch_raw_idx=branch_raw_idx,
        Pd_pu=Pd_pu,
        Pbus_pu=Pbus_pu,
        slack_bus_id=int(slack_id),
        gen_ids=gen_ids,
        gen_bus_ids=gen_bus_ids,
        Pg_pu=Pg_pu,
        M=M,
        D=D,
        xd_prime_pu=xd_prime_pu,
        K=K,
        Peq=Peq,
        delta_guess=delta_guess,
    )
