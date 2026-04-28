from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from kpg_swing.engine.dcflow import build_B_and_meta


@dataclass(frozen=True)
class BusRestoreMap:
    """
    v2(B) formulation에서 internal node(발전기) 상태로부터 bus 각도/주파수를 복원하는 선형 맵.

    - slack bus angle은 기준각 0으로 고정
    - 복원식(슬랙 제외 버스 e, internal k):
        theta_e = -Bee^{-1} Bek * theta_k
      따라서 omega도:
        omega_e = -Bee^{-1} Bek * omega_k
    """
    bus_ids: np.ndarray          # (nb,) MATPOWER bus id (1-based)
    slack_bus_id: int
    slack_idx: int              # bus array index
    elim_bus_idx: np.ndarray     # (nb-1,) slack 제외 bus index
    T_bus_from_int: np.ndarray   # (nb-1, ng)

    def restore_bus_angle(self, delta_int: np.ndarray) -> np.ndarray:
        delta_int = np.asarray(delta_int, dtype=float)

        nb = int(self.bus_ids.shape[0])
        if delta_int.ndim == 1:
            theta_elim = self.T_bus_from_int @ delta_int  # (nb-1,)
            theta_bus = np.zeros((nb,), dtype=float)
            theta_bus[self.elim_bus_idx] = theta_elim
            theta_bus[self.slack_idx] = 0.0
            return theta_bus

        if delta_int.ndim != 2:
            raise ValueError("delta_int must be (ng,) or (nt, ng)")

        nt, ng = delta_int.shape
        if self.T_bus_from_int.shape[1] != ng:
            raise ValueError(f"ng mismatch: map expects {self.T_bus_from_int.shape[1]}, got {ng}")

        theta_elim = (self.T_bus_from_int @ delta_int.T).T  # (nt, nb-1)
        theta_bus = np.zeros((nt, nb), dtype=float)
        theta_bus[:, self.elim_bus_idx] = theta_elim
        theta_bus[:, self.slack_idx] = 0.0
        return theta_bus

    def restore_bus_omega(self, omega_int: np.ndarray) -> np.ndarray:
        omega_int = np.asarray(omega_int, dtype=float)

        nb = int(self.bus_ids.shape[0])
        if omega_int.ndim == 1:
            omega_elim = self.T_bus_from_int @ omega_int
            omega_bus = np.zeros((nb,), dtype=float)
            omega_bus[self.elim_bus_idx] = omega_elim
            omega_bus[self.slack_idx] = 0.0
            return omega_bus

        if omega_int.ndim != 2:
            raise ValueError("omega_int must be (ng,) or (nt, ng)")

        nt, ng = omega_int.shape
        if self.T_bus_from_int.shape[1] != ng:
            raise ValueError(f"ng mismatch: map expects {self.T_bus_from_int.shape[1]}, got {ng}")

        omega_elim = (self.T_bus_from_int @ omega_int.T).T
        omega_bus = np.zeros((nt, nb), dtype=float)
        omega_bus[:, self.elim_bus_idx] = omega_elim
        omega_bus[:, self.slack_idx] = 0.0
        return omega_bus


def build_bus_restore_map_from_arrays(
    *,
    bus: np.ndarray,
    branch: np.ndarray,
    gen_bus_ids: np.ndarray,    # (ng,) MATPOWER bus id (1-based), duplicates allowed
    xd_prime_pu: float | np.ndarray,
    slack_bus_id: int,
) -> BusRestoreMap:
    """
    v2 internal_kron에서 쓰는 것과 동일한 방식으로 Bext를 구성하고,
    theta_bus = -Bee^{-1}Bek * theta_int 맵을 만든다.
    """
    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)

    bus_ids = bus[:, 0].astype(int)
    nb = int(bus.shape[0])

    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids.tolist())}
    slack_bus_id = int(slack_bus_id)
    if slack_bus_id not in bus_id_to_idx:
        raise ValueError(f"slack_bus_id={slack_bus_id} not found in current bus table")

    slack_idx = int(bus_id_to_idx[slack_bus_id])

    # Bbus
    Bbus, _meta = build_B_and_meta(bus, branch)

    # generator bus indices (0-based)
    gen_bus_ids = np.asarray(gen_bus_ids, dtype=int).reshape(-1)
    ng = int(gen_bus_ids.shape[0])
    gen_bus_idx = np.array([bus_id_to_idx[int(b)] for b in gen_bus_ids.tolist()], dtype=int)

    # xd' vector
    if np.isscalar(xd_prime_pu):
        xd_vec = np.full(ng, float(xd_prime_pu), dtype=float)
    else:
        xd_vec = np.asarray(xd_prime_pu, dtype=float).reshape(-1)
        if xd_vec.shape[0] != ng:
            raise ValueError(f"xd_prime_pu length mismatch: got {xd_vec.shape[0]}, expected {ng}")
    if np.any(xd_vec <= 0):
        raise ValueError("xd_prime_pu must be positive")

    # Bext: [bus nb] + [internal ng]
    n_ext = nb + ng
    Bext = np.zeros((n_ext, n_ext), dtype=float)
    Bext[:nb, :nb] = Bbus

    # connect internal node to terminal bus via xd'
    for k, bi in enumerate(gen_bus_idx):
        gi = nb + k
        b_xd = -1.0 / float(xd_vec[k])
        Bext[bi, gi] += b_xd
        Bext[gi, bi] += b_xd
        Bext[bi, bi] -= b_xd
        Bext[gi, gi] -= b_xd

    # eliminate: all buses except slack
    elim_bus = np.array([i for i in range(nb) if i != slack_idx], dtype=int)
    keep_int = np.arange(nb, n_ext, dtype=int)

    Bee = Bext[np.ix_(elim_bus, elim_bus)]
    Bek = Bext[np.ix_(elim_bus, keep_int)]

    try:
        X = np.linalg.solve(Bee, Bek)
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(Bee) @ Bek

    T = -X  # (nb-1, ng)
    return BusRestoreMap(
        bus_ids=bus_ids,
        slack_bus_id=slack_bus_id,
        slack_idx=slack_idx,
        elim_bus_idx=elim_bus,
        T_bus_from_int=T,
    )
