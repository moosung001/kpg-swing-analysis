from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.integrate import solve_ivp
except Exception as e:  # scipy가 없을 수도 있으니 에러 메시지를 명확히
    solve_ivp = None
    _SCIPY_IMPORT_ERROR = e
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class SwingConfig:
    t0: float = 0.0
    t1: float = 20.0
    dt: float = 0.01
    method: str = "RK45"      # 필요하면 "Radau" 등으로 변경
    rtol: float = 1e-7
    atol: float = 1e-9


@dataclass(frozen=True)
class SwingResult:
    t: np.ndarray            # (nt,)
    delta: np.ndarray        # (nt, ng)
    omega: np.ndarray        # (nt, ng)
    success: bool
    message: str


def _build_rhs(K: np.ndarray, Peq: np.ndarray, M: np.ndarray, D: np.ndarray):
    """
    상태 y = [delta(ng), omega(ng)].
    d(delta)/dt = omega
    d(omega)/dt = M^{-1} ( Peq - D*omega - sum_j K_ij sin(delta_i - delta_j) )
    """
    K = np.asarray(K, dtype=float)
    Peq = np.asarray(Peq, dtype=float)
    M = np.asarray(M, dtype=float)
    D = np.asarray(D, dtype=float)

    ng = K.shape[0]
    if K.shape != (ng, ng):
        raise ValueError(f"K shape invalid: {K.shape}")
    for name, v in [("Peq", Peq), ("M", M), ("D", D)]:
        if v.shape != (ng,):
            raise ValueError(f"{name} shape invalid: {v.shape}, expected ({ng},)")

    if np.any(M <= 0):
        raise ValueError("M must be positive")
    Minv = 1.0 / M

    # 벡터화 계산을 위해 미리 K 복사
    Kmat = K.copy()

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        delta = y[:ng]
        omega = y[ng:]

        # coupling term: sum_j K_ij sin(delta_i - delta_j)
        # matrix form using broadcasting
        diff = delta[:, None] - delta[None, :]
        coupling = np.sum(Kmat * np.sin(diff), axis=1)

        domega = Minv * (Peq - D * omega - coupling)
        return np.concatenate([omega, domega])

    return rhs


def simulate_swing(
    K: np.ndarray,
    Peq: np.ndarray,
    M: np.ndarray,
    D: np.ndarray,
    *,
    delta0: Optional[np.ndarray] = None,
    omega0: Optional[np.ndarray] = None,
    cfg: SwingConfig = SwingConfig(),
) -> SwingResult:
    """
    최소 swing 적분기.
    - delta0 기본: 0
    - omega0 기본: 0
    """
    if solve_ivp is None:
        raise ImportError(
            "scipy가 필요합니다. `pip install scipy` 후 다시 실행하세요."
        ) from _SCIPY_IMPORT_ERROR

    K = np.asarray(K, dtype=float)
    ng = K.shape[0]

    if delta0 is None:
        delta0 = np.zeros(ng, dtype=float)
    else:
        delta0 = np.asarray(delta0, dtype=float).reshape(-1)
        if delta0.shape != (ng,):
            raise ValueError(f"delta0 shape mismatch: {delta0.shape} vs ({ng},)")

    if omega0 is None:
        omega0 = np.zeros(ng, dtype=float)
    else:
        omega0 = np.asarray(omega0, dtype=float).reshape(-1)
        if omega0.shape != (ng,):
            raise ValueError(f"omega0 shape mismatch: {omega0.shape} vs ({ng},)")

    y0 = np.concatenate([delta0, omega0])
    rhs = _build_rhs(K=K, Peq=Peq, M=M, D=D)

    t_eval = np.arange(cfg.t0, cfg.t1 + 0.5 * cfg.dt, cfg.dt, dtype=float)

    sol = solve_ivp(
        fun=rhs,
        t_span=(cfg.t0, cfg.t1),
        y0=y0,
        t_eval=t_eval,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    if not sol.success:
        return SwingResult(
            t=np.asarray(sol.t, dtype=float),
            delta=np.zeros((0, ng), dtype=float),
            omega=np.zeros((0, ng), dtype=float),
            success=False,
            message=str(sol.message),
        )

    Y = sol.y.T  # (nt, 2ng)
    delta = Y[:, :ng]
    omega = Y[:, ng:]
    return SwingResult(
        t=np.asarray(sol.t, dtype=float),
        delta=delta,
        omega=omega,
        success=True,
        message=str(sol.message),
    )



def solve_swing_ivp(
    *,
    K,
    Peq,
    M,
    D,
    delta0,
    omega0,
    t_span,
    t_eval,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "RK45",
    step_t: float | None = None,
    step_gen_idx: int | None = None,
    step_dPeq: float = 0.0,
    disturbance=None,
):
    """
    Public stable API (헌법): scipy.solve_ivp 스타일로 swing ODE를 적분한다.

    상태: y = [delta(ng); omega(ng)]
    동역학:
      d(delta)/dt = omega
      d(omega)/dt = (Peq_eff(t) + u(t) - Pe(delta) - D*omega) / M
      Pe(delta)_i = sum_j K_ij * sin(delta_i - delta_j)

    - step_* : 특정 시각 step_t 이후에 Peq에 발전기 인덱스 step_gen_idx로 step_dPeq를 더함
    - disturbance : callable(t)-> (ng,) 또는 scalar(0.0) 로 외란을 추가 주입(기본 0)
      (step과 disturbance는 합산된다)
    """
    import numpy as np
    from scipy.integrate import solve_ivp as _solve_ivp

    # ---- normalize inputs ----
    K = np.asarray(K, dtype=float)
    Peq = np.asarray(Peq, dtype=float).reshape(-1)
    M = np.asarray(M, dtype=float).reshape(-1)
    D = np.asarray(D, dtype=float).reshape(-1)

    ng = int(Peq.shape[0])
    if K.shape != (ng, ng):
        raise ValueError(f"K shape mismatch: {K.shape} vs ({ng},{ng})")
    if M.shape[0] != ng or D.shape[0] != ng:
        raise ValueError(f"M/D length mismatch: M={M.shape}, D={D.shape}, ng={ng}")

    delta0 = np.asarray(delta0, dtype=float).reshape(-1)
    omega0 = np.asarray(omega0, dtype=float).reshape(-1)
    if delta0.shape[0] != ng or omega0.shape[0] != ng:
        raise ValueError(f"delta0/omega0 length mismatch: {delta0.shape}, {omega0.shape}, ng={ng}")

    t0, tf = float(t_span[0]), float(t_span[1])
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size == 0:
        raise ValueError("t_eval must be non-empty")
    if not (t_eval[0] >= t0 - 1e-12 and t_eval[-1] <= tf + 1e-12):
        raise ValueError("t_eval must lie within t_span")

    y0 = np.concatenate([delta0, omega0], axis=0)

    # ---- step setup (optional) ----
    step_enabled = (step_t is not None) and (step_gen_idx is not None) and (float(step_dPeq) != 0.0)
    if step_enabled:
        step_t = float(step_t)
        gi = int(step_gen_idx)
        if not (0 <= gi < ng):
            raise ValueError(f"step_gen_idx out of range: {gi} (ng={ng})")
        dPeq_vec = np.zeros(ng, dtype=float)
        dPeq_vec[gi] = float(step_dPeq)
    else:
        step_t = None
        dPeq_vec = np.zeros(ng, dtype=float)

    # ---- ODE ----
    def f(t, y):
        delta = y[:ng]
        omega = y[ng:]

        # electrical coupling
        dmat = delta[:, None] - delta[None, :]
        Pe = np.sum(K * np.sin(dmat), axis=1)

        # disturbance u(t)
        u = 0.0
        if disturbance is not None:
            u = disturbance(float(t))
        if np.isscalar(u):
            u_vec = float(u)
        else:
            u_vec = np.asarray(u, dtype=float).reshape(-1)
            if u_vec.shape[0] != ng:
                raise ValueError(f"disturbance(t) must return (ng,), got {u_vec.shape}")

        # step modifies Peq after step_t
        if step_t is not None and (float(t) >= step_t):
            Peq_eff = Peq + dPeq_vec
        else:
            Peq_eff = Peq

        d_delta = omega
        if np.isscalar(u_vec):
            d_omega = (Peq_eff + u_vec - Pe - D * omega) / M
        else:
            d_omega = (Peq_eff + u_vec - Pe - D * omega) / M

        return np.concatenate([d_delta, d_omega], axis=0)

    sol = _solve_ivp(
        fun=f,
        t_span=(t0, tf),
        y0=y0,
        t_eval=t_eval,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    )
    return sol

