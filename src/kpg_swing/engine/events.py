from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from kpg_swing.engine.dcflow import build_B_and_meta
from kpg_swing.engine.swing_api import solve_swing_ivp
from kpg_swing.engine.internal_kron import compute_K_and_Peq_from_arrays


@dataclass(frozen=True)
class LineOutageResult:
    success: bool
    message: str
    t: np.ndarray
    y: np.ndarray                 # (nt, 2ng)  [delta, omega]
    K_post: np.ndarray | None     # outage 이후 K
    line_local_idx: int
    reason_invalid: str | None


@dataclass(frozen=True)
class GenTripResult:
    success: bool
    message: str
    t: np.ndarray
    y: np.ndarray                 # (nt, 2ng_post)  [delta, omega]  <- post 차원 기준으로 통일
    K_post: np.ndarray | None
    Peq_post: np.ndarray | None
    trip_gen_id: int
    trip_gen_local_idx: int
    reason_invalid: str | None

    # post-system meta (차원감소 때문에 필요)
    gen_ids_post: np.ndarray | None = None
    gen_bus_ids_post: np.ndarray | None = None
    Pg_pu_post: np.ndarray | None = None
    M_post: np.ndarray | None = None
    D_post: np.ndarray | None = None
    xd_prime_pu_post: np.ndarray | None = None
    slack_bus_id_post: int | None = None

    # deficit info (선택적으로 디버깅에 유용)
    Pbus_sum_post: float | None = None
    Peq_sum_post: float | None = None
    trip_Pg_pu: float | None = None
    trip_bus_id: int | None = None


def _compute_K_only_from_arrays(
    *,
    bus: np.ndarray,
    branch: np.ndarray,
    gen_bus_ids: np.ndarray,
    xd_prime_pu: np.ndarray,
    slack_bus_id: int,
) -> np.ndarray:
    """
    internal_kron.compute_K_and_Peq_from_arrays()에서 K 계산 부분만 떼어낸 버전.
    - K는 topology(Bext)로만 결정되므로 Pbus(Peq) 없이도 계산 가능.
    - 라인 아웃 이벤트에서 "K만 바뀌고 Peq는 고정" 규약을 만족시키기 위해 필요.
    """
    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)
    gen_bus_ids = np.asarray(gen_bus_ids, dtype=int).reshape(-1)
    xd_prime_pu = np.asarray(xd_prime_pu, dtype=float).reshape(-1)

    nb = int(bus.shape[0])
    ng = int(gen_bus_ids.shape[0])
    if xd_prime_pu.shape[0] != ng:
        raise ValueError(f"xd_prime_pu length mismatch: {xd_prime_pu.shape[0]} vs ng={ng}")

    bus_ids = bus[:, 0].astype(int)
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}
    slack_idx = bus_id_to_idx.get(int(slack_bus_id))
    if slack_idx is None:
        raise ValueError(f"slack_bus_id={slack_bus_id} not found in bus table")

    Bbus, _meta = build_B_and_meta(bus, branch)

    n_ext = nb + ng
    Bext = np.zeros((n_ext, n_ext), dtype=float)
    Bext[:nb, :nb] = Bbus

    for gi in range(ng):
        b = 1.0 / float(xd_prime_pu[gi])
        bi = bus_id_to_idx.get(int(gen_bus_ids[gi]))
        if bi is None:
            raise ValueError(f"gen_bus_id {int(gen_bus_ids[gi])} not found in bus table")
        ii = nb + gi

        Bext[bi, bi] += b
        Bext[ii, ii] += b
        Bext[bi, ii] += -b
        Bext[ii, bi] += -b

    elim_bus = np.array([i for i in range(nb) if i != int(slack_idx)], dtype=int)
    keep_int = np.arange(nb, n_ext, dtype=int)

    Bee = Bext[np.ix_(elim_bus, elim_bus)]
    Bek = Bext[np.ix_(elim_bus, keep_int)]
    Bke = Bext[np.ix_(keep_int, elim_bus)]
    Bkk = Bext[np.ix_(keep_int, keep_int)]

    try:
        X = np.linalg.solve(Bee, Bek)
    except np.linalg.LinAlgError:
        Bee_pinv = np.linalg.pinv(Bee)
        X = Bee_pinv @ Bek

    Bred = Bkk - Bke @ X
    Bred = 0.5 * (Bred + Bred.T)
    K = -Bred
    return K


def _count_components_of_K(K: np.ndarray, eps: float = 1e-9) -> int:
    K = np.asarray(K, dtype=float)
    K_off = K.copy()
    np.fill_diagonal(K_off, 0.0)
    adj = (np.abs(K_off) > float(eps))

    n = int(K.shape[0])
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def _resolve_gen_local_idx(gen_ids: np.ndarray, trip_gen_id: int) -> int:
    gen_ids = np.asarray(gen_ids, dtype=int).reshape(-1)
    trip_gen_id = int(trip_gen_id)
    hit = np.where(gen_ids == trip_gen_id)[0]
    if hit.size == 0:
        raise ValueError(f"trip_gen_id={trip_gen_id} not found in sysobj.gen_ids")
    return int(hit[0])


def simulate_line_outage_piecewise(
    *,
    sysobj,
    line_local_idx: int,
    t_event: float,
    t_final: float,
    dt: float,
    D_used: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "RK45",
) -> LineOutageResult:
    t_event = float(t_event)
    t_final = float(t_final)
    dt = float(dt)

    if not (0.0 < t_event < t_final):
        return LineOutageResult(
            success=False,
            message="invalid time window",
            t=np.zeros(0),
            y=np.zeros((0, 0)),
            K_post=None,
            line_local_idx=int(line_local_idx),
            reason_invalid="bad_time_window",
        )

    branch = np.asarray(sysobj.branch, dtype=float)
    nl = int(branch.shape[0])
    li = int(line_local_idx)
    if not (0 <= li < nl):
        return LineOutageResult(
            success=False,
            message=f"line_local_idx out of range: {li} (nl={nl})",
            t=np.zeros(0),
            y=np.zeros((0, 0)),
            K_post=None,
            line_local_idx=li,
            reason_invalid="bad_line_index",
        )

    branch2 = branch.copy()
    branch2[li, 10] = 0.0

    try:
        K1 = _compute_K_only_from_arrays(
            bus=sysobj.bus,
            branch=branch2,
            gen_bus_ids=sysobj.gen_bus_ids,
            xd_prime_pu=sysobj.xd_prime_pu,
            slack_bus_id=int(sysobj.slack_bus_id),
        )
    except Exception as e:
        return LineOutageResult(
            success=False,
            message=f"K recompute failed: {e}",
            t=np.zeros(0),
            y=np.zeros((0, 0)),
            K_post=None,
            line_local_idx=li,
            reason_invalid="K_recompute_failed",
        )

    comps = _count_components_of_K(K1, eps=1e-9)
    if comps != 1:
        return LineOutageResult(
            success=False,
            message=f"invalid: K has {comps} components after outage",
            t=np.zeros(0),
            y=np.zeros((0, 0)),
            K_post=K1,
            line_local_idx=li,
            reason_invalid="islanding_after_line_outage",
        )

    ng = int(sysobj.ng)
    delta0 = np.asarray(sysobj.delta_guess, dtype=float).reshape(-1)
    omega0 = np.zeros(ng, dtype=float)

    t_eval_pre = np.arange(0.0, t_event + 0.5 * dt, dt, dtype=float)
    t_eval_pre = t_eval_pre[(t_eval_pre >= 0.0) & (t_eval_pre <= t_event)]

    t_eval_post = np.arange(t_event, t_final + 0.5 * dt, dt, dtype=float)
    t_eval_post = t_eval_post[(t_eval_post >= t_event) & (t_eval_post <= t_final)]

    if t_eval_pre.size == 0 or abs(t_eval_pre[-1] - t_event) > 1e-12:
        t_eval_pre = np.append(t_eval_pre, t_event)
    if t_eval_post.size == 0 or abs(t_eval_post[0] - t_event) > 1e-12:
        t_eval_post = np.insert(t_eval_post, 0, t_event)
    if abs(t_eval_post[-1] - t_final) > 1e-12:
        t_eval_post = np.append(t_eval_post, t_final)

    sol0 = solve_swing_ivp(
        K=sysobj.K,
        Peq=sysobj.Peq,
        M=sysobj.M,
        D=D_used,
        delta0=delta0,
        omega0=omega0,
        t_span=(0.0, t_event),
        t_eval=t_eval_pre,
        rtol=rtol,
        atol=atol,
        method=method,
        disturbance=None,
    )
    if not bool(sol0.success):
        return LineOutageResult(
            success=False,
            message=str(getattr(sol0, "message", "pre segment failed")),
            t=np.asarray(sol0.t, dtype=float),
            y=np.asarray(sol0.y.T, dtype=float) if hasattr(sol0, "y") else np.zeros((0, 0)),
            K_post=K1,
            line_local_idx=li,
            reason_invalid="pre_segment_failed",
        )

    y_end = np.asarray(sol0.y[:, -1], dtype=float)
    delta1 = y_end[:ng]
    omega1 = y_end[ng:]

    sol1 = solve_swing_ivp(
        K=K1,
        Peq=sysobj.Peq,     # 규약: Peq 고정
        M=sysobj.M,
        D=D_used,
        delta0=delta1,
        omega0=omega1,
        t_span=(t_event, t_final),
        t_eval=t_eval_post,
        rtol=rtol,
        atol=atol,
        method=method,
        disturbance=None,
    )
    if not bool(sol1.success):
        return LineOutageResult(
            success=False,
            message=str(getattr(sol1, "message", "post segment failed")),
            t=np.asarray(sol1.t, dtype=float),
            y=np.asarray(sol1.y.T, dtype=float) if hasattr(sol1, "y") else np.zeros((0, 0)),
            K_post=K1,
            line_local_idx=li,
            reason_invalid="post_segment_failed",
        )

    t0 = np.asarray(sol0.t, dtype=float)
    Y0 = np.asarray(sol0.y.T, dtype=float)
    t1 = np.asarray(sol1.t, dtype=float)
    Y1 = np.asarray(sol1.y.T, dtype=float)

    if t0.size > 0 and t1.size > 0 and abs(float(t0[-1]) - float(t1[0])) < 1e-12:
        t = np.concatenate([t0, t1[1:]], axis=0)
        Y = np.concatenate([Y0, Y1[1:, :]], axis=0)
    else:
        t = np.concatenate([t0, t1], axis=0)
        Y = np.concatenate([Y0, Y1], axis=0)

    return LineOutageResult(
        success=True,
        message="ok",
        t=t,
        y=Y,
        K_post=K1,
        line_local_idx=li,
        reason_invalid=None,
    )


def simulate_gen_trip_piecewise(
    *,
    sysobj,
    trip_gen_id: int,           # 규약: 원본/영구 gen_id 입력
    t_event: float,
    t_final: float,
    dt: float,
    D_used: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "RK45",
) -> GenTripResult:
    """
    젠트립 이벤트 (정석 + 결손 유지 버전):
      - 발전기 내부노드 제거로 ng' = ng-1
      - post에서 (K1, Peq1)를 "다시 계산"
        * Pbus_post는 trip 발전기의 Pg를 해당 bus에서 제거하여 ΣP<0 결손을 남긴다.
        * balance_on_slack=False (slack이 결손을 자동 보상하지 않음)
        * center_Peq=False (평형 강제 평균제거로 결손이 지워지는 것을 방지)
        * solve_delta_guess=False (결손이면 비선형 평형각 풀이가 불필요/실패 가능)
      - 반환 y는 post 차원(ng') 기준으로 통일하여 pre 구간도 동일 차원으로 투영해서 concat

    invalid 정책:
      - trip 이후 K가 여러 component면 success=False, reason_invalid="islanding_after_gen_trip"
    """
    t_event = float(t_event)
    t_final = float(t_final)
    dt = float(dt)

    if not (0.0 < t_event < t_final):
        return GenTripResult(
            success=False, message="invalid time window",
            t=np.zeros(0), y=np.zeros((0, 0)),
            K_post=None, Peq_post=None,
            trip_gen_id=int(trip_gen_id), trip_gen_local_idx=-1,
            reason_invalid="bad_time_window",
        )

    try:
        gi = _resolve_gen_local_idx(sysobj.gen_ids, int(trip_gen_id))
    except Exception as e:
        return GenTripResult(
            success=False, message=str(e),
            t=np.zeros(0), y=np.zeros((0, 0)),
            K_post=None, Peq_post=None,
            trip_gen_id=int(trip_gen_id), trip_gen_local_idx=-1,
            reason_invalid="bad_gen_id",
        )

    ng0 = int(sysobj.ng)

    # ---- pre segment solve (ng0) ----
    delta0 = np.asarray(sysobj.delta_guess, dtype=float).reshape(-1)
    omega0 = np.zeros(ng0, dtype=float)

    t_eval_pre = np.arange(0.0, t_event + 0.5 * dt, dt, dtype=float)
    t_eval_pre = t_eval_pre[(t_eval_pre >= 0.0) & (t_eval_pre <= t_event)]
    if t_eval_pre.size == 0 or abs(t_eval_pre[-1] - t_event) > 1e-12:
        t_eval_pre = np.append(t_eval_pre, t_event)

    sol0 = solve_swing_ivp(
        K=sysobj.K,
        Peq=sysobj.Peq,
        M=sysobj.M,
        D=D_used,
        delta0=delta0,
        omega0=omega0,
        t_span=(0.0, t_event),
        t_eval=t_eval_pre,
        rtol=rtol, atol=atol, method=method,
        disturbance=None,
    )
    if not bool(sol0.success):
        return GenTripResult(
            success=False,
            message=str(getattr(sol0, "message", "pre segment failed")),
            t=np.asarray(sol0.t, dtype=float),
            y=np.asarray(sol0.y.T, dtype=float) if hasattr(sol0, "y") else np.zeros((0, 0)),
            K_post=None, Peq_post=None,
            trip_gen_id=int(trip_gen_id), trip_gen_local_idx=int(gi),
            reason_invalid="pre_segment_failed",
        )

    y_end = np.asarray(sol0.y[:, -1], dtype=float)
    delta_evt = y_end[:ng0]
    omega_evt = y_end[ng0:]

    # ---- build post-trip reduced system (ng1=ng0-1) ----
    keep = np.ones(ng0, dtype=bool)
    keep[int(gi)] = False

    gen_ids1 = np.asarray(sysobj.gen_ids, dtype=int).reshape(-1)[keep]
    gen_bus_ids1 = np.asarray(sysobj.gen_bus_ids, dtype=int).reshape(-1)[keep]
    xd1 = np.asarray(sysobj.xd_prime_pu, dtype=float).reshape(-1)[keep]

    Pg1 = np.asarray(sysobj.Pg_pu, dtype=float).reshape(-1)[keep]
    M1 = np.asarray(sysobj.M, dtype=float).reshape(-1)[keep]
    D1 = np.asarray(D_used, dtype=float).reshape(-1)[keep]

    delta1_0 = np.asarray(delta_evt, dtype=float).reshape(-1)[keep]
    omega1_0 = np.asarray(omega_evt, dtype=float).reshape(-1)[keep]

    # ---- 핵심: Pbus_post 만들기 (결손 유지) ----
    Pbus_post = np.asarray(sysobj.Pbus_pu, dtype=float).reshape(-1).copy()
    trip_bus_id = int(np.asarray(sysobj.gen_bus_ids, dtype=int).reshape(-1)[int(gi)])
    trip_Pg_pu = float(np.asarray(sysobj.Pg_pu, dtype=float).reshape(-1)[int(gi)])

    bus_ids = np.asarray(sysobj.bus[:, 0], dtype=int).reshape(-1)
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}
    bi = bus_id_to_idx.get(trip_bus_id)
    if bi is None:
        return GenTripResult(
            success=False,
            message=f"trip bus_id {trip_bus_id} not found in sysobj.bus table",
            t=np.zeros(0), y=np.zeros((0, 0)),
            K_post=None, Peq_post=None,
            trip_gen_id=int(trip_gen_id), trip_gen_local_idx=int(gi),
            reason_invalid="bad_trip_bus",
        )

    # 발전기 탈락: 해당 버스 주입에서 Pg만큼 제거 -> ΣPbus < 0
    Pbus_post[int(bi)] -= trip_Pg_pu

    # ---- post에서 (K1, Peq1) 재계산 (결손 유지 설정) ----
    try:
        K1, _delta_dummy, Peq1, slack_post = compute_K_and_Peq_from_arrays(
            bus=np.asarray(sysobj.bus, dtype=float),
            branch=np.asarray(sysobj.branch, dtype=float),
            gen_bus_ids=np.asarray(gen_bus_ids1, dtype=int),
            P_bus_pu=np.asarray(Pbus_post, dtype=float),
            xd_prime_pu=np.asarray(xd1, dtype=float),
            slack_bus_id=int(sysobj.slack_bus_id),
            balance_on_slack=False,         # 결손을 slack이 메우지 않게
            center_Peq=False,               # 평균제거로 결손이 지워지지 않게
            solve_delta_guess=False,        # 결손이면 평형각 풀이 불필요/실패 가능
        )
    except Exception as e:
        return GenTripResult(
            success=False,
            message=f"K/Peq recompute failed: {e}",
            t=np.zeros(0), y=np.zeros((0, 0)),
            K_post=None, Peq_post=None,
            trip_gen_id=int(trip_gen_id), trip_gen_local_idx=int(gi),
            reason_invalid="K_Peq_recompute_failed",
        )

    comps = _count_components_of_K(K1, eps=1e-9)
    if comps != 1:
        return GenTripResult(
            success=False,
            message=f"invalid: K has {comps} components after gen trip",
            t=np.zeros(0), y=np.zeros((0, 0)),
            K_post=K1, Peq_post=Peq1,
            trip_gen_id=int(trip_gen_id), trip_gen_local_idx=int(gi),
            reason_invalid="islanding_after_gen_trip",
        )

    # ---- post segment solve (ng1) ----
    t_eval_post = np.arange(t_event, t_final + 0.5 * dt, dt, dtype=float)
    t_eval_post = t_eval_post[(t_eval_post >= t_event) & (t_eval_post <= t_final)]
    if t_eval_post.size == 0 or abs(t_eval_post[0] - t_event) > 1e-12:
        t_eval_post = np.insert(t_eval_post, 0, t_event)
    if abs(t_eval_post[-1] - t_final) > 1e-12:
        t_eval_post = np.append(t_eval_post, t_final)

    sol1 = solve_swing_ivp(
        K=K1,
        Peq=Peq1,
        M=M1,
        D=D1,
        delta0=delta1_0,
        omega0=omega1_0,
        t_span=(t_event, t_final),
        t_eval=t_eval_post,
        rtol=rtol, atol=atol, method=method,
        disturbance=None,
    )
    if not bool(sol1.success):
        return GenTripResult(
            success=False,
            message=str(getattr(sol1, "message", "post segment failed")),
            t=np.asarray(sol1.t, dtype=float),
            y=np.asarray(sol1.y.T, dtype=float) if hasattr(sol1, "y") else np.zeros((0, 0)),
            K_post=K1, Peq_post=Peq1,
            trip_gen_id=int(trip_gen_id), trip_gen_local_idx=int(gi),
            reason_invalid="post_segment_failed",
            gen_ids_post=gen_ids1,
            gen_bus_ids_post=gen_bus_ids1,
            Pg_pu_post=Pg1,
            M_post=M1,
            D_post=D1,
            xd_prime_pu_post=xd1,
            slack_bus_id_post=int(slack_post),
            Pbus_sum_post=float(np.sum(Pbus_post)),
            Peq_sum_post=float(np.sum(Peq1)),
            trip_Pg_pu=float(trip_Pg_pu),
            trip_bus_id=int(trip_bus_id),
        )

    # ---- concat (post 차원 기준) ----
    t0 = np.asarray(sol0.t, dtype=float)
    Y0 = np.asarray(sol0.y.T, dtype=float)     # (T0, 2ng0)
    t1 = np.asarray(sol1.t, dtype=float)
    Y1 = np.asarray(sol1.y.T, dtype=float)     # (T1, 2ng1)

    Y0_delta = Y0[:, :ng0][:, keep]
    Y0_omega = Y0[:, ng0:][:, keep]
    Y0r = np.concatenate([Y0_delta, Y0_omega], axis=1)  # (T0, 2ng1)

    if t0.size > 0 and t1.size > 0 and abs(float(t0[-1]) - float(t1[0])) < 1e-12:
        t = np.concatenate([t0, t1[1:]], axis=0)
        Y = np.concatenate([Y0r, Y1[1:, :]], axis=0)
    else:
        t = np.concatenate([t0, t1], axis=0)
        Y = np.concatenate([Y0r, Y1], axis=0)

    return GenTripResult(
        success=True,
        message="ok",
        t=t,
        y=Y,
        K_post=K1,
        Peq_post=Peq1,
        trip_gen_id=int(trip_gen_id),
        trip_gen_local_idx=int(gi),
        reason_invalid=None,
        gen_ids_post=gen_ids1,
        gen_bus_ids_post=gen_bus_ids1,
        Pg_pu_post=Pg1,
        M_post=M1,
        D_post=D1,
        xd_prime_pu_post=xd1,
        slack_bus_id_post=int(slack_post),
        Pbus_sum_post=float(np.sum(Pbus_post)),
        Peq_sum_post=float(np.sum(Peq1)),
        trip_Pg_pu=float(trip_Pg_pu),
        trip_bus_id=int(trip_bus_id),
    )
