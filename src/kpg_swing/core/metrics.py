from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any, List
import numpy as np


# ============================================================
# Conventions (IMPORTANT)
# ============================================================
# - omega: rad/s (electrical speed deviation variable used in your swing model)
# - df = omega / (2*pi): Hz deviation
# - Pareto objectives: minimize all objectives (smaller is better)


# =========================
# Basic conversions
# =========================
def omega_to_hz(omega_rad_s: np.ndarray) -> np.ndarray:
    """rad/s -> Hz (frequency deviation)"""
    omega_rad_s = np.asarray(omega_rad_s, dtype=float)
    return omega_rad_s / (2.0 * np.pi)


def coi_weighted_average(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    COI weighted average over generators.
    x: (T, ng)
    w: (ng,)
    returns: (T,)
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("x must be 2D array (T, ng)")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"shape mismatch: x {x.shape}, w {w.shape}")
    wsum = float(np.sum(w))
    if wsum <= 0:
        raise ValueError("sum(w) must be positive")
    return (x @ w) / wsum


# =========================
# COI angles and synchrony
# =========================
def coi_angle(delta: np.ndarray, M: np.ndarray) -> np.ndarray:
    """delta_coi(t) = sum_i M_i * delta_i(t) / sum_i M_i"""
    return coi_weighted_average(delta, M)


def angle_rel_to_coi(delta: np.ndarray, M: np.ndarray) -> np.ndarray:
    """delta_rel(t,i) = delta(t,i) - delta_coi(t)"""
    delta = np.asarray(delta, dtype=float)
    dcoi = coi_angle(delta, M)  # (T,)
    return delta - dcoi[:, None]


def angle_spread_relative_to_coi(delta: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      min_rel(t), max_rel(t), spread(t)=max-min
    """
    rel = angle_rel_to_coi(delta, M)
    mn = np.min(rel, axis=1)
    mx = np.max(rel, axis=1)
    return mn, mx, (mx - mn)


def order_parameter_R(delta: np.ndarray, use_rel_to_coi: bool = True, M: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Kuramoto order parameter:
      R(t) = |(1/ng) * sum_i exp(j * delta_i(t))|

    If use_rel_to_coi is True, uses delta_rel = delta - delta_coi (recommended).
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2:
        raise ValueError("delta must be 2D array (T, ng)")

    if use_rel_to_coi:
        if M is None:
            raise ValueError("M must be provided when use_rel_to_coi=True")
        delta_use = angle_rel_to_coi(delta, M)
    else:
        delta_use = delta

    z = np.exp(1j * delta_use)
    return np.abs(np.mean(z, axis=1))


# =========================
# Frequency-response helpers
# =========================
def nadir(df: np.ndarray, t: Optional[np.ndarray] = None) -> Tuple[float, Optional[float], int]:
    """
    df: (T,) frequency deviation [Hz]
    returns: (nadir_value, nadir_time_or_None, index)
    """
    df = np.asarray(df, dtype=float).reshape(-1)
    k = int(np.argmin(df))
    if t is None:
        return float(df[k]), None, k
    t = np.asarray(t, dtype=float).reshape(-1)
    return float(df[k]), float(t[k]), k


def rocof_linear_fit(t: np.ndarray, df: np.ndarray, t0: float, window: float) -> float:
    """
    RoCoF via linear regression slope on [t0, t0+window].
    Returns slope in Hz/s. If insufficient samples, returns nan.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    df = np.asarray(df, dtype=float).reshape(-1)
    if t.shape[0] != df.shape[0]:
        raise ValueError("t and df must have same length")
    if len(t) < 3:
        return float("nan")

    i0 = int(np.searchsorted(t, float(t0)))
    i1 = int(np.searchsorted(t, float(t0) + float(window)))
    i1 = min(i1, len(t) - 1)
    if i1 - i0 < 2:
        return float("nan")

    tt = t[i0:i1 + 1]
    yy = df[i0:i1 + 1]
    tc = tt - float(np.mean(tt))
    denom = float(np.sum(tc * tc))
    if denom <= 0:
        return float("nan")
    slope = float(np.sum(tc * (yy - float(np.mean(yy)))) / denom)
    return slope


def rocof_1step(t: np.ndarray, df: np.ndarray, t_event: float) -> float:
    """
    Debug RoCoF using first sample at/after event and next sample.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    df = np.asarray(df, dtype=float).reshape(-1)
    k = int(np.searchsorted(t, float(t_event)))
    if k + 1 >= len(t):
        return float("nan")
    dt = float(t[k + 1] - t[k])
    if dt <= 0:
        return float("nan")
    return float((df[k + 1] - df[k]) / dt)


def settling_time(
    t: np.ndarray,
    x: np.ndarray,
    eps: float,
    window: float,
    t_start: float,
    x_final: Optional[float] = None,
) -> float:
    """
    Settling time: smallest tau >= t_start such that for all times in [tau, tau+window],
    |x(t)-x_final| <= eps. Returns nan if not settled.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if t.shape[0] != x.shape[0] or t.shape[0] < 2:
        return float("nan")
    if x_final is None:
        x_final = float(x[-1])

    i0 = int(np.searchsorted(t, float(t_start)))
    if i0 >= len(t):
        return float("nan")

    for i in range(i0, len(t)):
        t_end = float(t[i] + float(window))
        j = int(np.searchsorted(t, t_end))
        if j >= len(t):
            break
        if np.all(np.abs(x[i:j + 1] - float(x_final)) <= float(eps)):
            return float(t[i])
    return float("nan")


def predicted_steady_df_hz(step_dPeq_pu_total: float, sum_D_used: float) -> float:
    """
    With your convention:
      omega_ss = step_dPeq_total / sum(D_used)    [rad/s]
      df_ss = omega_ss / (2*pi)                   [Hz]
    """
    if sum_D_used <= 0:
        return float("nan")
    return float(step_dPeq_pu_total / (2.0 * np.pi * float(sum_D_used)))


# =========================
# Pareto-friendly metrics
# =========================
@dataclass
class FrequencyResponseMetrics:
    # COI metrics
    coi_nadir_hz: float
    coi_nadir_time_s: float
    coi_steady_hz: float
    coi_rocof_fit_hz_per_s: float
    coi_rocof_1step_hz_per_s: float
    coi_settle_eps_1mHz_s: float
    coi_settle_eps_5mHz_s: float

    # Worst-case generator metrics
    worst_gen_nadir_hz: float
    worst_gen_nadir_time_s: float
    worst_gen_nadir_local_idx: int
    worst_gen_nadir_gen_id: int
    worst_gen_nadir_bus_id: int

    worst_gen_rocof_fit_hz_per_s: float
    worst_gen_rocof_local_idx: int
    worst_gen_rocof_gen_id: int
    worst_gen_rocof_bus_id: int

    # Synchrony metrics
    R_min: float
    R_final: float
    angle_spread_rel_coi_max_rad: float

    # Sanity scalars
    sum_M: float
    sum_D_used: float
    pred_steady_df_hz: float

    # [PATCH] bookkeeping for trip exclusion (optional)
    excluded_gen_local_idx: int


# [PATCH] helper: build mask
def _make_mask_excluding_gen(ng: int, exclude_gen_local_idx: Optional[int]) -> np.ndarray:
    mask = np.ones(int(ng), dtype=bool)
    if exclude_gen_local_idx is None:
        return mask
    gi = int(exclude_gen_local_idx)
    if gi < 0 or gi >= int(ng):
        # out-of-range => treat as no exclusion (safer for backward compat)
        return mask
    mask[gi] = False
    # if everything excluded (ng=1 edge), revert to no exclusion
    if not np.any(mask):
        return np.ones(int(ng), dtype=bool)
    return mask


def compute_frequency_response_metrics(
    t: np.ndarray,
    delta: np.ndarray,
    omega: np.ndarray,
    M: np.ndarray,
    D_used: np.ndarray,
    t_event: float,
    step_dPeq_total_pu: float,
    gen_ids: Optional[np.ndarray] = None,
    gen_bus_ids: Optional[np.ndarray] = None,
    rocof_fit_window: float = 0.5,
    settle_window: float = 5.0,
    include_series: bool = False,
    # [PATCH] new optional arg (default keeps old behavior)
    exclude_gen_local_idx: Optional[int] = None,
) -> Tuple[FrequencyResponseMetrics, Optional[Dict[str, Any]]]:
    """
    Computes:
      - COI frequency response
      - worst-case generator frequency response
      - synchrony metrics (R, angle spread)

    If exclude_gen_local_idx is given, metrics are computed over generators
    excluding that local index (useful for gen-trip post-event interpretation).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    M = np.asarray(M, dtype=float).reshape(-1)
    D_used = np.asarray(D_used, dtype=float).reshape(-1)

    if delta.ndim != 2 or omega.ndim != 2:
        raise ValueError("delta and omega must be 2D arrays (T, ng)")
    if delta.shape != omega.shape:
        raise ValueError(f"delta shape {delta.shape} must equal omega shape {omega.shape}")

    T, ng = omega.shape
    if M.shape[0] != ng or D_used.shape[0] != ng:
        raise ValueError("M and D_used must have length ng")

    if gen_ids is None:
        gen_ids = np.arange(ng, dtype=int)
    else:
        gen_ids = np.asarray(gen_ids, dtype=int).reshape(-1)

    if gen_bus_ids is None:
        gen_bus_ids = np.full(ng, -1, dtype=int)
    else:
        gen_bus_ids = np.asarray(gen_bus_ids, dtype=int).reshape(-1)

    if gen_ids.shape[0] != ng or gen_bus_ids.shape[0] != ng:
        raise ValueError("gen_ids and gen_bus_ids must have length ng")

    # [PATCH] apply mask (view only)
    mask = _make_mask_excluding_gen(ng, exclude_gen_local_idx)

    delta_m = delta[:, mask]
    omega_m = omega[:, mask]
    M_m = M[mask]
    D_used_m = D_used[mask]
    gen_ids_m = gen_ids[mask]
    gen_bus_ids_m = gen_bus_ids[mask]

    # frequency traces
    df_gen = omega_to_hz(omega_m)                     # (T, ng_eff)
    omega_coi = coi_weighted_average(omega_m, M_m)    # (T,) rad/s
    df_coi = omega_to_hz(omega_coi)                   # (T,) Hz

    # COI metrics
    coi_nadir_val, coi_nadir_t, _ = nadir(df_coi, t=t)
    coi_nadir_t = float(coi_nadir_t) if coi_nadir_t is not None else float("nan")
    coi_steady = float(df_coi[-1])
    coi_rocof_fit = rocof_linear_fit(t, df_coi, t0=float(t_event), window=float(rocof_fit_window))
    coi_rocof_1 = rocof_1step(t, df_coi, t_event=float(t_event))
    coi_st1 = settling_time(t, df_coi, eps=1e-3, window=float(settle_window), t_start=float(t_event), x_final=coi_steady)
    coi_st5 = settling_time(t, df_coi, eps=5e-3, window=float(settle_window), t_start=float(t_event), x_final=coi_steady)

    # Worst-case nadir among generators (masked set)
    gen_nadir_vals = np.min(df_gen, axis=0)  # (ng_eff,)
    gen_nadir_idx_m = int(np.argmin(gen_nadir_vals))
    worst_nadir_hz = float(gen_nadir_vals[gen_nadir_idx_m])
    t_idx = int(np.argmin(df_gen[:, gen_nadir_idx_m]))
    worst_nadir_time = float(t[t_idx])

    # [PATCH] map back to original local idx
    kept_idx = np.where(mask)[0]
    worst_nadir_local = int(kept_idx[gen_nadir_idx_m])
    worst_nadir_gen_id = int(gen_ids_m[gen_nadir_idx_m])
    worst_nadir_bus_id = int(gen_bus_ids_m[gen_nadir_idx_m])

    # Worst-case RoCoF among generators (masked set; most negative slope)
    ng_eff = int(df_gen.shape[1])
    rocof_fit_each = np.full(ng_eff, np.nan, dtype=float)
    for i in range(ng_eff):
        rocof_fit_each[i] = rocof_linear_fit(t, df_gen[:, i], t0=float(t_event), window=float(rocof_fit_window))

    finite = np.isfinite(rocof_fit_each)
    if np.any(finite):
        idx_all = np.where(finite)[0]
        worst_rocof_idx_m = int(idx_all[np.argmin(rocof_fit_each[finite])])
        worst_rocof_val = float(rocof_fit_each[worst_rocof_idx_m])
        worst_rocof_local = int(kept_idx[worst_rocof_idx_m])
        worst_rocof_gen_id = int(gen_ids_m[worst_rocof_idx_m])
        worst_rocof_bus_id = int(gen_bus_ids_m[worst_rocof_idx_m])
    else:
        worst_rocof_val = float("nan")
        worst_rocof_local = -1
        worst_rocof_gen_id = -1
        worst_rocof_bus_id = -1

    # synchrony metrics (masked set)
    R = order_parameter_R(delta_m, use_rel_to_coi=True, M=M_m)
    R_min = float(np.min(R))
    R_final = float(R[-1])
    _mn, _mx, spread = angle_spread_relative_to_coi(delta_m, M_m)
    angle_spread_max = float(np.max(spread))

    # sanity (keep reporting sums of USED set, since metrics are on masked system)
    sum_M = float(np.sum(M_m))
    sum_D_used = float(np.sum(D_used_m))
    pred_steady = predicted_steady_df_hz(step_dPeq_pu_total=float(step_dPeq_total_pu), sum_D_used=float(sum_D_used))

    out = FrequencyResponseMetrics(
        coi_nadir_hz=float(coi_nadir_val),
        coi_nadir_time_s=float(coi_nadir_t),
        coi_steady_hz=float(coi_steady),
        coi_rocof_fit_hz_per_s=float(coi_rocof_fit),
        coi_rocof_1step_hz_per_s=float(coi_rocof_1),
        coi_settle_eps_1mHz_s=float(coi_st1),
        coi_settle_eps_5mHz_s=float(coi_st5),

        worst_gen_nadir_hz=float(worst_nadir_hz),
        worst_gen_nadir_time_s=float(worst_nadir_time),
        worst_gen_nadir_local_idx=int(worst_nadir_local),
        worst_gen_nadir_gen_id=int(worst_nadir_gen_id),
        worst_gen_nadir_bus_id=int(worst_nadir_bus_id),

        worst_gen_rocof_fit_hz_per_s=float(worst_rocof_val),
        worst_gen_rocof_local_idx=int(worst_rocof_local),
        worst_gen_rocof_gen_id=int(worst_rocof_gen_id),
        worst_gen_rocof_bus_id=int(worst_rocof_bus_id),

        R_min=float(R_min),
        R_final=float(R_final),
        angle_spread_rel_coi_max_rad=float(angle_spread_max),

        sum_M=float(sum_M),
        sum_D_used=float(sum_D_used),
        pred_steady_df_hz=float(pred_steady),

        excluded_gen_local_idx=int(exclude_gen_local_idx) if exclude_gen_local_idx is not None else -1,
    )

    series = None
    if include_series:
        # NOTE: series는 "masked" 기준임 (ng_eff). 배치 기본은 include_series=False 권장.
        series = {
            "t": t.copy(),
            "df_coi_hz": df_coi.copy(),
            "df_gen_hz": df_gen.copy(),
            "R_t": R.copy(),
            "angle_spread_rel_coi": spread.copy(),
            "rocof_fit_each_hz_per_s": rocof_fit_each.copy(),
            "gen_nadir_each_hz": gen_nadir_vals.copy(),
            "mask_used": mask.copy(),
            "excluded_gen_local_idx": int(exclude_gen_local_idx) if exclude_gen_local_idx is not None else -1,
        }

    return out, series


def metrics_to_dict(m: FrequencyResponseMetrics) -> Dict[str, Any]:
    """Flatten metrics to dict (JSON/CSV-friendly)."""
    return asdict(m)


# ============================================================
# Pareto objective keys (fixed schema)
# ============================================================
OBJECTIVE_KEYS: List[str] = [
    "obj1_coi_nadir_drop_hz",
    "obj2_worst_nadir_drop_hz",
    "obj3_coi_rocof_abs_hz_per_s",
    "obj4_worst_rocof_abs_hz_per_s",
    "obj5_settle_1mHz_s",
    "obj6_one_minus_Rmin",
    "obj7_angle_spread_max_rad",
]


def build_objectives(m: FrequencyResponseMetrics) -> Dict[str, float]:
    """
    Returns objective dict. All objectives are to be MINIMIZED.
    """
    coi_drop = float(abs(m.coi_nadir_hz))
    worst_drop = float(abs(m.worst_gen_nadir_hz))

    coi_rocof_abs = float(abs(m.coi_rocof_fit_hz_per_s))
    worst_rocof_abs = float(abs(m.worst_gen_rocof_fit_hz_per_s)) if np.isfinite(m.worst_gen_rocof_fit_hz_per_s) else float("nan")

    settle_1mHz = float(m.coi_settle_eps_1mHz_s) if np.isfinite(m.coi_settle_eps_1mHz_s) else float("nan")
    one_minus_Rmin = float(1.0 - m.R_min)
    angle_spread = float(m.angle_spread_rel_coi_max_rad)

    return {
        "obj1_coi_nadir_drop_hz": coi_drop,
        "obj2_worst_nadir_drop_hz": worst_drop,
        "obj3_coi_rocof_abs_hz_per_s": coi_rocof_abs,
        "obj4_worst_rocof_abs_hz_per_s": worst_rocof_abs,
        "obj5_settle_1mHz_s": settle_1mHz,
        "obj6_one_minus_Rmin": one_minus_Rmin,
        "obj7_angle_spread_max_rad": angle_spread,
    }


# ============================================================
# CSV schema for batch runner
# ============================================================
BATCH_CSV_COLUMNS: List[str] = [
    "run_id",
    "created_at",
    "case_mfile",
    "dyn_csv",
    "t_event_s",
    "t_final_s",
    "dt_s",
    "rocof_fit_window_s",
    "settle_window_s",
    "event_type",
    "step_gen_local_idx",
    "step_dPeq_pu",
    "baseMVA",
    "step_MW",
    "D_scale",
    "nb",
    "ng",
    "slack_bus_id",
    "sum_M",
    "sum_D_used",
    "pred_steady_df_hz",
    "coi_nadir_hz",
    "coi_nadir_time_s",
    "coi_steady_hz",
    "coi_rocof_fit_hz_per_s",
    "coi_rocof_1step_hz_per_s",
    "coi_settle_eps_1mHz_s",
    "coi_settle_eps_5mHz_s",
    "worst_gen_nadir_hz",
    "worst_gen_nadir_time_s",
    "worst_gen_nadir_local_idx",
    "worst_gen_nadir_gen_id",
    "worst_gen_nadir_bus_id",
    "worst_gen_rocof_fit_hz_per_s",
    "worst_gen_rocof_local_idx",
    "worst_gen_rocof_gen_id",
    "worst_gen_rocof_bus_id",
    "R_min",
    "R_final",
    "angle_spread_rel_coi_max_rad",
    # [PATCH] new
    "excluded_gen_local_idx",
] + OBJECTIVE_KEYS


def make_batch_row(
    *,
    run_id: str,
    created_at: str,
    case_mfile: str,
    dyn_csv: str,
    t_event_s: float,
    t_final_s: float,
    dt_s: float,
    rocof_fit_window_s: float,
    settle_window_s: float,
    event_type: str,
    step_gen_local_idx: int,
    step_dPeq_pu: float,
    baseMVA: float,
    D_scale: float,
    nb: int,
    ng: int,
    slack_bus_id: int,
    metrics: FrequencyResponseMetrics,
) -> Dict[str, Any]:
    """
    Create one flat row that matches BATCH_CSV_COLUMNS exactly.
    """
    md = metrics_to_dict(metrics)
    obj = build_objectives(metrics)

    row = {
        "run_id": run_id,
        "created_at": created_at,
        "case_mfile": case_mfile,
        "dyn_csv": dyn_csv,
        "t_event_s": float(t_event_s),
        "t_final_s": float(t_final_s),
        "dt_s": float(dt_s),
        "rocof_fit_window_s": float(rocof_fit_window_s),
        "settle_window_s": float(settle_window_s),
        "event_type": str(event_type),
        "step_gen_local_idx": int(step_gen_local_idx),
        "step_dPeq_pu": float(step_dPeq_pu),
        "baseMVA": float(baseMVA),
        "step_MW": float(step_dPeq_pu * baseMVA),
        "D_scale": float(D_scale),
        "nb": int(nb),
        "ng": int(ng),
        "slack_bus_id": int(slack_bus_id),
        "sum_M": float(metrics.sum_M),
        "sum_D_used": float(metrics.sum_D_used),
        "pred_steady_df_hz": float(metrics.pred_steady_df_hz),
    }

    row.update(md)
    row.update(obj)

    missing = [c for c in BATCH_CSV_COLUMNS if c not in row]
    if missing:
        raise KeyError(f"missing columns in batch row: {missing}")
    return row
