from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from kpg_swing.paths import get_paths
from kpg_swing.core.loader import load_system
from kpg_swing.engine.swing_api import solve_swing_ivp
from kpg_swing.engine.disturbance import step_on_generator
from kpg_swing.core.metrics import compute_frequency_response_metrics, make_batch_row


def run_one_step(
    *,
    sysobj,
    t_event: float,
    t_final: float,
    dt: float,
    d_scale: float,
    rocof_fit_window: float,
    settle_window: float,
    step_gen_local_idx: int,
    step_dPeq_pu: float,
):
    # IC
    delta0 = np.asarray(sysobj.delta_guess, dtype=float)
    omega0 = np.zeros(sysobj.ng, dtype=float)

    # time grid
    t0, tf = 0.0, float(t_final)
    t_eval = np.arange(t0, tf + 0.5 * float(dt), float(dt), dtype=float)

    # D used
    D_used = np.asarray(sysobj.D, dtype=float).reshape(-1) * float(d_scale)

    # disturbance
    disturbance = step_on_generator(
        ng=sysobj.ng,
        t0=float(t_event),
        gen_idx=int(step_gen_local_idx),
        dPeq=float(step_dPeq_pu),
    )

    # integrate
    sol = solve_swing_ivp(
        K=sysobj.K,
        Peq=sysobj.Peq,
        M=sysobj.M,
        D=D_used,
        delta0=delta0,
        omega0=omega0,
        t_span=(t0, tf),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
        disturbance=disturbance,
    )

    if not bool(sol.success):
        return None, f"solve_ivp failed: {getattr(sol, 'message', '')}"

    t = np.asarray(sol.t, dtype=float)
    Y = np.asarray(sol.y.T, dtype=float)

    ng = int(sysobj.ng)
    delta = Y[:, :ng]
    omega = Y[:, ng:]

    # metrics (배치는 series 미포함)
    m, _series = compute_frequency_response_metrics(
        t=t,
        delta=delta,
        omega=omega,
        M=np.asarray(sysobj.M, dtype=float),
        D_used=D_used,
        t_event=float(t_event),
        step_dPeq_total_pu=float(step_dPeq_pu),
        gen_ids=np.asarray(sysobj.gen_ids, dtype=int),
        gen_bus_ids=np.asarray(sysobj.gen_bus_ids, dtype=int),
        rocof_fit_window=float(rocof_fit_window),
        settle_window=float(settle_window),
        include_series=False,
        exclude_gen_local_idx=None,
    )
    return m, ""


def main():
    # -----------------------------
    # Batch knobs (여기만 조절)
    # aligned with debug.py defaults
    # -----------------------------
    T_EVENT = 2.0
    T_FINAL = 100.0
    DT = 0.02

    D_SCALE = 0.05
    ROCOF_FIT_WINDOW = 0.5
    SETTLE_WINDOW = 5.0

    # step size set
    # pu 기준. baseMVA=100이면 -1.0 pu = -100MW
    STEP_DPEQ_LIST = [-1.0, -2.0, -5.0, -10.0]  # 필요하면 하나만 두면 됨

    # generator sweep
    # 기본: 전체
    GEN_LOCAL_IDXS = None  # None이면 0..ng-1 전체

    # 저장 최소화
    SAVE_ONLY_AGGREGATE = True

    # -----------------------------
    # Load once
    # -----------------------------
    paths = get_paths(validate=True, ensure_outputs=True)
    case_mfile = paths.case_mfile
    dyn_csv = paths.static_dir / "dyn_params.csv"

    sysobj = load_system(case_mfile, dyn_csv)

    if GEN_LOCAL_IDXS is None:
        GEN_LOCAL_IDXS = list(range(int(sysobj.ng)))

    # -----------------------------
    # Prepare outputs
    # -----------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = paths.aggregates_dir / f"batch_step_{ts}.csv"
    out_meta = paths.aggregates_dir / f"batch_step_{ts}_meta.json"

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case_mfile": str(case_mfile),
        "dyn_csv": str(dyn_csv),
        "sim": {
            "t_event": float(T_EVENT),
            "t_final": float(T_FINAL),
            "dt": float(DT),
            "D_scale": float(D_SCALE),
            "rocof_fit_window": float(ROCOF_FIT_WINDOW),
            "settle_window": float(SETTLE_WINDOW),
        },
        "event": {
            "type": "step",
            "step_dPeq_list_pu": [float(x) for x in STEP_DPEQ_LIST],
            "gen_local_indices": [int(x) for x in GEN_LOCAL_IDXS],
        },
        "system": {
            "nb": int(sysobj.nb),
            "ng": int(sysobj.ng),
            "slack_bus_id": int(sysobj.slack_bus_id),
            "baseMVA": float(sysobj.baseMVA),
        },
        "notes": "Batch runner uses step disturbance only. No gen_trip/line_outage.",
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # -----------------------------
    # Run batch
    # -----------------------------
    rows = []
    created_at = meta["created_at"]

    total = len(GEN_LOCAL_IDXS) * len(STEP_DPEQ_LIST)
    k = 0

    for gi in GEN_LOCAL_IDXS:
        for dPeq in STEP_DPEQ_LIST:
            k += 1
            run_id = f"step_{ts}_g{int(gi):04d}_d{float(dPeq):.3f}"

            m, err = run_one_step(
                sysobj=sysobj,
                t_event=float(T_EVENT),
                t_final=float(T_FINAL),
                dt=float(DT),
                d_scale=float(D_SCALE),
                rocof_fit_window=float(ROCOF_FIT_WINDOW),
                settle_window=float(SETTLE_WINDOW),
                step_gen_local_idx=int(gi),
                step_dPeq_pu=float(dPeq),
            )

            if m is None:
                # 실패도 남기고 싶으면 row를 최소로 기록
                rows.append({
                    "run_id": run_id,
                    "created_at": created_at,
                    "case_mfile": str(case_mfile),
                    "dyn_csv": str(dyn_csv),
                    "event_type": "step",
                    "step_gen_local_idx": int(gi),
                    "step_dPeq_pu": float(dPeq),
                    "baseMVA": float(sysobj.baseMVA),
                    "success": 0,
                    "message": str(err),
                })
                continue

            row = make_batch_row(
                run_id=run_id,
                created_at=created_at,
                case_mfile=str(case_mfile),
                dyn_csv=str(dyn_csv),
                t_event_s=float(T_EVENT),
                t_final_s=float(T_FINAL),
                dt_s=float(DT),
                rocof_fit_window_s=float(ROCOF_FIT_WINDOW),
                settle_window_s=float(SETTLE_WINDOW),
                event_type="step",
                step_gen_local_idx=int(gi),
                step_dPeq_pu=float(dPeq),
                baseMVA=float(sysobj.baseMVA),
                D_scale=float(D_SCALE),
                nb=int(sysobj.nb),
                ng=int(sysobj.ng),
                slack_bus_id=int(sysobj.slack_bus_id),
                metrics=m,
            )
            row["success"] = 1
            row["message"] = ""
            rows.append(row)

            if (k % 20) == 0:
                print(f"[progress] {k}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()
