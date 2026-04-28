from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kpg_swing.engine.busmap import make_busmap


@dataclass(frozen=True)
class SanityReport:
    ok: bool
    messages: list[str]


def _msg(report: SanityReport, text: str) -> SanityReport:
    msgs = list(report.messages)
    msgs.append(text)
    return SanityReport(ok=report.ok, messages=msgs)


def run_sanity_checks(sys, *, strict: bool = True) -> SanityReport:
    """
    LoadedSystem(또는 동일한 속성 구조)을 대상으로 보수적 점검을 수행.
    - bus id(1-based) vs row index(0-based) 혼용을 방지하기 위한 체크에 집중.

    strict=True면 실패 시 RuntimeError를 발생시킨다.
    strict=False면 report만 반환한다.
    """
    report = SanityReport(ok=True, messages=[])

    # --- basic shapes ---
    try:
        nb = int(sys.bus.shape[0])
        ng = int(sys.gen_ids.shape[0])
    except Exception as e:
        if strict:
            raise RuntimeError(f"sys has no expected arrays: {e}")
        return SanityReport(ok=False, messages=[f"sys has no expected arrays: {e}"])

    # --- bus id integrity ---
    bus_ids = sys.bus[:, 0].astype(int)
    if bus_ids.shape[0] != nb:
        report = _msg(report, f"[FAIL] bus_ids length mismatch: {bus_ids.shape[0]} vs nb={nb}")
        report = SanityReport(ok=False, messages=report.messages)

    if np.unique(bus_ids).shape[0] != nb:
        report = _msg(report, "[FAIL] duplicate bus ids found in sys.bus[:,0]")
        report = SanityReport(ok=False, messages=report.messages)

    # sys.bus_ids consistency
    if hasattr(sys, "bus_ids"):
        if not np.array_equal(sys.bus_ids.astype(int), bus_ids):
            report = _msg(report, "[FAIL] sys.bus_ids is not identical to sys.bus[:,0]")
            report = SanityReport(ok=False, messages=report.messages)
    else:
        report = _msg(report, "[WARN] sys has no attribute bus_ids (recommended)")

    # --- branch endpoints are valid bus ids ---
    branch_f = sys.branch[:, 0].astype(int)
    branch_t = sys.branch[:, 1].astype(int)
    bus_id_set = set(bus_ids.tolist())

    bad_f = [int(x) for x in np.unique(branch_f) if int(x) not in bus_id_set]
    bad_t = [int(x) for x in np.unique(branch_t) if int(x) not in bus_id_set]
    if bad_f or bad_t:
        report = _msg(report, f"[FAIL] branch references missing bus ids: f-missing={bad_f[:10]} t-missing={bad_t[:10]}")
        report = SanityReport(ok=False, messages=report.messages)

    # --- generator bus ids are valid bus ids ---
    gen_bus_ids = sys.gen_bus_ids.astype(int)
    if not set(gen_bus_ids.tolist()).issubset(bus_id_set):
        missing = sorted(list(set(gen_bus_ids.tolist()) - bus_id_set))
        report = _msg(report, f"[FAIL] gen_bus_ids include missing bus ids: {missing[:20]}")
        report = SanityReport(ok=False, messages=report.messages)

    # --- reduced shapes ---
    if sys.K.shape != (ng, ng):
        report = _msg(report, f"[FAIL] K shape mismatch: {sys.K.shape} vs (ng,ng)=({ng},{ng})")
        report = SanityReport(ok=False, messages=report.messages)
    if sys.Peq.shape[0] != ng:
        report = _msg(report, f"[FAIL] Peq length mismatch: {sys.Peq.shape} vs ng={ng}")
        report = SanityReport(ok=False, messages=report.messages)

    if hasattr(sys, "delta_guess"):
        if sys.delta_guess.shape[0] != ng:
            report = _msg(report, f"[FAIL] delta_guess length mismatch: {sys.delta_guess.shape} vs ng={ng}")
            report = SanityReport(ok=False, messages=report.messages)

    # --- optional numeric sanity ---
    peq_sum = float(np.sum(sys.Peq))
    if abs(peq_sum) > 1e-8:
        report = _msg(report, f"[WARN] Peq sum not near zero: {peq_sum:.3e} (may be OK depending on balance_on_slack)")

    # --- enforce mapping usage warning ---
    # This isn't a strict failure; it's just a reminder to use BusMap.
    try:
        _ = make_busmap(sys.bus)
    except Exception as e:
        report = _msg(report, f"[FAIL] failed to build BusMap from sys.bus: {e}")
        report = SanityReport(ok=False, messages=report.messages)

    if strict and (not report.ok):
        raise RuntimeError("Sanity checks failed:\n" + "\n".join(report.messages))

    if report.ok:
        report = _msg(report, "[OK] sanity checks passed")
    return report
