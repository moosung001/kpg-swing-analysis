from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _solve_delta_dc_init(
    K: np.ndarray,
    Peq: np.ndarray,
    *,
    ref_idx: int,
    fixed_idx: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """
    선형화(DC) 초기값 생성:
      Peq ≈ L(K) * delta
    ref_idx는 기준각(delta=0)으로 고정.
    """
    K = np.asarray(K, dtype=float)
    Peq = np.asarray(Peq, dtype=float).reshape(-1)
    ng = Peq.shape[0]
    if K.shape != (ng, ng):
        raise ValueError("K shape mismatch in _solve_delta_dc_init")

    # Laplacian L = diag(sum K_ij) - K (with zero diag in K assumed OK)
    L = -K.copy()
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -np.sum(L, axis=1))

    ref_idx = int(ref_idx)
    if not (0 <= ref_idx < ng):
        raise ValueError("ref_idx out of range")


    fixed = np.zeros(ng, dtype=bool)
    fixed[ref_idx] = True
    if fixed_idx is not None:
        fixed_idx_arr = np.asarray(fixed_idx, dtype=int).reshape(-1)
        for k in fixed_idx_arr.tolist():
            if 0 <= int(k) < ng:
                fixed[int(k)] = True

    mask = ~fixed


    Lr = L[np.ix_(mask, mask)]
    pr = Peq[mask]

    # 수치 안정성(연결성/특이 대비): 아주 작은 ridge
    eps = 1e-12
    Lr = Lr + eps * np.eye(Lr.shape[0])

    delta = np.zeros(ng, dtype=float)
    delta[mask] = np.linalg.solve(Lr, pr)
    delta[ref_idx] = 0.0
    delta[fixed] = 0.0

    return delta


def solve_delta_nonlinear_equilibrium(
    K: np.ndarray,
    Peq: np.ndarray,
    *,
    ref_idx: int,
    fixed_idx: np.ndarray | list[int] | None = None,
    delta0: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-10,
    step_damping: float = 1.0,
) -> np.ndarray:
    """
    비선형 평형(사인 결합) 해:
      F(delta) = Peq - sum_j K_ij sin(delta_i - delta_j) = 0
    ref_idx를 기준각 0으로 고정하고 나머지 (ng-1)개만 푼다.

    - delta0 없으면 DC 초기값으로 시작.
    - 뉴턴법 + 감쇠(step_damping) 사용.
    """
    K = np.asarray(K, dtype=float)
    Peq = np.asarray(Peq, dtype=float).reshape(-1)
    ng = Peq.shape[0]
    if K.shape != (ng, ng):
        raise ValueError("K shape mismatch in solve_delta_nonlinear_equilibrium")

    ref_idx = int(ref_idx)
    if not (0 <= ref_idx < ng):
        raise ValueError("ref_idx out of range")

    if delta0 is None:
        # continuation의 첫 단계(작은 a)에 맞춰서 초기값을 잡기 위해
        # 일단 0으로 시작 (고정 노드는 나중에 0으로 강제)
        delta = np.zeros(ng, dtype=float)
        delta[ref_idx] = 0.0
    else:
        delta = np.asarray(delta0, dtype=float).reshape(-1).copy()
        if delta.shape[0] != ng:
            raise ValueError("delta0 length mismatch")
        delta[ref_idx] = 0.0

    # fixed indices: ref_idx + additional fixed_idx (e.g., isolated nodes)
    fixed = np.zeros(ng, dtype=bool)
    fixed[ref_idx] = True
    if fixed_idx is not None:
        fixed_idx_arr = np.asarray(fixed_idx, dtype=int).reshape(-1)
        for k in fixed_idx_arr.tolist():
            if 0 <= int(k) < ng:
                fixed[int(k)] = True

    mask = ~fixed  # unknowns to solve
    delta[fixed] = 0.0



    # continuation: gradually scale Peq (0 -> 1)
    alphas = np.linspace(0.0, 1.0, 21)  
    for si, a in enumerate(alphas[1:], start=1):
        
        Peq_a = float(a) * Peq

        # 첫 stage에서만 DC init (이후에는 continuation: 이전 stage 해를 그대로 사용)
        if si == 1:
            delta = _solve_delta_dc_init(
                K, Peq_a, ref_idx=ref_idx, fixed_idx=np.where(fixed)[0]
            )
            delta[fixed] = 0.0



        for _ in range(int(max_iter)):
            dmat = delta[:, None] - delta[None, :]
            S = np.sin(dmat)
            C = np.cos(dmat)

            Pe = np.sum(K * S, axis=1)
            F = Peq_a - Pe
            Fr = F[mask]

            fnorm = float(np.max(np.abs(Fr)))
            if fnorm <= float(tol):
                break

            J = K * C
            J_full = J.copy()
            np.fill_diagonal(J_full, 0.0)
            diag = -np.sum(J_full, axis=1)
            np.fill_diagonal(J_full, diag)

            Jr = J_full[np.ix_(mask, mask)]
            Jr = Jr + 1e-12 * np.eye(Jr.shape[0])

            d = np.linalg.solve(Jr, -Fr)

            # backtracking line search: residual이 줄어드는 step만 채택
            base_norm = fnorm
            lam = float(step_damping)

            accepted = False
            for _ls in range(20):
                trial = delta.copy()
                trial[mask] = trial[mask] + lam * d
                trial[ref_idx] = 0.0
                trial[fixed] = 0.0


                dmat_t = trial[:, None] - trial[None, :]
                Pe_t = np.sum(K * np.sin(dmat_t), axis=1)
                F_t = Peq_a - Pe_t
                norm_t = float(np.max(np.abs(F_t[mask])))

                if norm_t < base_norm:
                    delta = trial
                    delta[fixed] = 0.0
                    accepted = True
                    break
                lam *= 0.5

            if not accepted:
                # 더 이상 줄어드는 step을 못 찾으면 이번 stage는 실패로 처리
                break

                    


        # stage convergence check (each alpha must converge)
        dmat = delta[:, None] - delta[None, :]
        Pe = np.sum(K * np.sin(dmat), axis=1)
        F = Peq_a - Pe
        fnorm = float(np.max(np.abs(F[mask])))

        # stage 통과 기준: 작은 a에서는 잔차가 Peq_a 크기 수준까지 남을 수 있어서
        # a에 비례하는 허용오차를 둔다. (초기각 seed 생성 목적)
        stage_tol = max(1e-6, 2e-3 * float(a))  # a=0.05면 1e-4

        if fnorm > stage_tol:
            raise RuntimeError(
                f"continuation stage a={a:.2f} did not converge: max|F|={fnorm:.3e} (stage_tol={stage_tol:.3e})"
            )



    # final delta is for alpha=1
    delta[ref_idx] = 0.0
    return delta





from kpg_swing.engine.dcflow import parse_mfile, build_B_and_meta


@dataclass(frozen=True)
class KronResult:
    K: np.ndarray           # (ng, ng)  K = -Bred (offdiag >= 0, diag <= 0)
    delta_guess: np.ndarray # (ng,)      초기각(라디안)
    Peq: np.ndarray         # (ng,)      등가 주입(평균 제거 후)
    slack_bus_id: int       # MATPOWER bus id (1-based)
    baseMVA: float


def compute_K_and_Peq_from_case(
    mfile: str | Path,
    gen_bus_ids: Iterable[int],
    P_bus_pu: np.ndarray,
    xd_prime_pu: float | np.ndarray = 0.3,
    slack_bus_id: int | None = None,
    balance_on_slack: bool = True,
) -> KronResult:
    bus, branch, baseMVA = parse_mfile(Path(mfile))
    K, delta_guess, Peq, slack_id = compute_K_and_Peq_from_arrays(
        bus=bus,
        branch=branch,
        gen_bus_ids=np.asarray(list(gen_bus_ids), dtype=int),
        P_bus_pu=P_bus_pu,
        xd_prime_pu=xd_prime_pu,
        slack_bus_id=slack_bus_id,
        balance_on_slack=balance_on_slack,
    )
    return KronResult(K=K, delta_guess=delta_guess, Peq=Peq, slack_bus_id=slack_id, baseMVA=float(baseMVA))


def compute_K_and_Peq_from_arrays(
    bus: np.ndarray,
    branch: np.ndarray,
    gen_bus_ids: np.ndarray,     # (ng,) MATPOWER bus id (1-based), 중복 허용
    P_bus_pu: np.ndarray,        # (nb,)
    xd_prime_pu: float | np.ndarray = 0.3,  # float 또는 (ng,)
    slack_bus_id: int | None = None,        # MATPOWER bus id (1-based)
    balance_on_slack: bool = True,
    *,
    center_Peq: bool = True,          # NEW: Peq 평균제거(평형 만들기용)
    solve_delta_guess: bool = True,    # NEW: delta_guess를 비선형 평형으로 풀지 여부
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)

    bus_ids = bus[:, 0].astype(int)  # MATPOWER bus id
    nb = bus.shape[0]

    # slack 선택
    bus_type = bus[:, 1].astype(int)
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}

    if slack_bus_id is None:
        slack_rows = np.where(bus_type == 3)[0]
        slack_bus_id = int(bus_ids[int(slack_rows[0])]) if slack_rows.size > 0 else int(bus_ids[0])

    if int(slack_bus_id) not in bus_id_to_idx:
        raise ValueError(f"slack_bus_id={slack_bus_id} not found in bus table")
    slack_idx = bus_id_to_idx[int(slack_bus_id)]

    # Bbus 구성
    Bbus, _meta = build_B_and_meta(bus, branch)

    # 발전기 버스 인덱스(0-based)
    gen_bus_ids = np.asarray(gen_bus_ids, dtype=int).reshape(-1)
    ng = gen_bus_ids.shape[0]
    gen_bus_idx = np.array([bus_id_to_idx[int(b)] for b in gen_bus_ids], dtype=int)

    # xd' 처리: float 또는 벡터
    if np.isscalar(xd_prime_pu):
        xd_vec = np.full(ng, float(xd_prime_pu), dtype=float)
    else:
        xd_vec = np.asarray(xd_prime_pu, dtype=float).reshape(-1)
        if xd_vec.shape[0] != ng:
            raise ValueError(f"xd_prime_pu length mismatch: got {xd_vec.shape[0]}, expected {ng}")
    if np.any(xd_vec <= 0):
        raise ValueError("xd_prime_pu는 양수여야 합니다")

    # 확장 Bext: [bus nb] + [internal ng]
    n_ext = nb + ng
    Bext = np.zeros((n_ext, n_ext), dtype=float)
    Bext[:nb, :nb] = Bbus

    # internal 연결(발전기별 xd')
    for k, bi in enumerate(gen_bus_idx):
        gi = nb + k
        b_xd = -1.0 / float(xd_vec[k])  # offdiag는 음수
        Bext[bi, gi] += b_xd
        Bext[gi, bi] += b_xd
        Bext[bi, bi] -= b_xd
        Bext[gi, gi] -= b_xd

    # 버스 주입 벡터
    P = np.asarray(P_bus_pu, dtype=float).reshape(-1)
    if P.shape[0] != nb:
        raise ValueError(f"P_bus_pu length mismatch: got {P.shape[0]}, expected {nb}")

    P_adj = P.copy()
    if balance_on_slack:
        P_adj[slack_idx] -= float(np.sum(P_adj))

    Pext = np.zeros(n_ext, dtype=float)
    Pext[:nb] = P_adj

    # 제거: slack 제외 모든 bus
    elim_bus = np.array([i for i in range(nb) if i != slack_idx], dtype=int)
    keep_int = np.arange(nb, n_ext, dtype=int)

    Bee = Bext[np.ix_(elim_bus, elim_bus)]
    Bek = Bext[np.ix_(elim_bus, keep_int)]
    Bke = Bext[np.ix_(keep_int, elim_bus)]
    Bkk = Bext[np.ix_(keep_int, keep_int)]

    Pe = Pext[elim_bus]
    Pk = Pext[keep_int]  # internal은 0

    # Schur
    try:
        X = np.linalg.solve(Bee, Bek)
        y = np.linalg.solve(Bee, Pe)
    except np.linalg.LinAlgError:
        Bee_pinv = np.linalg.pinv(Bee)
        X = Bee_pinv @ Bek
        y = Bee_pinv @ Pe

    Bred = Bkk - Bke @ X
    Peq = Pk - Bke @ y

    # 수치 정리
    Bred = 0.5 * (Bred + Bred.T)
    K = -Bred

    # ---- isolate detection on K (generator coupling graph) ----
    K_off = K.copy()
    np.fill_diagonal(K_off, 0.0)
    row_cap = np.sum(np.abs(K_off), axis=1)

    eps_iso = 1e-9
    iso_idx = np.where(row_cap <= eps_iso)[0]  # isolated generators (no coupling)

    # (선택) Peq centering: 평형(ΣPeq=0)을 강제하고 싶을 때만
    if bool(center_Peq):
        keep = np.ones(Peq.shape[0], dtype=bool)
        keep[iso_idx] = False
        if np.any(keep):
            Peq = Peq.copy()
            Peq[keep] = Peq[keep] - float(np.mean(Peq[keep]))

    # isolated nodes must have Peq=0 (always)
    if iso_idx.size > 0:
        Peq = Peq.copy()
        Peq[iso_idx] = 0.0
    # ---------------------------------------------

    # ===== diagnostics =====
    K_off = K.copy()
    np.fill_diagonal(K_off, 0.0)

    row_cap = np.sum(np.abs(K_off), axis=1)
    pe_abs = np.abs(Peq)

    print("[diag] K_off abs max:", float(np.max(np.abs(K_off))))
    print("[diag] Peq abs max:", float(np.max(pe_abs)))
    print("[diag] min row_cap, max row_cap:", float(np.min(row_cap)), float(np.max(row_cap)))
    print("[diag] max |Peq|/row_cap:", float(np.max(pe_abs / (row_cap + 1e-12))))

    eps = 1e-9
    adj = (np.abs(K_off) > eps)
    seen = np.zeros(ng, dtype=bool)
    sizes = []
    for s in range(ng):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cnt = 0
        while stack:
            u = stack.pop()
            cnt += 1
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        sizes.append(cnt)
    sizes.sort(reverse=True)
    print("[diag] K components:", len(sizes), "sizes:", sizes[:10])
    # ===== end diagnostics =====

    # delta_guess: 결손(ΣP≠0) 시나리오에서는 의미가 없거나 실패할 수 있으므로 옵션으로 끈다.
    if bool(solve_delta_guess):
        ref_candidates = np.where(gen_bus_ids.astype(int) == int(slack_bus_id))[0]
        ref_idx = int(ref_candidates[0]) if ref_candidates.size > 0 else 0

        delta_guess = solve_delta_nonlinear_equilibrium(
            K=K,
            Peq=Peq,
            ref_idx=ref_idx,
            fixed_idx=iso_idx,
            delta0=None,
            max_iter=50,
            tol=1e-10,
            step_damping=1.0,
        )
    else:
        delta_guess = np.zeros(ng, dtype=float)

    return K, delta_guess, Peq, int(slack_bus_id)
