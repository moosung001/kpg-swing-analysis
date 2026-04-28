#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analyze_batch_step.py

Default behavior (no args):
- Find the most recent batch_step_*.csv under outputs/aggregates/
- If matching meta json exists, copy it into analysis folder
- Produce:
  - analysis/summary_by_step.csv
  - analysis/ranking_step_<dPeq>.csv
  - analysis/fig_hist_worst_nadir_drop.png
  - analysis/fig_scatter_rocof_vs_nadir.png
  - analysis/fig_top20_worst_nadir_step_<dPeq>.png
  - analysis/fig_linear_scaling_check.png

Usage (PowerShell, from project root):
  $env:PYTHONPATH="src"
  python .\scripts\analyze_batch_step.py

Optional:
  python .\scripts\analyze_batch_step.py --csv .\outputs\aggregates\batch_step_20260130_000819.csv
  python .\scripts\analyze_batch_step.py --dir .\outputs\aggregates
  python .\scripts\analyze_batch_step.py --rank_step -2.0
  python .\scripts\analyze_batch_step.py --outdir .\outputs\aggregates\analysis_custom
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = [
    "step_dPeq_pu",
    "step_MW",
    "step_gen_local_idx",
    "obj2_worst_nadir_drop_hz",
    "obj4_worst_rocof_abs_hz_per_s",
    "coi_nadir_hz",
    "coi_rocof_fit_hz_per_s",
    "coi_settle_eps_1mHz_s",
    "worst_gen_nadir_local_idx",
    "worst_gen_rocof_local_idx",
    "worst_gen_nadir_gen_id",
    "worst_gen_nadir_bus_id",
    "worst_gen_nadir_time_s",
    "R_min",
    "angle_spread_rel_coi_max_rad",
]


def _ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _q(x: pd.Series, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join([f"  - {c}" for c in missing])
            + "\n\n(You may be using a CSV with a different schema.)"
        )


def _project_root_from_this_file() -> Path:
    # scripts/analyze_batch_step.py -> project root = parent of scripts
    return Path(__file__).resolve().parents[1]


def find_latest_batch_csv(search_dir: Path) -> Path:
    files = sorted(search_dir.glob("batch_step_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No batch_step_*.csv found in: {search_dir}")
    return files[0]


def guess_meta_for_csv(csv_path: Path) -> Path | None:
    # batch_step_YYYYMMDD_HHMMSS.csv -> batch_step_YYYYMMDD_HHMMSS_meta.json
    stem = csv_path.stem  # e.g., batch_step_20260130_000819
    meta = csv_path.with_name(f"{stem}_meta.json")
    return meta if meta.exists() else None


def summarize_by_step(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("step_dPeq_pu", sort=True)
    rows = []
    for step_pu, sub in g:
        coi_nadir = float(sub["coi_nadir_hz"].iloc[0])
        coi_rocof = float(sub["coi_rocof_fit_hz_per_s"].iloc[0])

        rows.append(
            {
                "step_dPeq_pu": float(step_pu),
                "step_MW": float(sub["step_MW"].iloc[0]),
                "n_cases": int(len(sub)),
                "coi_nadir_hz": coi_nadir,
                "coi_rocof_fit_hz_per_s": float(coi_rocof),
                "coi_settle_1mHz_s": float(sub["coi_settle_eps_1mHz_s"].iloc[0]),
                "worst_nadir_drop_min_hz": float(sub["obj2_worst_nadir_drop_hz"].min()),
                "worst_nadir_drop_p50_hz": float(sub["obj2_worst_nadir_drop_hz"].median()),
                "worst_nadir_drop_p95_hz": _q(sub["obj2_worst_nadir_drop_hz"], 0.95),
                "worst_nadir_drop_max_hz": float(sub["obj2_worst_nadir_drop_hz"].max()),
                "worst_rocof_abs_p50_hz_per_s": float(sub["obj4_worst_rocof_abs_hz_per_s"].median()),
                "worst_rocof_abs_p95_hz_per_s": _q(sub["obj4_worst_rocof_abs_hz_per_s"], 0.95),
                "worst_rocof_abs_max_hz_per_s": float(sub["obj4_worst_rocof_abs_hz_per_s"].max()),
                "Rmin_min": float(sub["R_min"].min()),
                "angle_spread_max_rad": float(sub["angle_spread_rel_coi_max_rad"].max()),
                "ratio_max_worst_nadir_to_coi_nadir": float(
                    sub["obj2_worst_nadir_drop_hz"].max() / (abs(coi_nadir) if abs(coi_nadir) > 0 else np.nan)
                ),
                "ratio_max_worst_rocof_to_coi_rocof": float(
                    sub["obj4_worst_rocof_abs_hz_per_s"].max() / (abs(coi_rocof) if abs(coi_rocof) > 0 else np.nan)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("step_dPeq_pu").reset_index(drop=True)


def make_ranking(df: pd.DataFrame, rank_step: float) -> pd.DataFrame:
    sub = df[df["step_dPeq_pu"] == float(rank_step)].copy()
    if len(sub) == 0:
        steps = sorted(df["step_dPeq_pu"].unique().tolist())
        raise ValueError(f"No rows for step_dPeq_pu={rank_step}. Available: {steps}")

    sub["is_worst_nadir_same_as_step"] = (sub["worst_gen_nadir_local_idx"] == sub["step_gen_local_idx"]).astype(int)
    sub["is_worst_rocof_same_as_step"] = (sub["worst_gen_rocof_local_idx"] == sub["step_gen_local_idx"]).astype(int)

    cols = [
        "step_gen_local_idx",
        "worst_gen_nadir_gen_id",
        "worst_gen_nadir_bus_id",
        "step_MW",
        "obj2_worst_nadir_drop_hz",
        "obj4_worst_rocof_abs_hz_per_s",
        "worst_gen_nadir_time_s",
        "R_min",
        "angle_spread_rel_coi_max_rad",
        "is_worst_nadir_same_as_step",
        "is_worst_rocof_same_as_step",
    ]
    rank = sub[cols].sort_values("obj2_worst_nadir_drop_hz", ascending=False).reset_index(drop=True)
    rank.index += 1
    rank.rename_axis("rank", inplace=True)
    return rank


def plot_hist_worst_nadir(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    for step, sub in df.groupby("step_dPeq_pu", sort=True):
        plt.hist(sub["obj2_worst_nadir_drop_hz"], bins=30, alpha=0.5, label=f"dPeq={step:g} pu")
    plt.xlabel("Worst generator nadir drop (Hz)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_scatter_rocof_vs_nadir(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    for step, sub in df.groupby("step_dPeq_pu", sort=True):
        plt.scatter(
            sub["obj2_worst_nadir_drop_hz"],
            sub["obj4_worst_rocof_abs_hz_per_s"],
            s=12,
            alpha=0.7,
            label=f"dPeq={step:g} pu",
        )
    plt.xlabel("Worst generator nadir drop (Hz)")
    plt.ylabel("Worst generator |RoCoF| (Hz/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_top20_bar(ranking: pd.DataFrame, outpath: Path) -> None:
    top = ranking.head(20).copy()
    x = np.arange(len(top))
    y = np.asarray(top["obj2_worst_nadir_drop_hz"], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.xticks(x, [str(int(i)) for i in top["step_gen_local_idx"]], rotation=90)
    plt.xlabel("Disturbed generator local index")
    plt.ylabel("Worst nadir drop (Hz)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_linear_scaling_check(df: pd.DataFrame, outpath: Path) -> None:
    piv = df.pivot_table(
        index="step_gen_local_idx",
        columns="step_dPeq_pu",
        values="obj2_worst_nadir_drop_hz",
        aggfunc="mean",
    )

    plt.figure()
    if (-1.0 in piv.columns) and (-2.0 in piv.columns):
        x = np.asarray(piv[-1.0], dtype=float)
        y = np.asarray(piv[-2.0] / 2.0, dtype=float)
        plt.scatter(x, y, s=12, alpha=0.7, label="(-2pu)/2 vs (-1pu)")
    if (-0.5 in piv.columns) and (-1.0 in piv.columns):
        x = np.asarray(piv[-1.0], dtype=float)
        y = np.asarray(piv[-0.5] * 2.0, dtype=float)
        plt.scatter(x, y, s=12, alpha=0.7, label="(-0.5pu)*2 vs (-1pu)")

    plt.xlabel("Worst nadir drop at -1.0 pu (Hz)")
    plt.ylabel("Scaled comparison (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, type=str, help="Path to batch_step_*.csv (optional)")
    ap.add_argument(
        "--dir",
        default=None,
        type=str,
        help="Directory to search latest batch_step_*.csv (optional). Default: <project_root>/outputs/aggregates",
    )
    ap.add_argument("--meta", default=None, type=str, help="Optional meta json path (optional)")
    ap.add_argument("--rank_step", default=-1.0, type=float, help="Step size (pu) for ranking (default: -1.0)")
    ap.add_argument("--outdir", default=None, type=str, help="Output directory (optional). Default: <csv_dir>/analysis")
    args = ap.parse_args()

    proj = _project_root_from_this_file()

    # Decide CSV path
    if args.csv:
        csv_path = Path(args.csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = (proj / csv_path).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
    else:
        search_dir = Path(args.dir).expanduser() if args.dir else (proj / "outputs" / "aggregates")
        if not search_dir.is_absolute():
            search_dir = (proj / search_dir).resolve()
        csv_path = find_latest_batch_csv(search_dir)

    # Decide meta path
    if args.meta:
        meta_path = Path(args.meta).expanduser()
        if not meta_path.is_absolute():
            meta_path = (proj / meta_path).resolve()
        if not meta_path.exists():
            meta_path = None
    else:
        meta_path = guess_meta_for_csv(csv_path)

    print(f"[info] csv:  {csv_path}")
    if meta_path:
        print(f"[info] meta: {meta_path}")
    else:
        print("[info] meta: (not found / not provided)")

    df = pd.read_csv(csv_path)
    _check_columns(df)

    # Filter to successes if column exists
    if "success" in df.columns:
        df_ok = df[df["success"] == 1].copy()
    else:
        df_ok = df.copy()
    if len(df_ok) == 0:
        raise ValueError("No successful rows to analyze.")

    # Output directory
    outdir = Path(args.outdir).expanduser() if args.outdir else (csv_path.parent / "analysis")
    if not outdir.is_absolute():
        outdir = (proj / outdir).resolve()
    outdir = _ensure_outdir(outdir)

    # Copy meta into analysis folder if available
    if meta_path and meta_path.exists():
        (outdir / meta_path.name).write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Summary
    s = summarize_by_step(df_ok)
    s_csv = outdir / "summary_by_step.csv"
    s.to_csv(s_csv, index=False, encoding="utf-8-sig")
    print(f"[saved] {s_csv}")

    # Ranking
    r = make_ranking(df_ok, rank_step=float(args.rank_step))
    r_csv = outdir / f"ranking_step_{args.rank_step:g}.csv"
    r.to_csv(r_csv, encoding="utf-8-sig")
    print(f"[saved] {r_csv}")

    # Figures
    plot_hist_worst_nadir(df_ok, outdir / "fig_hist_worst_nadir_drop.png")
    plot_scatter_rocof_vs_nadir(df_ok, outdir / "fig_scatter_rocof_vs_nadir.png")
    plot_top20_bar(r, outdir / f"fig_top20_worst_nadir_step_{args.rank_step:g}.png")
    plot_linear_scaling_check(df_ok, outdir / "fig_linear_scaling_check.png")
    print(f"[done] outputs in: {outdir}")


if __name__ == "__main__":
    main()
