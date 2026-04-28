from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = ROOT / "outputs" / "aggregates"
DEFAULT_OUTDIR = AGG_DIR / "analysis_gen"


def pick_latest(pattern: str, root: Path) -> Path:
    cands = list(root.rglob(pattern))
    if not cands:
        raise FileNotFoundError(f"Cannot find {pattern} under {root}")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def corr_table(df: pd.DataFrame, x_cols: list[str], y_cols: list[str]) -> pd.DataFrame:
    rows = []
    for x in x_cols:
        if x not in df.columns:
            continue
        for y in y_cols:
            if y not in df.columns:
                continue
            a = df[[x, y]].dropna()
            n = len(a)
            if n < 5:
                rows.append({"x": x, "y": y, "n": n, "spearman_rho": np.nan, "pearson_r": np.nan})
                continue
            spearman = a[x].rank().corr(a[y].rank(), method="pearson")
            pearson = a[x].corr(a[y], method="pearson")
            rows.append({"x": x, "y": y, "n": n, "spearman_rho": float(spearman), "pearson_r": float(pearson)})

    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman_rho"].abs()
    out = out.sort_values(["abs_spearman", "n"], ascending=[False, False]).drop(columns=["abs_spearman"])
    return out


def scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str | None = None):
    a = df[[x, y]].dropna()
    if len(a) == 0:
        return
    plt.figure()
    plt.scatter(a[x].to_numpy(), a[y].to_numpy(), s=18)
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_summary", type=str, default=None, help="path to gen_axis_summary.csv (optional)")
    ap.add_argument("--batch_csv", type=str, default=None, help="path to batch_step_*.csv (optional)")
    ap.add_argument("--dyn_params", type=str, default=None, help="path to data_static/dyn_params.csv (optional override)")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gen_summary = Path(args.gen_summary) if args.gen_summary else (DEFAULT_OUTDIR / "gen_axis_summary.csv")
    if not gen_summary.exists():
        gen_summary = pick_latest("gen_axis_summary.csv", AGG_DIR)
    print("[info] gen_summary:", gen_summary)

    batch_csv = Path(args.batch_csv) if args.batch_csv else pick_latest("batch_step_*.csv", AGG_DIR)
    print("[info] batch_csv:", batch_csv)

    # dyn_params는 v2에서 고정 위치가 기본값
    dyn_params = Path(args.dyn_params) if args.dyn_params else (ROOT / "data_static" / "dyn_params.csv")
    if not dyn_params.exists():
        raise FileNotFoundError(f"dyn_params not found: {dyn_params} (expected at data_static/dyn_params.csv)")
    print("[info] dyn_params:", dyn_params)

    g = pd.read_csv(gen_summary)
    b = pd.read_csv(batch_csv, usecols=[c for c in ["slack_bus_id"] if c in pd.read_csv(batch_csv, nrows=1).columns])
    if "slack_bus_id" not in b.columns:
        raise KeyError("batch_step_*.csv must contain 'slack_bus_id'")
    slack_bus_id = int(pd.read_csv(batch_csv)["slack_bus_id"].iloc[0])
    print("[info] slack_bus_id:", slack_bus_id)

    d = pd.read_csv(dyn_params)

    # dyn_params 스키마 검증
    required = ["gen_id", "bus", "type", "S_base_MVA", "H_s", "M", "D"]
    miss = [c for c in required if c not in d.columns]
    if miss:
        raise KeyError(f"dyn_params missing columns: {miss}")

    # 슬랙버스 발전기 제거(순서 유지) 후 gen_local_idx 재구성
    d2 = d[d["bus"].astype(int) != slack_bus_id].copy()
    d2["gen_local_idx"] = np.arange(len(d2), dtype=int)

    # 보기 좋은 이름으로 일부 rename
    d2 = d2.rename(columns={
        "bus": "bus_id",
        "S_base_MVA": "S_base_MVA",
        "H_s": "H",
    })

    # gen_axis_summary 쪽 필수키
    if "gen_local_idx" not in g.columns:
        raise KeyError("gen_axis_summary.csv must contain 'gen_local_idx'")

    # 숫자 변환
    g = coerce_numeric(g, [
        "coi_nadir_norm_mean", "coi_rocof_norm_mean",
        "worst_nadir_norm_mean", "worst_rocof_norm_mean",
        "settle_1mHz_mean", "one_minus_Rmin_mean", "angle_spread_mean",
    ])
    d2 = coerce_numeric(d2, ["H", "M", "D", "Pg", "Qg", "S_base_MVA", "bus_id"])

    # merge
    df = g.merge(
        d2[["gen_local_idx", "gen_id", "bus_id", "type", "Pg", "Qg", "S_base_MVA", "H", "M", "D"]],
        on="gen_local_idx",
        how="left"
    )

    out_join = outdir / "gen_axis_join_params.csv"
    df.to_csv(out_join, index=False)
    print("[saved]", out_join)

    # 상관 계산: 네가 관심 갖는 축 위주
    x_cols = ["H", "M", "D", "Pg", "S_base_MVA"]
    y_cols = [
        "worst_nadir_norm_mean",
        "worst_rocof_norm_mean",
        "angle_spread_mean",
        "one_minus_Rmin_mean",
        "settle_1mHz_mean",
        "coi_nadir_norm_mean",
        "coi_rocof_norm_mean",
    ]

    df_corr = corr_table(df, x_cols=x_cols, y_cols=y_cols)
    out_corr = outdir / "corr_gen_params_vs_response.csv"
    df_corr.to_csv(out_corr, index=False)
    print("[saved]", out_corr)

    # 산점도(핵심)
    for x in ["H", "M", "D"]:
        if x in df.columns:
            scatter(df, x, "worst_nadir_norm_mean", outdir / f"scatter_{x}_vs_worst_nadir_norm.png",
                    title=f"{x} vs worst_nadir_norm_mean")
            scatter(df, x, "worst_rocof_norm_mean", outdir / f"scatter_{x}_vs_worst_rocof_norm.png",
                    title=f"{x} vs worst_rocof_norm_mean")
            scatter(df, x, "angle_spread_mean", outdir / f"scatter_{x}_vs_angle_spread.png",
                    title=f"{x} vs angle_spread_mean")

    print("\n[peek] corr top 12 by |spearman|")
    print(df_corr.head(12).to_string(index=False))

    # sanity check
    n_missing = int(df["gen_id"].isna().sum())
    if n_missing > 0:
        print(f"\n[warn] gen_id missing after join: {n_missing} rows (check gen_local_idx alignment)")
    else:
        print("\n[ok] join looks consistent (no missing gen_id)")


if __name__ == "__main__":
    main()
