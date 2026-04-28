#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


DEFAULT_METRICS_CSV = Path(
    "outputs/aggregates/analysis_busfield/figs_injbus/injbus_system_metrics_by_deltaP.csv"
)
DEFAULT_OUTDIR = DEFAULT_METRICS_CSV.parent


# ---------------------------
# Style (journal-ish, English only)
# ---------------------------
def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.linewidth": 1.0,
    })


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clean_xy(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    a = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    return a


def corr_row(df: pd.DataFrame, x: str, y: str) -> dict:
    a = _clean_xy(df, x, y)
    n = int(len(a))
    if n < 3:
        return {"n": n, "spearman_rho": np.nan, "pearson_r": np.nan}

    xs = a[x].to_numpy(dtype=float)
    ys = a[y].to_numpy(dtype=float)

    rho = float(spearmanr(xs, ys).statistic)
    r = float(pearsonr(xs, ys).statistic)
    return {"n": n, "spearman_rho": rho, "pearson_r": r}


def infer_centrality_cols(df: pd.DataFrame) -> list[str]:
    # 너가 논문에서 쓰기로 한 7개 세트 우선순위로 잡음
    candidates = [
        "degree",
        "kcore",
        "betweenness",
        "eigenvector",
        "strength",
        "closeness_x",
        "cf_closeness",   # current-flow closeness (네 파일에서 이렇게 저장했을 가능성이 큼)
    ]
    have = [c for c in candidates if c in df.columns]

    # 혹시 다른 이름으로 저장된 경우를 위한 보조 탐색
    if not have:
        # 예: centrality_* / cent_* 같은 패턴
        alt = [c for c in df.columns if c.startswith("centrality_") or c.startswith("cent_")]
        if alt:
            have = alt

    return have


def make_corr_table_multi_x(
    df: pd.DataFrame,
    xcols: list[str],
    ycols: list[str],
    split_by_deltaP: bool = True,
) -> pd.DataFrame:
    rows = []

    # pooled
    for x in xcols:
        for y in ycols:
            c = corr_row(df, x, y)
            rows.append({"group": "pooled", "deltaP": np.nan, "x": x, "y": y, **c})

    # by deltaP
    if split_by_deltaP and "deltaP" in df.columns:
        for dp, g in df.groupby("deltaP", sort=True):
            for x in xcols:
                for y in ycols:
                    c = corr_row(g, x, y)
                    rows.append({"group": "by_deltaP", "deltaP": float(dp), "x": x, "y": y, **c})

    return pd.DataFrame(rows)


def plot_corrbar_pooled_3panels_all_centrality(
    corr_df: pd.DataFrame,
    outpath: Path,
    xcols: list[str],
    sev_n: str,
    sev_r: str,
    clip_negative_to_zero: bool = True,
    title: str = "Spearman correlation: injection-bus centrality vs system-level response metrics",
) -> None:
    pooled = corr_df[corr_df["group"] == "pooled"].copy()
    if pooled.empty:
        raise RuntimeError("No pooled rows in corr table.")

    # (panel title) -> (y_nadir, y_rocof)
    panels = [
        ("Severity (max drop)", (sev_n, sev_r)),
        ("Average (mean drop)", ("mean_nadir_used", "mean_rocof_used")), # 추가
        ("Spread (Gini)", ("gini_nadir", "gini_rocof")),
        ("Concentration (Top 10%)", ("topshare_nadir", "topshare_rocof")), # 이름 수정
    ]

    def get_rho(x: str, y: str) -> float:
        hit = pooled[(pooled["x"] == x) & (pooled["y"] == y)]
        if len(hit) == 0:
            return float("nan")
        return float(hit.iloc[0]["spearman_rho"])

    safe_mkdir(outpath.parent)

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), dpi=200, sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # x-lims per panel (깔끔하게 보이도록 각 패널에서 자동)
    for ax, (ptitle, (yN, yR)) in zip(axes, panels):
        rn = np.array([get_rho(x, yN) for x in xcols], dtype=float)
        rr = np.array([get_rho(x, yR) for x in xcols], dtype=float)

        if clip_negative_to_zero:
            rn = np.clip(rn, 0.0, None)
            rr = np.clip(rr, 0.0, None)

        # 정렬: 해당 패널에서 max(rn, rr) 큰 순으로 보기 좋게
        rmax = np.nanmax(np.vstack([rn, rr]), axis=0)
        order = np.argsort(rmax)  # ascending -> barh 아래에서 위로 증가
        x_sorted = [xcols[i] for i in order]
        rn = rn[order]
        rr = rr[order]

        y_pos = np.arange(len(x_sorted))
        h = 0.36

        ax.grid(True, axis="x", alpha=0.25)
        ax.barh(
            y_pos - h/2, rn, height=h,
            edgecolor="black", facecolor="0.80", linewidth=0.9,
            label="Nadir-based"
        )
        ax.barh(
            y_pos + h/2, rr, height=h,
            edgecolor="black", facecolor="0.45", linewidth=0.9,
            label="RoCoF-based"
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(x_sorted)
        ax.set_xlabel("Spearman $\\rho$ (positive part)" if clip_negative_to_zero else "Spearman $\\rho$")
        ax.set_title(ptitle)

        xmax = float(np.nanmax(np.abs(np.r_[rn, rr]))) if np.isfinite(np.r_[rn, rr]).any() else 0.0
        ax.set_xlim(0.0 if clip_negative_to_zero else -1.0, max(0.06, 1.15 * xmax))

        # 숫자 라벨(너무 튀지 않게)
        for i in range(len(x_sorted)):
            if np.isfinite(rn[i]) and rn[i] > 0:
                ax.text(rn[i] + 0.005, i - h/2, f"{rn[i]:.3f}", va="center", fontsize=9)
            if np.isfinite(rr[i]) and rr[i] > 0:
                ax.text(rr[i] + 0.005, i + h/2, f"{rr[i]:.3f}", va="center", fontsize=9)

    # 범례는 한 번만
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, default=str(DEFAULT_METRICS_CSV))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--clip_neg", action="store_true", help="clip negative rho to 0 in the figure")
    ap.add_argument("--use_raw", action="store_true",
                    help="use raw severity (severity_nadir/severity_rocof) instead of normalized (severity_*_used)")
    ap.add_argument("--no_split", action="store_true",
                    help="do not compute per-deltaP correlations (pooled only)")
    args = ap.parse_args()

    set_plot_style()

    metrics_csv = Path(args.metrics_csv)
    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    print(f"[info] metrics_csv: {metrics_csv.resolve()}")
    print(f"[info] outdir:      {outdir.resolve()}")

    df = pd.read_csv(metrics_csv)

    # infer centrality columns (inj-bus centralities)
    xcols = infer_centrality_cols(df)
    if not xcols:
        raise ValueError(
            "Could not find centrality columns. "
            "Expected columns like: degree, kcore, betweenness, eigenvector, strength, closeness_x, cf_closeness."
        )
    print(f"[info] centrality cols: {xcols}")

    # choose severity columns
    if args.use_raw:
        sev_n = "severity_nadir"
        sev_r = "severity_rocof"
        sev_note = "raw"
    else:
        sev_n = "severity_nadir_used"
        sev_r = "severity_rocof_used"
        sev_note = "normalized"

    # check required y cols exist
    needed_y = [sev_n, sev_r, "gini_nadir", "gini_rocof", "topshare_nadir", "topshare_rocof", "mean_nadir_used", "mean_rocof_used"]
    miss = [c for c in needed_y if c not in df.columns]
    if miss:
        raise ValueError(f"metrics_csv missing required metric columns: {miss}")

    # make corr table (multi-x)
    corr = make_corr_table_multi_x(
        df=df,
        xcols=xcols,
        ycols=needed_y,
        split_by_deltaP=(not args.no_split),
    )

    corr_csv = outdir / "corr_injbus_allcentrality_vs_system_metrics.csv"
    corr.to_csv(corr_csv, index=False)
    print(f"[saved] {corr_csv}")

    # pooled figure: 3 panels (Severity / Gini / Topshare), each shows all centralities with 2 bars
    figpath = outdir / f"corrbar_injbus_allcentrality_3panels_{sev_note}.png"
    plot_corrbar_pooled_3panels_all_centrality(
        corr_df=corr,
        outpath=figpath,
        xcols=xcols,
        sev_n=sev_n,
        sev_r=sev_r,
        clip_negative_to_zero=args.clip_neg,
        title="Spearman correlation: injection-bus centrality vs system-level frequency response",
    )

    # quick peek (pooled)
    pooled = corr[corr["group"] == "pooled"].copy()
    pooled["abs_rho"] = pooled["spearman_rho"].abs()
    pooled = pooled.sort_values("abs_rho", ascending=False)
    print("\n[peek] pooled correlations (top 12 by |rho|)")
    cols = ["x", "y", "n", "spearman_rho", "pearson_r"]
    print(pooled[cols].head(12).to_string(index=False))

    print("[done]")


if __name__ == "__main__":
    main()
