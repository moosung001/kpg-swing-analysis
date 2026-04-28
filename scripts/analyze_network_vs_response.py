#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analyze_network_vs_response.py

Network-structure (degree/centrality) vs dynamic response (vulnerability) analysis.

Default behavior:
- Find latest batch_step_*.csv under outputs/aggregates/
- Use its paired meta json if exists
- Load MATPOWER m-file path from meta.json (case_mfile)
- Build bus graph from branch list (status==1)
- Compute network metrics on buses (degree, strength, betweenness, closeness, eigenvector, pagerank, kcore)
- Map generator -> bus via batch CSV columns (worst_gen_nadir_bus_id etc.)
- Create vulnerability per generator for each step:
    vuln_nadir = obj2_worst_nadir_drop_hz
    vuln_rocof = obj4_worst_rocof_abs_hz_per_s
    drift_flag = worst_gen_nadir_time_s > drift_frac * t_final
- Aggregate to bus-level (max over generators on same bus)
- Compute correlations (Spearman) between bus metrics and vulnerability
- Save tables and plots into <csv_dir>/analysis_net/

Usage:
  $env:PYTHONPATH="src"
  python .\scripts\analyze_network_vs_response.py

Optional:
  python .\scripts\analyze_network_vs_response.py --csv .\outputs\aggregates\batch_step_*.csv
  python .\scripts\analyze_network_vs_response.py --step -1
  python .\scripts\analyze_network_vs_response.py --weight unweighted
  python .\scripts\analyze_network_vs_response.py --drift_frac 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx


def project_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def find_latest_batch_csv(search_dir: Path) -> Path:
    files = sorted(search_dir.glob("batch_step_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No batch_step_*.csv found in: {search_dir}")
    return files[0]


def guess_meta_for_csv(csv_path: Path) -> Path | None:
    meta = csv_path.with_name(f"{csv_path.stem}_meta.json")
    return meta if meta.exists() else None


def ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_meta(meta_path: Path | None) -> dict:
    if meta_path is None:
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def parse_mfile_via_project(case_mfile: Path):
    """
    Prefer using the project's parser if available.
    Falls back to a minimal MATPOWER branch-only parser if import fails.
    """
    try:
        from kpg_swing.engine.dcflow import parse_mfile  # type: ignore
        bus, branch, baseMVA = parse_mfile(case_mfile)
        return np.asarray(bus, dtype=float), np.asarray(branch, dtype=float), float(baseMVA)
    except Exception:
        # Minimal fallback: parse only branch matrix from .m
        text = case_mfile.read_text(encoding="utf-8", errors="ignore")
        key = "mpc.branch"
        i0 = text.find(key)
        if i0 < 0:
            raise RuntimeError("Could not find mpc.branch in m-file.")
        i1 = text.find("[", i0)
        i2 = text.find("];", i1)
        block = text[i1 + 1 : i2]
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            line = line.split("%")[0].strip()
            line = line.replace(";", " ").strip()
            if not line:
                continue
            parts = line.split()
            rows.append([float(x) for x in parts])
        branch = np.array(rows, dtype=float)
        return None, branch, float("nan")


def build_bus_graph(branch: np.ndarray, weight_mode: str) -> nx.Graph:
    """
    MATPOWER branch columns (0-index):
      0 fbus, 1 tbus, 2 r, 3 x, ... , 10 status (MATPOWER uses col 11 1-indexed)
    """
    G = nx.Graph()

    if branch.ndim != 2 or branch.shape[1] < 11:
        raise ValueError("branch matrix does not look like MATPOWER branch (need >= 11 cols).")

    for row in branch:
        f = int(row[0])
        t = int(row[1])
        x = float(row[3])
        status = int(row[10])

        if status != 1:
            continue

        if weight_mode == "unweighted":
            w = 1.0
            dist = 1.0
        else:
            # stronger coupling for smaller |x|
            eps = 1e-6
            w = 1.0 / max(abs(x), eps)
            # For shortest paths, use "distance" as inverse of coupling
            dist = 1.0 / max(w, eps)

        # accumulate parallel lines: sum weights, keep min distance
        if G.has_edge(f, t):
            G[f][t]["weight"] += w
            G[f][t]["distance"] = min(G[f][t]["distance"], dist)
        else:
            G.add_edge(f, t, weight=w, distance=dist)

    return G


def compute_bus_metrics(G: nx.Graph) -> pd.DataFrame:
    # Basic
    deg = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    # Centralities
    betw = nx.betweenness_centrality(G, weight="distance", normalized=True)
    close = nx.closeness_centrality(G, distance="distance")

    # Eigenvector may fail on disconnected graphs; handle per component then merge
    eig = {}
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        if len(H) < 3:
            for n in H.nodes:
                eig[n] = 0.0
            continue
        try:
            ec = nx.eigenvector_centrality(H, weight="weight", max_iter=5000)
            eig.update(ec)
        except Exception:
            for n in H.nodes:
                eig[n] = 0.0

    pr = nx.pagerank(G, weight="weight")
    core = nx.core_number(G)

    df = pd.DataFrame(
        {
            "bus_id": list(G.nodes),
            "degree": [deg[n] for n in G.nodes],
            "strength": [strength[n] for n in G.nodes],
            "betweenness": [betw[n] for n in G.nodes],
            "closeness": [close[n] for n in G.nodes],
            "eigenvector": [eig.get(n, 0.0) for n in G.nodes],
            "pagerank": [pr[n] for n in G.nodes],
            "kcore": [core[n] for n in G.nodes],
        }
    ).sort_values("bus_id")
    return df


def spearman_corr_table(df_join: pd.DataFrame, ycol: str, xcols: list[str]) -> pd.DataFrame:
    # Spearman by ranking
    out = []
    y = df_join[ycol].astype(float)
    for xcol in xcols:
        x = df_join[xcol].astype(float)
        ok = (~y.isna()) & (~x.isna())
        if ok.sum() < 5:
            rho = np.nan
        else:
            rho = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank(), method="pearson")
        out.append({"x": xcol, "spearman_rho": float(rho), "n": int(ok.sum())})
    return pd.DataFrame(out).sort_values("spearman_rho", ascending=False)


def plot_scatter(df_join: pd.DataFrame, x: str, y: str, outpath: Path, title: str):
    plt.figure()
    plt.scatter(df_join[x].astype(float), df_join[y].astype(float), s=14, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, type=str, help="batch_step_*.csv (optional)")
    ap.add_argument("--dir", default=None, type=str, help="search dir (default outputs/aggregates)")
    ap.add_argument("--meta", default=None, type=str, help="meta json (optional)")
    ap.add_argument("--step", default=-1.0, type=float, help="step_dPeq_pu to analyze (default -1)")
    ap.add_argument("--weight", default="reactance", choices=["reactance", "unweighted"],
                    help="graph weight mode: reactance uses 1/|x|, unweighted uses 1")
    ap.add_argument("--drift_frac", default=0.9, type=float,
                    help="drift if worst_gen_nadir_time_s > drift_frac * t_final (default 0.9)")
    args = ap.parse_args()

    proj = project_root_from_this_file()

    # Locate CSV
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = (proj / csv_path).resolve()
    else:
        search_dir = Path(args.dir) if args.dir else (proj / "outputs" / "aggregates")
        if not search_dir.is_absolute():
            search_dir = (proj / search_dir).resolve()
        csv_path = find_latest_batch_csv(search_dir)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Locate meta
    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.is_absolute():
            meta_path = (proj / meta_path).resolve()
        if not meta_path.exists():
            meta_path = None
    else:
        meta_path = guess_meta_for_csv(csv_path)

    meta = load_meta(meta_path)

    print(f"[info] csv:  {csv_path}")
    print(f"[info] meta: {meta_path if meta_path else '(none)'}")

    # Load batch CSV and select a step
    df = pd.read_csv(csv_path)
    if "success" in df.columns:
        df = df[df["success"] == 1].copy()

    step = float(args.step)
    sub = df[df["step_dPeq_pu"] == step].copy()
    if len(sub) == 0:
        avail = sorted(df["step_dPeq_pu"].unique().tolist())
        raise ValueError(f"No rows for step={step}. Available steps: {avail}")

    t_final = float(sub["t_final_s"].iloc[0])
    drift_thr = float(args.drift_frac) * t_final

    # Vulnerability at generator-level (but we will aggregate to bus-level)
    sub["vuln_nadir"] = sub["obj2_worst_nadir_drop_hz"].astype(float)
    sub["vuln_rocof"] = sub["obj4_worst_rocof_abs_hz_per_s"].astype(float)
    sub["drift_flag"] = (sub["worst_gen_nadir_time_s"].astype(float) > drift_thr).astype(int)

    # Use bus id of worst nadir gen (in this dataset it equals disturbed gen's bus)
    if "worst_gen_nadir_bus_id" not in sub.columns:
        raise ValueError("CSV missing worst_gen_nadir_bus_id column.")
    sub["bus_id"] = sub["worst_gen_nadir_bus_id"].astype(int)

    # Aggregate to bus-level if multiple generators exist on same bus
    bus_v = (
        sub.groupby("bus_id", as_index=False)
        .agg(
            vuln_nadir=("vuln_nadir", "max"),
            vuln_rocof=("vuln_rocof", "max"),
            drift_flag=("drift_flag", "max"),
            n_gens=("bus_id", "size"),
        )
    )

    # Build graph from MATPOWER branch
    case_mfile = None
    if "case_mfile" in meta:
        case_mfile = Path(meta["case_mfile"])
        if not case_mfile.is_absolute():
            case_mfile = (proj / case_mfile).resolve()
    if case_mfile is None or (not case_mfile.exists()):
        # fallback: infer common location
        guess = proj / "KPG193_ver1_2" / "network" / "m" / "KPG193_ver1_2.m"
        if guess.exists():
            case_mfile = guess
        else:
            raise FileNotFoundError("case_mfile not found from meta and fallback path missing.")

    _, branch, _baseMVA = parse_mfile_via_project(case_mfile)
    G = build_bus_graph(branch, weight_mode=args.weight)
    bus_metrics = compute_bus_metrics(G)

    # Join
    join = bus_metrics.merge(bus_v, on="bus_id", how="inner")
    outdir = ensure_dir(csv_path.parent / "analysis_net")
    if meta_path and meta_path.exists():
        (outdir / meta_path.name).write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")

    join_csv = outdir / f"join_bus_metrics_step_{step:g}.csv"
    join.to_csv(join_csv, index=False, encoding="utf-8-sig")
    print(f"[saved] {join_csv}")

    xcols = ["degree", "strength", "betweenness", "closeness", "eigenvector", "pagerank", "kcore"]

    corr_nadir = spearman_corr_table(join, "vuln_nadir", xcols)
    corr_rocof = spearman_corr_table(join, "vuln_rocof", xcols)
    corr_drift = spearman_corr_table(join, "drift_flag", xcols)

    corr_nadir.to_csv(outdir / f"corr_spearman_vuln_nadir_step_{step:g}.csv", index=False, encoding="utf-8-sig")
    corr_rocof.to_csv(outdir / f"corr_spearman_vuln_rocof_step_{step:g}.csv", index=False, encoding="utf-8-sig")
    corr_drift.to_csv(outdir / f"corr_spearman_drift_flag_step_{step:g}.csv", index=False, encoding="utf-8-sig")

    # A few default scatter plots (top 3 by rho for nadir)
    topx = corr_nadir.dropna().head(3)["x"].tolist()
    for x in topx:
        plot_scatter(
            join,
            x=x,
            y="vuln_nadir",
            outpath=outdir / f"scatter_{x}_vs_vuln_nadir_step_{step:g}.png",
            title=f"{x} vs vuln_nadir (step={step:g} pu)",
        )

    # Save drift cases list
    drift_cases = join[join["drift_flag"] == 1].sort_values("vuln_nadir", ascending=False)
    drift_cases.to_csv(outdir / f"drift_buses_step_{step:g}.csv", index=False, encoding="utf-8-sig")

    print(f"[done] outputs in: {outdir}")


if __name__ == "__main__":
    main()
