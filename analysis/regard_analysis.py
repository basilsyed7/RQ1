#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize regard scores across models and dimensions."""
from __future__ import annotations

from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_DIR = ROOT / "metrics"
OUT_DIR = ROOT / "analysis" / "outputs"
FIG_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_regard_scores(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(metrics_dir.glob("*.csv")):
        if "_" not in csv_path.stem:
            continue
        model, dimension = csv_path.stem.split("_", 1)
        df = pd.read_csv(csv_path)
        if "regard_score" not in df.columns:
            raise SystemExit(f"Missing regard_score in {csv_path}")
        scores = pd.to_numeric(df["regard_score"], errors="coerce").dropna()
        if scores.empty:
            mean = 0.0
            std = 0.0
            n = 0
        else:
            mean = float(scores.mean())
            std = float(scores.std(ddof=0))
            n = int(scores.count())
        try:
            rel_path = csv_path.resolve().relative_to(ROOT)
        except ValueError:
            rel_path = csv_path

        rows.append({
            "model": model,
            "dimension": dimension,
            "regard_mean": mean,
            "regard_std": std,
            "n": n,
            "path": str(rel_path),
        })
    if not rows:
        raise SystemExit(f"No metric CSVs found in {metrics_dir}")
    return pd.DataFrame(rows)


def best_dimension(per_dim: pd.DataFrame) -> pd.DataFrame:
    idx = per_dim.groupby("model")["regard_mean"].idxmax()
    best = per_dim.loc[idx, ["model", "dimension", "regard_mean", "n"]].copy()
    best.rename(columns={"regard_mean": "best_regard_mean", "n": "sample_count"}, inplace=True)
    return best.sort_values(["best_regard_mean", "model"], ascending=[False, True])


def overall_regard(per_dim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in per_dim.groupby("model"):
        total_n = group["n"].sum()
        weighted = 0.0 if total_n == 0 else float(np.average(group["regard_mean"], weights=group["n"]))
        equal = float(group["regard_mean"].mean()) if not group.empty else 0.0
        std = float(group["regard_mean"].std(ddof=0)) if len(group) > 1 else 0.0
        rows.append({
            "model": model,
            "overall_regard_weighted": weighted,
            "overall_regard_equal": equal,
            "dimension_std": std,
            "dimension_count": int(len(group)),
            "total_samples": int(total_n),
        })
    return pd.DataFrame(rows).sort_values("overall_regard_weighted", ascending=False)


def plot_by_dimension(per_dim: pd.DataFrame, out_path: Path):
    dims = sorted(per_dim["dimension"].unique())
    models = sorted(per_dim["model"].unique())
    x = np.arange(len(dims))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(1.6 * len(dims), 4.5))
    for i, model in enumerate(models):
        vals = []
        for dim in dims:
            subset = per_dim[(per_dim["dimension"] == dim) & (per_dim["model"] == model)]
            vals.append(float(subset["regard_mean"].iloc[0]) if not subset.empty else np.nan)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model)

    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=20)
    ax.set_ylabel("Regard score")
    ax.set_title("Regard by dimension and model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overall(overall: pd.DataFrame, out_path: Path):
    df = overall.sort_values("overall_regard_weighted", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(df)), 4.0))
    bars = ax.bar(df["model"], df["overall_regard_weighted"], color="#4e79a7")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_ylabel("Weighted regard score")
    ax.set_title("Overall regard by model")
    ax.margins(x=0.05)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row['overall_regard_weighted']:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(METRICS_DIR), help="Directory with per-sample metric CSVs")
    ap.add_argument("--out_dir", default=str(OUT_DIR), help="Where to write summary CSVs")
    ap.add_argument("--fig_dir", default=str(FIG_DIR), help="Where to write figures")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_dim = load_regard_scores(metrics_dir)
    per_dim = per_dim.sort_values(["dimension", "model"]).reset_index(drop=True)

    best = best_dimension(per_dim)
    overall = overall_regard(per_dim)

    per_dim_path = out_dir / "regard_by_model_dimension.csv"
    best_path = out_dir / "regard_best_dimension.csv"
    overall_path = out_dir / "regard_overall.csv"
    per_dim.to_csv(per_dim_path, index=False)
    best.to_csv(best_path, index=False)
    overall.to_csv(overall_path, index=False)

    plot_by_dimension(per_dim, fig_dir / "regard_by_dimension.png")
    plot_overall(overall, fig_dir / "regard_overall.png")

    print(f"Wrote {per_dim_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {fig_dir / 'regard_by_dimension.png'}")
    print(f"Wrote {fig_dir / 'regard_overall.png'}")


if __name__ == "__main__":
    main()
