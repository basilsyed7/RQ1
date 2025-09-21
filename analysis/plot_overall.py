#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick bar chart of overall model ranking from analysis/outputs/overall_ranking.csv

Usage:
  python3 analysis/plot_overall.py
  python3 analysis/plot_overall.py --csv analysis/outputs/overall_ranking.csv --out analysis/figures/overall_ranking.png
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "analysis" / "outputs" / "overall_ranking.csv"
OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_overall(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    # If file has no header, set expected columns
    if list(df.columns)[0].lower() != "model":
        df.columns = [
            "model",
            "composite_fairness",
            "total_significant_diffs",
            "n_cells",
            "overall_score",
            "rank",
        ]
    return df


def plot_overall(df: pd.DataFrame, out_path: Path):
    df = df.sort_values(["overall_score", "model"]).reset_index(drop=True)
    models = df["model"].tolist()
    vals = df["overall_score"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(models)), 4.5))
    bars = ax.bar(models, vals, color="#4e79a7")
    ax.set_ylabel("overall_score (lower is better)")
    ax.set_title("Overall Model Fairness Ranking")
    ax.margins(x=0.05)
    # annotate with composite_fairness and sig diffs
    if {"composite_fairness", "total_significant_diffs"}.issubset(df.columns):
        for b, (_, row) in zip(bars, df.iterrows()):
            label = f"cf={row['composite_fairness']:.3f}\nsig={int(row['total_significant_diffs'])}"
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), label,
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(OUT_DIR / "overall_ranking.png"))
    args = ap.parse_args()

    df = load_overall(Path(args.csv))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_overall(df, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

