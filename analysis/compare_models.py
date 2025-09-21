#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank models by fairness-like metrics per dimension.

Inputs (from analysis/analyze_rq1.py):
- analysis/outputs/tests_bias_score.csv         (required for bias_score)
- analysis/outputs/tests_all_metrics.csv        (optional; for favorability/refusal)

Outputs:
- analysis/outputs/model_ranking.csv  (all models ranked per dimension×metric)
- analysis/outputs/best_models.csv    (best per dimension×metric)

Fairness definition (lower is better):
- Gender (paired tests): fairness = abs(delta_mean)
- Other dimensions (one-vs-rest per attribute):
    fairness = weighted mean of |delta| over attributes, weights = n_attr

Also reports the number of statistically significant differences (p<0.05) per
model×dimension×metric as a cautionary indicator.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "analysis" / "outputs"


def load_tests():
    bias_path = OUT_DIR / "tests_bias_score.csv"
    ext_path  = OUT_DIR / "tests_all_metrics.csv"
    if not bias_path.exists():
        raise SystemExit("Missing tests_bias_score.csv. Run analysis/analyze_rq1.py first.")
    t_bias = pd.read_csv(bias_path)
    t_ext = pd.read_csv(ext_path) if ext_path.exists() else pd.DataFrame()
    return t_bias, t_ext


def rank_gender(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # df columns: model, dimension=gender, metric, delta_mean, p_value, n_pairs
    rows = []
    for model, g in df.groupby("model"):
        # Should be a single row per model×metric
        dm = float(g["delta_mean"].iloc[0])
        pv = float(g["p_value"].iloc[0])
        fairness = abs(dm)
        rows.append({
            "dimension": "gender",
            "metric": metric,
            "model": model,
            "fairness_score": fairness,
            "significant_diffs": int(pv < 0.05),
        })
    out = pd.DataFrame(rows).sort_values(["fairness_score", "significant_diffs", "model"]) 
    out["rank"] = out.groupby(["dimension", "metric"]).cumcount() + 1
    return out


def rank_non_gender(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # df columns: model, dimension, attribute_value, delta, p_value, n_attr, n_rest
    rows = []
    for (model, dim), g in df.groupby(["model", "dimension"]):
        # Weighted mean absolute delta across attributes
        # Some files may not carry n_attr; default to 1
        if "n_attr" in g.columns:
            w = g["n_attr"].fillna(1).astype(float).values
        else:
            w = np.ones(len(g), dtype=float)
        fairness = np.average(np.abs(g["delta"].astype(float).values), weights=w)
        sig = int((g["p_value"].astype(float) < 0.05).sum())
        rows.append({
            "dimension": dim,
            "metric": metric,
            "model": model,
            "fairness_score": float(fairness),
            "significant_diffs": sig,
        })
    out = pd.DataFrame(rows).sort_values(["dimension", "fairness_score", "significant_diffs", "model"]) 
    out["rank"] = out.groupby(["dimension", "metric"]).cumcount() + 1
    return out


def parse_metric_weights(arg_list):
    weights = {}
    for item in arg_list or []:
        if ":" in item:
            k, v = item.split(":", 1)
            try:
                weights[k.strip()] = float(v)
            except ValueError:
                pass
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="*", default=["bias_score", "favorability", "refusal"],
                    help="Metrics to rank: bias_score, favorability, refusal")
    ap.add_argument("--aggregate", action="store_true", help="Also compute overall ranking across all dimensions/metrics")
    ap.add_argument("--metric-weights", nargs="*", metavar="METRIC:W", help="Optional weights per metric for aggregation, e.g., bias_score:1 favorability:1 refusal:0.5")
    ap.add_argument("--sig-penalty", type=float, default=0.0, help="Penalty added per significant difference when aggregating (e.g., 0.05)")
    args = ap.parse_args()

    t_bias, t_ext = load_tests()

    rankings = []

    # bias_score: gender from tests_bias; non-gender from tests_bias
    if "bias_score" in args.metrics:
        # Gender
        g_bias = t_bias[(t_bias["dimension"] == "gender")]
        if not g_bias.empty:
            # ensure column 'metric' exists for concat compatibility
            g_bias = g_bias.copy()
            g_bias["metric"] = "bias_score"
            rankings.append(rank_gender(g_bias[["model","delta_mean","p_value"]].assign(dimension="gender", metric="bias_score"), "bias_score"))
        # Non-gender
        ng_bias = t_bias[(t_bias["dimension"] != "gender")]
        if not ng_bias.empty:
            ng_bias = ng_bias.copy()
            ng_bias["metric"] = "bias_score"
            rankings.append(rank_non_gender(ng_bias, "bias_score"))

    # favorability/refusal from tests_all_metrics
    for met in (m for m in args.metrics if m in {"favorability", "refusal"}):
        if t_ext.empty:
            continue
        # Gender
        g = t_ext[(t_ext["dimension"] == "gender") & (t_ext["metric"] == met)]
        if not g.empty:
            rankings.append(rank_gender(g[["model","delta_mean","p_value"]].assign(dimension="gender", metric=met), met))
        # Non-gender
        ng = t_ext[(t_ext["dimension"] != "gender") & (t_ext["metric"] == met)]
        if not ng.empty:
            rankings.append(rank_non_gender(ng, met))

    if not rankings:
        raise SystemExit("No rankings computed. Ensure tests CSVs exist and contain requested metrics.")

    rank_df = pd.concat(rankings, ignore_index=True)
    rank_out = OUT_DIR / "model_ranking.csv"
    rank_df.to_csv(rank_out, index=False)

    # Best per dimension×metric
    best = rank_df.sort_values(["dimension","metric","rank"]).groupby(["dimension","metric"], as_index=False).first()
    best_out = OUT_DIR / "best_models.csv"
    best.to_csv(best_out, index=False)

    print(f"Wrote {rank_out}")
    print(f"Wrote {best_out}")

    # Optional: overall aggregation across dimensions and metrics
    if args.aggregate:
        mw = parse_metric_weights(args.metric_weights)
        # default weight 1.0 for listed metrics
        def w(row):
            return mw.get(row["metric"], 1.0)

        tmp = rank_df.copy()
        tmp["weight"] = tmp.apply(w, axis=1)
        # Composite score = weighted mean fairness + penalty * total significant_diffs
        agg = tmp.groupby("model").apply(
            lambda g: pd.Series({
                "composite_fairness": float(np.average(g["fairness_score"].values, weights=g["weight"].values if (g["weight"].values>0).any() else None)),
                "total_significant_diffs": int(g["significant_diffs"].sum()),
                "n_cells": int(len(g)),
            })
        ).reset_index()
        agg["overall_score"] = agg["composite_fairness"] + args.sig_penalty * agg["total_significant_diffs"]
        agg = agg.sort_values(["overall_score","composite_fairness","total_significant_diffs","model"]).reset_index(drop=True)
        agg["rank"] = np.arange(1, len(agg) + 1)
        overall_out = OUT_DIR / "overall_ranking.csv"
        agg.to_csv(overall_out, index=False)
        print(f"Wrote {overall_out}")


if __name__ == "__main__":
    main()
