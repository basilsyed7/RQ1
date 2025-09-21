#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregates per-attribute metrics, computes bootstrap 95% CIs, and
runs quick tests:
  - Gender: paired (male vs female) by (prompt_id, name_index)
  - Others: attribute vs pooled others (unpaired, permutation p)

Summaries now include: bias_score, favorability, sentiment, hedge_score,
refusal, word_count.

Outputs CSVs to analysis/outputs/.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json, os

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "metrics"
OUT_DIR = ROOT / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(7)

def bootstrap_ci_mean(x, B=2000, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    means = []
    n = len(x)
    for _ in range(B):
        sample = x[RNG.integers(0,n,size=n)]
        means.append(np.mean(sample))
    means = np.sort(means)
    lo = np.percentile(means, 100*alpha/2)
    hi = np.percentile(means, 100*(1-alpha/2))
    return (float(np.mean(x)), float(lo), float(hi))

def permutation_pvalue(a, b, B=5000, two_sided=True):
    # Mean difference
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    obs = np.mean(a) - np.mean(b)
    concat = np.concatenate([a, b])
    n_a = len(a)
    if len(b)==0 or len(a)==0:
        return np.nan
    count = 0
    for _ in range(B):
        RNG.shuffle(concat)
        a_s = concat[:n_a]; b_s = concat[n_a:]
        stat = np.mean(a_s) - np.mean(b_s)
        if two_sided:
            if abs(stat) >= abs(obs): count += 1
        else:
            if stat >= obs: count += 1
    return (count + 1) / (B + 1)

def summarize_by_attribute(df, metric_col="bias_score"):
    """
    Returns per-attribute mean and CI for a model×dimension.
    Assumes df has 'attribute_value'.
    """
    rows = []
    for attr, group in df.groupby("attribute_value"):
        m, lo, hi = bootstrap_ci_mean(group[metric_col].values)
        rows.append({"attribute_value":attr, "mean":m, "ci_lo":lo, "ci_hi":hi, "n":len(group)})
    return pd.DataFrame(rows).sort_values("attribute_value")

def analyze_gender(df, metric_col="bias_score"):
    # Pair male vs female by (prompt_id, name_index)
    male = df[df["attribute_value"].str.lower()=="male"]
    female = df[df["attribute_value"].str.lower()=="female"]
    key = ["prompt_id","name_index"]
    merged = pd.merge(male[key+[metric_col]], female[key+[metric_col]], on=key, suffixes=("_male","_female"))
    if merged.empty:
        return np.nan, np.nan, 0
    delta = merged[f"{metric_col}_male"].values - merged[f"{metric_col}_female"].values
    # permutation sign test (approx)
    # center by swapping signs at random
    obs = np.mean(delta)
    B=5000; count=0
    for _ in range(B):
        signs = RNG.choice([-1,1], size=delta.shape[0])
        stat = np.mean(delta*signs)
        if abs(stat) >= abs(obs): count += 1
    p = (count+1)/(B+1)
    return float(obs), float(np.mean(delta)), p

def analyze_one_vs_rest(df, metric_col="bias_score"):
    """
    For non-gender dims: attribute vs pooled others
    Returns a dataframe with attr, mean_attr, mean_rest, delta, p
    """
    out = []
    for attr, g in df.groupby("attribute_value"):
        rest = df[df["attribute_value"] != attr]
        m_attr = g[metric_col].mean()
        m_rest = rest[metric_col].mean() if len(rest) else np.nan
        p = permutation_pvalue(g[metric_col].values, rest[metric_col].values, two_sided=True)
        out.append({"attribute_value":attr, "mean_attr":m_attr, "mean_rest":m_rest, "delta": m_attr - m_rest, "p_value":p, "n_attr":len(g), "n_rest":len(rest)})
    return pd.DataFrame(out).sort_values("attribute_value")

def main():
    metric_files = sorted(METRICS_DIR.glob("*.csv"))
    if not metric_files:
        print("No metric CSVs found. Run score_metrics.py first.")
        return

    all_df = []
    for mf in metric_files:
        model, dimension = mf.stem.split("_", 1)
        df = pd.read_csv(mf)
        df["model"] = model
        df["dimension"] = dimension
        all_df.append(df)
    data = pd.concat(all_df, ignore_index=True)

    # summaries per model × dimension
    summaries = []
    tests_bias = []   # keep original bias_score-only outputs for compatibility
    tests_ext  = []   # extended tests across multiple metrics

    for (model, dim), g in data.groupby(["model","dimension"]):
        # basic per-attribute summaries (extended)
        for metric_col in ["bias_score","favorability","sentiment","hedge_score","refusal","word_count"]:
            summ = summarize_by_attribute(g, metric_col=metric_col)
            summ.insert(0, "metric", metric_col)
            summ.insert(0, "dimension", dim)
            summ.insert(0, "model", model)
            summaries.append(summ)

        # tests
        if dim == "gender":
            # bias_score (compat)
            _, mean_delta, p = analyze_gender(g, metric_col="bias_score")
            tests_bias.append(pd.DataFrame([{
                "model":model, "dimension":dim, "metric":"bias_score", "test":"paired male-female",
                "delta_mean": mean_delta, "p_value": p, "n_pairs": int(len(g)/2)
            }]))

            # extended: favorability and refusal
            for metric_col in ["favorability","refusal"]:
                _, mean_delta, p = analyze_gender(g, metric_col=metric_col)
                tests_ext.append(pd.DataFrame([{
                    "model":model, "dimension":dim, "metric":metric_col, "test":"paired male-female",
                    "delta_mean": mean_delta, "p_value": p, "n_pairs": int(len(g)/2)
                }]))
        else:
            # bias_score (compat)
            tdf_bias = analyze_one_vs_rest(g, metric_col="bias_score")
            tdf_bias.insert(0, "dimension", dim)
            tdf_bias.insert(0, "model", model)
            tdf_bias.insert(0, "metric", "bias_score")
            tdf_bias.insert(0, "test", "one-vs-rest")
            tests_bias.append(tdf_bias)

            # extended: favorability and refusal
            for metric_col in ["favorability","refusal"]:
                tdf = analyze_one_vs_rest(g, metric_col=metric_col)
                tdf.insert(0, "dimension", dim)
                tdf.insert(0, "model", model)
                tdf.insert(0, "metric", metric_col)
                tdf.insert(0, "test", "one-vs-rest")
                tests_ext.append(tdf)

    summary_df = pd.concat(summaries, ignore_index=True)
    tests_bias_df = pd.concat(tests_bias, ignore_index=True) if tests_bias else pd.DataFrame()
    tests_ext_df  = pd.concat(tests_ext,  ignore_index=True) if tests_ext  else pd.DataFrame()

    summary_path = OUT_DIR / "summary_per_attribute.csv"
    tests_path   = OUT_DIR / "tests_bias_score.csv"
    tests_ext_path = OUT_DIR / "tests_all_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    if not tests_bias_df.empty:
        tests_bias_df.to_csv(tests_path, index=False)
    if not tests_ext_df.empty:
        tests_ext_df.to_csv(tests_ext_path, index=False)
    print(f"Wrote {summary_path}")
    if not tests_bias_df.empty:
        print(f"Wrote {tests_path}")
    if not tests_ext_df.empty:
        print(f"Wrote {tests_ext_path}")

if __name__ == "__main__":
    main()
