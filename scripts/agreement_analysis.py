"""Agreement between retrieval test sets, from per-query results.

Input: a CSV with one row per (benchmark, system, query) and the rank of the gold
passage (1..100, 0 = not retrieved in the top 100), as written by the per-query
evaluation cell of kyrgyz-embeddings.ipynb (results_per_query.csv).

Reports, for each metric:
  - mean score per system and test set, with a 95% bootstrap interval over queries
  - Kendall's tau between the system rankings of each pair of test sets, with a bootstrap interval
  - system pairs whose significant order differs between two test sets

Usage:
    python scripts/agreement_analysis.py results_per_query.csv
    python scripts/agreement_analysis.py results_per_query.csv --compare heldout_llm belebele --n-boot 2000
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

METRICS = ("ndcg@10", "acc@1")


def load(path):
    pq = pd.read_csv(path, dtype={"qid": str})
    pq["acc@1"] = (pq["rank"] == 1).astype(float)
    pq["ndcg@10"] = np.where(pq["rank"].between(1, 10), 1 / np.log2(pq["rank"].clip(lower=1) + 1), 0.0)
    return pq


def matrix(pq, bench, metric, style=None):
    """Systems x queries matrix of per-query scores."""
    d = pq[pq.benchmark == bench]
    if style:
        d = d[d["style"] == style]
    return d.pivot(index="system", columns="qid", values=metric)


def boot_means(m, n_boot, rng):
    """Systems x n_boot bootstrap means; every system uses the same query resample."""
    X = m.values
    idx = rng.integers(0, X.shape[1], (n_boot, X.shape[1]))
    return np.stack([X[:, i].mean(1) for i in idx], axis=1)


def summary(pq, metric, n_boot, rng):
    out = {}
    for bench in pq.benchmark.unique():
        m = matrix(pq, bench, metric)
        lo, hi = np.percentile(boot_means(m, n_boot, rng), [2.5, 97.5], axis=1)
        out[bench] = pd.Series([f"{v:.3f} [{l:.3f}, {h:.3f}]" for v, l, h in zip(m.values.mean(1), lo, hi)], index=m.index)
    return pd.DataFrame(out)


def tau(pq, b1, b2, metric, n_boot, rng):
    m1, m2 = matrix(pq, b1, metric), matrix(pq, b2, metric)
    systems = m1.index.intersection(m2.index)
    m1, m2 = m1.loc[systems], m2.loc[systems]
    point = kendalltau(m1.values.mean(1), m2.values.mean(1))[0]
    boots = [kendalltau(a, b)[0] for a, b in zip(boot_means(m1, n_boot, rng).T, boot_means(m2, n_boot, rng).T)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, len(systems)


def significant_order(pq, bench, metric, n_boot, rng):
    """{(A, B): +1 if A > B significantly, -1 if A < B, 0 if not separable}."""
    m = matrix(pq, bench, metric)
    bm = boot_means(m, n_boot, rng)
    names = list(m.index)
    out = {}
    for i, j in itertools.combinations(range(len(names)), 2):
        lo, hi = np.percentile(bm[i] - bm[j], [2.5, 97.5])
        out[(names[i], names[j])] = 1 if lo > 0 else -1 if hi < 0 else 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_query_csv")
    ap.add_argument("--compare", nargs=2, metavar=("SET_A", "SET_B"),
                    help="two test sets for the pairwise-order check (default: the last two found)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pq = load(args.per_query_csv)
    rng = np.random.default_rng(args.seed)
    benches = list(pq.benchmark.unique())
    a, b = args.compare or benches[-2:]

    for metric in METRICS:
        print(f"\n=== {metric}: mean [95% CI] ===")
        print(summary(pq, metric, args.n_boot, rng).to_string())
        print(f"\nKendall's tau between test sets ({metric}):")
        for b1, b2 in itertools.combinations(benches, 2):
            t, lo, hi, n = tau(pq, b1, b2, metric, args.n_boot, rng)
            print(f"  {b1:14s} vs {b2:14s} tau = {t:.2f} [{lo:.2f}, {hi:.2f}]  ({n} systems)")
        oa = significant_order(pq, a, metric, args.n_boot, rng)
        ob = significant_order(pq, b, metric, args.n_boot, rng)
        flips = [p for p in oa if oa[p] * ob[p] == -1]
        differ = [p for p in oa if oa[p] != ob[p]]
        print(f"  {a} vs {b}: {len(oa)} system pairs, {len(flips)} significant opposite orders, {len(differ)} differ in significance")
        for p in flips:
            print(f"    flip: {p[0]} vs {p[1]}")


if __name__ == "__main__":
    main()
