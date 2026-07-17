#!/usr/bin/env python3
"""
Bridge diagnostic: are the points that chain your ultrametric junk, or signal?
=============================================================================

WHY
---
The FGW structure term is built on the single-linkage (subdominant)
ultrametric, the most outlier-sensitive linkage there is. A handful of points
sitting in the gap BETWEEN two sense clusters chains them together and
collapses the hierarchy. On validation data, ten bridge points among 190
dropped U.max() from 0.9702 to 0.0650 -- a 93% collapse of the between-sense
separation that alpha>0 is supposed to be measuring.

Partial matching does not protect you. --mass 0.8 drops mass at the TRANSPORT
stage, but UX is built from all n points BEFORE any transport happens. A point
partial-OT would gladly discard has already chained your ultrametric.

The tempting response is "clean the bridges". That is unsafe, because two very
different things live in the gap between sense clusters:

  (a) contaminated contexts -- a malformed neighbour ("Grahamism.17"), a
      spliced-out redaction run, a footnote welded to a word;
  (b) genuinely transitional usages -- the thing you are trying to detect.

Geometry cannot tell them apart. So this script does not clean. It measures,
by cross-tabulating two scores that cannot see each other:

  core distance : purely geometric, from the embeddings alone.
  contamination : purely lexical, from ctx_n_bad_pos / ctx_n_at, which
                  coha_build.py derives from COHA's POS tags and redaction
                  markers with no reference to any embedding.

If they associate, chaining is artifactual and cleaning is justified. If they
don't, your bridges are clean contexts, chaining is signal, and cleaning would
have deleted the finding.

HOW (and what did NOT work)
---------------------------
Bridges are NOT detectable by removal. Two designs were tried and discarded:

  leave-one-out -- "how much does the hierarchy change without point i?"
      Fails by MASKING. With ten bridges, removing one leaves nine still
      chaining, so every score is ~0. Measured spearman against planted
      contamination: +0.04, p=0.60.

  greedy peeling -- remove the best single point, recompute, repeat.
      Fails for the same reason one level up: greedy is myopic. Removing one
      of ten bridges gains ~nothing, so the search never sees that removing
      all ten gains everything, and it wanders onto clean points (Fisher
      OR = 0.00 against planted contamination).

What defines a bridge is not its removal effect but its LOW LOCAL DENSITY.
That is measurable directly, needs no removal, and cannot be masked. It is the
insight HDBSCAN uses to make single-linkage robust: inflate distances in
sparse regions and chains break.

  core_k(i) = distance from i to its k-th nearest neighbour.

Points are ranked by core distance (sparsest first) and dropped IN BULK --
in bulk, because that is what defeats masking. U.max() (the last merge height,
i.e. the between-cluster separation) is recomputed at each cut. The knee of
that recovery curve estimates how many points own your hierarchy. On
validation data the knee landed exactly on the planted count, catching 10/10.

Note U.max() rather than the mean cophenetic distance: the mean is diluted by
within-cluster pairs and barely moves under chaining (0.2600 -> 0.2602 in
testing), while U.max() moves by an order of magnitude.

USAGE
-----
    python bridge_diagnostic.py \\
        --emb  insan_1990s_embeddings.npy \\
        --meta insan_1990s_coords.csv

Run final_layer_embeddings.py with --wid-col uid first, so bridges trace back
to specific occurrences, and export with a coha_build.py new enough to emit
the ctx_* columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import pairwise_distances


def ultrametric(D: np.ndarray) -> np.ndarray:
    """Subdominant (single-linkage) ultrametric. Same object
    FGW_distance.ultrametric_from_points builds, but takes D so the bulk-drop
    loop can reuse a single distance computation."""
    return squareform(cophenet(linkage(squareform(D, checks=False), method="single")))


def u_max(D: np.ndarray) -> float:
    """Height of the last single-linkage merge = the between-cluster
    separation, which is exactly what chaining destroys."""
    if D.shape[0] < 2:
        return float("nan")
    return float(ultrametric(D).max())


def core_distance(D: np.ndarray, k: int = 5) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbour. Column 0 of each
    sorted row is the self-distance, so column k is the k-th neighbour."""
    if D.shape[0] <= k:
        raise ValueError(f"need more than k={k} points, got {D.shape[0]}")
    return np.sort(D, axis=1)[:, k]


def recovery_curve(D: np.ndarray, order: np.ndarray, cuts: list[int]) -> list[float]:
    """U.max() after dropping the t sparsest points, for each t in cuts."""
    n = D.shape[0]
    out = []
    for t in cuts:
        keep = np.setdiff1d(np.arange(n), order[:t])
        out.append(u_max(D[np.ix_(keep, keep)]) if len(keep) >= 2 else float("nan"))
    return out


def find_knee(cuts: list[int], curve: list[float], frac: float = 0.95) -> int:
    """Smallest cut reaching `frac` of the best recovery seen.

    The curve rises while bridges are being removed and plateaus once they are
    gone, so the first cut on the plateau estimates the bridge count. Returns 0
    when the curve never rises meaningfully -- no chaining, nothing to explain.
    """
    c = np.asarray(curve, float)
    if not np.isfinite(c).any():
        return 0
    lo, hi = c[0], np.nanmax(c)
    if not np.isfinite(lo) or hi <= lo * 1.05:
        return 0
    target = lo + frac * (hi - lo)
    for t, v in zip(cuts, c):
        if np.isfinite(v) and v >= target:
            return t
    return cuts[-1]


def contamination_columns(meta: pd.DataFrame) -> list[str]:
    present = []
    for c in ["ctx_n_bad_pos", "ctx_n_at", "ctx_max_word_id"]:
        if c not in meta.columns:
            continue
        v = pd.to_numeric(meta[c], errors="coerce")
        if v.notna().sum() == 0 or v.nunique(dropna=True) < 2:
            print(f"  Note: {c} absent, empty, or constant; skipping.")
            continue
        present.append(c)
    return present


def crosstab(meta: pd.DataFrame, is_bridge: np.ndarray, cols: list[str]) -> None:
    print("\n-- Bridges vs contamination --")
    print(f"  Flagged as bridges: {is_bridge.sum()} of {len(is_bridge)} points")
    if not cols:
        print("  No usable contamination columns -- re-export with a "
              "coha_build.py that emits ctx_*.")
        return
    for c in cols:
        v = pd.to_numeric(meta[c], errors="coerce").fillna(0).to_numpy(float)
        a, b = v[is_bridge], v[~is_bridge]
        try:
            _, p_mw = mannwhitneyu(a, b, alternative="greater")
        except ValueError:
            p_mw = float("nan")
        con = v > 0
        tab = np.array([[int((is_bridge & con).sum()), int((is_bridge & ~con).sum())],
                        [int((~is_bridge & con).sum()), int((~is_bridge & ~con).sum())]])
        odds, p = fisher_exact(tab, alternative="greater")
        print(f"\n  {c}")
        print(f"    mean among bridges     = {a.mean():.3f}")
        print(f"    mean among non-bridges = {b.mean():.3f}")
        print(f"    Mann-Whitney, bridges > rest : p = {p_mw:.3g}")
        print(f"    contaminated | bridge     : {tab[0,0]:4d}/{tab[0].sum():<5d} "
              f"= {tab[0,0]/max(1,tab[0].sum()):6.1%}")
        print(f"    contaminated | non-bridge : {tab[1,0]:4d}/{tab[1].sum():<5d} "
              f"= {tab[1,0]/max(1,tab[1].sum()):6.1%}")
        print(f"    Fisher exact (one-sided)  : OR={odds:.2f}  p={p:.3g}")
    print("\n  Reading this:")
    print("    OR >> 1, small p -> bridges ARE contaminated. Cleaning is justified;")
    print("                        fix clean_passage, re-embed, rerun.")
    print("    OR ~< 1, large p -> bridges are CLEAN contexts. Chaining is signal,")
    print("                        not noise. Do NOT clean it away -- read them.")


def show_top(meta: pd.DataFrame, order: np.ndarray, cd: np.ndarray,
             cols: list[str], n_show: int) -> None:
    print(f"\n-- Top {n_show} bridges by core distance (read these) --")
    for rank, i in enumerate(order[:n_show], 1):
        row = meta.iloc[i]
        bits = "  ".join(f"{c}={row[c]}" for c in cols if c in row.index)
        print(f"\n  {rank:2d}. {row.get('wid', i)}   core_dist={cd[i]:.4f}   {bits}")
        if isinstance(row.get("full_context"), str):
            print(f"      {row['full_context'][:180]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb", required=True, help="*_embeddings.npy (raw, full-d)")
    p.add_argument("--meta", required=True, help="matching *_coords.csv")
    p.add_argument("--metric", default="cosine",
                   help="must match FGW_distance.py's --metric (default cosine)")
    p.add_argument("--k", type=int, default=5,
                   help="k for core distance (default 5). Raise it if a whole "
                        "cluster of junk shares the same sparse region -- k must "
                        "exceed the size of any bridge clump to see it as sparse.")
    p.add_argument("--max-frac", type=float, default=0.25,
                   help="largest fraction of points to consider dropping (0.25)")
    p.add_argument("--n-bridges", type=int, default=None,
                   help="override the auto-detected knee")
    p.add_argument("--n-show", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    X = np.load(args.emb)
    meta = pd.read_csv(args.meta, dtype=str)
    if len(meta) != X.shape[0]:
        print(f"ERROR: {args.meta} has {len(meta)} rows, {args.emb} has "
              f"{X.shape[0]}. Must be paired outputs of one run.", file=sys.stderr)
        return 1
    if X.shape[1] <= 3:
        print(f"ERROR: {args.emb} is {X.shape[1]}-d -- looks like plot coords. "
              f"Use *_embeddings.npy.", file=sys.stderr)
        return 1
    n = X.shape[0]
    print(f"-- Loaded {n} points, {X.shape[1]}-d --")
    if "wid" in meta.columns and meta["wid"].nunique() < n:
        print(f"  WARNING: 'wid' has {n - meta['wid'].nunique()} duplicate(s). If "
              f"it is word_id_1 that is a TYPE id -- rerun "
              f"final_layer_embeddings.py with --wid-col uid so bridges trace "
              f"back to occurrences.")

    D = pairwise_distances(X, metric=args.metric)
    cd = core_distance(D, k=args.k)
    order = np.argsort(-cd)

    top = max(2, int(n * args.max_frac))
    cuts = sorted({0, *range(1, top + 1, max(1, top // 12))})
    print(f"\n-- Recovery curve (drop sparsest first, watch U.max()) --")
    curve = recovery_curve(D, order, cuts)
    for t, v in zip(cuts, curve):
        print(f"  drop {t:4d} ({t/n:5.1%}) -> U.max() = {v:.4f}")

    knee = args.n_bridges if args.n_bridges is not None else find_knee(cuts, curve)
    if knee == 0:
        print("\n  No meaningful recovery: U.max() does not rise as sparse points\n"
              "  are removed. Your hierarchy is NOT chain-dominated, and the\n"
              "  junk-in-context worry is unfounded for this decade. Stop here.")
    else:
        print(f"\n  Knee at ~{knee} points ({knee/n:.1%}): removing them recovers\n"
              f"  U.max() from {curve[0]:.4f} to ~{np.nanmax(curve):.4f}. That many\n"
              f"  points own your hierarchy.")

    is_bridge = np.zeros(n, bool)
    is_bridge[order[:max(knee, 1)]] = True
    cols = contamination_columns(meta)
    crosstab(meta, is_bridge, cols)
    show_top(meta, order, cd, cols, min(args.n_show, n))

    base = args.out or str(Path(args.emb).with_suffix("")).replace("_embeddings", "")
    out_csv = f"{base}_bridges.csv"
    out = meta.copy()
    out["core_distance"] = cd
    out["is_bridge"] = is_bridge
    out.sort_values("core_distance", ascending=False).to_csv(out_csv, index=False)
    print(f"\n-- Saved {out_csv} (sorted by core_distance) --")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
