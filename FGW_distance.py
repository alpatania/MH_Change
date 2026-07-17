"""
Partial Fused Gromov-Wasserstein semantic shift between two corpora
====================================================================
Loads precomputed embeddings from .npy files (use the *_embeddings.npy
full-dimensional matrices saved by final_layer_embeddings.py, NOT the
2D plot coordinates).

The objective interpolates between two signals via --alpha:

  alpha = 0.0 : pure feature cost   -> "did occurrences move in RoBERTa space?"
                (plain partial Wasserstein on cross-corpus cosine distances)
  alpha = 1.0 : pure structure cost -> "did the sense hierarchy rearrange?"
                (partial GW between single-linkage ultrametrics)
  0 < alpha < 1 : fused (default 0.5)

A plain partial-OT baseline (feature only) and pure partial-GW score
(structure only) are always reported alongside the fused score, so you
can decompose positional vs. structural shift.

Structure input options:
  Default: --struct1/--struct2 are unset, so the within-corpus
  ultrametrics UX, UY are built from the same raw *_embeddings.npy
  arrays used for the cross-corpus feature cost M. This is the
  original (and safest) mode.

  Path B: pass --struct1 <corpus1>_pca90_umap.npy --struct2 <corpus2>_pca90_umap.npy
  to build the ultrametrics from the UMAP-of-PCA-90 outputs instead.
  Motivation: cosine distances in raw 768-d BERT space suffer from
  concentration-of-measure, which can wash out the local neighbourhood
  structure that single-linkage clustering depends on. UMAP explicitly
  preserves local neighbourhoods. Note that per-decade UMAP fits live
  in DIFFERENT manifold warpings, so struct inputs are ONLY valid for
  the ultrametrics (each computed independently within one corpus); M
  still MUST come from the raw shared-space embeddings, and the code
  enforces this by using --emb1/--emb2 for M unconditionally. Also
  consider setting --struct-metric euclidean when passing UMAP outputs,
  since UMAP's low-d embedding is optimized for Euclidean geometry.

Usage:
    uv run FGW_distance.py --emb1 c1_embeddings.npy --emb2 c2_embeddings.npy
    uv run FGW_distance.py --emb1 ... --emb2 ... --meta1 c1_coords.csv --meta2 c2_coords.csv
    uv run FGW_distance.py --emb1 ... --emb2 ... --alpha 0.7 --mass 0.6
    uv run FGW_distance.py --emb1 c1_embeddings.npy --emb2 c2_embeddings.npy \\
        --struct1 c1_pca90_umap.npy --struct2 c2_pca90_umap.npy \\
        --struct-metric euclidean

Requirements:
    uv add scikit-learn scipy POT matplotlib numpy pandas
"""

import argparse

import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances


# -- ULTRAMETRIC ----------------------------------------------------------------

def ultrametric_from_points(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Subdominant (single-linkage) ultrametric: U[i,j] = max edge weight on the
    MST path between i and j == cophenetic distance of single-linkage clustering.

    Replaces the previous hand-rolled MST DFS, which treated scipy's one-sided
    MST matrix as directed and left most entries at 0.
    """
    D = pairwise_distances(X, metric=metric)
    Z = linkage(squareform(D, checks=False), method="single")
    return squareform(cophenet(Z))


def load_or_build_ultrametric(source_file, X, metric="cosine", rebuild=False):
    """
    Load a cached ultrametric if available; otherwise compute and cache it.

    The ultrametric is a pure function of (X, metric), and X is fully
    determined by source_file, so caching on (stem, metric) is sound -- as
    long as source_file itself hasn't changed underneath the cache. Two
    guards enforce that, because a stale cache is silent and poisons every
    downstream number:

      1. mtime: if source_file is newer than the cache, rebuild. This is the
         one that matters when you regenerate *_embeddings.npy after changing
         the upstream cleaning -- the filename does not change, so without
         this check you would keep scoring the OLD point cloud's hierarchy
         against the NEW feature cost, and nothing would warn you.
      2. shape: the cache must be (n, n) for n = len(X). Catches a source
         file whose row count changed but whose mtime was preserved (copies,
         restores, checkouts).

    Pass rebuild=True (CLI: --rebuild-ultrametrics) to ignore the cache
    entirely and overwrite it.

    Parameters
    ----------
    source_file : str or Path
        The embedding/structure file associated with X. The ultrametric cache
        is stored alongside it.
    X : np.ndarray
        Point cloud.
    metric : str
        Distance metric passed to ultrametric_from_points().
    rebuild : bool
        Force recomputation even if a cache exists.
    """
    source_file = Path(source_file)

    cache_file = source_file.with_name(
        f"{source_file.stem}_ultrametric_{metric}.npy"
    )

    if cache_file.exists() and not rebuild:
        reason = None
        if cache_file.stat().st_mtime < source_file.stat().st_mtime:
            reason = f"{source_file.name} is newer than the cache"
        else:
            U = np.load(cache_file)
            expected = (X.shape[0], X.shape[0])
            if U.shape != expected:
                reason = f"cached shape {U.shape} != expected {expected}"
            else:
                print(f"  Loading cached ultrametric: {cache_file.name}")
                return U
        print(f"  Stale ultrametric cache ({reason}); rebuilding "
              f"{cache_file.name}")

    print(f"  Building ultrametric: {cache_file.name}  (n={X.shape[0]}, "
          f"metric={metric})")
    U = ultrametric_from_points(X, metric=metric)

    np.save(cache_file, U)

    return U

# -- FGW COMPUTATION -------------------------------------------------------------

def compute_fgw(
    X: np.ndarray,
    Y: np.ndarray,
    emb1_path: str,
    emb2_path: str,
    mass_fraction: float = 0.8,
    alpha: float = 0.5,
    metric: str = "cosine",
    normalize_scale: bool = False,
    nb_dummies: int = 1,
    num_iter_max: int = 50000,
    X_struct: np.ndarray | None = None,
    Y_struct: np.ndarray | None = None,
    struct1_path: str | None = None,
    struct2_path: str | None = None,
    struct_metric: str | None = None,
    diagnostics: bool = False,
    rebuild_ultrametrics: bool = False,
) -> dict:
    """
    Partial fused Gromov-Wasserstein between two embedding matrices that live
    in the SAME feature space (e.g. both RoBERTa final layer).

    Args:
        X, Y            : embeddings, shapes (n, d) and (m, d) with equal d.
                          Used for the cross-corpus feature cost M AND (by
                          default) for the within-corpus ultrametrics.
        emb1_path,      : paths X and Y were loaded from. Used ONLY to site
        emb2_path         and name the ultrametric cache next to its source.
                          Passed explicitly rather than read off a global
                          `args`, so compute_fgw stays importable and testable
                          without the CLI.
        struct1_path,   : as above, for the struct arrays. Required when
        struct2_path      X_struct/Y_struct are given.
        diagnostics     : also compute the feature-only and structure-only
                          baselines. The structure-only (partial GW) baseline
                          is the expensive one -- benchmarked at ~100x the
                          fused solve, since pure GW gives the conditional
                          gradient no feature term to descend. The
                          feature-only baseline is ~free and always runs.
        rebuild_ultrametrics : ignore any cached ultrametric and recompute.
        mass_fraction   : fraction of total mass to transport (0-1);
                          lower = more permissive partial matching
        alpha           : structure/feature trade-off (0 = feature only,
                          1 = structure only)
        metric          : metric for the cross-corpus feature cost M, and
                          for the ultrametrics when struct inputs are unset
        normalize_scale : if True, divide each ultrametric by its own max
                          (erases overall-spread differences between corpora;
                          off by default so dispersion change counts as shift)
        nb_dummies      : number of reservoir points added by POT's partial
                          OT solvers to avoid EMD numerical instabilities;
                          POT's own default is 1, and its docs recommend
                          raising this for large point counts (raise this
                          first if you hit "Error in the EMD resolution")
        num_iter_max    : max iterations for the underlying EMD/CG solvers;
                          POT's own default is 10000
        X_struct,       : optional per-corpus arrays used to build the
        Y_struct          ultrametrics instead of X, Y. Each must have the
                          same number of rows as X, Y respectively (same
                          contexts, different feature space). If either is
                          None, both fall back to X, Y (original behaviour).
                          Path B use case: pass *_pca90_umap.npy arrays.
        struct_metric   : metric for the ultrametrics. Defaults to `metric`
                          when unset. When passing UMAP outputs as struct
                          inputs, consider 'euclidean' -- UMAP's low-d
                          embedding is optimized for Euclidean geometry, and
                          cosine on UMAP output is not what UMAP preserves.

    Returns dict with transport matrix, decomposed scores, matches, and
    per-row transported mass.
    """
    import ot

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Feature dimensions differ ({X.shape[1]} vs {Y.shape[1]}). "
            "The fused feature cost needs a shared space - pass the raw "
            "*_embeddings.npy files, not per-corpus PCA outputs."
        )

    # Resolve struct inputs. Either both are supplied or neither; a single
    # side alone would give the two ultrametrics inconsistent semantics.
    use_struct = X_struct is not None and Y_struct is not None
    if (X_struct is None) != (Y_struct is None):
        raise ValueError(
            "--struct1 and --struct2 must be supplied together or not at all."
        )
    if use_struct:
        if X_struct.shape[0] != X.shape[0]:
            raise ValueError(
                f"struct1 has {X_struct.shape[0]} rows but emb1 has "
                f"{X.shape[0]}; they must describe the same contexts."
            )
        if Y_struct.shape[0] != Y.shape[0]:
            raise ValueError(
                f"struct2 has {Y_struct.shape[0]} rows but emb2 has "
                f"{Y.shape[0]}; they must describe the same contexts."
            )

    ult_metric = struct_metric if struct_metric is not None else metric

    n, m = X.shape[0], Y.shape[0]
    p = np.ones(n) / n
    q = np.ones(m) / m

    if use_struct:
        if struct1_path is None or struct2_path is None:
            raise ValueError(
                "struct1_path and struct2_path are required when X_struct/"
                "Y_struct are supplied -- they site the ultrametric cache."
            )
        print(f"  Corpus 1 ultrametric (from struct, {X_struct.shape}, "
              f"metric={ult_metric})")
        UX = load_or_build_ultrametric(struct1_path, X_struct, ult_metric,
                                       rebuild=rebuild_ultrametrics)
        print(f"  Corpus 2 ultrametric (from struct, {Y_struct.shape}, "
              f"metric={ult_metric})")
        UY = load_or_build_ultrametric(struct2_path, Y_struct, ult_metric,
                                       rebuild=rebuild_ultrametrics)
    else:
        print(f"  Corpus 1 ultrametric (from emb, metric={ult_metric})")
        UX = load_or_build_ultrametric(emb1_path, X, ult_metric,
                                       rebuild=rebuild_ultrametrics)
        print(f"  Corpus 2 ultrametric (from emb, metric={ult_metric})")
        UY = load_or_build_ultrametric(emb2_path, Y, ult_metric,
                                       rebuild=rebuild_ultrametrics)

    if normalize_scale:
        UX = UX / UX.max()
        UY = UY / UY.max()
    else:
        # normalize both by a COMMON factor: keeps relative spread, tames scale
        c = max(UX.max(), UY.max())
        UX, UY = UX / c, UY / c

    # cross-corpus feature cost (meaningful because both are RoBERTa vectors)
    M = pairwise_distances(X, Y, metric=metric)
    M = M / M.max()

    print(f"  Partial FGW (mass={mass_fraction}, alpha={alpha}, "
          f"nb_dummies={nb_dummies}, numItermax={num_iter_max})...")
    T, log = ot.gromov.partial_fused_gromov_wasserstein(
        M, UX, UY, p, q,
        m=mass_fraction,
        alpha=alpha,
        loss_fun="square_loss",
        nb_dummies=nb_dummies,
        numItermax=num_iter_max,
        log=True,
    )
    fgw_cost = float(log["partial_fgw_dist"])

    # Feature-only baseline is ~free (benchmarked at ~0.2% of the fused solve
    # at n=m=400), so it always runs -- half the decomposition for nothing.
    print("  Baseline: partial Wasserstein (feature only)...")
    _, log_w = ot.partial.partial_wasserstein(
        p, q, M, m=mass_fraction, nb_dummies=nb_dummies,
        numItermax=num_iter_max, log=True,
    )
    feat_cost = float(log_w["partial_w_dist"])

    if diagnostics:
        # Structure-only baseline is the expensive half: ~100x the fused solve
        # at n=m=400, because pure GW leaves the conditional gradient with no
        # feature term to descend. Gated behind --diagnostics for that reason.
        print("  Baseline: partial GW (structure only) -- this is the slow one...")
        _, log_gw = ot.gromov.partial_gromov_wasserstein(
            UX, UY, p, q, m=mass_fraction, loss_fun="square_loss",
            nb_dummies=nb_dummies, numItermax=num_iter_max, log=True,
        )
        struct_cost = float(log_gw["partial_gw_dist"])
    else:
        struct_cost = np.nan

    # matches: mask rows that transport (almost) no mass under partial matching
    row_mass = T.sum(axis=1)
    active = row_mass > (1.0 / n) * 0.1          # <10% of a uniform row's mass
    matches = np.where(active, np.argmax(T, axis=1), -1)

    return {
        "transport_matrix": T,
        "matches": matches,          # -1 => unmatched (dropped mass)
        "row_mass": row_mass,
        "fgw_cost": fgw_cost,        # the actual solver objective
        "feature_cost": feat_cost,   # positional shift in shared space
        "structure_cost": struct_cost,  # hierarchy rearrangement
        "UX": UX,
        "UY": UY,
        "M": M,
    }


# -- VISUALISATION ----------------------------------------------------------------

def plot_transport(T: np.ndarray, row_mass: np.ndarray,
                   output_path: str = "fgw_transport.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(T, aspect="auto", cmap="viridis")
    axes[0].set_title("Partial FGW transport matrix")
    axes[0].set_xlabel("Corpus 2 contexts")
    axes[0].set_ylabel("Corpus 1 contexts")
    plt.colorbar(im, ax=axes[0])

    n = T.shape[0]
    axes[1].bar(range(n), row_mass, width=1.0)
    axes[1].axhline(1.0 / n, color="crimson", ls="--", lw=1,
                    label="full match (1/n)")
    axes[1].set_title("Transported mass per corpus-1 context\n"
                      "(low bars = contexts with no counterpart)")
    axes[1].set_xlabel("Context index (corpus 1)")
    axes[1].set_ylabel("Transported mass")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot to {output_path}")


# -- METADATA (optional, for readable matches) -------------------------------------

def load_labels(meta_path: str | None, n_expected: int) -> list[str]:
    if meta_path is None:
        return [str(i) for i in range(n_expected)]
    import pandas as pd
    meta = pd.read_csv(meta_path, dtype=str)
    if len(meta) != n_expected:
        print(f"  WARNING: {meta_path} has {len(meta)} rows, embeddings have "
              f"{n_expected}; falling back to indices.")
        return [str(i) for i in range(n_expected)]
    col = "wid" if "wid" in meta.columns else meta.columns[0]
    return meta[col].astype(str).tolist()


# -- CLI ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Partial FGW semantic shift between two corpora")
    parser.add_argument("--emb1", required=True,
                        help="Corpus 1 raw embeddings (.npy, n x d)")
    parser.add_argument("--emb2", required=True,
                        help="Corpus 2 raw embeddings (.npy, m x d)")
    parser.add_argument("--struct1", default=None,
                        help="Optional Corpus 1 struct array (.npy) used for "
                             "the ultrametric UX instead of --emb1. Typically "
                             "a *_pca90_umap.npy. Row count must match --emb1. "
                             "Feature cost M still uses --emb1/--emb2.")
    parser.add_argument("--struct2", default=None,
                        help="Optional Corpus 2 struct array (.npy) used for "
                             "the ultrametric UY instead of --emb2. Must be "
                             "supplied together with --struct1.")
    parser.add_argument("--struct-metric", default=None,
                        help="Metric for the ultrametrics. Defaults to "
                             "--metric when unset. When using UMAP outputs as "
                             "struct inputs, consider 'euclidean'.")
    parser.add_argument("--meta1", default=None,
                        help="Optional *_coords.csv from step 1 (corpus 1) for wid labels")
    parser.add_argument("--meta2", default=None,
                        help="Optional *_coords.csv from step 1 (corpus 2) for wid labels")
    parser.add_argument("--mass", type=float, default=0.8,
                        help="Mass fraction for partial matching (default 0.8)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Structure weight: 0=feature only, 1=structure only (default 0.5)")
    parser.add_argument("--metric", default="cosine",
                        help="Metric for feature cost and ultrametrics (default cosine)")
    parser.add_argument("--normalize-scale", action="store_true",
                        help="Normalize each ultrametric by its own max "
                             "(default: common factor, so spread change counts)")
    parser.add_argument("--nb-dummies", type=int, default=1,
                        help="Reservoir points added by POT's partial OT solvers "
                             "to avoid EMD instabilities (POT default: 1). POT's "
                             "own docs suggest raising this first, but in testing "
                             "against a 448x768-point problem it had no effect up "
                             "to 20 -- --num-iter-max was what actually mattered.")
    parser.add_argument("--num-iter-max", type=int, default=50000,
                        help="Max iterations for the underlying EMD/CG solvers "
                             "(POT's own default is 10000; raised here because "
                             "testing showed 10000 fails on corpora of a few "
                             "hundred points with 'Error in the EMD resolution', "
                             "while 50000 converged reproducibly). A higher cap "
                             "costs nothing on small problems -- the solver still "
                             "exits early once converged.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Also compute the structure-only (partial GW) "
                             "baseline. Benchmarked at ~100x the fused solve "
                             "(12.8s vs 0.11s at n=m=400), so it is off by "
                             "default -- this flag, not the ultrametric cache, "
                             "is what governs runtime. The feature-only "
                             "(partial Wasserstein) baseline is ~free and "
                             "always runs regardless.")
    parser.add_argument("--rebuild-ultrametrics", action="store_true",
                        help="Ignore any cached *_ultrametric_<metric>.npy and "
                             "recompute. Caches are auto-invalidated when the "
                             "source .npy is newer or its row count changed, "
                             "so this is only needed if you have edited "
                             "ultrametric_from_points itself.")
    parser.add_argument("--output", default="fgw_transport.png")
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'
    base_path = output_path[:-len('.png')]

    print("\n-- Loading embeddings --")
    X = np.load(args.emb1)
    Y = np.load(args.emb2)
    print(f"  Corpus 1: {X.shape}")
    print(f"  Corpus 2: {Y.shape}")
    if X.shape[1] <= 3 or Y.shape[1] <= 3:
        print("  WARNING: these look like 2D/3D plot coordinates, not raw "
              "embeddings. Use the *_embeddings.npy files from step 1.")

    X_struct = None
    Y_struct = None
    if args.struct1 or args.struct2:
        if not (args.struct1 and args.struct2):
            raise SystemExit(
                "ERROR: --struct1 and --struct2 must be supplied together."
            )
        X_struct = np.load(args.struct1)
        Y_struct = np.load(args.struct2)
        print(f"  Struct 1: {X_struct.shape} (from {args.struct1})")
        print(f"  Struct 2: {Y_struct.shape} (from {args.struct2})")
        if args.struct_metric is None:
            print(f"  Note: --struct-metric not set; ultrametrics will use "
                  f"--metric ({args.metric}). If struct inputs are UMAP "
                  f"outputs, 'euclidean' is usually more appropriate.")

    labels1 = load_labels(args.meta1, X.shape[0])
    labels2 = load_labels(args.meta2, Y.shape[0])

    print("\n-- Computing partial FGW --")
    res = compute_fgw(X, Y,
                      emb1_path=args.emb1, emb2_path=args.emb2,
                      mass_fraction=args.mass, alpha=args.alpha,
                      metric=args.metric, normalize_scale=args.normalize_scale,
                      nb_dummies=args.nb_dummies, num_iter_max=args.num_iter_max,
                      X_struct=X_struct, Y_struct=Y_struct,
                      struct1_path=args.struct1, struct2_path=args.struct2,
                      struct_metric=args.struct_metric,
                      diagnostics=args.diagnostics,
                      rebuild_ultrametrics=args.rebuild_ultrametrics)

    print("\n-- Results --")
    print(f"  Fused cost   (alpha={args.alpha}) : {res['fgw_cost']:.6f}")
    print(f"  Feature cost (positional shift)   : {res['feature_cost']:.6f}")
    print(f"  Structure cost (hierarchy shift)  : {res['structure_cost']:.6f}")
    print(f"  Transport matrix shape            : {res['transport_matrix'].shape}")

    T = res["transport_matrix"]
    matches, row_mass = res["matches"], res["row_mass"]

    matched = [(i, matches[i], T[i, matches[i]])
               for i in range(len(matches)) if matches[i] >= 0]
    matched.sort(key=lambda x: -x[2])
    print("\n  Top 10 matches (corpus1 -> corpus2):")
    for i, j, w in matched[:10]:
        print(f"    {labels1[i]:>12s} -> {labels2[j]:<12s}  weight: {w:.6f}")

    dropped = [labels1[i] for i in range(len(matches)) if matches[i] < 0]
    if dropped:
        print(f"\n  Unmatched corpus-1 contexts ({len(dropped)}) - candidate "
              f"sense loss/idiosyncratic usages:")
        print(f"    {dropped}")
    col_mass = T.sum(axis=0)
    dropped2 = [labels2[j] for j in range(T.shape[1])
                if col_mass[j] < (1.0 / T.shape[1]) * 0.1]
    if dropped2:
        print(f"\n  Unmatched corpus-2 contexts ({len(dropped2)}) - candidate "
              f"sense gain:")
        print(f"    {dropped2}")

    print("\n-- Saving outputs --")
    matrix_path = f'{base_path}_transport_matrix.npy'
    matches_path = f'{base_path}_matches.npy'
    np.save(matrix_path, T)
    np.save(matches_path, matches)
    print(f'   Transport matrix: {matrix_path}')
    print(f'   Matches: {matches_path}')
    plot_transport(T, row_mass, output_path=output_path)

    summary_path = f'{base_path}_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        import csv as csv_module
        writer = csv_module.writer(f)
        writer.writerow([
            'emb1', 'emb2', 'struct1', 'struct2', 'struct_metric',
            'meta1', 'meta2', 'alpha', 'mass', 'metric',
            'nb_dummies', 'num_iter_max',
            'n1', 'n2', 'fgw_cost', 'feature_cost', 'structure_cost',
        ])
        writer.writerow([
            args.emb1, args.emb2, args.struct1, args.struct2, args.struct_metric,
            args.meta1, args.meta2, args.alpha, args.mass,
            args.metric, args.nb_dummies, args.num_iter_max, X.shape[0], Y.shape[0],
            res['fgw_cost'], res['feature_cost'], res['structure_cost'],
        ])
    print(f'   Summary: {summary_path}')

    print(f"\n  Done. Fused shift score: {res['fgw_cost']:.6f}")
    print("  (Higher = greater shift; compare feature vs structure costs "
          "to see WHICH kind of shift.)")


if __name__ == "__main__":
    main()