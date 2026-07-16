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

Usage:
    uv run FGW_distance.py --emb1 c1_embeddings.npy --emb2 c2_embeddings.npy
    uv run FGW_distance.py --emb1 ... --emb2 ... --meta1 c1_coords.csv --meta2 c2_coords.csv
    uv run FGW_distance.py --emb1 ... --emb2 ... --alpha 0.7 --mass 0.6

Requirements:
    uv add scikit-learn scipy POT matplotlib numpy pandas
"""

import argparse

import numpy as np
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


# -- FGW COMPUTATION -------------------------------------------------------------

def compute_fgw(
    X: np.ndarray,
    Y: np.ndarray,
    mass_fraction: float = 0.8,
    alpha: float = 0.5,
    metric: str = "cosine",
    normalize_scale: bool = False,
    nb_dummies: int = 1,
    num_iter_max: int = 50000,
) -> dict:
    """
    Partial fused Gromov-Wasserstein between two embedding matrices that live
    in the SAME feature space (e.g. both RoBERTa final layer).

    Args:
        X, Y            : embeddings, shapes (n, d) and (m, d) with equal d
        mass_fraction   : fraction of total mass to transport (0-1);
                          lower = more permissive partial matching
        alpha           : structure/feature trade-off (0 = feature only,
                          1 = structure only)
        metric          : metric for both the cross-corpus feature cost and
                          the within-corpus ultrametrics
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

    n, m = X.shape[0], Y.shape[0]
    p = np.ones(n) / n
    q = np.ones(m) / m

    print("  Building ultrametric for corpus 1...")
    UX = ultrametric_from_points(X, metric=metric)
    print("  Building ultrametric for corpus 2...")
    UY = ultrametric_from_points(Y, metric=metric)

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

    # decomposition: feature-only and structure-only runs for context
    print("  Baseline: partial Wasserstein (feature only)...")
    _, log_w = ot.partial.partial_wasserstein(
        p, q, M, m=mass_fraction, nb_dummies=nb_dummies,
        numItermax=num_iter_max, log=True,
    )
    feat_cost = float(log_w["partial_w_dist"])

    print("  Baseline: partial GW (structure only)...")
    _, log_gw = ot.gromov.partial_gromov_wasserstein(
        UX, UY, p, q, m=mass_fraction, loss_fun="square_loss",
        nb_dummies=nb_dummies, numItermax=num_iter_max, log=True,
    )
    struct_cost = float(log_gw["partial_gw_dist"])

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

    labels1 = load_labels(args.meta1, X.shape[0])
    labels2 = load_labels(args.meta2, Y.shape[0])

    print("\n-- Computing partial FGW --")
    res = compute_fgw(X, Y, mass_fraction=args.mass, alpha=args.alpha,
                      metric=args.metric, normalize_scale=args.normalize_scale,
                      nb_dummies=args.nb_dummies, num_iter_max=args.num_iter_max)

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
            'emb1', 'emb2', 'meta1', 'meta2', 'alpha', 'mass', 'metric',
            'nb_dummies', 'num_iter_max',
            'n1', 'n2', 'fgw_cost', 'feature_cost', 'structure_cost',
        ])
        writer.writerow([
            args.emb1, args.emb2, args.meta1, args.meta2, args.alpha, args.mass,
            args.metric, args.nb_dummies, args.num_iter_max, X.shape[0], Y.shape[0],
            res['fgw_cost'], res['feature_cost'], res['structure_cost'],
        ])
    print(f'   Summary: {summary_path}')

    print(f"\n  Done. Fused shift score: {res['fgw_cost']:.6f}")
    print("  (Higher = greater shift; compare feature vs structure costs "
          "to see WHICH kind of shift.)")


if __name__ == "__main__":
    main()