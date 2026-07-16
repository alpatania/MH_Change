"""
Subsampling diagnostic for FGW_distance.py (v2: pooled calibration + testing)
================================================================================
Builds an empirical noise floor for FGW costs at a given sample size, pooled
across one or more calibration pairs, and optionally tests a specific
observed value against that noise floor.

Rationale: a decade pair with few matched occurrences can show an elevated
fgw_cost / feature_cost / structure_cost purely from sampling noise, not a
real difference. To check whether a specific observed value is likely real,
this script repeatedly subsamples one or more LARGE pairs -- ideally ones
you already believe have little true difference at full size -- down to the
target sample size and recomputes the costs each time. Pooling several
calibration pairs (rather than just one) avoids the noise floor being tied
to the idiosyncrasies of a single comparison.

If --observed is given, the script reports where that value falls relative
to the pooled noise floor two ways: a distribution-free rank-based bound
(exact under the null of "no more extreme than sampling noise", but coarse
with few trials) and a normal-approximation z-score/p-value (finer-grained,
but only trustworthy if the noise floor is reasonably normal -- a Shapiro-
Wilk test is reported so you can judge that for yourself rather than take
normality on faith).

This reuses compute_fgw() from FGW_distance.py directly, rather than
reimplementing the FGW computation, so results are directly comparable to
what FGW_distance.py itself would report.

Usage:
    # Calibration only -- just characterize the noise floor
    python fgw_subsample_diagnostic.py \\
        --pair results/1950s_embeddings.npy:results/1960s_embeddings.npy \\
        --pair results/1960s_embeddings.npy:results/1970s_embeddings.npy \\
        --pair results/1980s_embeddings.npy:results/1990s_embeddings.npy \\
        --target-n 59 --n-trials 200

    # Testing -- compare a specific observed value against the pooled noise floor
    python fgw_subsample_diagnostic.py \\
        --pair results/1950s_embeddings.npy:results/1960s_embeddings.npy \\
        --pair results/1960s_embeddings.npy:results/1970s_embeddings.npy \\
        --target-n 59 --n-trials 200 \\
        --observed 0.0014640982803421879 --observed-metric structure_cost

Requirements: same as FGW_distance.py (scikit-learn, scipy, POT, numpy,
pandas), plus FGW_distance.py itself importable from the same directory
or via --fgw-script-dir.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pool multiple subsampled calibration pairs into an "
                    "empirical noise floor for FGW costs, and optionally "
                    "test a specific observed value against it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pair", action="append", dest="pairs", required=True,
        metavar="EMB1_PATH:EMB2_PATH",
        help="A calibration pair, as two .npy paths joined by a colon. "
             "Repeat --pair for multiple calibration pairs; their trials "
             "are pooled into one noise-floor distribution. Pick pairs "
             "whose FULL-SIZE cost you already trust as a low/near-null "
             "baseline.",
    )
    parser.add_argument("--target-n", type=int, required=True,
                        help="Sample size to subsample every corpus down to "
                             "(e.g. the size of the small pair you're testing)")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Number of random subsamples to draw PER "
                             "calibration pair (default 200)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed; trial i of pair j uses "
                             "seed + j*100000 + i")
    parser.add_argument("--mass", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--normalize-scale", action="store_true")
    parser.add_argument("--nb-dummies", type=int, default=1)
    parser.add_argument("--num-iter-max", type=int, default=50000)
    parser.add_argument("--fgw-script-dir", default=None,
                        help="Directory containing FGW_distance.py, if not "
                             "alongside this script")
    parser.add_argument(
        "--observed", type=float, default=None,
        help="A specific observed cost value to test against the pooled "
             "noise floor. If omitted, the script only characterizes the "
             "noise floor and does not report a significance estimate.",
    )
    parser.add_argument(
        "--observed-metric", default="structure_cost",
        choices=["fgw_cost", "feature_cost", "structure_cost"],
        help="Which cost --observed corresponds to (default structure_cost)",
    )
    parser.add_argument("--output-csv", default="subsample_diagnostic.csv",
                        help="Where to write per-trial results")
    return parser.parse_args()


def load_pair(spec: str) -> tuple[Path, Path]:
    if ":" not in spec:
        raise ValueError(
            f"--pair value {spec!r} must be two paths joined by ':' "
            "(EMB1_PATH:EMB2_PATH)"
        )
    left, right = spec.split(":", 1)
    return Path(left), Path(right)


def summarize(name: str, arr: np.ndarray) -> None:
    print(f"  {name:15s} mean={arr.mean():.6f}  std={arr.std(ddof=1):.6f}  "
          f"min={arr.min():.6f}  max={arr.max():.6f}")


def main():
    args = parse_args()

    if args.fgw_script_dir:
        sys.path.insert(0, args.fgw_script_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from FGW_PCA_distance import compute_fgw

    pair_paths = [load_pair(spec) for spec in args.pairs]
    pairs = []
    for emb1_path, emb2_path in pair_paths:
        X_full = np.load(emb1_path)
        Y_full = np.load(emb2_path)
        print(f"Calibration pair: {emb1_path.name} ({X_full.shape}) vs "
              f"{emb2_path.name} ({Y_full.shape})")
        min_size = min(X_full.shape[0], Y_full.shape[0])
        if args.target_n > min_size:
            raise ValueError(
                f"--target-n {args.target_n} exceeds the smaller corpus in "
                f"pair {emb1_path.name}/{emb2_path.name} (size {min_size}). "
                "Pick a target-n at or below the smallest corpus across all "
                "calibration pairs."
            )
        pairs.append((emb1_path, emb2_path, X_full, Y_full))

    results = []  # (pair_index, trial, fgw, feat, struct)
    for pair_index, (emb1_path, emb2_path, X_full, Y_full) in enumerate(pairs):
        print(f"\n=== Calibration pair {pair_index + 1}/{len(pairs)}: "
              f"{emb1_path.name} vs {emb2_path.name} ===")
        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + pair_index * 100000 + trial)
            idx1 = rng.choice(X_full.shape[0], size=args.target_n, replace=False)
            idx2 = rng.choice(Y_full.shape[0], size=args.target_n, replace=False)
            X = X_full[idx1]
            Y = Y_full[idx2]

            res = compute_fgw(
                X, Y,
                mass_fraction=args.mass,
                alpha=args.alpha,
                metric=args.metric,
                normalize_scale=args.normalize_scale,
                nb_dummies=args.nb_dummies,
                num_iter_max=args.num_iter_max,
            )
            results.append((pair_index, trial, res["fgw_cost"],
                           res["feature_cost"], res["structure_cost"]))
            if (trial + 1) % max(1, args.n_trials // 5) == 0 or trial == args.n_trials - 1:
                print(f"  trial {trial + 1}/{args.n_trials}: "
                      f"fgw={res['fgw_cost']:.6f}  feat={res['feature_cost']:.6f}  "
                      f"struct={res['structure_cost']:.6f}")

    fgw = np.array([r[2] for r in results])
    feat = np.array([r[3] for r in results])
    struct = np.array([r[4] for r in results])
    metric_arrays = {"fgw_cost": fgw, "feature_cost": feat, "structure_cost": struct}

    print(f"\n=== Pooled noise floor across {len(pairs)} pair(s), "
          f"{len(results)} total trials (target_n={args.target_n}) ===")
    for name, arr in metric_arrays.items():
        summarize(name, arr)

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_index", "trial", "target_n", "fgw_cost",
                        "feature_cost", "structure_cost"])
        for pair_index, trial, fgw_c, feat_c, struct_c in results:
            writer.writerow([pair_index, trial, args.target_n, fgw_c, feat_c, struct_c])
    print(f"\nPer-trial results written to {args.output_csv}")

    if args.observed is not None:
        from scipy import stats

        arr = metric_arrays[args.observed_metric]
        n = len(arr)
        observed = args.observed

        skewness = stats.skew(arr)
        shapiro_p = stats.shapiro(arr).pvalue if n <= 5000 else float("nan")

        exceed_count = int((arr >= observed).sum())
        rank_bound = (exceed_count + 1) / (n + 1)  # standard permutation-style correction

        z = (observed - arr.mean()) / arr.std(ddof=1)
        normal_p = float(stats.norm.sf(z))  # one-sided: P(X >= observed)

        print(f"\n=== Testing observed {args.observed_metric} = {observed:.6f} "
              f"against pooled noise floor ({n} trials) ===")
        print(f"  Noise-floor skewness: {skewness:.3f}  "
              f"Shapiro-Wilk normality p-value: {shapiro_p:.3f} "
              f"({'consistent with normal' if shapiro_p > 0.05 else 'NOT consistent with normal -- distrust the z-score estimate below'})")
        print(f"  Trials at/above observed value: {exceed_count}/{n}")
        print(f"  Distribution-free rank-based bound: p <= {rank_bound:.5f}")
        print(f"  Normal-approximation: z = {z:.3f}, one-sided p = {normal_p:.6f}")
        print(f"  Observed value is {observed / arr.mean():.2f}x the noise-floor mean "
              f"and {observed / arr.max():.2f}x the noise-floor maximum.")


if __name__ == "__main__":
    main()