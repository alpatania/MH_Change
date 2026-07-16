"""
Save a decade's single-linkage matrix for tree-structure visualizations
============================================================================
FGW_distance.py computes each decade's single-linkage ultrametric
internally (inside compute_fgw / ultrametric_from_points) but only keeps
the derived cophenetic distance matrix -- the linkage matrix itself (Z,
the actual tree structure) is discarded. This script computes and saves
Z directly, once per decade, reusing the same linkage call FGW_distance.py
makes internally so the tree is identical to the one implicitly used
there.

Usage:
    python fgw_build_linkage.py --emb results/insan_1820s_embeddings.npy \\
        --metric cosine --output results/insan_1820s_linkage.npy
"""

import argparse

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save a decade's single-linkage matrix."
    )
    parser.add_argument("--emb", required=True, help="<decade>_embeddings.npy")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--output", required=True, help="Where to save the linkage matrix (.npy)")
    return parser.parse_args()


def main():
    args = parse_args()
    X = np.load(args.emb)
    print(f"Embeddings: {X.shape}")

    D = pairwise_distances(X, metric=args.metric)
    Z = linkage(squareform(D, checks=False), method="single")
    np.save(args.output, Z)
    print(f"Linkage matrix ({Z.shape[0]} merges for {X.shape[0]} leaves): {args.output}")


if __name__ == "__main__":
    main()
