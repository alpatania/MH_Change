"""
Save each decade's single-linkage matrix for tree-structure visualizations
============================================================================
FGW_distance.py computes each decade's single-linkage ultrametric internally
(inside compute_fgw / ultrametric_from_points) but keeps only the derived
cophenetic distance matrix -- the linkage matrix itself (Z, the tree
structure) is discarded. fgw_tanglegram.py needs Z. This script recomputes and
saves it, reusing the identical linkage call so the tree matches the one FGW
used internally.

Two modes:

  Batch (recommended) -- give a search term; every decade's embeddings in the
  search's results folder is turned into a linkage matrix:

      python fgw_build_linkage.py --search insan
      # reads  results/insan/insan_<decade>s_embeddings.npy
      # writes results/insan/insan_<decade>s_linkage.npy

  Single -- the original one-file form, still supported:

      python fgw_build_linkage.py \\
          --emb results/insan/insan_1820s_embeddings.npy \\
          --output results/insan/insan_1820s_linkage.npy

--metric must match the metric used for the FGW ultrametrics (default cosine).
"""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances


EMB_RE = re.compile(r"^(?P<search>.+)_(?P<decade>\d{4})s_embeddings\.npy$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save each decade's single-linkage matrix."
    )
    parser.add_argument("--search", default=None,
                        help="Batch mode: build linkages for every decade of "
                             "this search. Reads/writes results/<search>/ "
                             "unless --out-dir is given.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory holding the embeddings (default: "
                             "results/<search>/). Batch mode only.")
    parser.add_argument("--emb", default=None,
                        help="Single mode: one <decade>_embeddings.npy.")
    parser.add_argument("--output", default=None,
                        help="Single mode: where to save the linkage .npy. "
                             "Defaults to the --emb path with _embeddings "
                             "replaced by _linkage.")
    parser.add_argument("--metric", default="cosine",
                        help="Must match the FGW ultrametric metric "
                             "(default cosine).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild even if the linkage file already exists.")
    return parser.parse_args()


def build_one(emb_path: Path, out_path: Path, metric: str, rebuild: bool) -> None:
    if out_path.exists() and not rebuild:
        print(f"  skip (exists): {out_path.name}")
        return
    X = np.load(emb_path)
    D = pairwise_distances(X, metric=metric)
    Z = linkage(squareform(D, checks=False), method="single")
    np.save(out_path, Z)
    print(f"  {emb_path.name} ({X.shape[0]} leaves) -> {out_path.name}")


def main():
    args = parse_args()

    # Single-file mode
    if args.emb:
        emb_path = Path(args.emb)
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = emb_path.with_name(
                emb_path.name.replace("_embeddings.npy", "_linkage.npy"))
        build_one(emb_path, out_path, args.metric, args.rebuild)
        return

    # Batch mode
    if not args.search:
        raise SystemExit("ERROR: give either --search (batch) or --emb (single).")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", args.search)
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / slug
    if not out_dir.is_dir():
        raise SystemExit(f"ERROR: {out_dir} is not a directory. Run stage 2 first.")

    embs = sorted(p for p in out_dir.glob(f"{args.search}_*s_embeddings.npy")
                  if EMB_RE.match(p.name))
    if not embs:
        raise SystemExit(
            f"ERROR: no {args.search}_<decade>s_embeddings.npy in {out_dir}.")

    print(f"Building {len(embs)} linkage matrix/matrices in {out_dir}")
    for emb_path in embs:
        out_path = emb_path.with_name(
            emb_path.name.replace("_embeddings.npy", "_linkage.npy"))
        build_one(emb_path, out_path, args.metric, args.rebuild)
    print("Done.")


if __name__ == "__main__":
    main()