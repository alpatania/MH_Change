"""
Extract the node correspondence from a saved FGW transport matrix
=====================================================================
FGW_distance.py already saves the actual transport plan used to compute
the distance, as <base>_transport_matrix.npy -- an (n1 x n2) matrix where
entry [i, j] is the fractional mass moved from corpus-1 point i to
corpus-2 point j. That matrix IS the correspondence; matches.npy is only
a simplified one-best-match-per-row summary of it (top printed to stdout,
not saved as a file).

This script reads the transport matrix back in, attaches readable labels
from each corpus's _coords.csv (wid, word, a passage snippet, genre,
year), and writes a full correspondence table -- optionally the single
best match per row (matching FGW_distance.py's own "matches" logic), or
the top-k matches per row if you want the fuller picture the transport
matrix actually contains rather than a hard 1-best summary.

Usage:
    python fgw_correspondence.py \\
        --transport-matrix results/insan_fgw_1820_1830_transport_matrix.npy \\
        --meta1 results/insan_1820s_coords.csv \\
        --meta2 results/insan_1830s_coords.csv \\
        --top-k 1 --output results/insan_1820_1830_correspondence.csv
"""

import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Turn a saved FGW transport matrix into a labeled "
                    "node correspondence table."
    )
    parser.add_argument("--transport-matrix", required=True,
                        help="<base>_transport_matrix.npy from FGW_distance.py")
    parser.add_argument("--meta1", required=True,
                        help="corpus 1's _coords.csv from final_layer_embeddings.py")
    parser.add_argument("--meta2", required=True,
                        help="corpus 2's _coords.csv from final_layer_embeddings.py")
    parser.add_argument("--top-k", type=int, default=1,
                        help="How many corpus-2 matches to report per "
                             "corpus-1 point, ranked by transported mass "
                             "(default 1, matching FGW_distance.py's own "
                             "hard-match summary; use more for the fuller "
                             "picture the transport matrix actually holds)")
    parser.add_argument("--min-mass-fraction", type=float, default=0.1,
                        help="Same threshold FGW_distance.py uses to flag a "
                             "row as unmatched: row_mass <= this fraction of "
                             "a uniform row's mass (1/n1) is marked unmatched "
                             "(default 0.1)")
    parser.add_argument("--passage-snippet-chars", type=int, default=120,
                        help="How many characters of the passage column to "
                             "include per side, if present (default 120)")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_meta(path: str, n_expected: int) -> pd.DataFrame:
    meta = pd.read_csv(path, dtype=str)
    if len(meta) != n_expected:
        raise ValueError(
            f"{path} has {len(meta)} rows but the transport matrix expects "
            f"{n_expected} -- these files don't correspond to each other. "
            "Make sure --meta1/--meta2 are the exact *_coords.csv produced "
            "alongside the *_embeddings.npy that were passed to FGW_distance.py."
        )
    return meta.reset_index(drop=True)


def label_row(meta: pd.DataFrame, i: int, snippet_chars: int) -> dict:
    row = meta.iloc[i]
    out = {}
    for col in ("wid", "word", "genre", "year"):
        if col in meta.columns:
            out[col] = row[col]
    if "passage" in meta.columns:
        text = str(row["passage"])
        out["passage_snippet"] = text[:snippet_chars] + ("..." if len(text) > snippet_chars else "")
    return out


def main():
    args = parse_args()

    T = np.load(args.transport_matrix)
    n1, n2 = T.shape
    print(f"Transport matrix: {T.shape}")

    meta1 = load_meta(args.meta1, n1)
    meta2 = load_meta(args.meta2, n2)

    row_mass = T.sum(axis=1)
    threshold = (1.0 / n1) * args.min_mass_fraction
    rows = []

    for i in range(n1):
        label1 = label_row(meta1, i, args.passage_snippet_chars)
        matched = row_mass[i] > threshold

        if not matched:
            record = {"corpus1_index": i, **{f"corpus1_{k}": v for k, v in label1.items()},
                       "corpus1_row_mass": row_mass[i], "matched": False,
                       "rank": None, "corpus2_index": None, "transported_mass": None}
            rows.append(record)
            continue

        ranked_targets = np.argsort(-T[i])[: args.top_k]
        for rank, j in enumerate(ranked_targets, start=1):
            weight = T[i, j]
            if weight <= 0:
                continue
            label2 = label_row(meta2, j, args.passage_snippet_chars)
            record = {
                "corpus1_index": i, **{f"corpus1_{k}": v for k, v in label1.items()},
                "corpus1_row_mass": row_mass[i], "matched": True, "rank": rank,
                "corpus2_index": j, **{f"corpus2_{k}": v for k, v in label2.items()},
                "transported_mass": weight,
            }
            rows.append(record)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    n_matched = int(row_mass[row_mass > threshold].shape[0])
    n_unmatched = n1 - n_matched
    print(f"{n_matched}/{n1} corpus-1 points matched (row_mass > "
          f"{args.min_mass_fraction:.0%} of uniform); {n_unmatched} unmatched.")
    print(f"Wrote {len(out_df)} row(s) (top-{args.top_k} per matched point) "
          f"to {args.output}")


if __name__ == "__main__":
    main()
