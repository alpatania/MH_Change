"""
Sankey-style cross-decade correspondence diagram
====================================================
Visualizes how individual matched occurrences of a target word connect
across decades, using the transport matrices FGW_distance.py already
saved. Decades on the x-axis, one node per occurrence on the y-axis
(normalized to 0-1 within each decade so sparse and dense decades are
visually comparable), a line between two nodes in adjacent decades if
they were matched (weight = transported mass), nodes and their links
colored by word.

This is not a literal Plotly Sankey trace (go.Sankey aggregates flow
between category BINS, not individual entities, and its automatic layout
doesn't give the kind of fine control needed for thousands of individual
per-occurrence nodes). It's built from scatter markers (nodes) and batched
line segments (links), which scales much better to this many points and
gives direct control over vertical layout and color.

Reads, per decade:  <out-dir>/<search>_<decade>s_coords.csv
Reads, per pair:    <out-dir>/<search>_fgw_<decade1>_<decade2>_transport_matrix.npy
(both produced by the existing pipeline: run_embedding_and_distance.sh)

Usage:
    python fgw_sankey_diagram.py --out-dir results --search insan \\
        --output results/insan_sankey.html

    # Limit visual density for large corpora
    python fgw_sankey_diagram.py --out-dir results --search insan \\
        --sample-per-decade 60 --output results/insan_sankey.html
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DECADE_RE = re.compile(r"^(?P<search>.+)_(?P<decade>\d{4})s_coords\.csv$")
PAIR_RE = re.compile(
    r"^(?P<search>.+)_fgw_(?P<decade1>\d{4})_(?P<decade2>\d{4})_transport_matrix\.npy$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sankey-style diagram of matched occurrences across decades.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", required=True,
                        help="Directory containing the pipeline's coords "
                             "csvs and FGW transport matrices")
    parser.add_argument("--search", required=True,
                        help="Search term prefix (matches run_all_decades.sh "
                             "/ run_embedding_and_distance.sh's --search)")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Links drawn per node, ranked by transported "
                             "mass (default 1)")
    parser.add_argument("--min-mass-fraction", type=float, default=0.1,
                        help="Same unmatched threshold FGW_distance.py uses: "
                             "row_mass <= this fraction of a uniform row's "
                             "mass is treated as unmatched (default 0.1)")
    parser.add_argument("--sample-per-decade", type=int, default=None,
                        help="If a decade has more nodes than this, randomly "
                             "sample down to this many for display (does not "
                             "affect the underlying FGW computation, only "
                             "what's drawn). A link is only drawn if BOTH "
                             "its endpoints are in the sampled subset, so "
                             "some true matches may not be shown alongside "
                             "sampled decades. Default: show everything.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for --sample-per-decade")
    parser.add_argument("--barycenter-passes", type=int, default=4,
                        help="Alternating forward/backward smoothing passes "
                             "that reposition each decade's nodes toward the "
                             "mass-weighted average position of whatever "
                             "they're matched to in the neighboring decade, "
                             "minimizing total line height-difference "
                             "(0 disables this and falls back to a plain "
                             "sort-by-word layout; default 4)")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def discover_decades(out_dir: Path, search: str) -> list[int]:
    decades = []
    for path in out_dir.glob(f"{search}_*s_coords.csv"):
        match = DECADE_RE.match(path.name)
        if match and match["search"] == search:
            decades.append(int(match["decade"]))
    return sorted(decades)


def discover_pairs(out_dir: Path, search: str) -> dict[tuple[int, int], Path]:
    pairs = {}
    for path in out_dir.glob(f"{search}_fgw_*_*_transport_matrix.npy"):
        match = PAIR_RE.match(path.name)
        if match and match["search"] == search:
            pairs[(int(match["decade1"]), int(match["decade2"]))] = path
    return pairs


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    decades = discover_decades(out_dir, args.search)
    if not decades:
        raise ValueError(
            f"No {args.search}_<decade>s_coords.csv files found in {out_dir}"
        )
    print(f"Found decades: {decades}")

    pairs = discover_pairs(out_dir, args.search)
    if not pairs:
        raise ValueError(
            f"No {args.search}_fgw_<d1>_<d2>_transport_matrix.npy files "
            f"found in {out_dir}"
        )
    print(f"Found {len(pairs)} decade-pair transport matrices")

    # --- load each decade's coords, subsample for display if requested ---
    decade_meta: dict[int, pd.DataFrame] = {}
    decade_display_idx: dict[int, np.ndarray] = {}  # original row indices shown
    for decade in decades:
        meta = pd.read_csv(out_dir / f"{args.search}_{decade}s_coords.csv", dtype=str)
        decade_meta[decade] = meta
        n = len(meta)
        if args.sample_per_decade and n > args.sample_per_decade:
            idx = rng.choice(n, size=args.sample_per_decade, replace=False)
            idx.sort()
        else:
            idx = np.arange(n)
        decade_display_idx[decade] = idx

    # --- global, consistent color per word across every decade ---
    all_words = sorted(set(
        word for decade in decades
        for word in decade_meta[decade].iloc[decade_display_idx[decade]]["word"]
    ))
    palette = (px.colors.qualitative.Dark24 if len(all_words) <= 24
              else px.colors.qualitative.Alphabet)
    color_map = {w: palette[i % len(palette)] for i, w in enumerate(all_words)}
    word_rank = {w: i / max(1, len(all_words) - 1) for i, w in enumerate(all_words)}

    decade_shown_set: dict[int, set] = {d: set(decade_display_idx[d]) for d in decades}

    def get_pair_matches(T: np.ndarray, d1: int, d2: int) -> dict[int, list[tuple[int, float]]]:
        """Row -> ranked list of (col, weight) above the mass threshold,
        restricted to displayed indices on both sides. Single source of
        truth for both barycenter positioning and link drawing, so they
        can never disagree about which pairs count as "matched".
        """
        n1 = T.shape[0]
        row_mass = T.sum(axis=1)
        threshold = (1.0 / n1) * args.min_mass_fraction
        matches: dict[int, list[tuple[int, float]]] = {}
        for i in range(n1):
            if i not in decade_shown_set[d1] or row_mass[i] <= threshold:
                continue
            ranked = [
                (int(j), float(T[i, j])) for j in np.argsort(-T[i])[: args.top_k]
                if T[i, j] > 0 and int(j) in decade_shown_set[d2]
            ]
            if ranked:
                matches[i] = ranked
        return matches

    pair_matches: dict[tuple[int, int], dict[int, list[tuple[int, float]]]] = {}
    for (d1, d2), matrix_path in pairs.items():
        if d1 not in decades or d2 not in decades:
            continue
        T = np.load(matrix_path)
        meta1, meta2 = decade_meta[d1], decade_meta[d2]
        if T.shape[0] != len(meta1) or T.shape[1] != len(meta2):
            print(f"WARNING: skipping {d1}-{d2}: transport matrix shape "
                  f"{T.shape} doesn't match coords sizes "
                  f"({len(meta1)}, {len(meta2)})")
            continue
        pair_matches[(d1, d2)] = get_pair_matches(T, d1, d2)

    # --- initial y-position: rank within decade (sorted by word), 0-1 ---
    # This is the pass-0 layout, and also the permanent fallback for any
    # node that ends up with no usable match to inherit a position from.
    y_cont: dict[int, dict[int, float]] = {}
    for decade in decades:
        idx = decade_display_idx[decade]
        meta = decade_meta[decade]
        shown = meta.iloc[idx].copy()
        order = shown.sort_values("word", kind="stable").index
        n_shown = len(order)
        y_cont[decade] = {
            orig_idx: (rank / (n_shown - 1) if n_shown > 1 else 0.5)
            for rank, orig_idx in enumerate(order)
        }
    word_fallback: dict[int, dict[int, float]] = {
        decade: {
            orig_idx: word_rank.get(decade_meta[decade].iloc[orig_idx]["word"], 0.5)
            for orig_idx in decade_shown_set[decade]
        }
        for decade in decades
    }

    # --- barycenter smoothing: alternating forward/backward passes so each
    # decade's node positions track the (mass-weighted) average position of
    # whatever they're matched to in the neighboring decade, minimizing the
    # total y-difference across drawn links. Standard technique for
    # minimizing edge length/crossings in layered graph drawing (Sugiyama
    # et al.), applied here per-node rather than per-category. ---
    decades_sorted = sorted(decades)
    for pass_num in range(args.barycenter_passes):
        forward = (pass_num % 2 == 0)
        order = decades_sorted if forward else list(reversed(decades_sorted))
        for pos, decade in enumerate(order):
            if pos == 0:
                continue
            neighbor = order[pos - 1]
            if forward:
                pair_key = (neighbor, decade)  # neighbor is earlier
            else:
                pair_key = (decade, neighbor)  # neighbor is later
            matches = pair_matches.get(pair_key)
            if not matches:
                continue

            if forward:
                # reverse lookup: for each j in `decade`, average over the
                # i's in `neighbor` that chose j as a top-k target
                incoming: dict[int, list[tuple[float, float]]] = {}
                for i, targets in matches.items():
                    y_i = y_cont[neighbor][i]
                    for j, w in targets:
                        incoming.setdefault(j, []).append((y_i, w))
                for j in decade_shown_set[decade]:
                    contributions = incoming.get(j)
                    if contributions:
                        ys, ws = zip(*contributions)
                        y_cont[decade][j] = float(np.average(ys, weights=ws))
                    # else: leave at current value (previous pass or fallback)
            else:
                # direct lookup: for each i in `decade`, average over its
                # own chosen targets j in `neighbor`
                for i in decade_shown_set[decade]:
                    targets = matches.get(i)
                    if targets:
                        ys = [y_cont[neighbor][j] for j, _ in targets]
                        ws = [w for _, w in targets]
                        y_cont[decade][i] = float(np.average(ys, weights=ws))
                    # else: leave at current value

    # --- final layout: sort each decade by its converged continuous
    # position and assign evenly spaced 0-1 ranks, so nodes stay distinct
    # and readable while tracking the smoothed ordering as closely as
    # discretization allows ---
    decade_y: dict[int, dict[int, float]] = {}
    for decade in decades:
        idx = decade_display_idx[decade]
        keys = list(idx)
        keys.sort(key=lambda orig_idx: y_cont[decade].get(
            orig_idx, word_fallback[decade][orig_idx]))
        n_shown = len(keys)
        decade_y[decade] = {
            orig_idx: (rank / (n_shown - 1) if n_shown > 1 else 0.5)
            for rank, orig_idx in enumerate(keys)
        }

    # --- build node traces, one per word (so the legend can toggle by word) ---
    fig = go.Figure()

    def snippet(text, n=140):
        text = str(text)
        return text[:n] + ("..." if len(text) > n else "")

    # links first, so nodes render on top
    for (d1, d2), matches in pair_matches.items():
        meta1 = decade_meta[d1]

        # group link segments by source word for batched, legend-clean traces
        segments_by_word: dict[str, dict[str, list]] = {}
        for i, targets in matches.items():
            word = meta1.iloc[i]["word"]
            bucket = segments_by_word.setdefault(
                word, {"x": [], "y": [], "weight": []}
            )
            y1 = decade_y[d1][i]
            for j, weight in targets:
                y2 = decade_y[d2][j]
                bucket["x"].extend([d1, d2, None])
                bucket["y"].extend([y1, y2, None])
                bucket["weight"].append(weight)

        for word, bucket in segments_by_word.items():
            if not bucket["x"]:
                continue
            max_w = max(bucket["weight"]) if bucket["weight"] else 1.0
            fig.add_trace(go.Scattergl(
                x=bucket["x"], y=bucket["y"],
                mode="lines",
                line=dict(color=color_map.get(word, "#888888"), width=1),
                opacity=0.25,
                showlegend=False,
                hoverinfo="skip",
            ))

    for word in all_words:
        xs, ys, hover = [], [], []
        for decade in decades:
            idx = decade_display_idx[decade]
            meta = decade_meta[decade]
            shown = meta.iloc[idx]
            word_rows = shown[shown["word"] == word]
            for orig_idx, row in word_rows.iterrows():
                xs.append(decade)
                ys.append(decade_y[decade][orig_idx])
                parts = [f"<b>{row.get('wid', orig_idx)}</b> ({word})", f"decade: {decade}s"]
                for col in ("genre", "year"):
                    if col in row.index:
                        parts.append(f"{col}: {row[col]}")
                if "passage" in row.index:
                    parts.append(snippet(row["passage"]))
                hover.append("<br>".join(parts))

        if not xs:
            continue
        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            mode="markers",
            name=word,
            marker=dict(color=color_map[word], size=7,
                       line=dict(color="white", width=0.5)),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"Cross-decade correspondence of contextual occurrences "
                 f"of '{args.search}'<br>"
                 f"<sup>colour = matched word form -- lines connect matched "
                 f"occurrences in adjacent decades (top-{args.top_k}, "
                 f"weighted by transported mass) -- y-position is layout only</sup>",
            font=dict(size=15),
        ),
        xaxis=dict(title="Decade", tickmode="array", tickvals=decades,
                  ticktext=[f"{d}s" for d in decades]),
        yaxis=dict(title=None, showticklabels=False, zeroline=False),
        legend_title="Word form",
        hovermode="closest",
        template="plotly_white",
        width=max(1100, 90 * len(decades)),
        height=750,
    )

    fig.write_html(args.output, include_plotlyjs="cdn")
    total_nodes = sum(len(decade_display_idx[d]) for d in decades)
    print(f"\nWrote {args.output}")
    print(f"  {len(decades)} decades, {total_nodes} nodes shown, "
          f"{len(pairs)} decade-pair(s) with links drawn")


if __name__ == "__main__":
    main()