"""
Tree-structure-preserving cross-decade tanglegram
========================================================
Draws each decade's actual single-linkage dendrogram (not just a point
cloud), fanning out from a vertical leaf-line at that decade's x-position,
with tanglegram-style connector lines between matched leaves in adjacent
decades -- the same idea as the Sankey-style diagram, but preserving the
real tree structure each ultrametric encodes, so you can see whether close
clusters in one decade land near close clusters in the next.

Leaf order within each decade's tree is chosen greedily, one internal node
at a time: at each merge, the child with the smaller mean "target
position" is placed first. Target position comes from the neighboring
decade's (already-fixed) leaf positions, via the FGW correspondence --
this is the "one-sided" tanglegram untangling strategy, extended across a
whole chain of decades with alternating forward/backward smoothing passes
(same architecture as fgw_sankey_diagram.py's barycenter smoothing, with
"rotate the tree" replacing "move the point" as the operation, since a
dendrogram's leaves can only be reordered by rotating internal nodes, not
freely rearranged).

Requires, per decade, a saved linkage matrix (see fgw_build_linkage.py) in
addition to the usual coords csv and pairwise transport matrices:

    <out-dir>/<search>_<decade>s_linkage.npy       (fgw_build_linkage.py)
    <out-dir>/<search>_<decade>s_coords.csv        (final_layer_embeddings.py)
    <out-dir>/<search>_fgw_<d1>_<d2>_transport_matrix.npy  (FGW_distance.py)

Usage:
    python fgw_tanglegram.py --out-dir results --search insan \\
        --output results/insan_tanglegram.html
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DECADE_RE = re.compile(r"^(?P<search>.+)_(?P<decade>\d{4})s_coords\.csv$")
LINKAGE_RE = re.compile(r"^(?P<search>.+)_(?P<decade>\d{4})s_linkage\.npy$")
PAIR_RE = re.compile(
    r"^(?P<search>.+)_fgw_(?P<decade1>\d{4})_(?P<decade2>\d{4})_transport_matrix\.npy$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tree-structure-preserving tanglegram across decades.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--top-k", type=int, default=1,
                        help="Links considered per node when computing "
                             "target positions and drawing connectors "
                             "(default 1)")
    parser.add_argument("--min-mass-fraction", type=float, default=0.1,
                        help="Same unmatched threshold FGW_distance.py "
                             "uses (default 0.1)")
    parser.add_argument("--passes", type=int, default=4,
                        help="Alternating forward/backward untangling "
                             "passes (default 4; 0 = word-sorted initial "
                             "order only, no untangling)")
    parser.add_argument("--tree-width", type=float, default=3.0,
                        help="How far each decade's tree fans out in "
                             "x-units (decades are typically 10 apart; "
                             "default 3.0 leaves clear separation)")
    parser.add_argument("--collapse-fraction", type=float, default=0.10,
                        help="Collapse any subtree whose leaf count is at "
                             "or below this fraction of its decade's total "
                             "leaf count into a single wedge glyph, rather "
                             "than drawing its internal chaining structure "
                             "in full detail (default 0.10; 0 disables "
                             "collapsing and draws every leaf individually)")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def discover_decades(out_dir: Path, search: str) -> list[int]:
    decades = []
    for path in out_dir.glob(f"{search}_*s_coords.csv"):
        m = DECADE_RE.match(path.name)
        if m and m["search"] == search:
            decades.append(int(m["decade"]))
    return sorted(decades)


def discover_pairs(out_dir: Path, search: str) -> dict[tuple[int, int], Path]:
    pairs = {}
    for path in out_dir.glob(f"{search}_fgw_*_*_transport_matrix.npy"):
        m = PAIR_RE.match(path.name)
        if m and m["search"] == search:
            pairs[(int(m["decade1"]), int(m["decade2"]))] = path
    return pairs


def compute_swaps(Z: np.ndarray, n_leaves: int,
                  target_position: dict[int, float]) -> dict[int, bool]:
    """Bottom-up, iterative (no native recursion -- single linkage can
    produce very unbalanced/chained trees, so this avoids any recursion-
    depth risk). At each internal node, decide whether to swap [child0,
    child1] so the child with the smaller mean target position comes
    first. For a binary tree under a position-difference objective, this
    is equivalent to "does flipping reduce total cost, flip if so".
    """
    root = 2 * n_leaves - 2
    node_mean: dict[int, tuple[float, int]] = {}
    swap: dict[int, bool] = {}
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if node < n_leaves:
            node_mean[node] = (target_position.get(node, 0.5), 1)
            continue
        row_idx = node - n_leaves
        c0, c1 = int(Z[row_idx, 0]), int(Z[row_idx, 1])
        if not processed:
            stack.append((node, True))
            stack.append((c0, False))
            stack.append((c1, False))
        else:
            m0, s0 = node_mean[c0]
            m1, s1 = node_mean[c1]
            if m0 > m1:
                swap[node] = True
            node_mean[node] = ((m0 * s0 + m1 * s1) / (s0 + s1), s0 + s1)
    return swap


def apply_swaps(Z: np.ndarray, n_leaves: int, swap: dict[int, bool]) -> np.ndarray:
    Z_ordered = Z.copy()
    for node, do_swap in swap.items():
        if do_swap:
            row_idx = node - n_leaves
            Z_ordered[row_idx, [0, 1]] = Z_ordered[row_idx, [1, 0]]
    return Z_ordered


def dendrogram_geometry(Z: np.ndarray, n_leaves: int):
    """Leaf order and U-shaped drawing coordinates for every merge, computed
    with a single forward pass over Z -- no recursion of any kind, not even
    an explicit stack. This deliberately avoids scipy.cluster.hierarchy.
    dendrogram(), which implements this recursively and is documented (and
    reproduced against a real corpus in this project) to raise
    RecursionError on sufficiently large/deep trees -- a long-standing,
    unresolved scipy limitation, not something fixable by catching the
    error or raising sys.setrecursionlimit() safely.

    Safe here specifically because linkage matrices are topologically
    sorted by construction: every child (row or leaf) has a smaller id
    than the parent row that merges it, so a single pass over Z in row
    order guarantees both children of every merge are already resolved.

    Verified byte-for-byte against scipy's own dendrogram() output (leaf
    order and segment coordinates) on cases small enough for scipy to
    succeed, before being relied on here for cases large enough that it
    doesn't.

    Returns (leaf_order, segments), where segments is a list of
    (node_id, xs, ys) with xs/ys in scipy's own icoord/dcoord convention
    (leaf slots at 5, 15, 25, ...; y = merge height), so existing
    rescaling logic written against that convention needs no changes.
    """
    cache_leaves: dict[int, list[int]] = {i: [i] for i in range(n_leaves)}
    for k in range(len(Z)):
        c0, c1 = int(Z[k, 0]), int(Z[k, 1])
        cache_leaves[n_leaves + k] = cache_leaves.pop(c0) + cache_leaves.pop(c1)
    leaf_order = cache_leaves[2 * n_leaves - 2] if n_leaves > 1 else [0]

    node_x: dict[int, float] = {leaf: 5 + 10 * rank for rank, leaf in enumerate(leaf_order)}
    node_h: dict[int, float] = {i: 0.0 for i in range(n_leaves)}
    segments: list[tuple[int, list[float], list[float]]] = []
    for k in range(len(Z)):
        node = n_leaves + k
        c0, c1 = int(Z[k, 0]), int(Z[k, 1])
        h = float(Z[k, 2])
        x0, x1 = node_x[c0], node_x[c1]
        h0, h1 = node_h[c0], node_h[c1]
        segments.append((node, [x0, x0, x1, x1], [h0, h, h, h1]))
        node_x[node] = (x0 + x1) / 2.0
        node_h[node] = h
    return leaf_order, segments


def find_collapse_points(Z: np.ndarray, n_leaves: int, threshold: int) -> list[int]:
    """Top-down, iterative: find every MAXIMAL internal node whose subtree
    size is <= threshold -- i.e. stop descending as soon as a subtree is
    small enough, rather than continuing into its (even smaller) children.
    Never returns plain leaves (size-1), only real multi-leaf clusters.
    """
    if threshold < 2:
        return []
    root = 2 * n_leaves - 2
    collapse_points = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node < n_leaves:
            continue  # a bare leaf is never a collapse point
        size = int(Z[node - n_leaves, 3])
        if size <= threshold:
            collapse_points.append(node)
            continue  # stop descending -- this whole subtree is one glyph
        c0, c1 = int(Z[node - n_leaves, 0]), int(Z[node - n_leaves, 1])
        stack.append(c0)
        stack.append(c1)
    return collapse_points


def leaves_under(Z: np.ndarray, n_leaves: int, node: int) -> list[int]:
    """Iterative traversal collecting every leaf under `node`."""
    if node < n_leaves:
        return [node]
    result = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n < n_leaves:
            result.append(n)
            continue
        c0, c1 = int(Z[n - n_leaves, 0]), int(Z[n - n_leaves, 1])
        stack.append(c0)
        stack.append(c1)
    return result


def nodes_under(Z: np.ndarray, n_leaves: int, node: int) -> list[int]:
    """Iterative traversal collecting every node (leaf or internal,
    including `node` itself) under `node`."""
    result = []
    stack = [node]
    while stack:
        n = stack.pop()
        result.append(n)
        if n >= n_leaves:
            c0, c1 = int(Z[n - n_leaves, 0]), int(Z[n - n_leaves, 1])
            stack.append(c0)
            stack.append(c1)
    return result


def get_pair_matches(T: np.ndarray, top_k: int,
                     min_mass_fraction: float) -> dict[int, list[tuple[int, float]]]:
    """Row -> ranked list of (col, weight) above the mass threshold. No
    leaves are excluded here (this script doesn't support subsampling --
    pruning a dendrogram to a subset of leaves isn't a simple row
    selection, it changes the tree itself, which would work against the
    fidelity this script is specifically for).
    """
    n1 = T.shape[0]
    row_mass = T.sum(axis=1)
    threshold = (1.0 / n1) * min_mass_fraction
    matches: dict[int, list[tuple[int, float]]] = {}
    for i in range(n1):
        if row_mass[i] <= threshold:
            continue
        ranked = [(int(j), float(T[i, j])) for j in np.argsort(-T[i])[:top_k] if T[i, j] > 0]
        if ranked:
            matches[i] = ranked
    return matches


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    decades = discover_decades(out_dir, args.search)
    if not decades:
        raise ValueError(f"No {args.search}_<decade>s_coords.csv files found in {out_dir}")
    print(f"Found decades: {decades}")

    pairs = discover_pairs(out_dir, args.search)
    print(f"Found {len(pairs)} decade-pair transport matrices")

    decade_meta: dict[int, pd.DataFrame] = {}
    decade_Z: dict[int, np.ndarray] = {}
    decade_n: dict[int, int] = {}
    for decade in decades:
        meta = pd.read_csv(out_dir / f"{args.search}_{decade}s_coords.csv", dtype=str)
        linkage_path = out_dir / f"{args.search}_{decade}s_linkage.npy"
        if not linkage_path.exists():
            raise FileNotFoundError(
                f"{linkage_path} not found -- run fgw_build_linkage.py for "
                f"decade {decade} first (needs the same *_embeddings.npy "
                "final_layer_embeddings.py produced)."
            )
        Z = np.load(linkage_path)
        n = Z.shape[0] + 1
        if len(meta) != n:
            raise ValueError(
                f"{linkage_path.name} has {n} leaves but "
                f"{args.search}_{decade}s_coords.csv has {len(meta)} rows -- "
                "these must come from the same embeddings."
            )
        decade_meta[decade] = meta
        decade_Z[decade] = Z
        decade_n[decade] = n

    pair_matches: dict[tuple[int, int], dict[int, list[tuple[int, float]]]] = {}
    for (d1, d2), matrix_path in pairs.items():
        if d1 not in decades or d2 not in decades:
            continue
        T = np.load(matrix_path)
        if T.shape[0] != decade_n[d1] or T.shape[1] != decade_n[d2]:
            print(f"WARNING: skipping {d1}-{d2}: transport matrix shape "
                  f"{T.shape} doesn't match leaf counts "
                  f"({decade_n[d1]}, {decade_n[d2]})")
            continue
        pair_matches[(d1, d2)] = get_pair_matches(T, args.top_k, args.min_mass_fraction)

    all_words = sorted(set(
        word for decade in decades for word in decade_meta[decade]["word"]
    ))
    palette = (px.colors.qualitative.Dark24 if len(all_words) <= 24
              else px.colors.qualitative.Alphabet)
    color_map = {w: palette[i % len(palette)] for i, w in enumerate(all_words)}
    word_rank = {w: i / max(1, len(all_words) - 1) for i, w in enumerate(all_words)}

    def word_target(decade: int) -> dict[int, float]:
        meta = decade_meta[decade]
        return {i: word_rank.get(meta.iloc[i]["word"], 0.5) for i in range(decade_n[decade])}

    def rebuild(decade: int, target: dict[int, float]):
        Z = decade_Z[decade]
        n = decade_n[decade]
        swap = compute_swaps(Z, n, target)
        Z_ordered = apply_swaps(Z, n, swap)
        leaf_order, segments = dendrogram_geometry(Z_ordered, n)
        y = {leaf: (rank / (n - 1) if n > 1 else 0.5)
             for rank, leaf in enumerate(leaf_order)}
        return Z_ordered, y, segments

    # --- initial layout: word-sorted, respecting each decade's own tree ---
    decade_y: dict[int, dict[int, float]] = {}
    decade_dendro: dict[int, list] = {}
    decade_Z_ordered: dict[int, np.ndarray] = {}
    for decade in decades:
        Z_ordered, y, d = rebuild(decade, word_target(decade))
        decade_y[decade] = y
        decade_dendro[decade] = d
        decade_Z_ordered[decade] = Z_ordered

    # --- alternating forward/backward untangling passes ---
    decades_sorted = sorted(decades)
    for pass_num in range(args.passes):
        forward = (pass_num % 2 == 0)
        order = decades_sorted if forward else list(reversed(decades_sorted))
        for pos, decade in enumerate(order):
            if pos == 0:
                continue
            neighbor = order[pos - 1]
            pair_key = (neighbor, decade) if forward else (decade, neighbor)
            matches = pair_matches.get(pair_key)
            if not matches:
                continue

            target = word_target(decade)
            if forward:
                incoming: dict[int, list[tuple[float, float]]] = {}
                for i, targets in matches.items():
                    y_i = decade_y[neighbor][i]
                    for j, w in targets:
                        incoming.setdefault(j, []).append((y_i, w))
                for j, contributions in incoming.items():
                    ys, ws = zip(*contributions)
                    target[j] = float(np.average(ys, weights=ws))
            else:
                for i, targets in matches.items():
                    ys = [decade_y[neighbor][j] for j, _ in targets]
                    ws = [w for _, w in targets]
                    target[i] = float(np.average(ys, weights=ws))

            Z_ordered, y, d = rebuild(decade, target)
            decade_y[decade] = y
            decade_dendro[decade] = d
            decade_Z_ordered[decade] = Z_ordered

    # --- collapse small chained subtrees into single wedge glyphs ---
    decade_collapse_points: dict[int, list[int]] = {}
    decade_leaf_to_collapse: dict[int, dict[int, int]] = {}  # leaf -> its collapse-point node id
    decade_collapse_leaves: dict[int, dict[int, list[int]]] = {}  # collapse-point -> its leaves
    decade_suppressed_nodes: dict[int, set[int]] = {}  # nodes hidden from individual drawing
    for decade in decades:
        Z_ordered = decade_Z_ordered[decade]
        n = decade_n[decade]
        threshold = 0 if args.collapse_fraction <= 0 else max(2, int(np.ceil(args.collapse_fraction * n)))
        points = find_collapse_points(Z_ordered, n, threshold)
        decade_collapse_points[decade] = points

        leaf_to_collapse: dict[int, int] = {}
        collapse_leaves: dict[int, list[int]] = {}
        suppressed_nodes: set[int] = set()
        for point in points:
            leaves = leaves_under(Z_ordered, n, point)
            collapse_leaves[point] = leaves
            for leaf in leaves:
                leaf_to_collapse[leaf] = point
            suppressed_nodes.update(nodes_under(Z_ordered, n, point))
        decade_leaf_to_collapse[decade] = leaf_to_collapse
        decade_collapse_leaves[decade] = collapse_leaves
        decade_suppressed_nodes[decade] = suppressed_nodes

        if points:
            n_collapsed_leaves = sum(len(v) for v in collapse_leaves.values())
            print(f"  {decade}: collapsed {len(points)} subtree(s) "
                  f"(threshold {threshold} leaves), hiding "
                  f"{n_collapsed_leaves}/{n} individual leaves")

    # --- build figure ---
    fig = go.Figure()

    def snippet(text, n=140):
        text = str(text)
        return text[:n] + ("..." if len(text) > n else "")

    # tree structure, drawn in neutral gray, fanning right from each
    # decade's leaf line -- segments inside a collapsed subtree are skipped
    for decade in decades:
        segments = decade_dendro[decade]
        n = decade_n[decade]
        suppressed = decade_suppressed_nodes[decade]
        max_dcoord = max((max(seg_ys) for _, _, seg_ys in segments), default=0.0) or 1.0
        xs, ys = [], []
        for node_id, seg_xs, seg_ys in segments:
            if node_id in suppressed:
                continue
            for ic, dc in zip(seg_xs, seg_ys):
                y_val = (ic - 5) / (10 * (n - 1)) if n > 1 else 0.5
                x_val = decade + (dc / max_dcoord) * args.tree_width
                xs.append(x_val)
                ys.append(y_val)
            xs.append(None)
            ys.append(None)
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines",
            line=dict(color="#999999", width=1),
            opacity=0.6, showlegend=False, hoverinfo="skip",
        ))

    # wedge glyphs for collapsed subtrees: a filled triangle from the
    # subtree's merge height (apex) to its leaf-line span (base), tinted
    # by whichever word is most common inside it
    for decade in decades:
        Z_ordered = decade_Z_ordered[decade]
        n = decade_n[decade]
        segments = decade_dendro[decade]
        max_dcoord = max((max(seg_ys) for _, _, seg_ys in segments), default=0.0) or 1.0
        meta = decade_meta[decade]
        for point, leaves in decade_collapse_leaves[decade].items():
            ys_here = [decade_y[decade][leaf] for leaf in leaves]
            y_min, y_max = min(ys_here), max(ys_here)
            height = Z_ordered[point - n, 2]
            x_apex = decade + (height / max_dcoord) * args.tree_width
            words_here = meta.iloc[leaves]["word"]
            counts = words_here.value_counts()
            dominant = counts.index[0]
            composition = ", ".join(f"{w}: {c}" for w, c in counts.items())
            fig.add_trace(go.Scatter(
                x=[decade, x_apex, decade, decade],
                y=[y_min, (y_min + y_max) / 2, y_max, y_min],
                mode="lines", fill="toself",
                fillcolor=color_map.get(dominant, "#888888"),
                opacity=0.35,
                line=dict(color=color_map.get(dominant, "#888888"), width=1),
                showlegend=False,
                hoverinfo="text",
                text=f"<b>collapsed: {len(leaves)} occurrences</b><br>"
                     f"decade: {decade}s<br>dominant: {dominant}<br>{composition}",
            ))

    # tanglegram connector lines between decades, colored by source word
    for (d1, d2), matches in pair_matches.items():
        if d1 not in decades or d2 not in decades:
            continue
        meta1 = decade_meta[d1]
        segments_by_word: dict[str, dict[str, list]] = {}
        for i, targets in matches.items():
            word = meta1.iloc[i]["word"]
            bucket = segments_by_word.setdefault(word, {"x": [], "y": []})
            y1 = decade_y[d1][i]
            for j, weight in targets:
                y2 = decade_y[d2][j]
                bucket["x"].extend([d1, d2, None])
                bucket["y"].extend([y1, y2, None])
        for word, bucket in segments_by_word.items():
            fig.add_trace(go.Scattergl(
                x=bucket["x"], y=bucket["y"], mode="lines",
                line=dict(color=color_map.get(word, "#888888"), width=1),
                opacity=0.3, showlegend=False, hoverinfo="skip",
            ))

    # leaf markers, colored by word, one trace per word for a clean legend
    # -- leaves hidden inside a collapsed wedge are skipped (the wedge
    # already represents them)
    for word in all_words:
        xs, ys, hover = [], [], []
        for decade in decades:
            meta = decade_meta[decade]
            suppressed_leaves = decade_leaf_to_collapse[decade]
            word_rows = meta[meta["word"] == word]
            for orig_idx, row in word_rows.iterrows():
                if orig_idx in suppressed_leaves:
                    continue
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
            x=xs, y=ys, mode="markers", name=word,
            marker=dict(color=color_map[word], size=6,
                       line=dict(color="white", width=0.5)),
            customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"Cross-decade tanglegram for '{args.search}'<br>"
                 f"<sup>gray = single-linkage tree structure per decade "
                 f"(fanning right, root at the far edge) -- colored lines "
                 f"= matched occurrences (top-{args.top_k}) -- leaf order "
                 f"chosen to align with neighboring decades</sup>",
            font=dict(size=15),
        ),
        xaxis=dict(title="Decade", tickmode="array", tickvals=decades,
                  ticktext=[f"{d}s" for d in decades]),
        yaxis=dict(title=None, showticklabels=False, zeroline=False),
        legend_title="Word form",
        hovermode="closest",
        template="plotly_white",
        width=max(1100, 110 * len(decades)),
        height=800,
    )

    fig.write_html(args.output, include_plotlyjs="cdn")
    total_leaves = sum(decade_n[d] for d in decades)
    print(f"\nWrote {args.output}")
    print(f"  {len(decades)} decades, {total_leaves} leaves total, "
          f"{len(pair_matches)} decade-pair(s) with connectors drawn")


if __name__ == "__main__":
    main()
