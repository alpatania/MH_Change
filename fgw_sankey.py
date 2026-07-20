"""
Cluster-level FGW sense-drift Sankey
=====================================
For a given search term, cuts each decade's single-linkage dendrogram at a
per-decade height threshold, aggregates the FGW transport matrices between
consecutive decades from leaf-level to cluster-level, and draws a Sankey
where each column is a decade, each node is a cluster (labeled by its top
lemmas and member count), and each ribbon is the mass flowing between two
clusters. Cluster colors are inherited via mass-flow continuity so that a
stable sense keeps a single color across decades, a split shows as two
same-colored branches, and a genuinely new sense gets a fresh color.

Design notes:

- The per-decade cut is chosen so that each decade sees the cluster
  structure its own ultrametric supports rather than a globally-imposed
  k. Default rule: cut at --height-fraction * max_merge_height of that
  decade's linkage (default 0.5). Safety rails --min-clusters and
  --max-clusters keep pathological cases readable without imposing a
  fixed k across decades.

- Small clusters (<--min-cluster-size leaves) are pooled per decade into
  a single "other" bucket rendered in gray. This is display-only; the
  underlying mass flow into and out of "other" is preserved.

- Cluster colors are assigned via a first-decade seeding (largest first)
  and then propagated via plurality of received mass, thresholded at
  --continuity-threshold. Ribbons take the color of their source cluster.

- Node y positions within each decade come from a barycentric heuristic
  (mass-weighted mean of parent-decade y positions), stacked by that
  order with heights proportional to member count. Reproducible run-to-
  run; near-minimum-crossings without a full optimizer.

- Interactive HTML output for hover-driven exploration. Optional
  static PDF/PNG output via Plotly's kaleido backend.

Requires (per decade) the linkage matrix, coords CSV, and consecutive-
decade transport matrices produced by earlier pipeline steps:

    <out-dir>/<search>_<decade>s_linkage.npy       (fgw_build_linkage.py)
    <out-dir>/<search>_<decade>s_coords.csv        (final_layer_embeddings.py)
    <out-dir>/<search>_fgw_<d1>_<d2>_transport_matrix.npy  (FGW_distance.py)

Usage:
    python fgw_sankey.py --out-dir results --search depression \\
        --output results/depression_sankey.html
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import fcluster


# --- Constants -------------------------------------------------------------

# Okabe-Ito qualitative palette, colorblind-safe, 8 hues.
# Widely recommended default for categorical scientific figures.
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
# A couple of extra hues from Wong's extended palette for when 8 isn't enough
EXTRA_HUES = ["#882255", "#88CCEE", "#DDCC77", "#117733", "#332288", "#AA4499"]
PALETTE = OKABE_ITO + EXTRA_HUES

OTHER_COLOR = "#BBBBBB"
OTHER_LABEL = "other"


# Filename conventions match fgw_tanglegram.py exactly.
LINKAGE_RE = re.compile(r"^(?P<search>.+)_(?P<decade>\d{4})s_linkage\.npy$")
COORDS_RE  = re.compile(r"^(?P<search>.+)_(?P<decade>\d{4})s_coords\.csv$")
PAIR_RE    = re.compile(
    r"^(?P<search>.+)_fgw_(?P<decade1>\d{4})_(?P<decade2>\d{4})_transport_matrix\.npy$"
)


# --- Data loading ----------------------------------------------------------

def discover_decades(
    out_dir: Path, search: str,
    clustering: str = "ultrametric",
    cluster_input_suffix: str = "",
) -> list[int]:
    """Find every decade for which we have coords.csv AND the clustering
    input the requested method needs (linkage.npy for ultrametric, the
    <cluster_input_suffix> file for hdbscan). Decades missing either are
    silently dropped -- they can't participate in the Sankey."""
    seen: dict[int, dict[str, bool]] = defaultdict(dict)
    if clustering == "ultrametric":
        for path in out_dir.glob(f"{search}_*_linkage.npy"):
            m = LINKAGE_RE.match(path.name)
            if m and m.group("search") == search:
                seen[int(m.group("decade"))]["cluster_input"] = True
    else:  # hdbscan (or other future methods that consume the same array)
        suffix = cluster_input_suffix
        pattern = re.compile(
            rf"^{re.escape(search)}_(?P<decade>\d{{4}})s{re.escape(suffix)}$"
        )
        for path in out_dir.glob(f"{search}_*{suffix}"):
            m = pattern.match(path.name)
            if m:
                seen[int(m.group("decade"))]["cluster_input"] = True
    for path in out_dir.glob(f"{search}_*_coords.csv"):
        m = COORDS_RE.match(path.name)
        if m and m.group("search") == search:
            seen[int(m.group("decade"))]["coords"] = True
    return sorted(d for d, kinds in seen.items()
                  if kinds.get("cluster_input") and kinds.get("coords"))


def discover_pairs(out_dir: Path, search: str) -> dict[tuple[int, int], Path]:
    """Consecutive-decade transport matrices, keyed by (d1, d2)."""
    pairs: dict[tuple[int, int], Path] = {}
    for path in out_dir.glob(f"{search}_fgw_*_*_transport_matrix.npy"):
        m = PAIR_RE.match(path.name)
        if m and m.group("search") == search:
            d1, d2 = int(m.group("decade1")), int(m.group("decade2"))
            pairs[(d1, d2)] = path
    return pairs


def load_decade(out_dir: Path, search: str, decade: int) -> tuple[np.ndarray, pd.DataFrame]:
    Z = np.load(out_dir / f"{search}_{decade}s_linkage.npy")
    meta = pd.read_csv(out_dir / f"{search}_{decade}s_coords.csv")
    return Z, meta


# --- Clustering ------------------------------------------------------------

def cut_with_rails(Z: np.ndarray, height_fraction: float,
                   min_k: int, max_k: int) -> tuple[np.ndarray, str, float]:
    """Cut the linkage matrix at height_fraction * max(merge_heights).

    If the resulting cluster count is above max_k, cut higher (fewer clusters).
    If below min_k, cut lower (more clusters). Both rails use scipy's
    'maxclust' criterion.

    Returns (labels, mode, effective_height). Labels are 1-indexed.
    Mode is one of 'natural' / 'capped-max' / 'floored-min'.
    """
    if Z.shape[0] == 0:
        # Single leaf, no merges: trivially one cluster.
        return np.array([1]), "natural", 0.0
    max_h = float(Z[:, 2].max())
    natural_t = height_fraction * max_h
    labels = fcluster(Z, t=natural_t, criterion="distance")
    k = int(labels.max())
    if k > max_k:
        labels = fcluster(Z, t=max_k, criterion="maxclust")
        return labels.astype(int), "capped-max", natural_t
    if k < min_k:
        labels = fcluster(Z, t=min_k, criterion="maxclust")
        return labels.astype(int), "floored-min", natural_t
    return labels.astype(int), "natural", natural_t


def cluster_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
) -> tuple[np.ndarray, str, set[int]]:
    """HDBSCAN on the per-decade cluster-input matrix (typically the
    UMAP-of-PCA-90 output). Returns (labels_1indexed, mode, noise_ids).

    HDBSCAN produces cluster labels 0..k-1 for real clusters and -1 for
    noise. We remap to 1-indexed positive integers and reserve one ID
    for noise so it can be routed into the 'other' bucket downstream
    regardless of noise count. That way HDBSCAN's "these don't fit any
    cluster" verdict is preserved semantically rather than showing as a
    (potentially large) cluster called 'noise'.

    Mode is 'natural' by convention (HDBSCAN has no rails to fire) or
    'trivial' when there's only one leaf.
    """
    n = X.shape[0]
    if n < 2:
        return np.array([1], dtype=int), "trivial", set()
    from sklearn.cluster import HDBSCAN

    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    raw = hdb.fit_predict(X)
    # Remap: real clusters 0..k-1 -> 1..k; noise -1 -> k+1 (reserved).
    if (raw >= 0).any():
        max_real = int(raw.max())
        noise_id = max_real + 2  # 1-indexed cluster IDs are 1..max_real+1
    else:
        # Everything is noise; treat as a single 'other' cluster.
        max_real = -1
        noise_id = 1
    labels = np.where(raw < 0, noise_id, raw + 1).astype(int)
    noise_ids = {noise_id} if (raw < 0).any() else set()
    return labels, "natural", noise_ids


def pool_small_clusters(
    labels: np.ndarray, min_size: int,
    force_other_ids: set[int] | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Merge every cluster with fewer than min_size members into a single
    pseudo-cluster labeled OTHER_LABEL. Returns (relabeled, id_to_name)
    where the new labels are 1..K' with K' <= K and one of them may be
    the 'other' bucket.

    force_other_ids: cluster IDs (in the original label space) that are
    always pooled into 'other' regardless of their size. Used to route
    HDBSCAN's noise label (-1, remapped to a positive ID before this call)
    into the 'other' bucket even when noise points are numerous.
    """
    force_other_ids = force_other_ids or set()
    sizes = Counter(labels.tolist())
    keep_ids = [c for c, n in sizes.items()
                if n >= min_size and c not in force_other_ids]
    other_ids = [c for c, n in sizes.items()
                 if n < min_size or c in force_other_ids]

    # Relabel: keepers get new sequential IDs (largest first for stable ordering),
    # small clusters all merge into one ID.
    keep_ids_sorted = sorted(keep_ids, key=lambda c: -sizes[c])
    remap: dict[int, int] = {}
    id_to_name: dict[int, str] = {}
    for new_id, old_id in enumerate(keep_ids_sorted, start=1):
        remap[old_id] = new_id
        id_to_name[new_id] = f"c{new_id}"
    if other_ids:
        other_new_id = len(keep_ids_sorted) + 1
        for old_id in other_ids:
            remap[old_id] = other_new_id
        id_to_name[other_new_id] = OTHER_LABEL

    new_labels = np.array([remap[c] for c in labels], dtype=int)
    return new_labels, id_to_name


# --- Cluster-level mass flow -----------------------------------------------

def cluster_flow(T: np.ndarray, labels1: np.ndarray, labels2: np.ndarray
                 ) -> np.ndarray:
    """Aggregate the leaf-level transport matrix T (n1 x n2) into a
    cluster-level flow matrix (k1 x k2). Vectorized: constructs the two
    one-hot matrices and multiplies. Faster than np.add.at for the sizes
    we care about (~500 leaves per decade) and clearer to read.
    """
    n1, n2 = T.shape
    if labels1.shape[0] != n1 or labels2.shape[0] != n2:
        raise ValueError(
            f"Transport matrix shape {T.shape} inconsistent with label counts "
            f"({labels1.shape[0]}, {labels2.shape[0]})."
        )
    k1, k2 = int(labels1.max()), int(labels2.max())
    L1 = np.zeros((n1, k1))
    L1[np.arange(n1), labels1 - 1] = 1.0
    L2 = np.zeros((n2, k2))
    L2[np.arange(n2), labels2 - 1] = 1.0
    return L1.T @ T @ L2  # shape (k1, k2)


# --- Color assignment ------------------------------------------------------

def assign_colors_by_continuity(
    flows: dict[tuple[int, int], np.ndarray],
    decades: list[int],
    id_to_name_by_decade: dict[int, dict[int, str]],
    sizes_by_decade: dict[int, dict[int, int]],
    continuity_threshold: float,
) -> dict[tuple[int, int], str]:
    """Return {(decade, cluster_id) -> color}. First decade: colors by
    size rank. Later decades: inherit from the parent-decade cluster
    that contributed the plurality of received mass, unless that
    plurality is below continuity_threshold OR the parent was 'other',
    in which case a fresh palette color is drawn."""
    colors: dict[tuple[int, int], str] = {}
    next_palette_idx = 0

    # First decade: rank clusters by size (largest first), take palette in order.
    d0 = decades[0]
    ordered = sorted(sizes_by_decade[d0].items(), key=lambda kv: -kv[1])
    for cid, _ in ordered:
        name = id_to_name_by_decade[d0].get(cid, "")
        if name == OTHER_LABEL:
            colors[(d0, cid)] = OTHER_COLOR
        else:
            colors[(d0, cid)] = PALETTE[next_palette_idx % len(PALETTE)]
            next_palette_idx += 1

    # Subsequent decades: inherit or draw fresh.
    for d_prev, d_curr in zip(decades, decades[1:]):
        flow = flows.get((d_prev, d_curr))
        if flow is None:
            # No transport matrix available for this consecutive pair.
            # Fall back to fresh colors for every cluster in d_curr.
            for cid, name in id_to_name_by_decade[d_curr].items():
                if name == OTHER_LABEL:
                    colors[(d_curr, cid)] = OTHER_COLOR
                else:
                    colors[(d_curr, cid)] = PALETTE[next_palette_idx % len(PALETTE)]
                    next_palette_idx += 1
            continue

        k_prev, k_curr = flow.shape
        for c_curr in range(1, k_curr + 1):
            name_curr = id_to_name_by_decade[d_curr].get(c_curr, "")
            if name_curr == OTHER_LABEL:
                colors[(d_curr, c_curr)] = OTHER_COLOR
                continue
            received = flow[:, c_curr - 1]
            total = float(received.sum())
            if total <= 0:
                colors[(d_curr, c_curr)] = PALETTE[next_palette_idx % len(PALETTE)]
                next_palette_idx += 1
                continue
            top_prev_idx = int(np.argmax(received))
            top_prev_cid = top_prev_idx + 1
            plurality = received[top_prev_idx] / total
            parent_name = id_to_name_by_decade[d_prev].get(top_prev_cid, "")
            if plurality >= continuity_threshold and parent_name != OTHER_LABEL:
                colors[(d_curr, c_curr)] = colors[(d_prev, top_prev_cid)]
            else:
                colors[(d_curr, c_curr)] = PALETTE[next_palette_idx % len(PALETTE)]
                next_palette_idx += 1
    return colors


# --- Node layout -----------------------------------------------------------

def barycentric_y_order(
    flows: dict[tuple[int, int], np.ndarray],
    decades: list[int],
    id_to_name_by_decade: dict[int, dict[int, str]],
    sizes_by_decade: dict[int, dict[int, int]],
) -> dict[int, list[int]]:
    """For each decade, return the ordered list of cluster IDs (top of
    column first). Barycentric: cluster_c_y_score = weighted mean of
    parent-decade y-positions. First decade: largest first."""
    order: dict[int, list[int]] = {}
    # First decade: rank by size, largest first, but push 'other' to the bottom.
    d0 = decades[0]
    def _first_decade_key(cid: int) -> tuple[int, float]:
        name = id_to_name_by_decade[d0].get(cid, "")
        # (0, -size) for named clusters, (1, -size) for 'other' -> other last
        return (1 if name == OTHER_LABEL else 0, -sizes_by_decade[d0][cid])
    order[d0] = sorted(sizes_by_decade[d0].keys(), key=_first_decade_key)

    for d_prev, d_curr in zip(decades, decades[1:]):
        prev_positions = {cid: rank for rank, cid in enumerate(order[d_prev])}
        curr_ids = list(sizes_by_decade[d_curr].keys())
        flow = flows.get((d_prev, d_curr))
        scores: dict[int, float] = {}
        for cid in curr_ids:
            name = id_to_name_by_decade[d_curr].get(cid, "")
            if flow is None:
                scores[cid] = -float(sizes_by_decade[d_curr][cid])
                continue
            received = flow[:, cid - 1]
            total = float(received.sum())
            if total <= 0:
                # Push clusters with no incoming mass to the bottom.
                scores[cid] = float(len(prev_positions))
                continue
            weighted_sum = sum(
                prev_positions[i + 1] * received[i] for i in range(flow.shape[0])
            )
            scores[cid] = weighted_sum / total
            # 'other' pushed to bottom of column
            if name == OTHER_LABEL:
                scores[cid] += float(len(prev_positions))
        order[d_curr] = sorted(curr_ids, key=lambda c: (scores[c], -sizes_by_decade[d_curr][c]))
    return order


def color_y_order(
    decades: list[int],
    barycentric_order: dict[int, list[int]],
    colors: dict[tuple[int, int], str],
    lemma_to_color: dict[str, str],
) -> dict[int, list[int]]:
    """Order clusters lexicographically by (color_priority, barycentric_rank).

    Concretely: take the barycentric order as a reference, then STABLE-sort
    each column by color priority. Python's sorted() is stable, so
    within-color relative order from barycentric is preserved -- which
    means ribbon crossings within each color band are still minimized
    subject to the color-grouping constraint.

    Color priority follows the legend order (lemma_to_color insertion
    order, which is ranked by total cluster mass). OTHER_COLOR sorts last.
    """
    color_priority: dict[str, int] = {}
    for i, (_lemma, color) in enumerate(lemma_to_color.items()):
        color_priority.setdefault(color, i)
    other_priority = len(color_priority)
    color_priority[OTHER_COLOR] = other_priority

    order: dict[int, list[int]] = {}
    for decade in decades:
        cids = barycentric_order[decade]  # already sorted by barycentric score
        # sorted() is stable -> within-color barycentric rank is preserved
        order[decade] = sorted(
            cids,
            key=lambda cid: color_priority.get(
                colors.get((decade, cid), OTHER_COLOR),
                other_priority,
            ),
        )
    return order


# --- Node labels and hover -------------------------------------------------

def _clean_lemma(v) -> str:
    """Coerce a single value to a clean lowercase string. Handles None,
    NaN (both numpy float NaN and pandas' NA), integers, and everything
    else pandas might smuggle into a str column. Returns '' for missing
    values; empty strings are filtered before Counter aggregation so
    they don't appear as spurious cluster labels."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    try:
        # pd.NA raises TypeError on bool(); catch that too.
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).casefold()


def lemma_column(meta: pd.DataFrame) -> pd.Series:
    """Return a Series of best-effort canonical lemma per row, casefolded.
    Prefers a 'lemma' or 'match_lemma_1' column if present; falls back to
    'word' (also casefolded). Robust to NaN, mixed dtypes, and the pandas
    pyarrow string backend (which preserves NaN through .astype(str)).
    """
    for col in ("lemma", "match_lemma_1", "word"):
        if col in meta.columns:
            return meta[col].map(_clean_lemma)
    return pd.Series([""] * len(meta), index=meta.index)


def _lemma_counter(meta: pd.DataFrame) -> Counter:
    """Counter of lemma occurrences, dropping empty strings. Guarantees
    all keys are non-empty strings so downstream .join() calls are safe.
    """
    return Counter(v for v in lemma_column(meta) if v)


def _dominant_lemma(meta: pd.DataFrame, threshold: float) -> str | None:
    """Return the most-common lemma in `meta` if it makes up at least
    `threshold` of the non-empty-lemma count; otherwise None. `None`
    signals 'no clearly dominant lemma' and is used downstream to route
    the cluster to the gray 'other' color even if it has ordinary size.
    """
    counter = _lemma_counter(meta)
    if not counter:
        return None
    top_lemma, top_count = counter.most_common(1)[0]
    total = sum(counter.values())
    return top_lemma if (top_count / total) >= threshold else None


def assign_colors_by_lemma(
    decades: list[int],
    labels_by_decade: dict[int, np.ndarray],
    meta_by_decade: dict[int, pd.DataFrame],
    id_to_name_by_decade: dict[int, dict[int, str]],
    sizes_by_decade: dict[int, dict[int, int]],
    dominance_threshold: float,
    max_legend_colors: int,
) -> tuple[dict[tuple[int, int], str], dict[str, str], dict[str, int]]:
    """Color each cluster by the palette color of its dominant lemma.

    Returns (colors, lemma_to_color, lemma_totals) where:
      colors           : (decade, cid) -> hex color (palette hue or OTHER_COLOR)
      lemma_to_color   : lemma -> hex color, ordered by total mass descending
                          (only lemmas that made the palette cut are keys)
      lemma_totals     : lemma -> summed cluster mass across all decades
                          (all lemmas that had at least one dominant cluster
                          somewhere, including those routed to 'other')

    Ordering rule: rank lemmas by their total summed cluster mass across
    all decades; assign the top `max_legend_colors` (bounded by palette
    size) distinct palette hues to them; any remaining lemmas — plus the
    OTHER_LABEL bucket, plus any cluster whose dominant lemma didn't
    clear `dominance_threshold` — get OTHER_COLOR.
    """
    dominant_by_cluster: dict[tuple[int, int], str | None] = {}
    for decade in decades:
        meta = meta_by_decade[decade]
        labels = labels_by_decade[decade]
        for cid, name in id_to_name_by_decade[decade].items():
            if name == OTHER_LABEL:
                dominant_by_cluster[(decade, cid)] = None
                continue
            members = meta.iloc[np.where(labels == cid)[0]]
            dominant_by_cluster[(decade, cid)] = _dominant_lemma(
                members, dominance_threshold,
            )

    # Rank distinct dominant lemmas by summed cluster mass. Tie-break
    # alphabetically so runs are deterministic.
    lemma_totals: dict[str, int] = defaultdict(int)
    for (decade, cid), lemma in dominant_by_cluster.items():
        if lemma is not None:
            lemma_totals[lemma] += sizes_by_decade[decade][cid]
    ranked = sorted(lemma_totals.items(), key=lambda kv: (-kv[1], kv[0]))
    palette_cap = min(max_legend_colors, len(PALETTE))
    top_lemmas = [l for l, _ in ranked[:palette_cap]]
    lemma_to_color = {l: PALETTE[i] for i, l in enumerate(top_lemmas)}

    colors: dict[tuple[int, int], str] = {}
    for (decade, cid), lemma in dominant_by_cluster.items():
        if lemma is not None and lemma in lemma_to_color:
            colors[(decade, cid)] = lemma_to_color[lemma]
        else:
            colors[(decade, cid)] = OTHER_COLOR
    return colors, lemma_to_color, dict(lemma_totals)


def snippet(text: str, n: int = 140) -> str:
    text = str(text)
    return text[:n] + ("…" if len(text) > n else "")


def cluster_label(
    decade: int, cid: int, name: str,
    members: pd.DataFrame,
    label_mode: str = "full",
) -> tuple[str, str]:
    """Return (short_label, hover_html) for a single cluster node.

    label_mode:
      'none' -> short is an empty string (nothing renders on the node bar)
      'size' -> just the member count, e.g. 'n=239'
      'full' -> 'c1: queer/queerly (n=239)'  (previous behavior)
    Hover is always the full rich version regardless of label_mode.
    """
    n = len(members)
    if name == OTHER_LABEL:
        # Aggregate label for the small-cluster bucket.
        top_lemmas = _lemma_counter(members).most_common(4)
        top_str = ", ".join(f"{w}" for w, _ in top_lemmas) if top_lemmas else "-"
        if label_mode == "none":
            short = ""
        elif label_mode == "size":
            short = f"n={n}"
        else:
            short = f"other (n={n})"
        hover = (f"<b>other bucket</b><br>decade: {decade}s<br>"
                 f"n={n}<br>top lemmas: {top_str}")
        return short, hover

    lemma_counts = _lemma_counter(members)
    top = lemma_counts.most_common(3)
    top_words_short = "/".join(w for w, _ in top[:2]) if top else "?"
    # Cluster ID prefix so that when HDBSCAN finds two sub-senses with the
    # same dominant lemma (e.g. two 'queer' sub-clusters), they remain
    # distinguishable at a glance instead of showing identical labels.
    if label_mode == "none":
        short = ""
    elif label_mode == "size":
        short = f"n={n}"
    else:
        short = f"c{cid}: {top_words_short} (n={n})"

    # Rich hover: top-5 lemmas with counts + up to 3 passage snippets.
    top5 = lemma_counts.most_common(5)
    top_str = "<br>".join(f"  {w}: {c}" for w, c in top5)
    passages = []
    if "passage" in members.columns:
        for _, row in members.head(3).iterrows():
            passages.append(snippet(str(row["passage"])))
    passages_str = ("<br><br><i>example passages:</i><br>" + "<br>".join(
        f"• {p}" for p in passages)) if passages else ""
    hover = (f"<b>cluster {cid} ({decade}s)</b><br>n={n}<br>"
             f"<i>top lemmas:</i><br>{top_str}{passages_str}")
    return short, hover


# --- Build the figure ------------------------------------------------------

def build_sankey(
    decades: list[int],
    labels_by_decade: dict[int, np.ndarray],
    id_to_name_by_decade: dict[int, dict[int, str]],
    meta_by_decade: dict[int, pd.DataFrame],
    sizes_by_decade: dict[int, dict[int, int]],
    flows: dict[tuple[int, int], np.ndarray],
    colors: dict[tuple[int, int], str],
    order: dict[int, list[int]],
    cluster_modes: dict[int, str],
    search: str,
    clustering: str,
    height_fraction: float,
    continuity_threshold: float,
    min_link_fraction: float,
    min_label_fraction: float,
    label_angle: float,
    decade_labels: str,
    label_mode: str,
    coloring: str,
    lemma_to_color: dict[str, str],
    width: int,
    height: int,
) -> go.Figure:
    """Assemble the Plotly Sankey. Nodes are (decade, cluster_id) pairs
    flattened into a single index list; links are (source_idx, target_idx,
    value)."""

    node_records: list[dict] = []
    node_index: dict[tuple[int, int], int] = {}
    for decade in decades:
        for cid in order[decade]:
            members = meta_by_decade[decade].iloc[
                np.where(labels_by_decade[decade] == cid)[0]
            ]
            name = id_to_name_by_decade[decade][cid]
            short, hover = cluster_label(decade, cid, name, members, label_mode=label_mode)
            node_records.append({
                "decade": decade,
                "cid": cid,
                "label": short,
                "color": colors[(decade, cid)],
                "hover": hover,
                "size": sizes_by_decade[decade][cid],
            })
            node_index[(decade, cid)] = len(node_records) - 1

    # Label suppression: for each decade column, blank out visible labels
    # for clusters whose share of the column mass is below
    # min_label_fraction. The cluster still renders as a colored bar and
    # the full label is available on hover; this just prevents dense-column
    # decades (HDBSCAN can produce 10+ clusters in one decade) from
    # rendering as a stack of overlapping text.
    n_labels_hidden = 0
    if min_label_fraction > 0:
        for decade in decades:
            col_total = sum(sizes_by_decade[decade].values())
            if col_total <= 0:
                continue
            for cid in id_to_name_by_decade[decade]:
                share = sizes_by_decade[decade][cid] / col_total
                if share < min_label_fraction:
                    node_records[node_index[(decade, cid)]]["label"] = ""
                    n_labels_hidden += 1
    if n_labels_hidden:
        print(f"  Suppressed {n_labels_hidden} label(s) below "
              f"{min_label_fraction:.0%} of column mass (still visible on hover).")

    # Node x positions: evenly spaced across the horizontal axis, one column
    # per decade. Clamp to (epsilon, 1-epsilon) because Plotly Sankey pushes
    # nodes at exactly 0 or 1 into the plot margins.
    eps = 0.001
    if len(decades) > 1:
        x_map = {d: eps + (1 - 2 * eps) * i / (len(decades) - 1)
                 for i, d in enumerate(decades)}
    else:
        x_map = {decades[0]: 0.5}

    # Node y positions: within each decade column, stack in `order`. Node
    # heights are computed FROM FLOW (max of incoming and outgoing mass),
    # matching what Plotly Sankey renders internally. Sizing by member
    # count instead of flow would leave rendered-height mismatches: a
    # cluster whose FGW flow is smaller than its member share ends up with
    # less rendered height than allocated, so adjacent clusters' rendered
    # heights extend into the resulting gap and appear to overlap. This is
    # the specific fix for the "orange ribbons emerging from behind a blue
    # cluster" artifact -- an orange cluster with more flow than my
    # size-based slot allowed was extending into the blue cluster's slot.
    decade_to_index = {d: i for i, d in enumerate(decades)}
    flow_heights: dict[tuple[int, int], float] = {}
    for decade in decades:
        idx = decade_to_index[decade]
        for cid in id_to_name_by_decade[decade]:
            out_sum = 0.0
            in_sum = 0.0
            if idx < len(decades) - 1:
                nxt = decades[idx + 1]
                flow = flows.get((decade, nxt))
                if flow is not None and cid - 1 < flow.shape[0]:
                    out_sum = float(flow[cid - 1].sum())
            if idx > 0:
                prv = decades[idx - 1]
                flow = flows.get((prv, decade))
                if flow is not None and cid - 1 < flow.shape[1]:
                    in_sum = float(flow[:, cid - 1].sum())
            flow_heights[(decade, cid)] = max(out_sum, in_sum)

    # Increase the between-nodes gap from 2% total to about 4% (roughly
    # matching Plotly's default node.pad of 15px at height=700, which is
    # ~2.5% per gap). Split evenly across all gaps in the column.
    node_y = [0.0] * len(node_records)
    for decade in decades:
        cids = order[decade]
        # Floor to keep zero-flow clusters from collapsing to zero share
        # (which would place them at the same y as their neighbor). We use
        # a very small floor so they don't visually distort the layout.
        floor = 1e-4
        raw = [max(floor, flow_heights[(decade, cid)]) for cid in cids]
        total = sum(raw)
        # Total inter-node gap in paper coords: ~4% distributed across gaps.
        gap_total = 0.04 if len(cids) > 1 else 0.0
        avail = max(0.0, 1.0 - gap_total)
        per_gap = gap_total / (len(cids) - 1) if len(cids) > 1 else 0.0
        cursor = 0.0
        for cid, r in zip(cids, raw):
            share = r / total if total else 0.0
            h = share * avail
            node_y[node_index[(decade, cid)]] = cursor + h / 2
            cursor += h + per_gap

    node_x = [x_map[r["decade"]] for r in node_records]

    # Links: (source, target, value, color, hover)
    link_src: list[int] = []
    link_tgt: list[int] = []
    link_val: list[float] = []
    link_color: list[str] = []
    link_hover: list[str] = []

    n_links_dropped = 0
    for (d1, d2), flow in flows.items():
        k1, k2 = flow.shape
        # Outflow normalization (for percent-of-source hover)
        row_totals = flow.sum(axis=1)
        col_totals = flow.sum(axis=0)
        # Per-pair link cutoff: drop ribbons carrying less than
        # min_link_fraction of the largest ribbon in THIS pair. Per-pair,
        # not global, because FGW normalizes each pair's total mass to
        # mass_fraction (default 0.8); a fraction-of-max threshold is the
        # scale-invariant way to say "trivial ribbon".
        pair_cutoff = min_link_fraction * flow.max() if min_link_fraction > 0 else 0.0
        for i in range(k1):
            for j in range(k2):
                v = flow[i, j]
                if v <= 0 or v < pair_cutoff:
                    if v > 0 and v < pair_cutoff:
                        n_links_dropped += 1
                    continue
                src_cid = i + 1
                tgt_cid = j + 1
                src_idx = node_index.get((d1, src_cid))
                tgt_idx = node_index.get((d2, tgt_cid))
                if src_idx is None or tgt_idx is None:
                    continue
                link_src.append(src_idx)
                link_tgt.append(tgt_idx)
                link_val.append(float(v))
                # Ribbon color: source cluster's color, with some transparency
                # applied via rgba conversion below at add_trace time.
                link_color.append(_hex_to_rgba(colors[(d1, src_cid)], 0.45))
                pct_of_src = (v / row_totals[i]) * 100 if row_totals[i] else 0
                pct_of_tgt = (v / col_totals[j]) * 100 if col_totals[j] else 0
                src_name = id_to_name_by_decade[d1][src_cid]
                tgt_name = id_to_name_by_decade[d2][tgt_cid]
                link_hover.append(
                    f"<b>{d1}s c{src_cid} ({src_name}) → "
                    f"{d2}s c{tgt_cid} ({tgt_name})</b><br>"
                    f"mass: {v:.4f}<br>"
                    f"{pct_of_src:.1f}% of {d1}s cluster's outflow<br>"
                    f"{pct_of_tgt:.1f}% of {d2}s cluster's inflow"
                )
    if n_links_dropped:
        print(f"  Dropped {n_links_dropped} ribbon(s) below "
              f"{min_link_fraction:.0%} of their pair's max flow.")

    node_labels = [r["label"] for r in node_records]
    node_colors = [r["color"] for r in node_records]
    node_hover = [r["hover"] for r in node_records]

    # If label_angle is nonzero, render labels as annotations (which support
    # textangle) rather than as node labels (which don't). Node labels are
    # blanked out so they don't render twice. Suppressed labels (empty
    # string from --min-label-fraction) get no annotation.
    label_annotations: list[dict] = []
    if label_angle != 0:
        for i, r in enumerate(node_records):
            if not r["label"]:
                continue
            label_annotations.append(dict(
                x=node_x[i], y=node_y[i],
                xref="paper", yref="paper",
                text=r["label"],
                textangle=label_angle,
                showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(size=11),
                # Nudge above the node bar's y-center so the rotated text
                # column doesn't overlap the node/ribbon.
            ))
        node_labels_out = [""] * len(node_records)
    else:
        node_labels_out = node_labels

    # Note on any capped/floored decades in the subtitle.
    capped = [str(d) for d in decades if cluster_modes.get(d) == "capped-max"]
    floored = [str(d) for d in decades if cluster_modes.get(d) == "floored-min"]
    if clustering == "ultrametric":
        subtitle_bits = [
            f"single-linkage ultrametric, cut at {height_fraction:.0%} of "
            f"each decade's max merge height",
        ]
    else:  # hdbscan
        subtitle_bits = [
            "HDBSCAN on per-decade UMAP embeddings connected by FGW transport pairings",
        ]
    if coloring == "continuity":
        subtitle_bits.append(
            f"continuity coloring (threshold {continuity_threshold:.0%})"
        )
    else:
        subtitle_bits.append("colored by dominant lemma per cluster")
    if capped:
        subtitle_bits.append(f"capped at max: {', '.join(capped)}s")
    if floored:
        subtitle_bits.append(f"floored at min: {', '.join(floored)}s")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        domain=dict(x=[0, 1], y=[0.375, 1.0]), 
        node=dict(
            pad=1, thickness=5,
            line=dict(color="#333333", width=0.5),
            label=node_labels_out,
            color=node_colors,
            x=node_x, y=[v/2 for v in node_y],
            customdata=node_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=link_src, target=link_tgt, value=link_val,
            color=link_color,
            customdata=link_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    )])

    # x-axis decade tick labels: Plotly Sankey doesn't have a real x-axis,
    # so we add them as annotations. Position is caller-controlled because
    # with vertical node labels (--label-angle -90), labels extend above
    # each column and collide with top-placed decade labels -- so 'bottom'
    # is often the better choice in that case.
    annotations = list(label_annotations)
    top_y = 1.16
    bottom_y = -0.06
    if decade_labels in ("top", "both"):
        for decade in decades:
            annotations.append(dict(
                x=x_map[decade]-0.02, y=top_y, xref="paper", yref="paper",
                text=f"<b>{decade}s</b>", showarrow=False,
                font=dict(size=12),
            ))
    if decade_labels in ("bottom", "both"):
        for decade in decades:
            annotations.append(dict(
                x=x_map[decade]-0.02, y=bottom_y, xref="paper", yref="paper",
                text=f"<b>{decade}s</b>", showarrow=False,
                font=dict(size=12),
            ))

    # Adjust bottom margin to give room for bottom-placed decade labels
    # without cropping them. Top margin already has room from the title.
    bottom_margin = 30 if decade_labels in ("top", "none") else 50

    # Legend traces: one invisible Scatter per lemma (plus one for 'other')
    # so Plotly's built-in legend renders on the right. This is the
    # standard Sankey-legend workaround since go.Sankey ignores showlegend
    # on its own trace and won't participate in the legend.
    right_margin = 30
    show_legend = coloring == "lemma" and bool(lemma_to_color)
    if show_legend:
        for lemma, color in lemma_to_color.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=lemma, showlegend=True, hoverinfo="skip",
            ))
        # 'other' entry describes both size-pooled clusters and clusters
        # whose dominant lemma didn't clear the threshold or wasn't in the
        # top-N by mass.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=OTHER_COLOR, symbol="square"),
            name="other", showlegend=True, hoverinfo="skip",
        ))
        # Room for the legend on the right; longest lemma name determines width.
        longest = max((len(k) for k in list(lemma_to_color) + ["other"]), default=6)
        # ~7px per char at 11pt + padding
        right_margin = min(260, 40 + 7 * longest)

    legend_dict = dict(
        x=1.01, y=0.5, xanchor="left", yanchor="middle",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#CCCCCC", borderwidth=0.5,
        font=dict(size=11),
        title=dict(text="dominant lemma", font=dict(size=11)),
    )

    fig.update_layout(
        title=dict(
            text=(f"Sense-drift Sankey for '{search}'<br>"
                  f"<sup>{' · '.join(subtitle_bits)}</sup>"),
            font=dict(size=14),
        ),
        autosize=False,
        width=width, height=height,
        font=dict(family="Helvetica, Arial, sans-serif", size=11),
        margin=dict(t=40, b=bottom_margin, l=30, r=right_margin),
        annotations=annotations,
        template="plotly_white",
        showlegend=show_legend,
        legend=legend_dict,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    fig.update_yaxes(automargin='bottom')
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' -> 'rgba(r, g, b, alpha)'. Passthrough if the
    string already looks like an rgba(...)."""
    if hex_color.startswith("rgba"):
        return hex_color
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(128,128,128,{alpha})"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --- CLI ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster-level FGW sense-drift Sankey.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory holding linkage.npy, coords.csv, and "
                        "transport_matrix.npy files "
                        "(default: results/<search>/).")
    p.add_argument("--search", required=True,
                   help="Search term prefix for input files.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output HTML path "
                        "(default: <out-dir>/<search>_sankey.html).")
    p.add_argument("--output-pdf", type=Path, default=None,
                   help="Optional static PDF via Plotly's kaleido backend.")
    p.add_argument("--output-png", type=Path, default=None,
                   help="Optional static PNG via Plotly's kaleido backend.")
    p.add_argument("--output-svg", type=Path, default=None,
                   help="Optional static SVG via Plotly's kaleido backend. "
                        "Recommended for paper figures -- vector format, "
                        "scales cleanly at any resolution.")

    p.add_argument("--clustering", default="ultrametric",
                   choices=["ultrametric", "hdbscan"],
                   help="Clustering method. 'ultrametric' (default) cuts "
                        "each decade's single-linkage linkage.npy at a "
                        "per-decade height; this matches what FGW's "
                        "structure_cost compares. 'hdbscan' runs sklearn's "
                        "HDBSCAN on a per-decade embedding array (default "
                        "the UMAP-of-PCA-90 output). Density-based, so it "
                        "does not chain; unlike ultrametric it does not "
                        "correspond exactly to what FGW compares -- see "
                        "the figure subtitle.")
    p.add_argument("--cluster-input-suffix", default="_pca90_umap.npy",
                   help="For --clustering hdbscan: filename suffix (after "
                        "'<search>_<decade>s') of the per-decade array to "
                        "cluster. Default '_pca90_umap.npy'. Ignored when "
                        "--clustering ultrametric.")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=None,
                   help="HDBSCAN's own min_cluster_size (defaults to "
                        "--min-cluster-size when unset). This is the "
                        "algorithmic threshold; the Sankey's separate "
                        "--min-cluster-size runs as a downstream pooler.")
    p.add_argument("--hdbscan-min-samples", type=int, default=None,
                   help="HDBSCAN's min_samples parameter (default: unset, "
                        "which delegates to HDBSCAN's own default).")

    p.add_argument("--height-fraction", type=float, default=0.5,
                   help="Cut each decade's linkage at this fraction of its "
                        "own max merge height (default 0.5).")
    p.add_argument("--min-clusters", type=int, default=1,
                   help="Safety rail: minimum clusters per decade (default 1).")
    p.add_argument("--max-clusters", type=int, default=10,
                   help="Safety rail: maximum clusters per decade (default 10).")
    p.add_argument("--min-cluster-size", type=int, default=5,
                   help="Clusters below this size are pooled into 'other' "
                        "per decade (default 5).")
    p.add_argument("--continuity-threshold", type=float, default=0.15,
                   help="Fraction of a cluster's inflow that must come from "
                        "one parent for it to inherit that parent's color "
                        "(default 0.15).")
    p.add_argument("--coloring", default="lemma",
                   choices=["lemma", "continuity"],
                   help="How to color clusters. 'lemma' (default): each "
                        "cluster is colored by its dominant lemma; clusters "
                        "with no clear dominant lemma (below "
                        "--dominance-threshold) fall back to gray 'other'. "
                        "A legend on the right names the colors. "
                        "'continuity': each cluster inherits the color of "
                        "the previous-decade parent that sent it the most "
                        "mass (the previous behavior); no legend since "
                        "colors track lineage rather than a nameable "
                        "quantity.")
    p.add_argument("--dominance-threshold", type=float, default=0.5,
                   help="Fraction of a cluster's non-empty lemma count that "
                        "the top lemma must reach for the cluster to be "
                        "colored by that lemma. Below this, the cluster is "
                        "rendered gray as 'other'. Default 0.5 (majority).")
    p.add_argument("--node-labels", default="none",
                   choices=["none", "size", "full"],
                   help="Node label mode: 'none' (default): no text on the "
                        "node bar; identity is carried entirely by color and "
                        "hover. 'size': show only member count e.g. 'n=239'. "
                        "'full': show 'c1: queer/queerly (n=239)' (previous "
                        "behavior). Hover always shows the full breakdown.")
    p.add_argument("--max-legend-colors", type=int, default=None,
                   help="Cap the number of distinct lemma-colors in the "
                        "legend (default: use full palette, currently 14). "
                        "Lemmas ranked below this threshold by total mass "
                        "are folded into the gray 'other' category.")

    p.add_argument("--min-link-fraction", type=float, default=0.02,
                   help="Drop ribbons below this fraction of the max flow "
                        "for their consecutive-decade pair (default 0.02). "
                        "Set to 0 to keep every non-zero ribbon.")
    p.add_argument("--min-label-fraction", type=float, default=0.03,
                   help="Suppress the visible label of any cluster below "
                        "this fraction of its decade's total mass (default "
                        "0.03). The cluster still appears as a colored node "
                        "and its full label is available on hover; this "
                        "just prevents overlap in busy decades. Set to 0 "
                        "to always show all labels.")
    p.add_argument("--label-angle", type=float, default=0,
                   help="Rotate node labels by this angle in degrees "
                        "(default 0 = horizontal). Try -90 for vertical "
                        "labels reading bottom-to-top -- useful when there "
                        "are many decades in a narrow width and horizontal "
                        "labels bleed into adjacent columns. Note: uses "
                        "Plotly annotations because go.Sankey has no "
                        "native textangle; hover still works normally.")
    p.add_argument("--decade-labels", default="top",
                   choices=["top", "bottom", "both", "none"],
                   help="Where to place the '<decade>s' column labels "
                        "(default top). Use 'bottom' when --label-angle is "
                        "-90, since vertical node labels extend upward from "
                        "each column and collide with top-placed decade "
                        "labels.")
    p.add_argument("--order-by", default="barycentric",
                   choices=["barycentric", "color"],
                   help="How to order clusters vertically within each "
                        "decade column. 'barycentric' (default): place each "
                        "cluster near the weighted average of its parent-"
                        "decade positions; minimizes ribbon crossings. "
                        "'color': lexicographic (color_priority, "
                        "barycentric_rank) -- groups clusters into coherent "
                        "color bands, and within each band preserves the "
                        "crossings-minimizing barycentric order. Only "
                        "meaningful with --coloring lemma since "
                        "'continuity' colors don't have a natural priority "
                        "order.")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=700)
    return p.parse_args()


def main():
    args = parse_args()
    # Resolve search-derived defaults: out-dir -> results/<search>/,
    # output -> <out-dir>/<search>_sankey.html.
    search_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", args.search)
    out_dir = args.out_dir if args.out_dir else Path("results") / search_slug
    args.out_dir = out_dir
    if args.output is None:
        args.output = out_dir / f"{args.search}_sankey.html"

    if not out_dir.is_dir():
        raise SystemExit(f"ERROR: {out_dir} is not a directory")

    decades = discover_decades(
        out_dir, args.search,
        clustering=args.clustering,
        cluster_input_suffix=args.cluster_input_suffix,
    )
    if not decades:
        raise SystemExit(
            f"ERROR: no complete decade found for '{args.search}' in {out_dir}."
        )
    print(f"Clustering: {args.clustering}")
    print(f"Decades found: {decades}")

    pair_paths = discover_pairs(out_dir, args.search)
    print(f"Consecutive-pair transports found: {len(pair_paths)}")

    hdb_min_cluster_size = (
        args.hdbscan_min_cluster_size
        if args.hdbscan_min_cluster_size is not None
        else args.min_cluster_size
    )

    labels_by_decade: dict[int, np.ndarray] = {}
    id_to_name_by_decade: dict[int, dict[int, str]] = {}
    meta_by_decade: dict[int, pd.DataFrame] = {}
    sizes_by_decade: dict[int, dict[int, int]] = {}
    cluster_modes: dict[int, str] = {}

    for decade in decades:
        meta = pd.read_csv(out_dir / f"{args.search}_{decade}s_coords.csv")
        n_leaves = len(meta)

        if args.clustering == "ultrametric":
            Z = np.load(out_dir / f"{args.search}_{decade}s_linkage.npy")
            expected_leaves = Z.shape[0] + 1 if Z.shape[0] > 0 else 1
            if n_leaves != expected_leaves:
                print(f"  WARNING: decade {decade}: coords has {n_leaves} "
                      f"rows but linkage implies {expected_leaves}; using min.")
                n_leaves = min(n_leaves, expected_leaves)
                meta = meta.head(n_leaves).reset_index(drop=True)
            raw_labels, mode, _t = cut_with_rails(
                Z if Z.shape[0] > 0 else np.zeros((0, 4)),
                args.height_fraction, args.min_clusters, args.max_clusters,
            )
            if raw_labels.shape[0] != n_leaves:
                raw_labels = np.ones(n_leaves, dtype=int)
            labels, id_to_name = pool_small_clusters(
                raw_labels, args.min_cluster_size,
            )
        else:  # hdbscan
            cluster_input_path = (
                out_dir / f"{args.search}_{decade}s{args.cluster_input_suffix}"
            )
            X = np.load(cluster_input_path)
            if X.shape[0] != n_leaves:
                print(f"  WARNING: decade {decade}: coords has {n_leaves} "
                      f"rows but {cluster_input_path.name} has {X.shape[0]}; "
                      f"using min.")
                n_leaves = min(n_leaves, X.shape[0])
                meta = meta.head(n_leaves).reset_index(drop=True)
                X = X[:n_leaves]
            raw_labels, mode, noise_ids = cluster_hdbscan(
                X,
                min_cluster_size=hdb_min_cluster_size,
                min_samples=args.hdbscan_min_samples,
            )
            labels, id_to_name = pool_small_clusters(
                raw_labels, args.min_cluster_size,
                force_other_ids=noise_ids,
            )

        labels_by_decade[decade] = labels
        id_to_name_by_decade[decade] = id_to_name
        meta_by_decade[decade] = meta
        sizes_by_decade[decade] = dict(Counter(labels.tolist()))
        cluster_modes[decade] = mode
        n_named = sum(1 for v in id_to_name.values() if v != OTHER_LABEL)
        has_other = OTHER_LABEL in id_to_name.values()
        print(f"  {decade}s: {n_named} cluster(s)"
              f"{' + other' if has_other else ''}, "
              f"mode={mode}, n_leaves={n_leaves}")

    # Cluster-level flows for every consecutive pair we have.
    flows: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(decades) - 1):
        d1, d2 = decades[i], decades[i + 1]
        path = pair_paths.get((d1, d2))
        if path is None:
            print(f"  no transport matrix for {d1}s -> {d2}s; skipping ribbons")
            continue
        T = np.load(path)
        try:
            flow = cluster_flow(T, labels_by_decade[d1], labels_by_decade[d2])
        except ValueError as e:
            print(f"  {d1}s -> {d2}s: {e}; skipping")
            continue
        flows[(d1, d2)] = flow

    lemma_to_color: dict[str, str] = {}
    if args.coloring == "lemma":
        max_legend = args.max_legend_colors if args.max_legend_colors is not None else len(PALETTE)
        colors, lemma_to_color, lemma_totals = assign_colors_by_lemma(
            decades, labels_by_decade, meta_by_decade,
            id_to_name_by_decade, sizes_by_decade,
            args.dominance_threshold, max_legend,
        )
        n_lemmas_total = len(lemma_totals)
        n_lemmas_colored = len(lemma_to_color)
        n_lemmas_other = n_lemmas_total - n_lemmas_colored
        print(f"Coloring: by dominant lemma "
              f"(threshold {args.dominance_threshold:.0%})")
        print(f"  distinct dominant lemmas: {n_lemmas_total}")
        print(f"  colored in legend: {n_lemmas_colored}"
              + (f" (top by total cluster mass)" if n_lemmas_colored < n_lemmas_total else ""))
        if n_lemmas_other:
            print(f"  routed to 'other' (rank {n_lemmas_colored + 1} and below): "
                  f"{n_lemmas_other}")
    else:
        colors = assign_colors_by_continuity(
            flows, decades, id_to_name_by_decade, sizes_by_decade,
            args.continuity_threshold,
        )
        print(f"Coloring: by continuity (threshold {args.continuity_threshold:.0%})")
    barycentric_order = barycentric_y_order(
        flows, decades, id_to_name_by_decade, sizes_by_decade,
    )
    if args.order_by == "color" and args.coloring == "lemma" and lemma_to_color:
        order = color_y_order(
            decades, barycentric_order, colors, lemma_to_color,
        )
        print("Ordering: color-grouped, barycentric within each color band")
    else:
        if args.order_by == "color" and args.coloring != "lemma":
            print("Ordering: barycentric (color ordering requires --coloring lemma)")
        order = barycentric_order

    fig = build_sankey(
        decades, labels_by_decade, id_to_name_by_decade,
        meta_by_decade, sizes_by_decade, flows, colors, order, cluster_modes,
        args.search, args.clustering,
        args.height_fraction, args.continuity_threshold,
        args.min_link_fraction, args.min_label_fraction,
        args.label_angle,
        args.decade_labels,
        args.node_labels,
        args.coloring,
        lemma_to_color,
        args.width, args.height,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        args.output, include_plotlyjs="cdn",
        config={
            # Make the interactive HTML's toolbar "download" button emit
            # SVG by default instead of Plotly's default (low-res) PNG.
            # The user can still pick png/jpeg/webp from the menu, but SVG
            # is what you want for the paper.
            "toImageButtonOptions": {
                "format": "svg",
                "filename": args.output.stem,
                "scale": 1,
            },
        },
    )
    print(f"\nWrote {args.output}")

    if args.output_svg is not None:
        try:
            fig.write_image(str(args.output_svg), format="svg",
                            width=args.width, height=args.height)
            print(f"Wrote {args.output_svg}")
        except Exception as e:
            print(f"  SVG export skipped ({e}); install kaleido to enable "
                  f"static export: pip install -U kaleido")

    if args.output_pdf is not None:
        try:
            fig.write_image(str(args.output_pdf), format="pdf",
                            width=args.width, height=args.height, scale=2)
            print(f"Wrote {args.output_pdf}")
        except Exception as e:
            print(f"  PDF export skipped ({e}); install kaleido to enable "
                  f"static export: pip install -U kaleido")

    if args.output_png is not None:
        try:
            fig.write_image(str(args.output_png), format="png",
                            width=args.width, height=args.height, scale=2)
            print(f"Wrote {args.output_png}")
        except Exception as e:
            print(f"  PNG export skipped ({e}); install kaleido to enable "
                  f"static export: pip install -U kaleido")


if __name__ == "__main__":
    main()