"""
Sense-flow DAG and localized sense-split detection
==================================================
Turns the per-decade occurrence clouds + partial-FGW transport plans into a
directed acyclic sense-flow graph, then scores where a single sense
*bifurcates* into two across a decade boundary. The deliverable is one
long-format event table: "this sense split in this decade", with keyword
descriptors, separation measures, and significance columns.

WHERE THIS SITS IN THE PIPELINE
    final_layer_embeddings.py  ->  <search>_<decade>s_embeddings.npy   (raw 768-d token vectors)
                               ->  <search>_<decade>s_coords.csv       (has 'passage','wid','word')
    FGW_distance.py            ->  <search>_fgw_<d1>_<d2>_transport_matrix.npy   (n1 x n2, partial mass)
    THIS SCRIPT                ->  <search>_senses.csv        (labeled per-decade sense inventory)
                               ->  <search>_sense_splits.csv  (scored split/merge events)

DESIGN DECISIONS (documented for a methods section)
  1. Nodes are per-decade senses; edges are aggregated transport mass. The
     graph is acyclic by construction because FGW only ever links adjacent
     decades. Reading a source node's out-edges gives splits; reading a target
     node's in-edges gives merges. A bifurcation is therefore a *local* event
     at one (sense, decade-transition), which is exactly the requested readout.

  2. Clustering is BERTopic's backbone (dimensionality reduction -> HDBSCAN ->
     c-TF-IDF), run INDEPENDENTLY per decade, fed the precomputed *token*
     embeddings (not sentence embeddings) so clusters track word sense rather
     than passage topic. The `passage` column is handed in only as the text
     c-TF-IDF summarizes. Cross-decade correspondence is NOT BERTopic's
     topics-over-time (which assumes global senses); it comes from the FGW
     transport plan, preserving the per-decade-induction + OT-alignment design.

  3. The reducer defaults to PCA, not UMAP. UMAP is stochastic and clustering
     on an aggressively reduced UMAP projection is the step reviewers question;
     PCA keeps per-decade clustering deterministic and reproducible. `--reducer
     umap` is available (with `--seed`) for comparison.

  4. HDBSCAN outliers (label -1) are kept as an explicit sense node, never
     reassigned by default. The occurrences that look like outliers in an early
     decade are often the nascent uses that become a new sense; reduce_outliers
     would erase the sense-birth signal. Keeping them also conserves mass so the
     birth/death deficits below are exact.

  5. Split separation is reported in TWO spaces, because "the children moved
     apart" is ambiguous: `sep_feature` is cosine distance between child
     centroids in raw RoBERTa space (what FGW's feature cost sees), and
     `sep_structure` is the single-linkage ultrametric distance between the
     children (what FGW's structure cost sees). `--sep` selects which drives the
     headline `split_score`; both columns are always written.

MASS / MARGINAL ASSUMPTION
  Death and birth fractions assume the partial-FGW marginals are uniform
  (a_i = 1/n1, b_j = 1/n2), which is POT's default and what FGW_distance.py
  uses. death_frac(a) = 1 - n1 * F[a,:].sum() / size_a. If you change the
  marginals upstream, adjust `_deficit_frac` accordingly.

USAGE
    python fgw_sense_flow.py --search insan
    python fgw_sense_flow.py --search insan --min-cluster-size 15 --sep structure
    python fgw_sense_flow.py --search insan --reducer umap --seed 42 --n-perm 500

REQUIREMENTS
    pip install bertopic scikit-learn scipy numpy pandas   (umap-learn only for --reducer umap)
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances

OUTLIER_NAME = "outlier"
PASSAGE_CANDIDATES = ["passage", "full_context", "context", "matched_text"]

# Mirror fgw_sankey.py's filename grammar so this reads the same tree.
PAIR_RE = re.compile(
    r"^(?P<search>.+)_fgw_(?P<decade1>\d{4})_(?P<decade2>\d{4})_transport_matrix\.npy$"
)
COORDS_RE = re.compile(r"^(?P<search>.+)_(?P<decade>\d{4})s_coords\.csv$")


# --------------------------------------------------------------------------- #
# Discovery / IO                                                              #
# --------------------------------------------------------------------------- #

def discover_decades(out_dir: Path, search: str) -> list[int]:
    """Decades that have both a coords.csv and an embeddings.npy."""
    decades = []
    for path in out_dir.glob(f"{search}_*s_coords.csv"):
        m = COORDS_RE.match(path.name)
        if not m or m.group("search") != search:
            continue
        d = int(m.group("decade"))
        if (out_dir / f"{search}_{d}s_embeddings.npy").exists():
            decades.append(d)
    return sorted(decades)


def discover_pairs(out_dir: Path, search: str) -> dict[tuple[int, int], Path]:
    pairs: dict[tuple[int, int], Path] = {}
    for path in out_dir.glob(f"{search}_fgw_*_*_transport_matrix.npy"):
        m = PAIR_RE.match(path.name)
        if m and m.group("search") == search:
            pairs[(int(m.group("decade1")), int(m.group("decade2")))] = path
    return pairs


def load_decade(out_dir: Path, search: str, decade: int, context_col: str):
    """Return (embeddings [n,d], passages [n], meta DataFrame). Row order of all
    three matches the transport-matrix index for this decade."""
    emb = np.load(out_dir / f"{search}_{decade}s_embeddings.npy")
    meta = pd.read_csv(out_dir / f"{search}_{decade}s_coords.csv")
    if len(meta) != emb.shape[0]:
        raise ValueError(
            f"{decade}s: coords rows ({len(meta)}) != embedding rows "
            f"({emb.shape[0]}). Row alignment with the transport matrix is "
            f"required; regenerate embeddings/coords together."
        )
    col = context_col if context_col in meta.columns else next(
        (c for c in PASSAGE_CANDIDATES if c in meta.columns), None
    )
    if col is None:
        raise ValueError(
            f"{decade}s: no context column found. Tried {context_col!r} and "
            f"{PASSAGE_CANDIDATES}. Pass --context-col."
        )
    # Empty passages must NOT be dropped (would misalign with the transport
    # matrix). Replace with a placeholder so CountVectorizer is happy.
    passages = meta[col].fillna("").astype(str)
    passages = passages.mask(passages.str.strip() == "", "<empty>").tolist()
    return emb, passages, meta


# --------------------------------------------------------------------------- #
# Clustering (BERTopic backbone, per decade)                                  #
# --------------------------------------------------------------------------- #

def cluster_decade(
    emb: np.ndarray,
    passages: list[str],
    min_cluster_size: int,
    min_samples: int | None,
    reducer: str,
    reducer_dims: int,
    seed: int,
    extra_stopwords: list[str],
    n_examples: int,
    meta: pd.DataFrame,
):
    """Run BERTopic on precomputed token embeddings for one decade.

    Returns
        labels      : int array, 1-indexed contiguous 1..K (outliers get their
                      own id so cluster_flow keeps their mass).
        outlier_id  : the label id reserved for HDBSCAN noise, or None.
        info        : dict id -> {'keywords': str, 'size': int,
                                  'examples': [str], 'is_outlier': bool}
    """
    n = emb.shape[0]

    # Tiny decades: BERTopic/HDBSCAN degenerate. Treat as a single sense and
    # summarize with a plain count vectorizer so the row still carries keywords.
    if n < max(5, 2 * min_cluster_size):
        labels = np.ones(n, dtype=int)
        info = {1: {
            "keywords": _top_terms(passages, extra_stopwords),
            "size": n, "is_outlier": False,
            "examples": _examples(meta, np.arange(n), n_examples),
        }}
        return labels, None, info

    try:
        from bertopic import BERTopic
    except ImportError as e:
        raise SystemExit(
            "bertopic is not installed. `pip install bertopic scikit-learn`."
        ) from e
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    if reducer == "pca":
        from sklearn.decomposition import PCA
        nc = int(min(reducer_dims, n - 1, emb.shape[1]))
        dim_model = PCA(n_components=nc, random_state=seed)
    elif reducer == "umap":
        try:
            import umap
        except ImportError as e:
            raise SystemExit("--reducer umap needs umap-learn installed.") from e
        nc = int(min(reducer_dims, n - 2))
        dim_model = umap.UMAP(
            n_components=nc, n_neighbors=min(15, n - 1),
            min_dist=0.0, metric="cosine", random_state=seed,
        )
    else:
        raise ValueError(f"unknown reducer {reducer!r}")

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    vectorizer = CountVectorizer(
        stop_words=_english_plus(extra_stopwords), min_df=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topic_model = BERTopic(
            embedding_model=None,          # embeddings are precomputed
            umap_model=dim_model,          # PCA by default (deterministic)
            hdbscan_model=hdb,
            vectorizer_model=vectorizer,
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(passages, embeddings=emb)

    topics = np.asarray(topics, dtype=int)

    # Remap BERTopic/HDBSCAN ids to 1-indexed contiguous labels; real topics
    # 0..k-1 -> 1..k, then reserve the next id for outliers (-1) so their mass
    # is carried, not dropped.
    real_ids = sorted(t for t in set(topics.tolist()) if t >= 0)
    remap = {t: i + 1 for i, t in enumerate(real_ids)}
    outlier_id = None
    if (topics < 0).any():
        outlier_id = len(real_ids) + 1
        remap[-1] = outlier_id
    labels = np.array([remap[t] for t in topics], dtype=int)

    info: dict[int, dict] = {}
    kw_by_topic = {t: topic_model.get_topic(t) for t in real_ids}
    for old_id, new_id in remap.items():
        idx = np.where(topics == old_id)[0]
        is_out = old_id < 0
        if is_out:
            kw = OUTLIER_NAME
        else:
            terms = kw_by_topic.get(old_id) or []
            kw = ", ".join(w for w, _ in terms[:8]) or _top_terms(
                [passages[i] for i in idx], extra_stopwords)
        info[new_id] = {
            "keywords": kw, "size": int(idx.size), "is_outlier": is_out,
            "examples": _examples(meta, idx, n_examples),
        }
    return labels, outlier_id, info


def _english_plus(extra: list[str]):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return list(ENGLISH_STOP_WORDS.union(w.lower() for w in extra))


def _top_terms(passages: list[str], extra: list[str], k: int = 8) -> str:
    from sklearn.feature_extraction.text import CountVectorizer
    if not passages:
        return ""
    try:
        cv = CountVectorizer(stop_words=_english_plus(extra), min_df=1)
        X = cv.fit_transform(passages)
    except ValueError:
        return ""
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(cv.get_feature_names_out())
    order = np.argsort(-counts)[:k]
    return ", ".join(vocab[order].tolist())


def _examples(meta: pd.DataFrame, idx: np.ndarray, k: int) -> list[str]:
    col = next((c for c in PASSAGE_CANDIDATES if c in meta.columns), None)
    if col is None or idx.size == 0:
        return []
    picks = idx[:k]
    return [_snippet(str(meta.iloc[i][col])) for i in picks]


def _snippet(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "\u2026"


# --------------------------------------------------------------------------- #
# Flow + geometry                                                             #
# --------------------------------------------------------------------------- #

def cluster_flow(T: np.ndarray, labels1: np.ndarray, labels2: np.ndarray) -> np.ndarray:
    """Aggregate leaf-level transport T (n1 x n2) into a cluster-level flow
    matrix (k1 x k2). Mirrors fgw_sankey.cluster_flow so results are consistent.
    labels are 1-indexed; k = labels.max()."""
    n1, n2 = T.shape
    if labels1.shape[0] != n1 or labels2.shape[0] != n2:
        raise ValueError(
            f"Transport shape {T.shape} inconsistent with labels "
            f"({labels1.shape[0]}, {labels2.shape[0]})."
        )
    k1, k2 = int(labels1.max()), int(labels2.max())
    L1 = np.zeros((n1, k1)); L1[np.arange(n1), labels1 - 1] = 1.0
    L2 = np.zeros((n2, k2)); L2[np.arange(n2), labels2 - 1] = 1.0
    return L1.T @ T @ L2


def ultrametric(emb: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Single-linkage (subdominant) ultrametric on a cloud -- the same object
    FGW_distance builds for its structure cost. U[i,j] = cophenetic distance."""
    D = pairwise_distances(emb, metric=metric)
    Z = linkage(squareform(D, checks=False), method="single")
    return squareform(cophenet(Z))


def sep_feature(emb2: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> float:
    """Cosine distance between two child clusters' centroids in raw space."""
    c1 = emb2[idx1].mean(axis=0); c2 = emb2[idx2].mean(axis=0)
    denom = np.linalg.norm(c1) * np.linalg.norm(c2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.dot(c1, c2) / denom)


def sep_structure(U2: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> float:
    """Mean single-linkage ultrametric distance across the two child clusters."""
    if idx1.size == 0 or idx2.size == 0:
        return float("nan")
    return float(U2[np.ix_(idx1, idx2)].mean())


def _deficit_frac(transported: float, size: int, n_total: int) -> float:
    """Fraction of a cluster's own (uniform) mass that was NOT transported.
    Death for a source cluster, birth for a target cluster."""
    own = size / n_total
    if own == 0:
        return float("nan")
    return float(np.clip(1.0 - transported / own, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Split / merge scoring                                                       #
# --------------------------------------------------------------------------- #

def score_transition(
    F: np.ndarray, T: np.ndarray,
    labels1: np.ndarray, labels2: np.ndarray,
    outlier1: int | None, outlier2: int | None,
    emb2: np.ndarray, U2: np.ndarray,
    info1: dict, info2: dict,
    d1: int, d2: int, search: str,
    sep_choice: str, min_child_share: float, top2_floor: float,
    n_perm: int, n_boot: int, seed: int,
) -> list[dict]:
    """Score every real source cluster as a candidate split (rows of F) and
    every real target cluster as a candidate merge (cols of F)."""
    rng = np.random.default_rng(seed)
    n1, n2 = T.shape
    events: list[dict] = []

    real_tgt = [c for c in range(1, F.shape[1] + 1) if c != outlier2]
    real_src = [c for c in range(1, F.shape[0] + 1) if c != outlier1]
    idx2_by = {c: np.where(labels2 == c)[0] for c in range(1, int(labels2.max()) + 1)}
    idx1_by = {c: np.where(labels1 == c)[0] for c in range(1, int(labels1.max()) + 1)}

    # ---- splits: one source sense fanning to two target senses ----
    for a in real_src:
        row = F[a - 1]
        f_real = np.array([row[t - 1] for t in real_tgt])
        transported_real = f_real.sum()
        to_outlier = row[outlier2 - 1] if outlier2 else 0.0
        if transported_real <= 0 or len(real_tgt) < 2:
            continue
        p = f_real / transported_real
        order = np.argsort(-p)
        b1, b2 = real_tgt[order[0]], real_tgt[order[1]]
        p1, p2 = float(p[order[0]]), float(p[order[1]])
        sf = sep_feature(emb2, idx2_by[b1], idx2_by[b2])
        ss = sep_structure(U2, idx2_by[b1], idx2_by[b2])
        sep = sf if sep_choice == "feature" else ss
        score = (p2 / p1) * sep if (p1 > 0 and np.isfinite(sep)) else float("nan")
        candidate = (p2 >= min_child_share) and ((p1 + p2) >= top2_floor)

        p_perm = _perm_pvalue_split(
            T, labels1, labels2, a, real_tgt, emb2, U2, idx2_by,
            sep_choice, score, n_perm, rng,
        ) if (n_perm and candidate and np.isfinite(score)) else float("nan")
        boot = _boot_stability_split(
            T, labels1, labels2, a, real_tgt, {b1, b2}, n_boot, rng,
        ) if (n_boot and candidate) else float("nan")

        events.append(dict(
            word=search, kind="split", decade_from=d1, decade_to=d2,
            parent_id=a, parent_keywords=info1[a]["keywords"],
            parent_size=info1[a]["size"],
            child1_id=b1, child1_keywords=info2[b1]["keywords"], child1_share=round(p1, 4),
            child2_id=b2, child2_keywords=info2[b2]["keywords"], child2_share=round(p2, 4),
            top2_mass=round(p1 + p2, 4), to_outlier_frac=round(float(to_outlier / (transported_real + to_outlier)) if (transported_real + to_outlier) else 0.0, 4),
            sep_feature=round(sf, 4) if np.isfinite(sf) else np.nan,
            sep_structure=round(ss, 4) if np.isfinite(ss) else np.nan,
            split_score=round(score, 4) if np.isfinite(score) else np.nan,
            death_frac=round(_deficit_frac(row.sum(), info1[a]["size"], n1), 4),
            p_perm=round(p_perm, 4) if np.isfinite(p_perm) else np.nan,
            boot_stability=round(boot, 4) if np.isfinite(boot) else np.nan,
            is_candidate=candidate,
            parent_examples=" || ".join(info1[a]["examples"]),
            child1_examples=" || ".join(info2[b1]["examples"]),
            child2_examples=" || ".join(info2[b2]["examples"]),
        ))

    # ---- merges: two source senses converging on one target sense ----
    for b in real_tgt:
        col = F[:, b - 1]
        f_real = np.array([col[s - 1] for s in real_src])
        received_real = f_real.sum()
        if received_real <= 0 or len(real_src) < 2:
            continue
        p = f_real / received_real
        order = np.argsort(-p)
        a1, a2 = real_src[order[0]], real_src[order[1]]
        p1, p2 = float(p[order[0]]), float(p[order[1]])
        candidate = (p2 >= min_child_share) and ((p1 + p2) >= top2_floor)
        if not candidate:
            continue
        events.append(dict(
            word=search, kind="merge", decade_from=d1, decade_to=d2,
            parent_id=b, parent_keywords=info2[b]["keywords"],
            parent_size=info2[b]["size"],
            child1_id=a1, child1_keywords=info1[a1]["keywords"], child1_share=round(p1, 4),
            child2_id=a2, child2_keywords=info1[a2]["keywords"], child2_share=round(p2, 4),
            top2_mass=round(p1 + p2, 4), to_outlier_frac=np.nan,
            sep_feature=np.nan, sep_structure=np.nan, split_score=np.nan,
            death_frac=round(_deficit_frac(col.sum(), info2[b]["size"], n2), 4),  # birth_frac for merges
            p_perm=np.nan, boot_stability=np.nan, is_candidate=True,
            parent_examples=" || ".join(info2[b]["examples"]),
            child1_examples=" || ".join(info1[a1]["examples"]),
            child2_examples=" || ".join(info1[a2]["examples"]),
        ))
    return events


def _perm_pvalue_split(T, labels1, labels2, a, real_tgt, emb2, U2, idx2_by,
                       sep_choice, obs_score, n_perm, rng) -> float:
    """Null: shuffle decade-2 labels among decade-2 occurrences, keeping sizes,
    then recompute the split score for source `a`. Tests whether a's outgoing
    mass concentrates into two *separated* target senses more than chance."""
    n_ge = 0
    for _ in range(n_perm):
        perm = labels2[rng.permutation(labels2.size)]
        Fp = cluster_flow(T, labels1, perm)
        row = Fp[a - 1]
        f_real = np.array([row[t - 1] for t in real_tgt])
        tot = f_real.sum()
        if tot <= 0:
            continue
        p = f_real / tot; order = np.argsort(-p)
        b1, b2 = real_tgt[order[0]], real_tgt[order[1]]
        p1, p2 = float(p[order[0]]), float(p[order[1]])
        i1 = np.where(perm == b1)[0]; i2 = np.where(perm == b2)[0]
        sep = (sep_feature(emb2, i1, i2) if sep_choice == "feature"
               else sep_structure(U2, i1, i2))
        s = (p2 / p1) * sep if (p1 > 0 and np.isfinite(sep)) else 0.0
        if s >= obs_score:
            n_ge += 1
    return (1 + n_ge) / (1 + n_perm)


def _boot_stability_split(T, labels1, labels2, a, real_tgt, obs_pair, n_boot, rng) -> float:
    """Resample decade-2 occurrences with replacement (columns of T + labels2),
    recompute the top-2 targets of source `a`, and report how often the same
    child pair recurs. Measures sampling stability of the flow given fixed
    clusters (not clustering stability -- see --boot-recluster note in docstring)."""
    n2 = T.shape[1]
    hits = 0
    for _ in range(n_boot):
        cols = rng.integers(0, n2, size=n2)
        Tb = T[:, cols]; lb = labels2[cols]
        Fb = cluster_flow(Tb, labels1, lb)
        row = Fb[a - 1]
        f_real = np.array([row[t - 1] for t in real_tgt if t <= Fb.shape[1]])
        valid_tgt = [t for t in real_tgt if t <= Fb.shape[1]]
        if f_real.sum() <= 0 or len(valid_tgt) < 2:
            continue
        order = np.argsort(-f_real)
        pair = {valid_tgt[order[0]], valid_tgt[order[1]]}
        if pair == obs_pair:
            hits += 1
    return hits / n_boot


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Base of the results tree; reads <out-dir>/results/<search>/ (default ./).")
    ap.add_argument("--context-col", default="passage",
                    help="coords.csv column holding the occurrence context text (default 'passage').")
    ap.add_argument("--min-cluster-size", type=int, default=10,
                    help="HDBSCAN min_cluster_size: the sense-granularity knob. Held fixed across decades.")
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--reducer", choices=["pca", "umap"], default="pca")
    ap.add_argument("--reducer-dims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sep", choices=["feature", "structure"], default="feature",
                    help="Which separation drives split_score. Both are always stored.")
    ap.add_argument("--min-child-share", type=float, default=0.15,
                    help="Minimum share for the smaller child to count as a split.")
    ap.add_argument("--top2-floor", type=float, default=0.60,
                    help="Minimum combined share of the two children.")
    ap.add_argument("--n-perm", type=int, default=200, help="Permutation reps (0 to skip).")
    ap.add_argument("--n-boot", type=int, default=200, help="Bootstrap reps (0 to skip).")
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--extra-stopwords", default="",
                    help="Comma-separated extra stopwords (e.g. the surface forms of the target).")
    args = ap.parse_args()

    base = args.out_dir if args.out_dir else Path(".")
    out_dir = base / "results" / args.search
    if not out_dir.is_dir():
        raise SystemExit(f"ERROR: {out_dir} is not a directory.")

    extra_stop = [w.strip() for w in args.extra_stopwords.split(",") if w.strip()]
    decades = discover_decades(out_dir, args.search)
    if len(decades) < 2:
        raise SystemExit(f"Need >=2 decades with embeddings; found {decades}.")
    pairs = discover_pairs(out_dir, args.search)
    print(f"Decades: {decades}")
    print(f"Transport pairs: {sorted(pairs)}")

    # 1. Per-decade sense induction (BERTopic).
    labels_by, outlier_by, info_by, emb_by, U_by = {}, {}, {}, {}, {}
    sense_rows = []
    for d in decades:
        emb, passages, meta = load_decade(out_dir, args.search, d, args.context_col)
        labels, outlier_id, info = cluster_decade(
            emb, passages, args.min_cluster_size, args.min_samples,
            args.reducer, args.reducer_dims, args.seed, extra_stop,
            args.n_examples, meta,
        )
        labels_by[d], outlier_by[d], info_by[d], emb_by[d] = labels, outlier_id, info, emb
        n_real = sum(1 for i, v in info.items() if not v["is_outlier"])
        print(f"  {d}s: {n_real} sense(s)"
              f"{' + outlier' if outlier_id else ''}, n={emb.shape[0]}")
        for cid, v in info.items():
            sense_rows.append(dict(
                word=args.search, decade=d, sense_id=cid, size=v["size"],
                is_outlier=v["is_outlier"], keywords=v["keywords"],
                examples=" || ".join(v["examples"]),
            ))
    pd.DataFrame(sense_rows).to_csv(out_dir / f"{args.search}_senses.csv", index=False)
    print(f"  wrote {args.search}_senses.csv")

    # 2. Flow + scoring per consecutive pair.
    events = []
    for d1, d2 in zip(decades[:-1], decades[1:]):
        if (d1, d2) not in pairs:
            print(f"  no transport matrix for {d1}s->{d2}s; skipping.")
            continue
        T = np.load(pairs[(d1, d2)])
        l1, l2 = labels_by[d1], labels_by[d2]
        if T.shape != (l1.size, l2.size):
            print(f"  {d1}s->{d2}s: transport {T.shape} != labels "
                  f"({l1.size},{l2.size}); skipping.")
            continue
        F = cluster_flow(T, l1, l2)
        # ultrametric of the TARGET decade is what split separation reads.
        if d2 not in U_by:
            U_by[d2] = ultrametric(emb_by[d2])
        events += score_transition(
            F, T, l1, l2, outlier_by[d1], outlier_by[d2],
            emb_by[d2], U_by[d2], info_by[d1], info_by[d2],
            d1, d2, args.search, args.sep,
            args.min_child_share, args.top2_floor,
            args.n_perm, args.n_boot, args.seed,
        )

    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values(
            ["is_candidate", "split_score"], ascending=[False, False]
        ).reset_index(drop=True)
    out_csv = out_dir / f"{args.search}_sense_splits.csv"
    df.to_csv(out_csv, index=False)
    n_cand = int(df["is_candidate"].sum()) if not df.empty else 0
    print(f"  wrote {out_csv} ({len(df)} events, {n_cand} candidate splits/merges)")


if __name__ == "__main__":
    main()
