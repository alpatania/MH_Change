"""
Final-Layer Contextual Embeddings
===================================
Extracts the last hidden-state vector for a target word from RoBERTa,
applies PCA or UMAP, and produces an interactive Plotly scatter plot.

Run this first (Step 1) to see where each contextual usage lands — how many
clusters exist, which passages are outliers, which usages are semantically
close — then use contextual_embeddings.py (Step 2) on a chosen subset to
trace the layer trajectories of representative passages.

DATA FORMAT
-----------
Accepts two formats, auto-detected by whether the file has a header row.

Named-column format (e.g. output from corpus query tools):
    occurrence, search, text_id, ..., word_id_1, ..., match_word_1, ..., full_context, genre, year
    1, insan, 505250, ..., 5931, ..., insane, ..., "dog . Crowds...", MAG, 1855

Legacy three-column format (no header):
    wid0879, insane, "passage text..."

Column names are configurable; the defaults match the named-column format above.

NOISE CLEANING
--------------
@ placeholder tokens (e.g. "@ @ @ @ @ @ @ @ @ @") are stripped automatically.
Pass --no-clean to disable.

USAGE
-----
    # No --word needed: the matched word is read from matched_text for each row
    python final_layer_embeddings.py --input data.csv

    # --word is an optional row filter for when a CSV mixes multiple search terms
    python final_layer_embeddings.py --input data.csv --word insane
    python final_layer_embeddings.py --input data.csv --color-col genre --clusters 3 --cosine-matrix

REQUIREMENTS
------------
    pip install transformers torch scikit-learn plotly pandas numpy
    pip install umap-learn   # optional
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


# -- text cleaning --------------------------------------------------------------

def clean_passage(text: str) -> str:
    """Remove @ placeholder tokens and normalise whitespace."""
    text = re.sub(r'(?:@\s*)+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# -- data loading ----------------------------------------------------------------

def load_data(
    path: str,
    target_word: str | None,
    wid_col: str,
    word_col: str,
    passage_col: str,
    apply_cleaning: bool = True,
) -> pd.DataFrame:
    """
    Load CSV in either named-column (with header) or legacy three-column (no header) format.
    Detection: read zero data rows to get column names, then check whether the expected
    column names are present. This avoids the fragile digit-check heuristic.
    """
    peek = pd.read_csv(path, nrows=0, dtype=str)
    cols = [c.strip() for c in peek.columns]
    has_named_cols = all(c in cols for c in [wid_col, word_col, passage_col])

    if has_named_cols:
        df = pd.read_csv(path, header=0, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={wid_col: 'wid', word_col: 'word', passage_col: 'passage'})
    else:
        if cols and not cols[0].replace('-', '').replace('_', '').isdigit():
            raise ValueError(
                f"Expected columns not found in CSV.\n"
                f"  Looking for: wid={wid_col!r}, word={word_col!r}, passage={passage_col!r}\n"
                f"  Available:   {cols}\n"
                f"Use --wid-col, --word-col, --passage-col to specify the correct names."
            )
        # Truly headerless legacy format
        df = pd.read_csv(path, header=None, dtype=str)
        df = df.rename(columns={0: 'wid', 1: 'word', 2: 'passage'})
        for c in ['wid', 'word', 'passage']:
            df[c] = df[c].str.strip()
        df['passage'] = df['passage'].str.strip('"')

    if target_word:
        df = df[df['word'].str.lower() == target_word.lower()]

    if apply_cleaning:
        df['passage'] = df['passage'].apply(clean_passage)

    df = df.dropna(subset=['wid', 'word', 'passage']).reset_index(drop=True)
    print(f"Loaded {len(df)} rows  |  word forms: {df['word'].value_counts().to_dict()}")
    if 'genre' in df.columns:
        print(f"  Genres: {df['genre'].value_counts().to_dict()}")
    if 'year' in df.columns:
        print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
    return df


# -- token location ----------------------------------------------------------------

def find_target_token_span(tokenizer, passage: str, word: str) -> tuple[int, int] | None:
    m = re.search(r'\b' + re.escape(word) + r'\b', passage, re.IGNORECASE)
    if m is None:
        return None
    char_start, char_end = m.start(), m.end()
    enc = tokenizer(
        passage,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )
    tok_start = tok_end = None
    for i, (cs, ce) in enumerate(enc['offset_mapping']):
        if cs == 0 and ce == 0:
            continue
        if cs <= char_start < ce:
            tok_start = i
        if cs < char_end <= ce:
            tok_end = i
            break
    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end


# -- embedding extraction ----------------------------------------------------------

def extract_final_layer(
    tokenizer, model, passage: str, word: str, device: str = 'cpu'
) -> np.ndarray | None:
    import torch
    span = find_target_token_span(tokenizer, passage, word)
    if span is None:
        return None
    tok_start, tok_end = span
    enc = tokenizer(
        passage, return_tensors='pt', truncation=True,
        max_length=512, add_special_tokens=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**enc)
    vecs = outputs.last_hidden_state[0, tok_start:tok_end + 1, :]
    return vecs.mean(dim=0).cpu().numpy()


def build_embedding_matrix(
    df: pd.DataFrame, tokenizer, model, device: str = 'cpu'
) -> tuple[np.ndarray, pd.DataFrame]:
    vecs, keep_idx = [], []
    for idx, row in df.iterrows():
        print(f"  {row['wid']} ({row['word']}) ...", end=' ', flush=True)
        vec = extract_final_layer(tokenizer, model, row['passage'], row['word'], device)
        if vec is None:
            print("SKIPPED (word not found after cleaning)")
        else:
            print('ok')
            vecs.append(vec)
            keep_idx.append(idx)
    return np.array(vecs), df.loc[keep_idx].reset_index(drop=True)


# -- dimensionality reduction ----------------------------------------------------

def reduce_dimensions(matrix: np.ndarray, method: str = 'pca',
                      n_components: int = 2, seed: int = 42) -> np.ndarray:
    if method == 'pca':
        return PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
    elif method == 'umap':
        try:
            import umap
            return umap.UMAP(n_components=n_components, random_state=seed).fit_transform(matrix)
        except ImportError:
            print('umap-learn not installed - falling back to PCA.')
            return reduce_dimensions(matrix, 'pca', n_components, seed)
    raise ValueError(f"Unknown method '{method}'.")


def fit_umap_on_pca(
    pca_coords: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray | None:
    """Fit UMAP on PCA-reduced coordinates, per decade.

    Returns the UMAP embedding, or None if UMAP is skipped. Reasons to
    skip: umap-learn not installed, or too few samples to fit meaningfully.
    n_neighbors is clipped to n_samples-1 when necessary (UMAP's own hard
    constraint) with a warning; the fit still proceeds because the clipped
    value is at least a defensible one for tiny clouds. If n_samples is at
    or below n_components + 1 we skip entirely -- the low-d embedding would
    be geometrically degenerate.
    """
    n_samples = pca_coords.shape[0]
    if n_samples <= n_components + 1:
        print(f'   UMAP skipped: only {n_samples} samples '
              f'(need > n_components+1 = {n_components + 1})')
        return None
    try:
        import umap
    except ImportError:
        print('   UMAP skipped: umap-learn not installed')
        return None
    effective_n_neighbors = min(n_neighbors, n_samples - 1)
    if effective_n_neighbors < n_neighbors:
        print(f'   UMAP n_neighbors clipped {n_neighbors} -> '
              f'{effective_n_neighbors} (n_samples={n_samples})')
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(pca_coords)


# -- cosine similarity -------------------------------------------------------------

def save_cosine_matrix(matrix: np.ndarray, meta: pd.DataFrame, out_path: str) -> None:
    normed = normalize(matrix, norm='l2')
    sim = normed @ normed.T
    pd.DataFrame(sim, index=meta['wid'], columns=meta['wid']).to_csv(out_path)
    print(f'  Cosine-similarity matrix: {out_path}')


# -- clustering ----------------------------------------------------------------

def cluster_embeddings(matrix: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, random_state=seed, n_init='auto').fit_predict(matrix)


# -- plotting -----------------------------------------------------------------------

def make_figure(
    coords: np.ndarray, meta: pd.DataFrame,
    method: str, model_name: str, word: str,
    color_col: str | None = None,
    label_col: str | None = None,
    clusters: np.ndarray | None = None,
) -> go.Figure:
    df_plot = meta.copy()
    df_plot['x'] = coords[:, 0]
    df_plot['y'] = coords[:, 1]
    if coords.shape[1] == 3:
        df_plot['z'] = coords[:, 2]

    if clusters is not None:
        df_plot['_color'] = [f'Cluster {c}' for c in clusters]
        legend_title = 'Cluster'
    elif color_col and color_col in df_plot.columns:
        df_plot['_color'] = df_plot[color_col].astype(str)
        legend_title = color_col
    else:
        df_plot['_color'] = df_plot['word'].astype(str)
        legend_title = 'Passage (word)'

    if label_col and label_col in df_plot.columns:
        df_plot['_label'] = df_plot[label_col].astype(str)
    else:
        df_plot['_label'] = ''#df_plot['word'].astype(str)

    color_values = sorted(df_plot['_color'].unique())
    palette = px.colors.qualitative.Dark24 if len(color_values) <= 24 else px.colors.qualitative.Alphabet
    color_map = {v: palette[i % len(palette)] for i, v in enumerate(color_values)}

    is_3d = coords.shape[1] == 3
    fig = go.Figure()

    for group_val in color_values:
        sub = df_plot[df_plot['_color'] == group_val]
        color = color_map[group_val]

        # build hover: always show wid + snippet; optionally show metadata
        def hover_text(row):
            parts = [f"<b>{row['wid']}</b>  ({row['word']})"]
            for mc in [color_col, label_col]:
                if mc and mc in row.index and mc not in ('wid', 'word', 'passage'):
                    parts.append(f"{mc}: {row[mc]}")
            parts.append(row['passage'][:160] + '...')
            return '<br>'.join(parts)

        hover = [hover_text(r) for _, r in sub.iterrows()]

        common = dict(
            name=str(group_val),
            customdata=hover,
            hovertemplate='%{customdata}<extra></extra>',
            text=sub['_label'].tolist(),
        )
        marker_kwargs = dict(color=color, opacity=0.85,
                             line=dict(color='white', width=1))
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=sub['x'], y=sub['y'], z=sub['z'],
                mode='markers+text',
                marker=dict(size=8, **marker_kwargs),
                textposition='top center', textfont=dict(size=9),
                **common,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=sub['x'], y=sub['y'],
                mode='markers+text',
                marker=dict(size=12, **marker_kwargs),
                textposition='top center', textfont=dict(size=10),
                **common,
            ))

    ax = method.upper()
    color_desc = ('cluster' if clusters is not None
                  else (color_col if color_col else 'passage'))
    fig.update_layout(
        title=dict(
            text=f"Final-layer contextual embeddings of <i>'{word}'</i> ({model_name})<br>"
                 f"<sup>{ax} of last hidden state - colour = {color_desc} - hover for passage text</sup>",
            font=dict(size=15),
        ),
        xaxis_title=f'{ax} 1', yaxis_title=f'{ax} 2',
        legend_title=legend_title,
        legend=dict(itemsizing='constant'),
        hovermode='closest', width=1100, height=700, template='plotly_white',
    )
    return fig


# -- main -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',        required=True)
    p.add_argument('--word',         default=None)
    p.add_argument('--wid-col',      default='word_id_1')
    p.add_argument('--word-col',     default='matched_text')
    p.add_argument('--passage-col',  default='full_context')
    p.add_argument('--model',        default='roberta-base',
                   choices=['roberta-base', 'roberta-large'])
    p.add_argument('--method',       default='pca', choices=['pca', 'umap'])
    p.add_argument('--dims',         default=2, type=int, choices=[2, 3])
    p.add_argument('--color-col',    default=None,
                   help='Column to colour points by (e.g. genre, year)')
    p.add_argument('--label-col',    default=None,
                   help='Column to label points with')
    p.add_argument('--clusters',     default=None, type=int,
                   help='Number of k-means clusters')
    p.add_argument('--cosine-matrix', action='store_true')
    p.add_argument('--no-clean',     action='store_true',
                   help='Disable @ token removal')
    p.add_argument('--device',       default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--output',       default=None)
    # UMAP-on-PCA parameters (separate from --method umap, which controls the
    # low-d PLOT projection). These control the downstream analysis output
    # <base>_pca90_umap.npy fed to clustering / FGW.
    p.add_argument('--no-umap',       action='store_true',
                   help='Skip the UMAP-on-PCA-90 downstream step even if '
                        'umap-learn is installed.')
    p.add_argument('--umap-n-components', default=10, type=int,
                   help='UMAP output dimensionality for the downstream '
                        'clustering feed (default 10; use 2-3 only for '
                        'visualization).')
    p.add_argument('--umap-n-neighbors', default=15, type=int,
                   help="UMAP n_neighbors (default 15). Clipped to "
                        'n_samples-1 for small clouds.')
    p.add_argument('--umap-min-dist', default=0.0, type=float,
                   help='UMAP min_dist (default 0.0 for tight clusters; '
                        'raise to ~0.1 for visualization).')
    p.add_argument('--umap-metric',   default='cosine',
                   help='UMAP metric (default cosine, standard for text '
                        'embeddings).')
    p.add_argument('--umap-seed',     default=42, type=int,
                   help='UMAP random_state for reproducibility.')
    p.add_argument('--reuse-embeddings-if-exists', action='store_true',
                   help='If <output>_embeddings.npy and <output>_coords.csv '
                        'already exist, skip loading the CSV/model and skip '
                        'the BERT extraction step -- load the cached matrix '
                        'and metadata instead, and re-run only the PCA-90 '
                        'and UMAP steps. Requires --output to be set '
                        '(since the base path must be known before the CSV '
                        'is loaded). Intended for retrofitting UMAP outputs '
                        'onto runs that predate the UMAP patch, without '
                        'paying to re-embed everything through BERT.')
    args = p.parse_args()

    from transformers import RobertaModel, RobertaTokenizer

    # --- Cached-embeddings fast path -----------------------------------------
    # When --reuse-embeddings-if-exists is set and the two BERT-side outputs
    # already exist for this base_path, skip the (expensive) BERT step and
    # load matrix / meta from disk. We still run PCA-90 and UMAP downstream,
    # which is exactly what someone retrofitting the UMAP outputs onto a
    # pre-UMAP run wants. Requires --output because otherwise base_path
    # depends on the CSV data (via target_word derived from df).
    matrix = None
    meta = None
    base_path = None
    target_word = None
    if args.reuse_embeddings_if_exists:
        if not args.output:
            print('  Note: --reuse-embeddings-if-exists requires --output to '
                  'be set; running the full pipeline instead.', file=sys.stderr)
        else:
            output_path = args.output
            if not output_path.endswith('.html'):
                output_path += '.html'
            base_path = output_path[:-len('.html')]
            emb_cache = f'{base_path}_embeddings.npy'
            coords_cache = f'{base_path}_coords.csv'
            if os.path.exists(emb_cache) and os.path.exists(coords_cache):
                print(f'Reusing cached BERT outputs from {base_path}_*.')
                matrix = np.load(emb_cache)
                meta = pd.read_csv(coords_cache)
                # coords.csv also stored the plot coord columns (pc1/pc2/pc3).
                # Strip them so meta matches what build_embedding_matrix
                # would have returned.
                for c in ('pc1', 'pc2', 'pc3'):
                    if c in meta.columns:
                        meta = meta.drop(columns=c)
                target_word = args.word or (
                    meta['word'].iloc[0] if 'word' in meta.columns and len(meta)
                    else 'unknown'
                )
                print(f'  Cached matrix: {matrix.shape}, meta rows: {len(meta)}')

    if matrix is None:
        df = load_data(
            args.input, args.word,
            wid_col=args.wid_col, word_col=args.word_col, passage_col=args.passage_col,
            apply_cleaning=not args.no_clean,
        )
        if df.empty:
            print('No data found.')
            sys.exit(1)

        target_word = args.word or df['word'].iloc[0]
        output_path = args.output or f"{target_word}_final_layer.html"
        if not output_path.endswith('.html'):
            output_path += '.html'
        base_path = output_path[:-len('.html')]

        print(f'\nLoading {args.model} ...')
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
        model = RobertaModel.from_pretrained(args.model)
        model.eval().to(args.device)
        print(f'  Hidden size: {model.config.hidden_size}')

        print(f"\nExtracting final-layer embeddings ...")
        matrix, meta = build_embedding_matrix(df, tokenizer, model, device=args.device)
        if len(matrix) == 0:
            print('No embeddings extracted.')
            sys.exit(1)
        print(f'\nMatrix: {matrix.shape}')

        if args.cosine_matrix:
            save_cosine_matrix(matrix, meta, f'{base_path}_cosine.csv')

        cluster_labels = None
        if args.clusters:
            print(f'\nRunning k-means (k={args.clusters}) ...')
            cluster_labels = cluster_embeddings(matrix, args.clusters)
            meta['cluster'] = cluster_labels
            for k in range(args.clusters):
                members = meta[meta['cluster'] == k]['wid'].tolist()
                print(f'  Cluster {k}: {members}')

        print(f'Applying {args.method.upper()} -> {args.dims}D (for plot) ...')
        coords = reduce_dimensions(matrix, method=args.method, n_components=args.dims)

        print('Building figure ...')
        fig = make_figure(
            coords, meta, args.method, args.model, target_word,
            color_col=args.color_col, label_col=args.label_col,
            clusters=cluster_labels,
        )
        fig.write_html(output_path, include_plotlyjs='cdn')
        print(f'\nPlot: {output_path}')

        df_out = meta.copy()
        coord_cols = ['pc1', 'pc2'] if args.dims == 2 else ['pc1', 'pc2', 'pc3']
        df_out[coord_cols] = coords
        csv_path = f'{base_path}_coords.csv'
        df_out.to_csv(csv_path, index=False)
        print(f'   Coords: {csv_path}')
        npy_path = f'{base_path}_coords.npy'
        np.save(npy_path, coords)
        print(f'   {args.method.upper()} array (plot coords only): {npy_path}')
        # Raw full-dimensional embeddings -- feed THESE to FGW_distance.py
        emb_path = f'{base_path}_embeddings.npy'
        np.save(emb_path, matrix)
        print(f'   Raw embeddings ({matrix.shape[1]}-d, for FGW): {emb_path}')

    # PCA-to-90%-variance-explained: always produced regardless of --method
    # AND regardless of cache path -- the whole point of the reuse mode is
    # to retrofit these downstream outputs onto pre-UMAP runs cheaply.
    pca_full = PCA(n_components=0.90, svd_solver='full', random_state=42)
    coords_full = pca_full.fit_transform(matrix)
    n_dims = coords_full.shape[1]
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)[-1]
    pca90_path = f'{base_path}_pca90.npy'
    np.save(pca90_path, coords_full)
    print(f'   PCA ({n_dims} components, {cum_var * 100:.1f}% variance): {pca90_path}')

    # UMAP on top of PCA-90 for the downstream clustering / FGW feed.
    # Per-decade fit (not a joint fit across decades); this is the intended
    # design for the sheaf-and-FGW analysis, since FGW operates on the
    # pairwise metric structure within each cloud rather than on aligned
    # coordinates. If you later want a shared visualization space across
    # decades, that requires a joint fit or a projection through a reference,
    # which is not what this step produces.
    if not args.no_umap:
        umap_coords = fit_umap_on_pca(
            coords_full,
            n_components=args.umap_n_components,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.umap_seed,
        )
        if umap_coords is not None:
            umap_path = f'{base_path}_pca90_umap.npy'
            np.save(umap_path, umap_coords)
            print(f'   UMAP ({umap_coords.shape[1]}-d from PCA-90, '
                  f'n_neighbors={args.umap_n_neighbors}, '
                  f'min_dist={args.umap_min_dist}, '
                  f'metric={args.umap_metric}): {umap_path}')

if __name__ == '__main__':
    main()