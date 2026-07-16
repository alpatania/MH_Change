# MH_Change — Contextual Word Embeddings & Semantic Shift Analysis

Tools for analyzing semantic shifts in text corpora using contextual embeddings from RoBERTa.

**Available scripts:**

| Script | Purpose | Status |
|--------|---------|--------|
| `final_layer_embeddings.py` | **Step 1 — the map.** Extract final-layer embeddings and visualize where contextual usages cluster | ✓ Available |
| `FGW_distance.py` | **Semantic shift.** Compute Fused Gromov-Wasserstein distance between two sets of embeddings to quantify semantic change | ✓ Available |
| `contextual_embeddings.py` | **Step 2 — the journey.** *(Not included)* For a chosen subset of passages, trace how embeddings travel through all layers. Inspired by [Zimmerman et al., arXiv:2412.10924](https://arxiv.org/abs/2412.10924). | ⚠️ Missing |

---

## Installation

### Using UV (recommended)

```bash
uv sync
```

Or with pip in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows
pip install -e .
```

### Core dependencies

- `transformers` ≥ 5.12.1 — RoBERTa tokenizer & model
- `torch` ≥ 2.12.1 — PyTorch neural networks
- `scikit-learn` ≥ 1.9.0 — PCA, k-means, metrics
- `scipy` ≥ 1.14.0 — hierarchical clustering, distance metrics
- `pandas` ≥ 3.0.3 — CSV handling
- `numpy` ≥ 2.4.6 — numerical arrays
- `plotly` ≥ 6.8.0 — interactive visualizations
- `matplotlib` ≥ 3.11.0 — static plotting
- `pot` ≥ 0.9.0 — Python Optimal Transport (for FGW)
- `umap-learn` ≥ 0.5.6 — UMAP dimensionality reduction (optional; falls back to PCA)

---

## Data format

Both scripts accept the same CSV and auto-detect whether it has a header row.

### Named-column format (corpus query output, **with header**)

```
occurrence, search, text_id, document_file, genre, year, ..., word_id_1, matched_text, ..., full_context
1, insan, 505250, mag_1855_505250.txt, MAG, 1855, ..., 5931, insane, ..., "dog . Crowds of idlers ..."
```

Default column mapping (matches the format above; change with `--wid-col`, `--word-col`, `--passage-col`):

| Role | Default column name |
|------|---------------------|
| Passage ID | `word_id_1` |
| Matched word | `matched_text` |
| Passage text | `full_context` |

All other columns (e.g. `genre`, `year`, `match_pos_1`, `source`) are kept as metadata
and can be used for colouring and labelling points in `final_layer_embeddings.py`.

---

## Noise cleaning

Both scripts automatically remove `@` placeholder tokens before tokenisation.
These appear in corpus exports as variable-length sequences of space-separated
`@` signs (e.g. `@ @ @`, `@ @ @ @ @ @ @ @ @ @`) marking redacted text spans.
They are stripped and whitespace is normalised before the passage is passed to RoBERTa.

Pass `--no-clean` to disable this behaviour.

---

## Step 1 — `final_layer_embeddings.py`

Extracts one vector per passage (last RoBERTa layer, mean-pooled over the
word's tokens) and plots all passages in a single 2D or 3D scatter.

### What it produces

Each dot is one **passage**:

- **Position** — PCA or UMAP of the final hidden-state vector
- **Colour** — passage ID, a metadata column (e.g. `genre`, `year`), or k-means cluster
- **Hover** — passage ID, word form, metadata values, and a text snippet

### Minimal usage

```bash
python final_layer_embeddings.py --input data.csv
```

`--word` is not required. Each row already carries its matched word in the `matched_text`
column, and the script reads it from there. Pass `--word insane` only if your CSV mixes
multiple search terms and you want to process just one at a time.

Saves `insane_final_layer.html` and `insane_final_layer_coords.csv`.

### Full options

```bash
python final_layer_embeddings.py \
  --input        data.csv          \  # CSV file
  --word         insane            \  # target word (case-insensitive)
  --wid-col      word_id_1         \  # column for passage ID (default: word_id_1)
  --word-col     matched_text      \  # column for matched word (default: matched_text)
  --passage-col  full_context      \  # column for passage text (default: full_context)
  --model        roberta-base      \  # roberta-base (default) or roberta-large
  --method       pca               \  # pca (default) or umap
  --dims         2                 \  # 2 (default) or 3
  --color-col    genre             \  # colour points by this column (e.g. genre, year)
  --label-col    year              \  # label points with this column
  --clusters     3                 \  # run k-means and colour by cluster
  --cosine-matrix                  \  # save pairwise cosine-similarity matrix as CSV
  --no-clean                       \  # disable @ token removal
  --device       cpu               \  # cpu (default) or cuda
  --output       insane_map.html      # output filename
```

### Recommended usage with real corpus data

```bash
# Colour by genre, discover natural groupings with k-means
python final_layer_embeddings.py --input insan_results.csv --word insane \
    --color-col genre --clusters 3 --cosine-matrix

# Colour by decade for diachronic analysis
python final_layer_embeddings.py --input insan_results.csv --word insane \
    --color-col year --method umap
```

The console prints which passage IDs landed in each cluster — copy those
into a subset CSV to feed into Step 2 (if available).

### Output files

- `{word}_final_layer.html` — Interactive scatter plot
- `{word}_final_layer_coords.csv` — Reduced coordinates + metadata
- `{word}_final_layer_coords.npy` — NumPy array of coordinates
- `{word}_final_layer_pca90.npy` — Full PCA (90% variance) array *if using `--method pca`*
- `{word}_final_layer_cosine.csv` — Pairwise cosine similarity *if using `--cosine-matrix`*

---

## Computing Semantic Shift — `FGW_distance.py` ✓

Computes the Fused Gromov-Wasserstein distance between two sets of pre-computed embeddings
to measure how much a word's semantic usage has changed between corpora or time periods.

### Use case
After extracting embeddings for the same word in two different corpora (e.g., 1850s vs. 1950s),
compute the transport-based distance to quantify semantic shift.

### Input

Two NumPy `.npy` files containing embedding matrices:
- Corpus 1: shape `[n_contexts, embedding_dim]` (e.g., 768 for RoBERTa-base)
- Corpus 2: shape `[m_contexts, embedding_dim]`

### Minimal usage

```bash
python FGW_distance.py --emb1 corpus_1850s.npy --emb2 corpus_1950s.npy
```

### Full options

```bash
python FGW_distance.py \
  --emb1 corpus1_embeddings.npy    \  # required: corpus 1 embeddings
  --emb2 corpus2_embeddings.npy    \  # required: corpus 2 embeddings
  --mass 0.8                       \  # mass fraction for partial GW (default: 0.8)
  --metric euclidean               \  # distance metric: euclidean or cosine (default: euclidean)
  --output fgw_transport.png          # output plot filename (default: fgw_transport.png)
```

### Output

- **PNG plot** — Transport matrix heatmap + mass distribution
- **Console output** — Semantic shift score + top 10 sentence matches between corpora

### Example workflow

```bash
# Extract embeddings for corpus 1 (1850s)
python final_layer_embeddings.py --input corpus_1850s.csv \
    --output insane_1850s.html

# Extract embeddings for corpus 2 (1950s)
python final_layer_embeddings.py --input corpus_1950s.csv \
    --output insane_1950s.html

# Compute semantic shift
python FGW_distance.py \
    --emb1 insane_1850s_coords.npy \
    --emb2 insane_1950s_coords.npy
```

---

## Step 2 — `contextual_embeddings.py` ⚠️ (Not included)

Extracts the hidden state at **every layer** for each passage, stacks them,
and plots each (passage × layer) pair. Lines connect the same passage across
layers, tracing its embedding trajectory through the model.

### What it produces

Each dot is one **(passage × layer)** pair:

- **Position** — PCA or UMAP of the hidden-state vector at that layer
- **Colour** — layer depth (dark = early, bright = late)
- **Connected lines** — trace the same passage across all layers
- **Hover** — passage ID, layer number, and a text snippet

### Minimal usage

```bash
python contextual_embeddings.py --input subset.csv
```

**This script is not included in this repository.**

If available, it would produce the following:

Saves `insane_embeddings.html` and `insane_embeddings_coords.csv`.

### Full options (if script exists)

```bash
python contextual_embeddings.py \
  --input        subset.csv        \  # CSV file (ideally a cluster subset from Step 1)
  --word         insane            \  # target word
  --wid-col      word_id_1         \  # column for passage ID (default: word_id_1)
  --word-col     matched_text      \  # column for matched word (default: matched_text)
  --passage-col  full_context      \  # column for passage text (default: full_context)
  --model        roberta-base      \  # roberta-base (default) or roberta-large
  --method       pca               \  # pca (default) or umap
  --dims         2                 \  # 2 (default) or 3
  --layers       all               \  # 'all', '0,4,8,12', or '0:13:2'
  --no-clean                       \  # disable @ token removal
  --device       cpu               \  # cpu (default) or cuda
  --output       insane_traj.html     # output filename
```

### Layer subsets (less crowded, faster)

```bash
# Every other layer
python contextual_embeddings.py --input subset.csv --word insane --layers 0:13:2

# Only the last four layers
python contextual_embeddings.py --input subset.csv --word insane --layers 9,10,11,12
```

---

## Recommended end-to-end workflow

```bash
# 1. Map all passages in the final layer; colour by genre; find 3 clusters
python final_layer_embeddings.py --input insan_results.csv --word insane \
    --color-col genre --clusters 3 --cosine-matrix

# 2. Inspect console output to see which word_id_1 values land in each cluster.
#    Filter the CSV to one representative per cluster and save as subset.csv.

# 3. Examine trajectories for those representatives
python contextual_embeddings.py --input subset.csv --word insane

# 4. Optionally use roberta-large for 24-layer richer trajectories
python contextual_embeddings.py --input subset.csv --word insane \
    --model roberta-large --device cuda
```

---

## How each script works internally

### `final_layer_embeddings.py`
1. Load CSV (auto-detect header); strip `@` sequences; filter to target word
2. Tokenise each passage with RoBERTa's BPE tokenizer
3. Locate the target word's token(s) via character-offset mapping
4. Forward pass → `last_hidden_state`; mean-pool over the word's tokens → one 768-dim vector per passage
5. Optional k-means on the full-dimensional matrix
6. PCA / UMAP → 2D or 3D coordinates
7. Plotly scatter, saved as self-contained HTML

### `contextual_embeddings.py`
Steps 1–3 identical, then for each layer:

4. Forward pass with `output_hidden_states=True`; mean-pool at every layer → one vector per (passage, layer)
5. Stack into matrix of shape `(n_passages × n_layers, hidden_dim)`
6. PCA / UMAP → 2D or 3D coordinates
7. Plotly scatter with dotted lines connecting layers within the same passage

---

## Tips

- **roberta-large** has 24 layers and 1024-dim hidden states (vs. 12 layers / 768-dim for base). Richer trajectories, but ~2× slower and needs ~1.4 GB RAM.
- Add `--device cuda` if you have a GPU.
- Word matching is case-insensitive, so `insane`, `Insane`, and `INSANE` are all found by `--word insane`.
- If a passage doesn't contain the target word after `@` removal, a warning is printed and that row is skipped.
- Both scripts save a `*_coords.csv` alongside the HTML for post-processing in R or other tools.
- The cosine-similarity matrix (`--cosine-matrix`) can be used with `hclust` in R or `scipy.cluster.hierarchy` in Python to build a dendrogram of passage similarity.
