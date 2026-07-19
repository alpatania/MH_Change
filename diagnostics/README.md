# MH_Change — Diagnostics & Analysis

These scripts are **not** part of the production run described in `README.md`.
They exist to answer one question in different ways: *is a measured FGW shift
real, or an artifact of sample size, corpus noise, or the single-linkage
structure term?* Run them on results you already have; none of them are
required to produce those results.

They fall into three groups: validate the distances, inspect the
correspondences, and plan the term list.

---

## 1. Validating the distances

### `fgw_subsample_diagnostic.py` — is an observed cost above the noise floor?

A decade pair with few matched occurrences can show an elevated cost purely
from sampling noise. This script builds an empirical noise floor by repeatedly
subsampling one or more large calibration pairs down to a target size and
recomputing the FGW costs, then optionally tests a specific observed value
against that floor (both a distribution-free rank bound and a normal-
approximation z-score, with a Shapiro-Wilk check so you can judge whether the
z-score is trustworthy).

```bash
# characterise the noise floor at n=59, pooled across three "quiet" pairs
python fgw_subsample_diagnostic.py \
    --pair results/insan_1950s_embeddings.npy:results/insan_1960s_embeddings.npy \
    --pair results/insan_1960s_embeddings.npy:results/insan_1970s_embeddings.npy \
    --pair results/insan_1980s_embeddings.npy:results/insan_1990s_embeddings.npy \
    --target-n 59 --n-trials 200

# then test one observed structure cost against that floor
python fgw_subsample_diagnostic.py \
    --pair ... --pair ... \
    --target-n 59 --n-trials 200 \
    --observed 0.00146 --observed-metric structure_cost
```

Pick calibration pairs whose full-size cost you already believe is near-null,
and set `--target-n` at or below your smallest corpus.

**Note:** this script currently imports `compute_fgw` from `FGW_PCA_distance`,
the older pre-UMAP variant. The production pipeline uses `FGW_distance.py`. If
you want the noise floor computed with the exact solver the pipeline uses,
point the import at `FGW_distance` instead — the `compute_fgw` signatures need
to match for this to be a drop-in change.

### `bridge_diagnostic.py` — are the points that chain the hierarchy junk or signal?

The FGW structure term is a single-linkage ultrametric, the most outlier-
sensitive linkage there is: a few points sitting between two sense clusters can
chain them and collapse the hierarchy, and partial matching does not protect
against it because the ultrametric is built from all points before any mass is
dropped. This script finds those bridge points by local density (core
distance), confirms they own the hierarchy via a recovery curve, then
cross-tabulates them against the lexical contamination columns
(`ctx_n_bad_pos`, `ctx_n_at`) from Stage 1.

```bash
python bridge_diagnostic.py \
    --emb  results/insan_1990s_embeddings.npy \
    --meta results/insan_1990s_coords.csv
```

The two scores are independent — one purely geometric, one purely lexical — so
their association is informative. If bridges are contaminated (Fisher OR >> 1),
cleaning is justified. If they are clean contexts, the chaining is signal and
should not be cleaned away; read the top bridges by hand. A knee at zero means
the hierarchy is not chain-dominated and there is nothing to explain.

For this to be readable, the coords CSV should carry a per-occurrence key. If
`wid` is a type ID (duplicated across occurrences), the script warns; regenerate
with a per-occurrence `uid` if you want bridges traced back to specific rows.

---

## 2. Inspecting the correspondences

### `fgw_correspondence.py` — turn a transport matrix into a labelled table

`FGW_distance.py` saves the transport plan as `<base>_transport_matrix.npy`, an
`(n1 x n2)` matrix of fractional mass moved from each corpus-1 point to each
corpus-2 point. That matrix *is* the correspondence; the console "matches" are
only a 1-best summary of it. This script reads the matrix back, attaches
readable labels (word, passage snippet, genre, year) from each side's
`_coords.csv`, and writes a full correspondence CSV — 1-best per row, or top-k
if you want the fuller picture.

```bash
python fgw_correspondence.py \
    --transport-matrix results/insan_fgw_1820_1830_transport_matrix.npy \
    --meta1 results/insan_1820s_coords.csv \
    --meta2 results/insan_1830s_coords.csv \
    --top-k 1 \
    --output results/insan_1820_1830_correspondence.csv
```

`--min-mass-fraction` (default 0.1) sets the same unmatched threshold
`FGW_distance.py` uses: a corpus-1 point transporting less than that fraction of
a uniform row's mass is flagged unmatched — a candidate sense loss or
idiosyncratic usage.

### `fgw_sankey_diagram.py` — per-occurrence correspondence (variant of Stage 3)

A second, finer Sankey: instead of aggregating to sense clusters like
`fgw_sankey.py`, this draws one node per individual occurrence and links
matched occurrences across adjacent decades. It reads the same per-decade
`_coords.csv` and per-pair transport matrices the pipeline already produced.
Useful when you want to trace individual passages rather than clusters; heavier
to read for dense decades.

```bash
python fgw_sankey_diagram.py \
    --out-dir results --search insan \
    --output results/insan_sankey_occurrences.html
```

---

## 3. Planning the term list

### `token_count_sweep.py` — how many tokens does each term have, per decade?

Before committing to a term list, this counts how many matches each
`search_arg` in a `search_terms.csv` would return in each decade's SQLite DB,
reusing `coha_build.py`'s `prefix_bounds` so the counts match what the pipeline
would actually pull. It writes a term-by-decade matrix plus, per term, how many
decades clear a `--floor`, and flags terms that pass in zero decades as removal
candidates.

```bash
python token_count_sweep.py \
    --terms-csv search_terms.csv \
    --sqlite-dir results \
    --out-csv token_counts.csv \
    --floor 50
```

The output feeds `run_pipeline_from_csv.py --token-counts`, which then skips
terms too sparse to be worth embedding.

---

## Note on `FGW_PCA_distance.py`

`FGW_PCA_distance.py` is an **older version** of the FGW computation that
predates the UMAP struct-input path. It is kept only because
`fgw_subsample_diagnostic.py` currently imports from it. For any new work use
`FGW_distance.py`, which is what the production pipeline runs. If the two
`compute_fgw` signatures have diverged, reconcile them before treating a
subsample noise floor as directly comparable to a pipeline result.

---

## Which of these is safe to set aside

If you are only running the pipeline and not auditing it, none of the scripts
in this file need to be present. The ones most worth keeping close are
`fgw_correspondence.py` (makes any transport matrix readable) and
`bridge_diagnostic.py` (the check on whether the structure term is trustworthy
for a given decade). `FGW_PCA_distance.py` can be retired entirely once
`fgw_subsample_diagnostic.py` is repointed at `FGW_distance.py`.