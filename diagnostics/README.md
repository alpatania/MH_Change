# MH_Change — Diagnostics & Analysis

These scripts are **not** part of the production run described in `README.md`.
They live in `diagnostics/` and are run from the project root, so they can
import `coha_build.py` and `FGW_distance.py` from the parent directory:

```bash
uv run python diagnostics/<script>.py ...
```

Two of them you will reach for constantly — the term sweep (before a batch
run) and the NLTK check (when cleaning silently stops working). The rest
answer the question *is a measured FGW shift real, or an artifact of sample
size, corpus noise, or the single-linkage structure term?*

---

## 1. Planning a run

### `token_count_sweep.py` — how many tokens does each term have, per decade?

Run this **before** any batch job. It counts how many matches each
`search_arg` in `search_terms.csv` would return in each decade, reusing
`coha_build.py`'s own `prefix_bounds` so the counts are exactly what the
pipeline would pull. Output is a term-by-decade matrix plus, per term, how
many decades clear `--floor`, and it flags terms passing in zero decades.

```bash
uv run python diagnostics/token_count_sweep.py \
    --terms-csv search_terms.csv \
    --sqlite-dir results/_db \
    --out-csv token_counts.csv \
    --floor 50
```

`--sqlite-dir` is the **shared** DB folder (`results/_db`), not a per-search
results folder — the decade DBs are search-agnostic and are cached there once.
Stage 1 must have run at least once so the DBs exist.

The output feeds `run_pipeline_from_csv.py`. Because `token_counts.csv`
carries every column `--terms-csv` needs, it can serve as **both** inputs,
which removes any chance of the two files drifting apart:

```bash
uv run python run_pipeline_from_csv.py \
    --terms-csv token_counts.csv --token-counts token_counts.csv \
    --min-decades-pass 3 \
    --db-dir coha-db-003 --lexicon-txt coha-lexicon.txt \
    --sources-txt coha-sources.txt
```

**Choosing `--floor` is a research decision, not a default.** A floor of 50
per decade excludes terms that are historically diffuse rather than absent —
in one sweep, `sanatorium` (274 total, max 48 in any decade), `madhouse` (341
total, max 39), and `feeble-minded` (246 total, max 47) all scored zero
passing decades despite being substantively central. If terms like those
matter to your question, either lower the floor and rerun the sweep, or keep
the floor and pass `--min-decades-pass 1` so a term qualifies on its best
decade. Rerunning the sweep is usually better, because it lets you see the
per-term tradeoff rather than hiding it behind a threshold.

### `check_nltk.py` — is run-on splitting actually working?

`coha_build.py`'s dictionary splitter (which turns `insanewould` into
`insane`) needs NLTK plus the `wordnet`, `omw-1.4`, and `words` corpora. If
they are missing the splitter **degrades to a no-op**: the pipeline still
runs, but run-ons are silently left whole and `clean_status` never contains
`split`. This script tells you which state you are in.

```bash
uv run python diagnostics/check_nltk.py
```

It tests the corpora the way the splitter actually loads them
(`wordnet.ensure_loaded()` and a real query) rather than checking for a
folder, because a corpus can be present as an unextracted `.zip` that looks
missing to a path check but is still half-broken. If it reports a corpus as
present-but-not-loadable, that is the zip case:

```bash
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
# or, if it stays zipped:
cd ~/nltk_data/corpora && unzip -o wordnet.zip && unzip -o omw-1.4.zip
```

Run it in **each** environment separately — a green check on a laptop says
nothing about whether the HPC node's corpora are extracted. `coha_build.py`
also prints a loud warning at the top of every run when splitting is
unavailable, and `--require-nltk` turns that warning into a hard abort for
reproducible batch runs.

---

## 2. Validating the distances

### `fgw_subsample_diagnostic.py` — is an observed cost above the noise floor?

A decade pair with few matched occurrences can show an elevated cost purely
from sampling noise. This builds an empirical noise floor by repeatedly
subsampling large calibration pairs down to a target size and recomputing the
FGW costs, then optionally tests an observed value against it (a
distribution-free rank bound and a normal-approximation z-score, with a
Shapiro-Wilk check so you can judge whether the z-score is trustworthy).

```bash
# characterise the noise floor at n=59, pooled across three "quiet" pairs
uv run python diagnostics/fgw_subsample_diagnostic.py \
    --pair results/insan/insan_1950s_embeddings.npy:results/insan/insan_1960s_embeddings.npy \
    --pair results/insan/insan_1960s_embeddings.npy:results/insan/insan_1970s_embeddings.npy \
    --pair results/insan/insan_1980s_embeddings.npy:results/insan/insan_1990s_embeddings.npy \
    --target-n 59 --n-trials 200

# then test one observed structure cost against that floor
uv run python diagnostics/fgw_subsample_diagnostic.py \
    --pair ... --pair ... \
    --target-n 59 --n-trials 200 \
    --observed 0.00146 --observed-metric structure_cost
```

Pick calibration pairs whose full-size cost you already believe is near-null,
and set `--target-n` at or below your smallest corpus.

**Note:** this script currently imports `compute_fgw` from `FGW_PCA_distance`,
the older pre-UMAP variant, while the pipeline runs `FGW_distance.py`. For a
noise floor computed with the same solver the pipeline uses, repoint the
import — the two `compute_fgw` signatures must match for that to be a drop-in
change.

### `bridge_diagnostic.py` — are the chaining points junk or signal?

The FGW structure term is a single-linkage ultrametric, the most
outlier-sensitive linkage there is: a few points in the gap between two sense
clusters chain them and collapse the hierarchy, and partial matching does not
protect against it because the ultrametric is built from all points before any
mass is dropped. This finds those bridge points by local density, confirms
they own the hierarchy via a recovery curve, then cross-tabulates them against
the lexical contamination columns (`ctx_n_bad_pos`, `ctx_n_at`) from Stage 1.

```bash
uv run python diagnostics/bridge_diagnostic.py \
    --emb  results/insan/insan_1990s_embeddings.npy \
    --meta results/insan/insan_1990s_coords.csv
```

The two scores are independent — one purely geometric, one purely lexical — so
their association is informative. If bridges are contaminated (Fisher OR >> 1),
cleaning is justified. If they are clean contexts, the chaining is signal and
should not be cleaned away; read the top bridges by hand. A knee at zero means
the hierarchy is not chain-dominated and there is nothing to explain.

---

## 3. Inspecting the correspondences

### `fgw_correspondence.py` — turn a transport matrix into a labelled table

`FGW_distance.py` saves the transport plan as `<base>_transport_matrix.npy`,
an `(n1 x n2)` matrix of fractional mass moved from each corpus-1 point to
each corpus-2 point. That matrix *is* the correspondence; the console
"matches" are only a 1-best summary. This reads it back, attaches readable
labels (word, passage snippet, genre, year) from each side's `_coords.csv`,
and writes a full correspondence CSV — 1-best per row, or top-k.

```bash
uv run python diagnostics/fgw_correspondence.py \
    --transport-matrix results/insan/insan_fgw_1820_1830_transport_matrix.npy \
    --meta1 results/insan/insan_1820s_coords.csv \
    --meta2 results/insan/insan_1830s_coords.csv \
    --top-k 1 \
    --output results/insan/insan_1820_1830_correspondence.csv
```

`--min-mass-fraction` (default 0.1) is the same unmatched threshold
`FGW_distance.py` uses: a corpus-1 point transporting less than that fraction
of a uniform row's mass is flagged unmatched — a candidate sense loss or
idiosyncratic usage.

### `fgw_sankey_diagram.py` — per-occurrence correspondence

A finer variant of the pipeline's Sankey: instead of aggregating to sense
clusters, it draws one node per occurrence and links matched occurrences
across adjacent decades. Reads the same coords and transport matrices the
pipeline already produced. Useful for tracing individual passages; heavy to
read for dense decades.

```bash
uv run python diagnostics/fgw_sankey_diagram.py \
    --out-dir . --search insan \
    --output results/insan/insan_sankey_occurrences.html
```

---

## Note on `FGW_PCA_distance.py`

`FGW_PCA_distance.py` is an **older version** of the FGW computation,
predating the UMAP struct-input path. It is kept only because
`fgw_subsample_diagnostic.py` imports from it. Use `FGW_distance.py` for
anything new; `FGW_PCA_distance.py` can be retired once the subsample
diagnostic is repointed.

---

## What is safe to set aside

If you are only running the pipeline and not auditing it, nothing in this
folder is required — except `token_count_sweep.py`, which you will want before
any multi-term batch, and `check_nltk.py`, which is how you find out that
cleaning has silently stopped working. The most useful of the rest are
`fgw_correspondence.py` (makes any transport matrix readable) and
`bridge_diagnostic.py` (checks whether the structure term is trustworthy for a
given decade).