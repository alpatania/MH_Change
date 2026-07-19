# MH_Change — Contextual Word Embeddings & Semantic Shift

Measures how a word's meaning shifts across decades of the Corpus of Historical
American English (COHA), using RoBERTa contextual embeddings and partial Fused
Gromov-Wasserstein (FGW) optimal transport between consecutive decades.

The core idea: embed every occurrence of a target term in each decade, then ask
how the decade-to-decade point clouds correspond. FGW compares them on two axes
at once — where occurrences sit in RoBERTa space (feature term) and how each
decade's internal sense hierarchy is organised (structure term) — so a result
can be read as positional drift, structural rearrangement, or both.

---

## The pipeline

Four stages, run in order. The first two are shell wrappers over per-decade
Python; the last two are visualisations.

```
run_all_decades.sh            ->  per-decade search-result CSVs
run_embedding_and_distance.sh ->  per-decade embeddings + per-pair FGW distances
fgw_sankey.py                 ->  cluster-level sense-drift Sankey
fgw_tanglegram.py             ->  side-by-side decade dendrograms
```

Everything for one search term is prefixed with that term's slug (e.g.
`insan_1950s_results.csv`), so multiple terms can share one `results/`
directory without colliding.

### Stage 1 — build per-decade match CSVs

`run_all_decades.sh` discovers which decades exist from the `db_*.zip`
filenames in `--db-dir`, then runs `coha_build.py` once per decade. Each run
builds (or reuses) a per-decade SQLite database and writes a CSV of every
context window matching the search term.

```bash
./run_all_decades.sh \
    --db-dir coha-db-003 \
    --lexicon-txt coha-lexicon.txt \
    --sources-txt coha-sources.txt \
    --search insan \
    --out-dir results \
    --context 20
```

Matching is a **case-folded prefix range** on either the word form or its
lemma, so `--search insan` returns `insane`, `insanity`, `insanely`,
`Insane`, `INSANE`, and so on. Narrow with `--pos-filter` (CLAWS7 tag
prefixes, e.g. `JJ,NN`) if a prefix pulls in unwanted parts of speech.
The SQLite DB is search-agnostic and reused across terms by default; pass
`--rebuild` to force a fresh build.

Each decade's CSV carries the matched token, its context window, per-token
metadata, and a set of contamination-scoring columns (`ctx_n_at`,
`ctx_n_bad_pos`, `ctx_max_word_id`) used by the diagnostics — see
`README_diagnostics.md`.

### Stage 2 — embeddings and FGW distances

`run_embedding_and_distance.sh` discovers the Stage 1 CSVs, runs
`final_layer_embeddings.py` on each decade, then runs `FGW_distance.py` on
every consecutive decade pair.

```bash
./run_embedding_and_distance.sh \
    --search insan --csv-dir results --out-dir results \
    --struct-suffix _pca90_umap.npy \
    --struct-metric cosine \
    --rebuild
```

Per decade this produces `_embeddings.npy` (raw 768-d RoBERTa vectors, one per
occurrence), `_coords.csv` (2-D plot coordinates plus all metadata),
`_pca90.npy`, and — unless `--no-umap` — `_pca90_umap.npy`. Per pair it
produces an FGW transport matrix, a matches array, a summary CSV, and a
transport heatmap PNG. All per-pair summaries are combined into
`<search>_fgw_summary_all_pairs.csv`.

`--struct-suffix` selects **Path B**: the FGW structure term is built from the
named per-decade arrays (here the UMAP outputs) instead of the raw embeddings.
If a struct file is missing for a pair, that pair falls back to Path A with a
warning rather than failing. Leaving `--struct-suffix` empty runs Path A for
every pair.

Two things worth knowing about this stage. The FGW feature term needs a shared
coordinate space, so it always reads the raw `_embeddings.npy`; only the
structure term uses the struct arrays. And the `--metric` you pass here must
match the metric used anywhere the same clouds are compared downstream (the
tanglegram's linkage, the diagnostics), or you are comparing two different
geometries.

### Stage 3 — cluster-level Sankey

`fgw_sankey.py` cuts each decade's single-linkage dendrogram at a per-decade
height, aggregates the leaf-level FGW transport up to cluster level, and draws
a Sankey where each column is a decade, each node a sense cluster (labelled by
its top lemmas), and each ribbon the mass flowing between clusters. Colours
follow mass-flow continuity, so a stable sense keeps its colour, a split shows
as two same-coloured branches, and a genuinely new sense gets a fresh colour.

```bash
python fgw_sankey.py \
    --out-dir results --search insan \
    --output results/insan_sankey.html
```

Useful knobs: `--height-fraction` (where to cut each dendrogram, default 0.5 of
max merge height), `--min-clusters`/`--max-clusters` (readability rails),
`--coloring` (default `lemma`), and `--clustering` (`ultrametric` by default,
or HDBSCAN). Run `python fgw_sankey.py --help` for the full set.

### Stage 4 — tanglegram

`fgw_tanglegram.py` draws each decade's actual dendrogram side by side and
connects leaves across decades by their FGW correspondence, so you can see
which branches map onto which.

```bash
python fgw_tanglegram.py \
    --out-dir results --search insan \
    --output results/insan_tanglegram.html
```

**This stage needs prebuilt linkage matrices.** `FGW_distance.py` builds each
decade's single-linkage tree internally but keeps only the derived cophenetic
distances, discarding the tree itself. `fgw_build_linkage.py` recomputes and
saves the tree, using the identical linkage call so the saved tree matches the
one FGW used. Run it once per decade first:

```bash
for d in 1820 1830 1840 1850 1860 1870 1880 1890 1900 1910 \
         1920 1930 1940 1950 1960 1970 1980 1990 2000 2010; do
    python fgw_build_linkage.py \
        --emb results/insan_${d}s_embeddings.npy \
        --metric cosine \
        --output results/insan_${d}s_linkage.npy
done
```

Use the same `--metric` here as in Stage 2. If a linkage was built from a
different (older) embeddings file than the one the tanglegram loads, you will
get a leaf-count mismatch error; delete the stale `_linkage.npy` and rebuild.

---

## Running many terms at once

`run_pipeline_from_csv.py` drives Stages 1–2 for every row of a
`search_terms.csv`, optionally filtered by a token-count floor (see
`token_count_sweep.py` in the diagnostics README). It reruns each term through
both shell scripts, reuses the search-agnostic DBs across terms, records a
manifest, and continues past individual failures. See its module docstring for
the full argument list.

---

## Word-form normalisation (current state)

The matched target word is lightly repaired before embedding: endnote markers
and trailing punctuation are stripped (`insanity.42` -> `insanity`), camelCase
run-ons are split, and the result is lowercased so `INSANE`, `Insane`, and
`insane` collapse into one legend entry while morphology (`insane` /
`insanity` / `insanely`) is preserved.

Two malformed classes are **not** yet folded: slash-joined tokens
(`insane/Jack`) and boundary-less run-ons (`insanityof`, `insant`). These
still appear as separate legend entries. Folding them requires normalising
upstream in `coha_build.py` (a `word_clean` column read by every downstream
consumer) rather than in the plot script; that change is planned but not yet
made. Until then, treat the rarest legend entries as probable artifacts and
cross-check against the contamination columns.

Redaction handling: COHA replaces ten consecutive tokens every two hundred with
`@` for copyright. These runs are stripped before RoBERTa sees the passage.
Stripping splices the two sides together, so the resulting context reads as
fluent but is missing material; the `ctx_n_at` column records how many `@`
tokens a window contained before stripping, so contaminated windows remain
countable. Pass `--no-clean` to disable both the `@` removal and the target
repair.

---

## Output files, per search term

| File | Produced by | Contents |
|------|-------------|----------|
| `<term>_<decade>s_results.csv` | Stage 1 | matched context windows + metadata |
| `corpus_search_<decade>.sqlite` | Stage 1 | per-decade token DB (reused across terms) |
| `<term>_<decade>s_embeddings.npy` | Stage 2 | raw 768-d vectors, one per occurrence |
| `<term>_<decade>s_coords.csv` | Stage 2 | 2-D coords + metadata (drives plots) |
| `<term>_<decade>s_pca90.npy` | Stage 2 | PCA to 90% variance |
| `<term>_<decade>s_pca90_umap.npy` | Stage 2 | UMAP of the PCA output (Path B struct input) |
| `<term>_fgw_<d1>_<d2>_transport_matrix.npy` | Stage 2 | (n1 x n2) transport plan |
| `<term>_fgw_<d1>_<d2>_summary.csv` | Stage 2 | fused / feature / structure costs for the pair |
| `<term>_fgw_summary_all_pairs.csv` | Stage 2 | all pairs combined |
| `<term>_<decade>s_linkage.npy` | `fgw_build_linkage.py` | single-linkage tree for the tanglegram |
| `<term>_sankey.html` | Stage 3 | cluster-level drift Sankey |
| `<term>_tanglegram.html` | Stage 4 | side-by-side decade dendrograms |

---

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`. From the project root:

```bash
uv sync
```

This creates `.venv/` and installs the locked dependency set (Python >= 3.12;
RoBERTa via `transformers` + `torch`, optimal transport via `pot`, `umap-learn`
for the struct arrays, `plotly` for the HTML visualisations).

Run scripts either through uv, which uses the project environment automatically:

```bash
uv run python fgw_tanglegram.py --out-dir results --search insan \
    --output results/insan_tanglegram.html
```

or by activating the environment once per shell:

```bash
source .venv/bin/activate
python fgw_tanglegram.py --out-dir results --search insan \
    --output results/insan_tanglegram.html
```

**If you move the project folder**, the virtual environment breaks — a venv
hard-codes its own absolute path. Recreate it:

```bash
deactivate 2>/dev/null
rm -rf .venv
uv sync
```

**Do not keep `.venv/` in a cloud-synced folder** (OneDrive, Dropbox, iCloud).
Sync mangles the thousands of small binary files a venv contains and produces
corrupted-package errors (e.g. `ModuleNotFoundError: No module named
'yaml.error'`) that look like code bugs but are really half-synced files. Keep
the code in the synced folder and the environment outside it:

```bash
export UV_PROJECT_ENVIRONMENT=~/venvs/MH_Change   # add to ~/.zshrc to persist
uv sync
```

If `uv` warns that `VIRTUAL_ENV` does not match the project environment path,
an old venv is still activated in your shell; run `deactivate` (or open a fresh
terminal) so the active environment and uv's target agree.

A `requirements.txt` is also provided for non-uv installs (`pip install -r
requirements.txt` into a manually created venv), but `uv.lock` is the source of
truth.

---

## Diagnostics and analysis

The scripts above are the production pipeline. A second tier of scripts checks
whether the results are real — noise floors, corpus-size bias, chaining
artifacts, and readable correspondence tables. Those are documented separately
in **`README_diagnostics.md`** so this file stays focused on the run itself.