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

## Quick start

Everything for one search term lives in `results/<search>/`, and every stage
derives its paths from `--search`. Stage 1 also needs to know where the COHA
data is; nothing after it does.

```bash
# 1. build per-decade match CSVs (needs the COHA data locations)
./run_all_decades.sh \
    --db-dir coha-db-003 --lexicon-txt coha-lexicon.txt \
    --sources-txt coha-sources.txt --search insan

# 2. embeddings + pairwise FGW distances
./run_embedding_and_distance.sh --search insan --struct-suffix _pca90_umap.npy

# 3. linkage matrices for the tanglegram (all decades at once)
python fgw_build_linkage.py --search insan

# 4. visualisations
python fgw_tanglegram.py --search insan
python fgw_sankey.py --search insan
```

All outputs land in `./results/insan/`, each file prefixed `insan_`. To rerun
just the embeddings and distances after changing something upstream, the whole
command is `./run_embedding_and_distance.sh --search insan` — no paths to
repeat.

`--out-dir` sets the **base** location under which the `results/<search>/`
tree is created; it defaults to the current directory (`.`). So
`--out-dir /data/experiments` writes to
`/data/experiments/results/insan/`. Because `results/<search>/` is always
appended, two searches can never collide in one flat folder regardless of the
base you choose.

---

## Output layout

```
results/
  insan/
    insan_1820s_results.csv            # cleaned matches (what RoBERTa reads)
    insan_1820s_results_uncleaned.csv  # raw matches (provenance)
    corpus_search_1820.sqlite          # per-decade token DB (reused across searches)
    insan_1820s_embeddings.npy         # raw 768-d vectors, one per occurrence
    insan_1820s_coords.csv             # 2-D coords + metadata (drives plots)
    insan_1820s_pca90.npy
    insan_1820s_pca90_umap.npy         # Path B struct input
    insan_1820s_linkage.npy            # single-linkage tree (fgw_build_linkage.py)
    insan_fgw_1820_1830_transport_matrix.npy
    insan_fgw_1820_1830_summary.csv
    insan_fgw_summary_all_pairs.csv
    insan_tanglegram.html
    insan_sankey.html
    ... (one set per decade)
```

Because each search is a self-contained folder, you can archive or delete a
whole search at once, and two searches can never collide.

---

## The pipeline in detail

### Stage 1 — build per-decade match CSVs

`run_all_decades.sh` discovers which decades exist from the `db_*.zip`
filenames in `--db-dir`, then runs `coha_build.py` once per decade. Each run
builds (or reuses) a per-decade SQLite database and writes the matches for the
search term. Output defaults to `results/<search>/`.

```bash
./run_all_decades.sh \
    --db-dir coha-db-003 --lexicon-txt coha-lexicon.txt \
    --sources-txt coha-sources.txt --search insan \
    --context 20
```

Matching is a **case-folded prefix range** on either the word form or its
lemma, so `--search insan` returns `insane`, `insanity`, `insanely`, `Insane`,
`INSANE`, and so on. Narrow with `--pos-filter` (CLAWS7 tag prefixes, e.g.
`JJ,NN`). The SQLite DB is search-agnostic and reused across terms by default;
pass `--rebuild` to force a fresh build.

**Two CSVs per decade.** `coha_build.py` writes both a cleaned and an uncleaned
file:

- `<search>_<decade>s_results.csv` — the target form is normalised
  (`word_clean`) and the context is cleaned (`@` runs stripped, malformed
  tokens repaired), so this file is what RoBERTa reads directly.
- `<search>_<decade>s_results_uncleaned.csv` — the raw output, nothing touched,
  kept for provenance.

Both carry identical columns and identical row order/`uid`, so they join 1:1.
The cleaned file adds `word_clean` (the normalised target) and `clean_status`
(how it was derived: `lemma`, `regex`, `split`, `surface`, or `unresolved`), plus the
contamination-scoring columns (`ctx_n_at`, `ctx_n_bad_pos`, `ctx_max_word_id`)
used by the diagnostics.

### Stage 2 — embeddings and FGW distances

`run_embedding_and_distance.sh` reads the Stage 1 CSVs from `results/<search>/`,
runs `final_layer_embeddings.py` on each decade, then runs `FGW_distance.py` on
every consecutive decade pair. Both `--csv-dir` and `--out-dir` default to
`results/<search>/`.

```bash
./run_embedding_and_distance.sh --search insan \
    --struct-suffix _pca90_umap.npy --struct-metric cosine
```

Per decade: `_embeddings.npy` (raw 768-d vectors), `_coords.csv` (2-D coords +
metadata), `_pca90.npy`, and — unless `--no-umap` — `_pca90_umap.npy`. Per
pair: a transport matrix, matches array, summary CSV, and heatmap PNG. All
per-pair summaries combine into `<search>_fgw_summary_all_pairs.csv`.

`--struct-suffix` selects **Path B**: the FGW structure term is built from the
named per-decade arrays (the UMAP outputs) instead of the raw embeddings. A
missing struct file for a pair falls back to Path A with a warning. Empty
`--struct-suffix` runs Path A throughout.

Because the embedding script reads the cleaned Stage 1 CSV, cleaning is **not**
repeated here: `final_layer_embeddings.py` auto-detects the `word_clean` column
and groups on it, and reads the already-cleaned `full_context`. In-script
cleaning is available as an opt-in fallback (`--clean-in-script`) only for
legacy CSVs that predate upstream cleaning.

Two invariants to respect: the FGW feature term needs a shared space, so it
always reads the raw `_embeddings.npy`; and the `--metric` here must match the
metric used anywhere the same clouds are compared downstream (the tanglegram's
linkage, the diagnostics).

### Stage 3 — linkage matrices

`FGW_distance.py` builds each decade's single-linkage tree internally but keeps
only the derived cophenetic distances, discarding the tree. The tanglegram
needs the tree, so `fgw_build_linkage.py` recomputes and saves it using the
identical linkage call. Batch mode does every decade at once:

```bash
python fgw_build_linkage.py --search insan
```

This reads `results/insan/insan_<decade>s_embeddings.npy` and writes
`results/insan/insan_<decade>s_linkage.npy`. It skips linkages that already
exist unless you pass `--rebuild` — which prevents the stale-linkage
leaf-count mismatch that arises if embeddings are regenerated without rebuilding
the trees. Use the same `--metric` as Stage 2 (default cosine). A single-file
mode (`--emb` / `--output`) is also still available.

### Stage 4 — visualisations

`fgw_tanglegram.py` draws each decade's dendrogram side by side and connects
leaves across decades by their FGW correspondence. `fgw_sankey.py` cuts each
decade's dendrogram into sense clusters and draws the mass flow between them as
a Sankey, colours following mass-flow continuity so a stable sense keeps its
colour and a new sense gets a fresh one.

```bash
python fgw_tanglegram.py --search insan
python fgw_sankey.py --search insan
```

Both default `--out-dir` to `results/<search>/` and the output filename to
`<search>_tanglegram.html` / `<search>_sankey.html`. The tanglegram requires the
Stage 3 linkage matrices. Useful sankey knobs: `--height-fraction` (dendrogram
cut height, default 0.5), `--min-clusters`/`--max-clusters`, `--coloring`
(default `lemma`), `--clustering` (`ultrametric` or HDBSCAN). See
`--help` on each for the full set.

---

## Running many terms at once

`run_pipeline_from_csv.py` drives Stages 1–2 for every row of a
`search_terms.csv`, optionally filtered by a token-count floor (see
`token_count_sweep.py` in the diagnostics README). It reuses the
search-agnostic DBs across terms, records a manifest, and continues past
individual failures. See its module docstring for the argument list.

---

## Word-form and context normalisation

Normalisation happens **upstream in `coha_build.py`**, saved into the cleaned
`*_results.csv`, so every downstream consumer reads the same clean values.

The target form is resolved through a ladder, most-trusted signal first, and
the `clean_status` column records which rung each row landed on:

1. **`lemma`** — the tagger already resolved a clean in-family lemma. Because
   COHA's tagger saw the full sentence, this handles most malformed tokens for
   free: `insanity.42`, `insanit4`, `insaneand`, `insanityof` all fold to their
   base form, case variants collapse (`INSANE`/`Insane`/`insane` → `insane`),
   and morphology is preserved (`insane` / `insanity` / `insanely` stay
   distinct).
2. **`regex`** — no usable lemma, but rule-based repair of the surface form
   recovers it (endnote/punctuation fusion, camelCase, page-marker residue).
3. **`split`** — a boundary-less run-on the rules can't touch
   (`insanewould`, `insanegiggle`), resolved by a dictionary splitter (below).
4. **`surface` / `unresolved`** — a clean in-family surface form, or a token
   nothing could resolve. The `unresolved` tail is your review list: typically
   OCR garbage to drop and real compounds (`insane-asylum`) to keep.

Nothing is folded blindly, and the raw surface form is always preserved in
`match_word_1` and in the uncleaned CSV.

**The run-on splitter.** COHA has run-ons with no case or punctuation signal —
`insanewould` (a dropped space), `nightgownssoft`, `beforehanded` — that regex
cannot split because the split point is only knowable from vocabulary. The
splitter uses two dictionaries together (WordNet plus NLTK's 235k `words`
list, since neither alone is complete: WordNet omits pronouns like
`something`, the words list omits nothing but is over-permissive) and restricts
to a **single split point**, matching the actual corruption. Guards keep it
conservative: capitalized tokens are skipped (proper nouns no dictionary
reliably holds, e.g. `Gaddon`), real words are never touched (`nightgowns`,
`something` stay whole), and a split is accepted only when both halves are
known words. For the target it additionally keeps only the piece beginning with
the search prefix. This runs on both the target and the context.

If NLTK or its corpora are not installed, the splitter degrades to a no-op —
the pipeline still runs, it just leaves run-ons whole (still flagged in
`ctx_bad_forms`). See Installation for the one-time corpus download.

Context is cleaned the same way: `@` redaction runs are stripped, malformed
tokens repaired, and run-ons split, before the passage is saved. Note that
stripping a `@` run splices the two sides together — the result reads fluently
but is missing the redacted words — so the `ctx_n_at` column records how many
`@` tokens a window contained, keeping contaminated windows countable. Roughly
a quarter of ±20-token windows intersect a redaction block.

---

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`:

```bash
uv sync
```

The run-on splitter in `coha_build.py` needs NLTK and three corpora. These are
a one-time download (the splitter no-ops without them, so the pipeline still
runs, just without run-on splitting):

```bash
uv run python -m nltk.downloader wordnet omw-1.4 words
```

Run scripts through uv (uses the project environment automatically):

```bash
uv run python fgw_tanglegram.py --search insan
```

or activate once per shell:

```bash
source .venv/bin/activate
python fgw_tanglegram.py --search insan
```

**If you move the project folder**, the venv breaks (it hard-codes its own
absolute path). Recreate it:

```bash
deactivate 2>/dev/null
rm -rf .venv
uv sync
```

**Do not keep `.venv/` in a cloud-synced folder** (OneDrive, Dropbox, iCloud).
Sync mangles the many small binary files a venv contains and produces
corrupted-package errors (e.g. `ModuleNotFoundError: No module named
'yaml.error'`) that look like code bugs but are half-synced files. Keep the
code in the synced folder and the environment outside it:

```bash
export UV_PROJECT_ENVIRONMENT=~/venvs/MH_Change   # add to ~/.zshrc to persist
uv sync
```

The parent (`~/venvs/`) must exist; `uv sync` creates the leaf. If uv warns that
`VIRTUAL_ENV` does not match the project environment path, an old venv is still
active in your shell — run `deactivate` or open a fresh terminal.

**On an HPC login node / VS Code tunnel**, the integrated terminal may not
source `~/.zshrc`, so set the variable in VS Code's remote
`settings.json` instead (use a full path, not `~`):

```json
"terminal.integrated.env.linux": {
    "UV_PROJECT_ENVIRONMENT": "/home/USER/venvs/MH_Change"
}
```

Run `uv sync` on the login node (which has network), not inside a batch job.

---

## Diagnostics and analysis

The scripts above are the production pipeline. A second tier checks whether the
results are real — noise floors, corpus-size bias, single-linkage chaining, and
readable correspondence tables — documented separately in
**`diagnostics\README.md`**.