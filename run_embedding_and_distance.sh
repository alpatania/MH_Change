#!/usr/bin/env bash
#
# Stage 2+3 of the pipeline: given the per-decade search-result CSVs that
# run_all_decades.sh already produced, run final_layer_embeddings.py on
# each one, then run FGW_distance.py on every pair of CONSECUTIVE decades
# (1820s vs 1830s, 1830s vs 1840s, ...) using the resulting embeddings and
# coords files. Decades are discovered from the CSV filenames in --csv-dir,
# matching <search>_<decade>s_results.csv, the same naming run_all_decades.sh
# uses.
#
# --search is REQUIRED and is prefixed onto every output filename this
# script produces (embeddings, coords, FGW plots/matrices/matches, the
# combined summary) and used to filter which CSVs are picked up from
# --csv-dir. Without this, running the pipeline for a second search term
# into the same --out-dir would silently overwrite every file from the
# first term's run, since decade numbers alone aren't a unique filename.
#
# Usage:
#   ./run_embedding_and_distance.sh --search insan --csv-dir results --out-dir results \
#     [--embed-script final_layer_embeddings.py] [--fgw-script FGW_distance.py] \
#     [--model roberta-base] [--method pca] [--dims 2] [--device cpu] \
#     [--alpha 0.5] [--mass 0.8] [--metric cosine] [--normalize-scale] \
#     [--no-umap] [--umap-n-components 10] [--umap-n-neighbors 15] \
#     [--umap-min-dist 0.0] [--umap-metric cosine] [--umap-seed 42] \
#     [--struct-suffix _pca90_umap.npy] [--struct-metric euclidean] \
#     [--rebuild] [--list-only]
#
# UMAP pass-through flags (--no-umap, --umap-*) are forwarded to
# final_layer_embeddings.py verbatim; see that script's --help for their
# semantics.
#
# --struct-suffix / --struct-metric enable Path B of the FGW step: when
# --struct-suffix is set (e.g. "_pca90_umap.npy"), for each pair the shell
# constructs "${BASE1}${STRUCT_SUFFIX}" and "${BASE2}${STRUCT_SUFFIX}" and
# passes them as --struct1/--struct2 to FGW_distance.py -- so the
# ultrametrics are built from those arrays instead of from the raw
# embeddings. If either struct file is missing for a pair (e.g. UMAP was
# skipped for a marginal decade), that pair falls back to Path A (no
# struct inputs) with a warning rather than failing. Empty --struct-suffix
# (the default) preserves original behaviour exactly.
#
# --struct-metric is the metric for the ultrametrics when Path B is
# active; default is --metric. If you pass UMAP outputs as struct inputs,
# 'euclidean' is usually more appropriate than 'cosine' because UMAP's
# low-d embedding is optimized for Euclidean geometry.
#
# Note: an earlier version of this script put --normalize-scale in the
# EMBED extra-args list, where it would have been rejected by
# final_layer_embeddings.py had anyone actually set it. It has been moved
# to the FGW block, which is what the FGW script actually accepts.
#
# Each decade's embedding outputs (plot, coords csv, embeddings npy, pca90
# npy) and each pair's FGW outputs (plot, transport matrix, matches,
# per-pair summary csv) land in --out-dir, all prefixed with <search>_. A
# failure in one decade's embedding step skips that decade for pairing
# (with a warning) rather than stopping the run; a failure in one pair's
# FGW step is reported and skipped the same way. If every attempted pair
# succeeds, their per-pair summary csvs are combined into
# out-dir/<search>_fgw_summary_all_pairs.csv with explicit decade1/decade2
# columns.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EMBED_SCRIPT="$SCRIPT_DIR/final_layer_embeddings.py"
FGW_SCRIPT="$SCRIPT_DIR/FGW_distance.py"
SEARCH=""
CSV_DIR=""
OUT_DIR=""
MODEL="roberta-base"
METHOD="pca"
DIMS=2
DEVICE="cpu"
ALPHA=0.5
MASS=0.8
METRIC="cosine"
NB_DUMMIES=1
NUM_ITER_MAX=50000
NORMALIZE_SCALE=0
REBUILD=0
LIST_ONLY=0

# UMAP pass-through defaults (empty = not passed; falls back to embed script's own defaults)
NO_UMAP=0
UMAP_N_COMPONENTS=""
UMAP_N_NEIGHBORS=""
UMAP_MIN_DIST=""
UMAP_METRIC=""
UMAP_SEED=""

# FGW Path B pass-through defaults (empty = not passed; runs Path A / original behaviour)
STRUCT_SUFFIX=""
STRUCT_METRIC=""

while [ $# -gt 0 ]; do
  case "$1" in
    --search) SEARCH="$2"; shift 2 ;;
    --csv-dir) CSV_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --embed-script) EMBED_SCRIPT="$2"; shift 2 ;;
    --fgw-script) FGW_SCRIPT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --dims) DIMS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --mass) MASS="$2"; shift 2 ;;
    --metric) METRIC="$2"; shift 2 ;;
    --nb-dummies) NB_DUMMIES="$2"; shift 2 ;;
    --num-iter-max) NUM_ITER_MAX="$2"; shift 2 ;;
    --normalize-scale) NORMALIZE_SCALE=1; shift 1 ;;
    --no-umap) NO_UMAP=1; shift 1 ;;
    --umap-n-components) UMAP_N_COMPONENTS="$2"; shift 2 ;;
    --umap-n-neighbors) UMAP_N_NEIGHBORS="$2"; shift 2 ;;
    --umap-min-dist) UMAP_MIN_DIST="$2"; shift 2 ;;
    --umap-metric) UMAP_METRIC="$2"; shift 2 ;;
    --umap-seed) UMAP_SEED="$2"; shift 2 ;;
    --struct-suffix) STRUCT_SUFFIX="$2"; shift 2 ;;
    --struct-metric) STRUCT_METRIC="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift 1 ;;
    --list-only) LIST_ONLY=1; shift 1 ;;
    -h|--help)
      echo "Usage: $0 --search WORD --csv-dir DIR --out-dir DIR [--embed-script PATH] [--fgw-script PATH] [--model M] [--method pca|umap] [--dims 2|3] [--device cpu|cuda] [--alpha A] [--mass M] [--metric M] [--nb-dummies N] [--num-iter-max N] [--normalize-scale] [--no-umap] [--umap-n-components N] [--umap-n-neighbors N] [--umap-min-dist F] [--umap-metric M] [--umap-seed N] [--struct-suffix S] [--struct-metric M] [--rebuild] [--list-only]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$SEARCH" ]; then
  echo "Usage: $0 --search WORD [--csv-dir DIR] [--out-dir DIR] [options] (see --help)" >&2
  echo "  --csv-dir and --out-dir both default to results/<search>/" >&2
  exit 1
fi

# Per-search home: default both csv-dir (where stage 1 wrote) and out-dir (where
# stage 2/3 write) to results/<search>/, so the only required argument is
# --search. Either can still be overridden explicitly.
SEARCH_SLUG=$(printf '%s' "$SEARCH" | tr -c 'A-Za-z0-9_-' '_')
DEFAULT_DIR="${OUT_DIR_BASE:-results}/${SEARCH_SLUG}"
[ -z "$CSV_DIR" ] && CSV_DIR="$DEFAULT_DIR"
[ -z "$OUT_DIR" ] && OUT_DIR="$DEFAULT_DIR"


if [ ! -d "$CSV_DIR" ]; then
  echo "ERROR: $CSV_DIR is not a directory" >&2
  exit 1
fi

if [ ! -f "$EMBED_SCRIPT" ]; then
  echo "ERROR: embedding script not found at $EMBED_SCRIPT (use --embed-script)" >&2
  exit 1
fi

if [ ! -f "$FGW_SCRIPT" ]; then
  echo "ERROR: FGW script not found at $FGW_SCRIPT (use --fgw-script)" >&2
  exit 1
fi

# Discover decades from CSV filenames like insan_1820s_results.csv, scoped
# specifically to the given --search term so a --csv-dir containing CSVs
# for multiple search terms doesn't get them mixed up.
DECADES=$(find "$CSV_DIR" -maxdepth 1 -name "${SEARCH}_*_results.csv" -print 2>/dev/null \
  | sed -E 's/.*_([0-9]{4})s_results\.csv$/\1/' \
  | grep -E '^[0-9]{4}$' \
  | sort -un)

if [ -z "$DECADES" ]; then
  echo "No ${SEARCH}_<decade>s_results.csv files found in $CSV_DIR" >&2
  exit 1
fi

echo "Found decades:"
echo "$DECADES" | tr '\n' ' '
echo
echo

if [ "$LIST_ONLY" -eq 1 ]; then
  exit 0
fi

mkdir -p "$OUT_DIR"

# --- Stage 2: embeddings, one run per decade ---------------------------------
EMBED_FAILED=""
EMBED_OK=""
for DECADE in $DECADES; do
  CSV=$(find "$CSV_DIR" -maxdepth 1 -name "${SEARCH}_${DECADE}s_results.csv" -print | head -n 1)
  BASE="$OUT_DIR/${SEARCH}_${DECADE}s"
  LOG="$OUT_DIR/${SEARCH}_${DECADE}s_embed.log"

  if [ "$REBUILD" -eq 0 ]; then
    # Decide whether the embed step needs to run at all. Required outputs:
    #   _embeddings.npy, _coords.csv, _pca90.npy
    # plus _pca90_umap.npy when UMAP is enabled. If ANY are missing, we
    # invoke the embed script, but pass --reuse-embeddings-if-exists so
    # the expensive BERT step is skipped when only the newer PCA-90 /
    # UMAP outputs are missing. This lets us retrofit UMAP onto old runs
    # for free.
    NEED_RUN=0
    NEED_BERT=0
    if [ ! -f "${BASE}_embeddings.npy" ] || [ ! -f "${BASE}_coords.csv" ]; then
      NEED_RUN=1
      NEED_BERT=1
    fi
    if [ ! -f "${BASE}_pca90.npy" ]; then
      NEED_RUN=1
    fi
    if [ "$NO_UMAP" -eq 0 ] && [ ! -f "${BASE}_pca90_umap.npy" ]; then
      NEED_RUN=1
    fi
    if [ "$NEED_RUN" -eq 0 ]; then
      echo "=== Decade ${DECADE} (embeddings) === all outputs exist, skipping"
      EMBED_OK="$EMBED_OK $DECADE"
      continue
    fi
    if [ "$NEED_BERT" -eq 0 ]; then
      echo "=== Decade ${DECADE} (embeddings) === reusing cached BERT, running PCA-90/UMAP only"
    fi
  fi

  echo "=== Decade ${DECADE} (embeddings) ==="
  EXTRA_ARGS=()
  # Retrofit: reuse cached BERT outputs when the raw embeddings and coords
  # already exist but the PCA-90 / UMAP outputs are missing. Set inside
  # the skip-logic block above; safe to pass unconditionally since the
  # embed script only activates the cache path when both cached files
  # actually exist on disk.
  if [ "$REBUILD" -eq 0 ] && [ -n "${NEED_BERT:-}" ] && [ "${NEED_BERT:-1}" -eq 0 ]; then
    EXTRA_ARGS+=(--reuse-embeddings-if-exists)
  fi
  # UMAP pass-through: each optional flag is only appended if set. This lets
  # us forward exactly what the user asked for without inheriting stale
  # defaults from this script.
  if [ "$NO_UMAP" -eq 1 ]; then
    EXTRA_ARGS+=(--no-umap)
  fi
  if [ -n "$UMAP_N_COMPONENTS" ]; then
    EXTRA_ARGS+=(--umap-n-components "$UMAP_N_COMPONENTS")
  fi
  if [ -n "$UMAP_N_NEIGHBORS" ]; then
    EXTRA_ARGS+=(--umap-n-neighbors "$UMAP_N_NEIGHBORS")
  fi
  if [ -n "$UMAP_MIN_DIST" ]; then
    EXTRA_ARGS+=(--umap-min-dist "$UMAP_MIN_DIST")
  fi
  if [ -n "$UMAP_METRIC" ]; then
    EXTRA_ARGS+=(--umap-metric "$UMAP_METRIC")
  fi
  if [ -n "$UMAP_SEED" ]; then
    EXTRA_ARGS+=(--umap-seed "$UMAP_SEED")
  fi

  if python3 "$EMBED_SCRIPT" \
      --input "$CSV" \
      --model "$MODEL" \
      --method "$METHOD" \
      --dims "$DIMS" \
      --device "$DEVICE" \
      --output "${BASE}.html" \
      "${EXTRA_ARGS[@]}" > "$LOG" 2>&1; then
    echo "  ok -- $(tail -n 1 "$LOG")"
    EMBED_OK="$EMBED_OK $DECADE"
  else
    echo "  FAILED -- see $LOG" >&2
    EMBED_FAILED="$EMBED_FAILED $DECADE"
  fi
done

echo
if [ -n "$EMBED_FAILED" ]; then
  echo "Decades whose embedding step failed (excluded from pairing):$EMBED_FAILED" >&2
fi

EMBED_OK_COUNT=$(echo $EMBED_OK | wc -w)
if [ "$EMBED_OK_COUNT" -lt 2 ]; then
  echo "Only ${EMBED_OK_COUNT} decade(s) have usable embeddings -- need at least 2 to form a consecutive pair."
  exit 0
fi

# --- Stage 3: FGW distance, one run per consecutive pair of successful decades ---
PAIR_FAILED=""
SUMMARY_FILES=""
PREV=""
for DECADE in $EMBED_OK; do
  if [ -n "$PREV" ]; then
    BASE1="$OUT_DIR/${SEARCH}_${PREV}s"
    BASE2="$OUT_DIR/${SEARCH}_${DECADE}s"
    FGW_BASE="$OUT_DIR/${SEARCH}_fgw_${PREV}_${DECADE}"
    LOG="$OUT_DIR/${SEARCH}_fgw_${PREV}_${DECADE}.log"

    echo "=== FGW: ${PREV}s vs ${DECADE}s ==="
    FGW_EXTRA=()
    if [ "$NORMALIZE_SCALE" -eq 1 ]; then
      FGW_EXTRA+=(--normalize-scale)
    fi
    # Path B: if --struct-suffix is set, construct the per-corpus struct
    # paths from the two base paths and check both exist before passing
    # them through. If only one exists (or neither), fall back to Path A
    # for this pair with a warning rather than failing the pair -- letting
    # a single missing pca90_umap.npy silently break FGW for the pair would
    # be unfriendly, since UMAP is legitimately skipped for marginal decades.
    if [ -n "$STRUCT_SUFFIX" ]; then
      STRUCT1="${BASE1}${STRUCT_SUFFIX}"
      STRUCT2="${BASE2}${STRUCT_SUFFIX}"
      if [ -f "$STRUCT1" ] && [ -f "$STRUCT2" ]; then
        FGW_EXTRA+=(--struct1 "$STRUCT1" --struct2 "$STRUCT2")
        if [ -n "$STRUCT_METRIC" ]; then
          FGW_EXTRA+=(--struct-metric "$STRUCT_METRIC")
        fi
      else
        echo "  WARNING: struct file(s) missing for this pair -- falling back to Path A." >&2
        [ ! -f "$STRUCT1" ] && echo "           missing: $STRUCT1" >&2
        [ ! -f "$STRUCT2" ] && echo "           missing: $STRUCT2" >&2
      fi
    fi

    if python3 "$FGW_SCRIPT" \
        --emb1 "${BASE1}_embeddings.npy" \
        --emb2 "${BASE2}_embeddings.npy" \
        --meta1 "${BASE1}_coords.csv" \
        --meta2 "${BASE2}_coords.csv" \
        --alpha "$ALPHA" \
        --mass "$MASS" \
        --metric "$METRIC" \
        --nb-dummies "$NB_DUMMIES" \
        --num-iter-max "$NUM_ITER_MAX" \
        --output "${FGW_BASE}.png" \
        "${FGW_EXTRA[@]}" > "$LOG" 2>&1; then
      echo "  ok -- $(grep 'Fused shift score' "$LOG")"
      SUMMARY_FILES="$SUMMARY_FILES ${PREV}:${DECADE}:${FGW_BASE}_summary.csv"
    else
      echo "  FAILED -- see $LOG" >&2
      PAIR_FAILED="$PAIR_FAILED ${PREV}-${DECADE}"
    fi
  fi
  PREV="$DECADE"
done

echo
if [ -n "$PAIR_FAILED" ]; then
  echo "Pairs that failed:$PAIR_FAILED" >&2
fi

if [ -z "$SUMMARY_FILES" ]; then
  echo "No successful FGW pairs -- nothing to combine."
  exit 1
fi

echo "Combining per-pair summaries into $OUT_DIR/${SEARCH}_fgw_summary_all_pairs.csv"
python3 - "$OUT_DIR/${SEARCH}_fgw_summary_all_pairs.csv" $SUMMARY_FILES <<'PYEOF'
import csv
import sys

combined_path = sys.argv[1]
entries = sys.argv[2:]  # each is "decade1:decade2:path"

with open(combined_path, "w", newline="", encoding="utf-8-sig") as out_f:
    writer = None
    for entry in entries:
        decade1, decade2, path = entry.split(":", 2)
        with open(path, newline="", encoding="utf-8-sig") as in_f:
            reader = csv.DictReader(in_f)
            row = next(reader)
        row = {"decade1": decade1, "decade2": decade2, **row}
        if writer is None:
            writer = csv.DictWriter(out_f, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
print(f"Combined {len(entries)} pair(s) -> {combined_path}")
PYEOF