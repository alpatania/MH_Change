#!/usr/bin/env bash
#
# Run coha_build.py once per decade, discovering which decades are actually
# present in --db-dir from the db_<genre>_<decade>.zip filenames rather than
# assuming a fixed range -- this adapts to whatever you have on disk instead
# of requiring you to know or type out every decade COHA covers.
#
# Usage:
#   ./run_all_decades.sh --db-dir coha-db-003 --lexicon-txt coha-lexicon.txt \
#     --sources-txt coha-sources.txt --search insan \
#     [--genre acad] [--out-dir results] [--script coha_build.py] \
#     [--context 20] [--pos-filter JJ] [--rebuild] [--list-only]
#
# Each decade's sqlite database, CSV, and build log are written into
# --out-dir, kept as separate per-decade files. A decade that fails does
# not stop the others -- failures are collected and reported at the end,
# along with each one's log file. There is no combined CSV: downstream
# steps (final_layer_embeddings.py, FGW_distance.py) consume each decade's
# CSV individually, and a merged file would just be unused output.
#
# --pos-filter is passed through to coha_build.py verbatim; see that
# script's --help for the CLAWS7 POS-prefix syntax (e.g. "JJ" for
# adjectives, "JJ,NN" for adjectives or any noun). Omit to disable.
#
# --rebuild is now OPT-IN. By default the SQLite DB for a decade is
# reused if it already exists in --out-dir, which is what you want when
# running the same decade for many different search terms in a row:
# the DB content is search-agnostic, so rebuilding it every time would
# waste time. Pass --rebuild to force a fresh build (e.g. if the
# underlying db_*.zip files changed).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SCRIPT="$SCRIPT_DIR/coha_build.py"
OUT_DIR=""
OUT_DIR_EXPLICIT=""
CONTEXT=20
GENRE=""
DB_DIR=""
LEXICON_TXT=""
SOURCES_TXT=""
SEARCH=""
POS_FILTER=""
REBUILD=0
LIST_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --db-dir) DB_DIR="$2"; shift 2 ;;
    --lexicon-txt) LEXICON_TXT="$2"; shift 2 ;;
    --sources-txt) SOURCES_TXT="$2"; shift 2 ;;
    --search) SEARCH="$2"; shift 2 ;;
    --genre) GENRE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; OUT_DIR_EXPLICIT=1; shift 2 ;;
    --script) SCRIPT="$2"; shift 2 ;;
    --context) CONTEXT="$2"; shift 2 ;;
    --pos-filter) POS_FILTER="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift 1 ;;
    --list-only) LIST_ONLY=1; shift 1 ;;
    -h|--help)
      echo "Usage: $0 --db-dir DIR --lexicon-txt FILE --sources-txt FILE --search WORD [--genre G] [--out-dir DIR] [--script PATH] [--context N] [--pos-filter TAGS] [--rebuild] [--list-only]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$DB_DIR" ] || [ -z "$LEXICON_TXT" ] || [ -z "$SOURCES_TXT" ] || [ -z "$SEARCH" ]; then
  echo "Usage: $0 --db-dir DIR --lexicon-txt FILE --sources-txt FILE --search WORD [--genre G] [--out-dir DIR] [--script PATH] [--context N] [--pos-filter TAGS] [--rebuild] [--list-only]" >&2
  exit 1
fi

# Per-search home: unless --out-dir was given explicitly, everything for this
# search lands in results/<search-slug>/ so distinct searches never collide and
# a whole search can be archived or deleted as one folder. Files keep the
# <search>_ prefix inside the folder, so they stay self-identifying if moved.
SEARCH_SLUG=$(printf '%s' "$SEARCH" | tr -c 'A-Za-z0-9_-' '_')
if [ -z "$OUT_DIR_EXPLICIT" ]; then
  OUT_DIR="${OUT_DIR_BASE:-results}/${SEARCH_SLUG}"
fi

if [ ! -d "$DB_DIR" ]; then
  echo "ERROR: $DB_DIR is not a directory" >&2
  exit 1
fi

# Discover decades from filenames like db_acad_1820.zip. This intentionally
# does not filter by genre at discovery time, even if --genre is set later,
# since a genre might be absent for some decades and present for others --
# coha_build.py itself will report "no files found" per decade if that combo
# doesn't exist, rather than this script silently guessing which decades a
# given genre covers.
DECADES=$(find "$DB_DIR" -maxdepth 1 -name 'db_*.zip' -print 2>/dev/null \
  | sed -E 's/.*db_[^_]+_([0-9]{4})\.zip$/\1/' \
  | sort -un)

if [ -z "$DECADES" ]; then
  echo "No db_*.zip files found in $DB_DIR" >&2
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

FAILED=""
for DECADE in $DECADES; do
  echo "=== Decade $DECADE ==="
  LOG="$OUT_DIR/${DECADE}.log"
  CSV="$OUT_DIR/${SEARCH_SLUG}_${DECADE}s_results.csv"

  EXTRA_ARGS=()
  if [ -n "$GENRE" ]; then
    EXTRA_ARGS+=(--genre "$GENRE")
  fi
  if [ -n "$POS_FILTER" ]; then
    EXTRA_ARGS+=(--pos-filter "$POS_FILTER")
  fi
  if [ "$REBUILD" -eq 1 ]; then
    EXTRA_ARGS+=(--rebuild)
  fi

  if python3 "$SCRIPT" \
      --db-dir "$DB_DIR" \
      --lexicon-txt "$LEXICON_TXT" \
      --sources-txt "$SOURCES_TXT" \
      --decade "$DECADE" \
      "${EXTRA_ARGS[@]}" \
      --search "$SEARCH" \
      --context "$CONTEXT" \
      --sqlite "$OUT_DIR/corpus_search_${DECADE}.sqlite" \
      --csv "$CSV" > "$LOG" 2>&1; then
    echo "  ok -- $(tail -n 1 "$LOG")"
  else
    echo "  FAILED -- see $LOG" >&2
    FAILED="$FAILED $DECADE"
  fi
done

echo

if [ -n "$FAILED" ]; then
  echo "Decades that failed:$FAILED" >&2
  exit 1
fi

echo "All decades completed."