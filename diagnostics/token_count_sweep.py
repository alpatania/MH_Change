#!/usr/bin/env python3
"""Run every search_arg in search_terms.csv against every decade's SQLite
DB and produce a token-count matrix.

Design principle: reuse coha_build.py's prefix_bounds so normalization is
identical to what the pipeline will actually match. The count query is a
COUNT(*) version of find_matches's WHERE clause, so a cell in the output
is exactly the number of token-index positions in that decade's DB that
find_matches would return for that search_arg (with the same POS filter
that will be applied downstream).

Usage:
  python3 token_count_sweep.py \\
      --terms-csv search_terms.csv \\
      --sqlite-dir results \\
      --out-csv token_counts.csv \\
      [--floor 50]

Inputs:
  --terms-csv    CSV with columns surface_form, search_arg, kind, pos_filter,
                 ambiguity_flag, idiom_family, notes.
  --sqlite-dir   Directory containing corpus_search_<decade>.sqlite files
                 (as produced by run_all_decades.sh). Decades are discovered
                 from filenames.
  --out-csv      Output CSV. Rows = one per search_arg. Columns = surface_form,
                 search_arg, pos_filter, ambiguity_flag, then one column per
                 decade (labeled like "1820"), then n_decades_pass (number of
                 decades whose count >= --floor), then min_count / max_count /
                 total_count.
  --floor        Minimum tokens per decade to count that decade as "passing".
                 Default 50.

Failures on individual (search_arg, decade) cells are recorded as -1 in
the output and printed to stderr, without stopping the whole sweep.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

# Reuse the exact same prefix_bounds the pipeline uses. Sys-path hack so we
# can import from the user's coha_build.py without moving files around.
# Reuse the exact same prefix_bounds the pipeline uses, so the counts here are
# what find_matches would actually return. This script normally lives in
# diagnostics/ while coha_build.py sits in the project root, so BOTH the script
# dir and its parent go on sys.path -- searching only the script dir would fail
# silently into the fallback below and risk drifting from the real matcher.
_here = Path(__file__).resolve().parent
for _candidate in (_here, _here.parent):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
try:
    from coha_build import prefix_bounds
except ImportError:
    # Fallback: inline reimplementation. Must match coha_build.py exactly.
    # If you hit this, the counts are still produced but are no longer
    # guaranteed to track coha_build -- check that the two agree.
    print("WARNING: could not import prefix_bounds from coha_build.py; "
          "using an inline copy that may drift from the real matcher.",
          file=sys.stderr)

    def prefix_bounds(value: str) -> tuple[str, str]:
        value = value.casefold()
        return value, value + "\U0010ffff"


SQLITE_NAME_RE = re.compile(r"^corpus_search_(\d{4})\.sqlite$")


def discover_decades(sqlite_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted [(decade, path), ...] of corpus_search_<decade>.sqlite."""
    found: list[tuple[int, Path]] = []
    for path in sqlite_dir.iterdir():
        m = SQLITE_NAME_RE.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    found.sort(key=lambda x: x[0])
    return found


def build_count_sql(search_arg: str, pos_filter: str) -> tuple[str, list[object]]:
    """Return (sql, params) for a COUNT(*) query matching find_matches's
    predicates for this search_arg. If pos_filter is non-empty, every token
    slot must match one of the comma-separated CLAWS7 POS prefixes.

    We do NOT join to documents here: within one corpus_search_<decade>.sqlite,
    every document is in-decade by construction, so the year predicate is a
    no-op and skipping the join speeds up the sweep considerably.
    """
    terms = search_arg.split()
    if not terms:
        raise ValueError("empty search_arg")

    aliases = [f"t{i}" for i in range(len(terms))]
    joins = " ".join(
        f"JOIN tokens {aliases[i]} ON {aliases[i]}.text_id=t0.text_id "
        f"AND {aliases[i]}.token_index=t0.token_index+{i}"
        for i in range(1, len(terms))
    )

    pos_prefixes = [p.strip() for p in pos_filter.split(",") if p.strip()]

    predicates: list[str] = []
    parameters: list[object] = []
    for alias, term in zip(aliases, terms):
        low, high = prefix_bounds(term)
        predicates.append(
            f"(({alias}.word_norm>=? AND {alias}.word_norm<?) "
            f"OR ({alias}.lemma_norm>=? AND {alias}.lemma_norm<?))"
        )
        parameters.extend((low, high, low, high))
        if pos_prefixes:
            pos_clause = " OR ".join(f"{alias}.pos LIKE ?" for _ in pos_prefixes)
            predicates.append(f"({pos_clause})")
            parameters.extend(f"{p}%" for p in pos_prefixes)

    sql = f"SELECT COUNT(*) FROM tokens t0 {joins} WHERE {' AND '.join(predicates)}"
    return sql, parameters


def count_matches(conn: sqlite3.Connection, search_arg: str, pos_filter: str) -> int:
    sql, params = build_count_sql(search_arg, pos_filter)
    (n,) = conn.execute(sql, params).fetchone()
    return int(n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms-csv", type=Path, required=True)
    parser.add_argument("--sqlite-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--floor", type=int, default=50)
    args = parser.parse_args()

    if not args.terms_csv.is_file():
        print(f"ERROR: {args.terms_csv} not found", file=sys.stderr)
        return 1
    if not args.sqlite_dir.is_dir():
        print(f"ERROR: {args.sqlite_dir} is not a directory", file=sys.stderr)
        return 1

    decades = discover_decades(args.sqlite_dir)
    if not decades:
        print(
            f"ERROR: no corpus_search_<decade>.sqlite in {args.sqlite_dir}",
            file=sys.stderr,
        )
        return 1
    print(f"Found {len(decades)} decades: "
          f"{decades[0][0]}s..{decades[-1][0]}s", file=sys.stderr)

    with args.terms_csv.open("r", encoding="utf-8-sig", newline="") as f:
        terms = list(csv.DictReader(f))
    print(f"Loaded {len(terms)} search terms", file=sys.stderr)

    # Open all decade connections once; SQLite handles read-only fine.
    conns: dict[int, sqlite3.Connection] = {}
    for decade, path in decades:
        conns[decade] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    decade_cols = [str(d) for d, _ in decades]
    out_fields = [
        "surface_form", "search_arg", "kind", "pos_filter", "ambiguity_flag",
        "idiom_family",
        *decade_cols,
        "n_decades_pass", "min_count", "max_count", "total_count", "notes",
    ]

    n_below_floor_cells = 0
    n_failed_cells = 0
    n_terms_all_below = 0

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for i, term in enumerate(terms, 1):
            search_arg = term["search_arg"]
            pos_filter = term.get("pos_filter", "") or ""
            counts: dict[str, int] = {}
            errors: list[str] = []
            for decade, _ in decades:
                try:
                    counts[str(decade)] = count_matches(
                        conns[decade], search_arg, pos_filter,
                    )
                except sqlite3.Error as e:
                    counts[str(decade)] = -1
                    errors.append(f"{decade}:{e}")
                    n_failed_cells += 1

            valid = [c for c in counts.values() if c >= 0]
            passing = [c for c in valid if c >= args.floor]
            n_pass = len(passing)
            n_below_floor_cells += sum(
                1 for c in valid if 0 <= c < args.floor
            )
            if n_pass == 0:
                n_terms_all_below += 1

            row = {
                "surface_form": term["surface_form"],
                "search_arg": search_arg,
                "kind": term["kind"],
                "pos_filter": pos_filter,
                "ambiguity_flag": term["ambiguity_flag"],
                "idiom_family": term.get("idiom_family", ""),
                **counts,
                "n_decades_pass": n_pass,
                "min_count": min(valid) if valid else -1,
                "max_count": max(valid) if valid else -1,
                "total_count": sum(valid) if valid else -1,
                "notes": term.get("notes", ""),
            }
            writer.writerow(row)

            if errors:
                print(f"[{i}/{len(terms)}] {search_arg}: errors: {errors}",
                      file=sys.stderr)

            if i % 20 == 0:
                print(f"...processed {i}/{len(terms)}", file=sys.stderr)

    for conn in conns.values():
        conn.close()

    print(file=sys.stderr)
    print(f"Done. Wrote {args.out_csv}", file=sys.stderr)
    print(f"  {n_below_floor_cells} (search_arg, decade) cells "
          f"below floor={args.floor}", file=sys.stderr)
    print(f"  {n_failed_cells} cells failed with SQL errors", file=sys.stderr)
    print(f"  {n_terms_all_below} search_args pass in ZERO decades "
          f"(candidates for removal)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())