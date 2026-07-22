#!/usr/bin/env python3
"""Drive the full Sheaf-NLP pipeline from search_terms.csv.

For every row in --terms-csv, this runs the same four stages the README
documents:
  1. run_all_decades.sh  --search <search_arg> [--pos-filter <pos>]
     (builds per-decade CSVs of matched contexts; reuses SQLite DBs
     across terms since DB content is search-agnostic)
  2. run_embedding_and_distance.sh  --search <slug>
     (per-decade RoBERTa embeddings + pairwise FGW distances)
  3. fgw_build_linkage.py --search <slug>
     (single-linkage trees, required by the tanglegram)
  4. fgw_tanglegram.py / fgw_sankey.py --search <slug>
     (the two visualisations)

Stages 3-4 can be skipped with --no-viz if you only want the numbers.

PATHS: --out-dir is the BASE location, matching the shell scripts and the
leaf Python scripts. Each stage writes to <out-dir>/results/<slug>/. The
default is '.', i.e. ./results/<slug>/. Do NOT pass '--out-dir results' --
that would produce results/results/<slug>/.

Slug computation replicates run_all_decades.sh's SEARCH_SLUG rule
(non-[A-Za-z0-9_-] characters replaced with '_') so that later stages find
the files stage 1 produced. Passing the slug rather than the raw search
to stage 2 is what makes phrase searches ('mental hospital') work.

Optional --token-counts filters rows using the sweep output: only rows
whose n_decades_pass >= --min-decades-pass are run. Without --token-counts,
every row in --terms-csv is run.

Failures on individual terms do not stop the run. A summary is printed
at the end listing failed terms and their log files.

Usage:
  python3 run_pipeline_from_csv.py \\
      --terms-csv search_terms.csv \\
      --token-counts token_counts.csv \\
      --min-decades-pass 3 \\
      --db-dir coha-db-003 \\
      --lexicon-txt coha-lexicon.txt \\
      --sources-txt coha-sources.txt \\
      [--out-dir .] \\
      [--genre acad] [--context 20] \\
      [--alpha 0.5] [--mass 0.8] \\
      [--struct-suffix _pca90_umap.npy] [--struct-metric cosine] \\
      [--no-viz]     # skip stages 3-4
      [--limit N]    # only process first N surviving terms (for pilots)
      [--dry-run]    # print planned invocations, do not execute
      [--stage1-only]  # skip stages 2-4
      [--rebuild-db]   # force stage 1 DB rebuild on first term
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def slugify(search: str) -> str:
    """Replicate run_all_decades.sh's SEARCH_SLUG rule exactly:
        printf '%s' "$SEARCH" | tr -c 'A-Za-z0-9_-' '_'
    `tr` operates on bytes, not Unicode code points, so multi-byte UTF-8
    characters produce one underscore PER BYTE. That matters for entries
    like 'séance' where 'é' is two bytes and bash produces 's__ance'.
    We match that byte-level behavior so the slug the wrapper computes
    is identical to the one run_all_decades.sh writes to disk; otherwise
    stage 2 would look for a file stage 1 never created.
    """
    allowed = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    out = bytearray()
    for b in search.encode("utf-8"):
        out.append(b if b in allowed else ord("_"))
    return out.decode("ascii")


def load_terms(
    terms_csv: Path,
    token_counts: Path | None,
    min_pass: int,
) -> list[dict]:
    """Load terms from search_terms.csv, optionally filtered by the sweep."""
    with terms_csv.open("r", encoding="utf-8-sig", newline="") as f:
        terms = list(csv.DictReader(f))

    if token_counts is None:
        return terms

    with token_counts.open("r", encoding="utf-8-sig", newline="") as f:
        counts = {r["search_arg"]: r for r in csv.DictReader(f)}

    kept: list[dict] = []
    dropped_missing = 0
    dropped_below = 0
    for t in terms:
        arg = t["search_arg"]
        if arg not in counts:
            dropped_missing += 1
            continue
        try:
            n_pass = int(counts[arg].get("n_decades_pass", "0"))
        except ValueError:
            n_pass = 0
        if n_pass < min_pass:
            dropped_below += 1
            continue
        # Enrich with sweep summary for logging
        t = dict(t)
        t["_n_decades_pass"] = n_pass
        t["_total_count"] = counts[arg].get("total_count", "?")
        kept.append(t)

    print(f"Loaded {len(terms)} terms; kept {len(kept)}, "
          f"dropped {dropped_below} below floor, "
          f"{dropped_missing} missing from token_counts", file=sys.stderr)
    return kept


def run_command(
    cmd: list[str],
    log_path: Path,
    dry_run: bool,
    label: str,
) -> tuple[bool, float]:
    """Run cmd, teeing output to log_path. Returns (success, elapsed_seconds)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"    [DRY-RUN] {label}: {' '.join(cmd)}")
        print(f"    [DRY-RUN] log -> {log_path}")
        return True, 0.0
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    return result.returncode == 0, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terms-csv", type=Path, required=True)
    p.add_argument("--token-counts", type=Path, default=None)
    p.add_argument("--min-decades-pass", type=int, default=1)

    p.add_argument("--db-dir", type=Path, required=True)
    p.add_argument("--lexicon-txt", type=Path, required=True)
    p.add_argument("--sources-txt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("."),
                   help="BASE location; each stage writes to "
                        "<out-dir>/results/<slug>/ (default: '.'). Do not "
                        "pass 'results' -- that yields results/results/<slug>/.")
    p.add_argument("--genre", default=None)
    p.add_argument("--context", type=int, default=20)

    p.add_argument("--stage1-script", type=Path, default=Path("./run_all_decades.sh"))
    p.add_argument("--stage2-script", type=Path,
                   default=Path("./run_embedding_and_distance.sh"))
    p.add_argument("--coha-script", type=Path, default=Path("./coha_build.py"))
    p.add_argument("--embed-script", type=Path,
                   default=Path("./final_layer_embeddings.py"))
    p.add_argument("--fgw-script", type=Path, default=Path("./FGW_distance.py"))
    p.add_argument("--linkage-script", type=Path,
                   default=Path("./fgw_build_linkage.py"))
    p.add_argument("--tanglegram-script", type=Path,
                   default=Path("./fgw_tanglegram.py"))
    p.add_argument("--sankey-script", type=Path, default=Path("./fgw_sankey.py"))
    p.add_argument("--no-viz", action="store_true",
                   help="Skip stages 3-4 (linkage + tanglegram + sankey); "
                        "run only the CSVs, embeddings and FGW distances.")
    p.add_argument("--struct-suffix", default="_pca90_umap.npy",
                   help="Passed to stage 2 as --struct-suffix (FGW Path B). "
                        "Set to '' to run Path A. Default matches the README.")
    p.add_argument("--struct-metric", default="cosine",
                   help="Passed to stage 2 as --struct-metric when "
                        "--struct-suffix is non-empty.")
    p.add_argument("--metric", default="cosine",
                   help="Metric for the linkage step; must match the metric "
                        "used for the FGW ultrametrics.")

    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mass", type=float, default=0.8)

    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N surviving terms (0 = all)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stage1-only", action="store_true",
                   help="Skip stage 2 (embeddings + FGW)")
    p.add_argument("--rebuild-db", action="store_true",
                   help="Pass --rebuild to stage 1 for the FIRST term only "
                        "(subsequent terms reuse the fresh DB)")
    args = p.parse_args()

    if not args.terms_csv.is_file():
        print(f"ERROR: {args.terms_csv} not found", file=sys.stderr)
        return 1

    terms = load_terms(args.terms_csv, args.token_counts, args.min_decades_pass)
    if args.limit > 0:
        terms = terms[:args.limit]
        print(f"Limiting to first {len(terms)} terms", file=sys.stderr)

    if not terms:
        print("No terms to process.", file=sys.stderr)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    driver_log_dir = args.out_dir / "driver_logs"
    driver_log_dir.mkdir(parents=True, exist_ok=True)

    # Per-run manifest of what got processed
    manifest_path = args.out_dir / "pipeline_manifest.csv"
    manifest_fields = [
        "surface_form", "search_arg", "slug", "pos_filter", "kind",
        "idiom_family", "n_decades_pass", "stage1_status",
        "stage1_seconds", "stage2_status", "stage2_seconds",
        "viz_status", "viz_seconds",
        "stage1_log", "stage2_log",
    ]
    if args.dry_run:
        manifest_file = None
        manifest_writer = None
    else:
        manifest_file = manifest_path.open("w", encoding="utf-8", newline="")
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        manifest_writer.writeheader()

    failed: list[tuple[str, str, str]] = []  # (search_arg, stage, log_path)
    n_ok = 0

    for i, t in enumerate(terms, 1):
        search_arg = t["search_arg"]
        pos_filter = (t.get("pos_filter") or "").strip()
        slug = slugify(search_arg)
        kind = t.get("kind", "")
        family = t.get("idiom_family", "")
        n_pass = t.get("_n_decades_pass", "?")

        header = (f"\n=== [{i}/{len(terms)}] {t['surface_form']!r} "
                  f"(arg={search_arg!r}, slug={slug!r}, pos={pos_filter or '-'}, "
                  f"kind={kind}, n_pass={n_pass}) ===")
        print(header)

        stage1_log = driver_log_dir / f"{slug}_stage1.log"
        stage2_log = driver_log_dir / f"{slug}_stage2.log"

        # --- Stage 1: run_all_decades.sh
        stage1_cmd = [
            "bash", str(args.stage1_script),
            "--db-dir", str(args.db_dir),
            "--lexicon-txt", str(args.lexicon_txt),
            "--sources-txt", str(args.sources_txt),
            "--search", search_arg,
            "--out-dir", str(args.out_dir),
            "--script", str(args.coha_script),
            "--context", str(args.context),
        ]
        if args.genre:
            stage1_cmd += ["--genre", args.genre]
        if pos_filter:
            stage1_cmd += ["--pos-filter", pos_filter]
        # Rebuild only on the first term if requested
        if args.rebuild_db and i == 1:
            stage1_cmd += ["--rebuild"]

        stage1_ok, stage1_secs = run_command(
            stage1_cmd, stage1_log, args.dry_run, "stage1")
        if stage1_ok:
            print(f"  stage1 ok ({stage1_secs:.1f}s)")
        else:
            print(f"  stage1 FAILED (see {stage1_log})", file=sys.stderr)
            failed.append((search_arg, "stage1", str(stage1_log)))

        # --- Stage 2: run_embedding_and_distance.sh
        stage2_ok = None
        stage2_secs = 0.0
        if args.stage1_only:
            stage2_status = "skipped"
        elif not stage1_ok and not args.dry_run:
            stage2_status = "skipped_stage1_failed"
            print(f"  stage2 skipped (stage1 failed)")
        else:
            stage2_cmd = [
                "bash", str(args.stage2_script),
                "--search", slug,
                "--out-dir", str(args.out_dir),
                "--embed-script", str(args.embed_script),
                "--fgw-script", str(args.fgw_script),
                "--alpha", str(args.alpha),
                "--mass", str(args.mass),
                "--metric", str(args.metric),
            ]
            # FGW Path B, matching the README's manual invocation. Without
            # this, batch runs would silently use Path A and produce a
            # different structure term than manual runs.
            if args.struct_suffix:
                stage2_cmd += ["--struct-suffix", args.struct_suffix,
                               "--struct-metric", args.struct_metric]
            stage2_ok, stage2_secs = run_command(
                stage2_cmd, stage2_log, args.dry_run, "stage2")
            if stage2_ok:
                stage2_status = "ok"
                print(f"  stage2 ok ({stage2_secs:.1f}s)")
            else:
                stage2_status = "failed"
                print(f"  stage2 FAILED (see {stage2_log})", file=sys.stderr)
                failed.append((search_arg, "stage2", str(stage2_log)))

        # --- Stages 3-4: linkage matrices, then the two visualisations.
        # Gated on stage 2, because all three read the embeddings/coords it
        # produces. The tanglegram additionally requires stage 3's linkages,
        # so a linkage failure skips the tanglegram but not the sankey.
        viz_status = "skipped"
        viz_secs = 0.0
        if args.no_viz or args.stage1_only:
            viz_status = "skipped"
        elif stage2_ok is False and not args.dry_run:
            viz_status = "skipped_stage2_failed"
            print("  viz skipped (stage2 failed)")
        elif stage2_ok or args.dry_run:
            viz_log = driver_log_dir / f"{slug}_viz.log"
            viz_ok = True

            linkage_cmd = [
                sys.executable, str(args.linkage_script),
                "--search", slug,
                "--out-dir", str(args.out_dir),
                "--metric", str(args.metric),
            ]
            link_ok, link_secs = run_command(
                linkage_cmd, driver_log_dir / f"{slug}_linkage.log",
                args.dry_run, "linkage")
            viz_secs += link_secs
            if link_ok:
                print(f"  linkage ok ({link_secs:.1f}s)")
            else:
                print(f"  linkage FAILED (see {slug}_linkage.log)",
                      file=sys.stderr)
                failed.append((search_arg, "linkage",
                               str(driver_log_dir / f"{slug}_linkage.log")))
                viz_ok = False

            # Tanglegram needs the linkages; skip it if they failed.
            if link_ok or args.dry_run:
                tang_cmd = [
                    sys.executable, str(args.tanglegram_script),
                    "--search", slug,
                    "--out-dir", str(args.out_dir),
                ]
                t_ok, t_secs = run_command(
                    tang_cmd, driver_log_dir / f"{slug}_tanglegram.log",
                    args.dry_run, "tanglegram")
                viz_secs += t_secs
                print(f"  tanglegram {'ok' if t_ok else 'FAILED'} "
                      f"({t_secs:.1f}s)")
                if not t_ok:
                    failed.append((search_arg, "tanglegram",
                                   str(driver_log_dir / f"{slug}_tanglegram.log")))
                    viz_ok = False

            # Sankey reads coords + transport matrices, not linkages, so it
            # runs even if the linkage step failed.
            sankey_cmd = [
                sys.executable, str(args.sankey_script),
                "--search", slug,
                "--out-dir", str(args.out_dir),
            ]
            s_ok, s_secs = run_command(
                sankey_cmd, driver_log_dir / f"{slug}_sankey.log",
                args.dry_run, "sankey")
            viz_secs += s_secs
            print(f"  sankey {'ok' if s_ok else 'FAILED'} ({s_secs:.1f}s)")
            if not s_ok:
                failed.append((search_arg, "sankey",
                               str(driver_log_dir / f"{slug}_sankey.log")))
                viz_ok = False

            viz_status = "ok" if viz_ok else "failed"

        if manifest_writer is not None:
            manifest_writer.writerow({
                "surface_form": t["surface_form"],
                "search_arg": search_arg,
                "slug": slug,
                "pos_filter": pos_filter,
                "kind": kind,
                "idiom_family": family,
                "n_decades_pass": n_pass,
                "stage1_status": "ok" if stage1_ok else "failed",
                "stage1_seconds": f"{stage1_secs:.1f}",
                "stage2_status": stage2_status,
                "stage2_seconds": f"{stage2_secs:.1f}",
                "viz_status": viz_status,
                "viz_seconds": f"{viz_secs:.1f}",
                "stage1_log": str(stage1_log),
                "stage2_log": str(stage2_log),
            })
            manifest_file.flush()

        if stage1_ok and (args.stage1_only or stage2_ok) and \
                viz_status in ('ok', 'skipped'):
            n_ok += 1

    if manifest_file is not None:
        manifest_file.close()

    print()
    print(f"=== Pipeline driver summary ===")
    print(f"  Total terms:   {len(terms)}")
    print(f"  Fully ok:      {n_ok}")
    print(f"  Failures:      {len(failed)}")
    if failed:
        print("  Failed terms:")
        for arg, stage, log in failed:
            print(f"    - {arg} ({stage}) -> {log}")
    if manifest_writer is not None:
        print(f"  Manifest:      {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())