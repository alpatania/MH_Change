#!/usr/bin/env python3
"""Build a searchable COHA sample database and export n-gram occurrences.

Reads the corpusdata.org five-file distribution format, with the db tar
already extracted to a folder of per-genre-decade zip files, and the
lexicon and sources files already unzipped to plain text:

  - <db-dir>/           : a folder of db_<genre>_<decade>.zip files, each
                          holding one .txt of (textID, corpus_token_id,
                          wordID) rows, in document order.
  - <lexicon>.txt       : (wordID, word, lemma, PoS), tab-delimited.
  - <sources>.txt       : per-text metadata (textID, #words, genre, year,
                          source, title, pubInfo or similar; column set
                          may vary), tab-delimited with a header row.

wlp and text archives are NOT read. db + lexicon already reconstruct
word/lemma/PoS for every token, and the CSV export never uses raw_text,
so skipping wlp/text avoids parsing several GB of redundant data. If your
downstream pipeline needs the literal original document text (rather than
text reconstructed from tokens), that's a small addition to make later --
flag it and it can be wired back in.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# Change this value to search for another word or whitespace-separated n-gram.
# Prefix matching means "insan" finds insane, insanely, insanity, etc.
SEARCH_WORD = "insan"
CONTEXT_WORDS = 20

# Matches the per-genre-decade zip filenames inside the db folder, e.g.
# db_acad_1820.zip
DB_FILE_RE = re.compile(r"^db_(?P<genre>[^_]+)_(?P<decade>\d{4})\.zip$")


def decode(data: bytes) -> str:
    """Decode corpus files while tolerating the occasional legacy byte/NUL."""
    try:
        return data.decode("utf-8-sig").replace("\x00", "")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace").replace("\x00", "")


def split_records(text: str) -> list[str]:
    """Split into logical rows on real line terminators only (\\r\\n or a
    lone \\n) -- never on a bare \\r. Unlike str.splitlines(), which treats
    a lone \\r as its own line boundary, this preserves rows that contain
    embedded bare-CR characters inside a field value (confirmed present in
    the sources file, e.g. a soft line break inside a title field) instead
    of shredding one well-formed row into several bogus fragments.
    """
    return text.replace("\r\n", "\n").split("\n")


def _single_txt_member(zf: zipfile.ZipFile, label: str) -> str:
    names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
    if len(names) != 1:
        raise ValueError(f"Expected exactly one .txt file inside {label}, found {names}")
    return names[0]


def read_lexicon(lexicon_txt_path: Path) -> dict[int, tuple[str, str, str]]:
    """Read wordID -> (word, lemma, PoS) from the plain-text lexicon file."""
    data = lexicon_txt_path.read_bytes()

    lexicon: dict[int, tuple[str, str, str]] = {}
    for line_number, line in enumerate(split_records(decode(data)), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"Malformed lexicon line {line_number}: {line!r}")
        word_id, word, lemma, pos = fields
        lexicon[int(word_id)] = (word, lemma, pos)
    return lexicon


def read_sources(sources_txt_path: Path) -> dict[int, dict[str, str]]:
    """Read textID -> metadata dict from the plain-text sources file
    (tab-delimited, header row present). Column set varies across
    corpusdata.org releases, so this reads whatever headers are actually
    present rather than assuming a fixed schema.
    """
    data = sources_txt_path.read_bytes()

    lines = split_records(decode(data))
    if not lines:
        raise ValueError(f"{sources_txt_path.name} is empty")

    headers = [re.sub(r"^#+\s*", "", cell.strip().lower()) for cell in lines[0].split("\t")]
    metadata: dict[int, dict[str, str]] = {}
    skipped = 0
    for line_number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split("\t")
        fields = [re.sub(r"\s*[\r\n]+\s*", " ", field).strip() for field in fields]
        record = dict(zip(headers, fields))
        text_id = record.get("textid")
        if text_id:
            try:
                metadata[int(text_id)] = record
            except ValueError:
                skipped += 1
                print(
                    f"WARNING: skipping unparseable sources line {line_number}: {line!r}",
                    file=sys.stderr,
                )
    if skipped:
        print(
            f"WARNING: skipped {skipped:,} malformed row(s) in {sources_txt_path.name}; "
            "documents referencing those textIDs will have blank metadata.",
            file=sys.stderr,
        )
    return metadata


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE documents (
            text_id       INTEGER PRIMARY KEY,
            document_file TEXT NOT NULL,
            genre         TEXT,
            year          INTEGER,
            word_count    INTEGER,
            title         TEXT,
            author        TEXT,
            source        TEXT,
            raw_text      TEXT NOT NULL
        );

        CREATE TABLE tokens (
            text_id         INTEGER NOT NULL REFERENCES documents(text_id),
            token_index     INTEGER NOT NULL,
            corpus_token_id INTEGER NOT NULL,
            word_id         INTEGER NOT NULL,
            word            TEXT NOT NULL,
            lemma           TEXT NOT NULL,
            pos             TEXT NOT NULL,
            word_norm       TEXT NOT NULL,
            lemma_norm      TEXT NOT NULL,
            PRIMARY KEY (text_id, token_index)
        ) WITHOUT ROWID;

        CREATE INDEX tokens_word_norm ON tokens(word_norm);
        CREATE INDEX tokens_lemma_norm ON tokens(lemma_norm);
        """
    )


def find_db_files(
    db_dir: Path, decade: int | None, genre: str | None
) -> list[tuple[str, int, Path]]:
    """Select db_<genre>_<decade>.zip files matching the requested decade/genre.
    Both filters are optional: omitting --decade processes every decade in
    the folder, which is a lot of data; omitting --genre processes every
    genre for the requested decade.
    """
    matches: list[tuple[str, int, Path]] = []
    for path in db_dir.glob("db_*.zip"):
        match = DB_FILE_RE.match(path.name)
        if not match:
            continue
        file_genre = match["genre"]
        file_decade = int(match["decade"])
        if decade is not None and file_decade != decade:
            continue
        if genre is not None and file_genre.lower() != genre.lower():
            continue
        matches.append((file_genre, file_decade, path))
    return sorted(matches, key=lambda item: (item[1], item[0]))


def read_db_rows_for_file(zip_path: Path) -> dict[int, list[tuple[int, int]]]:
    """Read one db_<genre>_<decade>.zip and parse its (textID,
    corpus_token_id, wordID) rows, grouped by textID in file order.
    """
    with zipfile.ZipFile(zip_path) as zf:
        txt_name = _single_txt_member(zf, zip_path.name)
        data = zf.read(txt_name)

    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for line_number, line in enumerate(split_records(decode(data)), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"Malformed {zip_path.name} line {line_number}: {line!r}")
        text_id, corpus_token_id, word_id = map(int, fields)
        rows[text_id].append((corpus_token_id, word_id))
    return rows


def strip_leading_marker(
    text_id: int,
    rows: list[tuple[int, int]],
    lexicon: dict[int, tuple[str, str, str]],
) -> list[tuple[int, int]]:
    """Drop leading structural marker row(s) whose resolved word is exactly
    "@@<textID>" (the document-boundary token). Inline "@" placeholder runs
    elsewhere in the body -- a known artifact of this corpus family -- are
    left in place as ordinary tokens, matching the original script's intent.
    """
    marker = f"@@{text_id}"
    first_real = 0
    while first_real < len(rows):
        _, word_id = rows[first_real]
        resolved = lexicon.get(word_id)
        if resolved is None:
            raise ValueError(f"wordID {word_id} (text {text_id}) not found in lexicon")
        if resolved[0] != marker:
            break
        first_real += 1
    return rows[first_real:]


def build_database(
    database_path: Path,
    db_dir: Path,
    lexicon_txt_path: Path,
    sources_txt_path: Path,
    decade: int | None = None,
    genre: str | None = None,
) -> None:
    if database_path.exists():
        database_path.unlink()

    lexicon = read_lexicon(lexicon_txt_path)
    metadata = read_sources(sources_txt_path)

    with sqlite3.connect(database_path) as connection:
        create_schema(connection)
        files = find_db_files(db_dir, decade, genre)
        if not files:
            scope = f"decade {decade}" if decade is not None else "any decade"
            if genre is not None:
                scope += f", genre {genre}"
            raise ValueError(f"No db_*.zip files found for {scope} in {db_dir}")

        insert_token = "INSERT INTO tokens VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        document_count = token_count = skipped_documents = 0

        for file_genre, file_decade, zip_path in files:
            rows_by_text = read_db_rows_for_file(zip_path)

            for text_id, rows in sorted(rows_by_text.items()):
                token_rows = strip_leading_marker(text_id, rows, lexicon)
                if not token_rows:
                    skipped_documents += 1
                    print(
                        f"WARNING: text {text_id} in {zip_path.name} has no tokens "
                        "after the marker -- skipping this document",
                        file=sys.stderr,
                    )
                    continue

                meta = metadata.get(text_id, {})
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        text_id,
                        f"{file_genre}_{file_decade}_{text_id}.txt",
                        meta.get("genre", file_genre).upper(),
                        int(meta["year"]) if meta.get("year") else file_decade,
                        int(meta["words"]) if meta.get("words") else None,
                        meta.get("title", ""),
                        meta.get("author", ""),
                        meta.get("source", ""),
                        "",  # raw_text intentionally left empty; see module docstring
                    ),
                )

                def token_rows_gen(text_id=text_id, token_rows=token_rows):
                    for index, (corpus_token_id, word_id) in enumerate(token_rows, 1):
                        word, lemma, pos = lexicon[word_id]
                        yield (
                            text_id,
                            index,
                            corpus_token_id,
                            word_id,
                            word,
                            lemma,
                            pos,
                            word.casefold(),
                            lemma.casefold(),
                        )

                connection.executemany(insert_token, token_rows_gen())
                document_count += 1
                token_count += len(token_rows)

            connection.commit()
            print(
                f"Indexed {file_genre} {file_decade}: {document_count:,} documents so far",
                flush=True,
            )

        connection.execute("ANALYZE")
        connection.execute("PRAGMA optimize")
        print(f"Built {database_path}: {document_count:,} documents, {token_count:,} tokens")
        if skipped_documents:
            print(
                f"WARNING: skipped {skipped_documents:,} document(s) with no tokens "
                "after the marker; see warnings above for the specific text IDs.",
                file=sys.stderr,
            )


def prefix_bounds(value: str) -> tuple[str, str]:
    value = value.casefold()
    return value, value + "\U0010ffff"


def find_matches(
    connection: sqlite3.Connection,
    search: str,
    decade: int | None = None,
    pos_filter: str | None = None,
):
    """Find matches for the search string, optionally restricting each token
    slot to a set of CLAWS7 POS-tag prefixes.

    pos_filter, when set, is a comma-separated list of POS prefixes (e.g.
    "JJ,NN"). A token qualifies if its `pos` column starts with any listed
    prefix -- so "JJ" matches JJ, JJR, JJT (all adjective subclasses) and
    "NN" matches NN, NN1, NN2, NNL, NNO, etc. Whitespace around individual
    prefixes is stripped. An empty list, empty string, or None disables the
    filter entirely and preserves the pre-filter behavior exactly.

    For phrase searches, the same filter is applied to every token slot,
    which is the right default for most ambiguity cases but can be
    generalized later if per-slot filtering is needed.
    """
    terms = search.split()
    if not terms:
        raise ValueError("The search value cannot be empty")

    pos_prefixes = [p.strip() for p in (pos_filter or "").split(",") if p.strip()]

    aliases = [f"t{i}" for i in range(len(terms))]
    joins = " ".join(
        f"JOIN tokens {aliases[i]} ON {aliases[i]}.text_id=t0.text_id "
        f"AND {aliases[i]}.token_index=t0.token_index+{i}"
        for i in range(1, len(terms))
    )
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

    if decade is not None:
        predicates.append("(d.year>=? AND d.year<?)")
        parameters.extend((decade, decade + 10))

    selected_tokens = ", ".join(
        f"{alias}.word AS match_word_{i + 1}, {alias}.lemma AS match_lemma_{i + 1}, "
        f"{alias}.pos AS match_pos_{i + 1}, {alias}.corpus_token_id AS corpus_token_id_{i + 1}, "
        f"{alias}.word_id AS word_id_{i + 1}"
        for i, alias in enumerate(aliases)
    )
    sql = f"""
        SELECT d.text_id, d.document_file, d.genre, d.year, d.word_count,
               d.title, d.author, d.source, t0.token_index, {selected_tokens}
        FROM tokens t0
        {joins}
        JOIN documents d ON d.text_id=t0.text_id
        WHERE {' AND '.join(predicates)}
        ORDER BY d.year, d.text_id, t0.token_index
    """
    return connection.execute(sql, parameters), len(terms)


def export_csv(
    database_path: Path,
    csv_path: Path,
    search: str,
    context_words: int,
    decade: int | None = None,
    pos_filter: str | None = None,
) -> int:
    with sqlite3.connect(database_path) as connection, csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        matches, ngram_size = find_matches(connection, search, decade, pos_filter)
        match_fields: list[str] = []
        for i in range(1, ngram_size + 1):
            match_fields.extend(
                (f"match_word_{i}", f"match_lemma_{i}", f"match_pos_{i}",
                 f"corpus_token_id_{i}", f"word_id_{i}")
            )
        fields = [
            "occurrence", "search", "text_id", "document_file", "genre", "year",
            "document_word_count", "title", "author", "source", "token_index",
            *match_fields, "context_before", "matched_text", "context_after",
            "full_context",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        count = 0
        for row in matches:
            count += 1
            base = dict(zip(
                ["text_id", "document_file", "genre", "year", "document_word_count",
                 "title", "author", "source", "token_index", *match_fields],
                row,
            ))
            start = int(base["token_index"])
            end = start + ngram_size - 1
            nearby = connection.execute(
                "SELECT token_index, word FROM tokens WHERE text_id=? AND token_index "
                "BETWEEN ? AND ? ORDER BY token_index",
                (base["text_id"], max(1, start - context_words), end + context_words),
            ).fetchall()
            before = " ".join(word for index, word in nearby if index < start)
            matched = " ".join(word for index, word in nearby if start <= index <= end)
            after = " ".join(word for index, word in nearby if index > end)
            base.update(
                occurrence=count,
                search=search,
                context_before=before,
                matched_text=matched,
                context_after=after,
                full_context=" ".join(part for part in (before, matched, after) if part),
            )
            writer.writerow(base)
    print(f"Wrote {count:,} matches to {csv_path}")
    return count


def parse_args() -> argparse.Namespace:
    desktop = Path.home() / "Desktop"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=desktop / "coha-db-003")
    parser.add_argument("--lexicon-txt", type=Path, default=desktop / "coha-lexicon.txt")
    parser.add_argument("--sources-txt", type=Path, default=desktop / "coha-sources.txt")
    parser.add_argument(
        "--decade",
        type=int,
        default=None,
        help="Restrict to one decade, e.g. 1950. Omitting this processes every "
        "decade in the folder, which is much slower and rarely what you want.",
    )
    parser.add_argument(
        "--genre",
        default=None,
        help="Restrict to one genre/type, e.g. acad or fic. Omitting this "
        "processes every genre for the requested decade.",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=None,
        help="Defaults to corpus_search.sqlite, or corpus_search_<decade>.sqlite "
        "when --decade is set, so different decades don't overwrite each other.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Defaults to <search>_results.csv, or <search>_<decade>s_results.csv "
        "when --decade is set.",
    )
    parser.add_argument("--search", default=SEARCH_WORD)
    parser.add_argument("--context", type=int, default=CONTEXT_WORDS)
    parser.add_argument(
        "--pos-filter",
        default=None,
        help="Optional comma-separated CLAWS7 POS-tag prefixes to keep at "
        "every token slot, e.g. 'JJ' (adjectives only) or 'JJ,NN' "
        "(adjectives and any noun). Prefix match: 'JJ' matches JJ/JJR/JJT; "
        "'NN' matches NN/NN1/NN2/NNL/NNO. Omit to disable filtering.",
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.context < 0:
        raise ValueError("--context must be zero or greater")
    if args.decade is not None and args.decade % 10 != 0:
        raise ValueError("--decade must be the first year of a decade, e.g. 1950")

    search_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", args.search)
    if args.sqlite is not None:
        sqlite_path = args.sqlite
    elif args.decade is not None:
        sqlite_path = Path(f"corpus_search_{args.decade}.sqlite")
    else:
        sqlite_path = Path("corpus_search.sqlite")

    if args.csv is not None:
        csv_path = args.csv
    elif args.decade is not None:
        csv_path = Path(f"{search_slug}_{args.decade}s_results.csv")
    else:
        csv_path = Path(f"{search_slug}_results.csv")

    if args.rebuild or not sqlite_path.exists():
        if not args.db_dir.is_dir():
            raise FileNotFoundError(f"{args.db_dir} is not a directory")
        for path in (args.lexicon_txt, args.sources_txt):
            if not path.is_file():
                raise FileNotFoundError(path)
        build_database(
            sqlite_path,
            args.db_dir,
            args.lexicon_txt,
            args.sources_txt,
            args.decade,
            args.genre,
        )
    else:
        print(f"Reusing existing database {sqlite_path}")
    export_csv(
        sqlite_path, csv_path, args.search, args.context, args.decade,
        args.pos_filter,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)