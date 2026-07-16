#!/usr/bin/env python3
"""Build a searchable COHA sample database and export n-gram occurrences.

This script uses database.zip, wordLemPoS.zip, and text.zip without extracting
them.  It requires only the Python standard library.
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
from xml.etree import ElementTree as ET


# Change this value to search for another word or whitespace-separated n-gram.
# Prefix matching means "insan" finds insane, insanely, insanity, etc.
SEARCH_WORD = "insan"
CONTEXT_WORDS = 20

FILE_RE = re.compile(r"^(?P<genre>[^_]+)_(?P<year>\d{4})_(?P<text_id>\d+)\.txt$")
XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def decode(data: bytes) -> str:
    """Decode corpus files while tolerating the occasional legacy byte/NUL."""
    try:
        return data.decode("utf-8-sig").replace("\x00", "")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace").replace("\x00", "")


def excel_column(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference).group(0)
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def read_source_metadata(database_zip: zipfile.ZipFile) -> dict[int, dict[str, str]]:
    """Read sourcesSample.xlsx from inside database.zip using stdlib XML."""
    xlsx_bytes = database_zip.read("sourcesSample.xlsx")
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(xlsx_bytes)) as workbook:
        shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(f"{{{XLSX_NS}}}t"))
            for item in shared_root.findall(f"{{{XLSX_NS}}}si")
        ]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet2.xml"))

    rows: list[list[str]] = []
    for row in sheet.findall(f".//{{{XLSX_NS}}}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{{{XLSX_NS}}}c"):
            value_node = cell.find(f"{{{XLSX_NS}}}v")
            if value_node is None:
                value = ""
            elif cell.get("t") == "s":
                value = shared[int(value_node.text)]
            else:
                value = value_node.text or ""
            values[excel_column(cell.get("r", "A1"))] = value
        if values:
            rows.append([values.get(i, "") for i in range(max(values) + 1)])

    headers = [value.strip().lower().replace("# ", "") for value in rows[0]]
    metadata: dict[int, dict[str, str]] = {}
    for row in rows[1:]:
        record = dict(zip(headers, row))
        if record.get("textid"):
            metadata[int(record["textid"])] = record
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


def database_rows_for_decade(
    archive: zipfile.ZipFile, decade: int
) -> dict[int, list[tuple[int, int]]]:
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for line_number, line in enumerate(decode(archive.read(f"{decade}.txt")).splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"Malformed {decade}.txt line {line_number}: {line!r}")
        text_id, corpus_token_id, word_id = map(int, fields)
        rows[text_id].append((corpus_token_id, word_id))
    return rows


def tagged_tokens(data: bytes) -> tuple[list[tuple[str, str, str]], int]:
    tokens: list[tuple[str, str, str]] = []
    leading_markers = 0
    for line in decode(data).splitlines():
        if not line:
            continue
        if line.startswith("@@"):
            if tokens:
                raise ValueError("Unexpected structural marker inside tagged text")
            leading_markers += 1
            continue
        fields = line.split("\t")
        if fields[0] == "q!":  # fixed-width end padding, absent from database.zip
            continue
        if len(fields) < 3 or any(fields[3:]):
            raise ValueError(f"Malformed word/lemma/POS row: {line!r}")
        tokens.append((fields[0], fields[1], fields[2]))
    return tokens, leading_markers


def build_database(
    database_path: Path,
    database_zip_path: Path,
    tagged_zip_path: Path,
    text_zip_path: Path,
) -> None:
    if database_path.exists():
        database_path.unlink()

    with (
        zipfile.ZipFile(database_zip_path) as database_zip,
        zipfile.ZipFile(tagged_zip_path) as tagged_zip,
        zipfile.ZipFile(text_zip_path) as text_zip,
        sqlite3.connect(database_path) as connection,
    ):
        create_schema(connection)
        metadata = read_source_metadata(database_zip)
        text_members = set(text_zip.namelist())
        parsed_members: list[tuple[int, int, str, str]] = []
        for member in tagged_zip.namelist():
            match = FILE_RE.match(member)
            if match:
                parsed_members.append(
                    (int(match["year"]), int(match["text_id"]), match["genre"], member)
                )

        by_decade: dict[int, list[tuple[int, int, str, str]]] = defaultdict(list)
        for item in parsed_members:
            by_decade[(item[0] // 10) * 10].append(item)

        insert_token = """
            INSERT INTO tokens VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        document_count = token_count = 0
        for decade in sorted(by_decade):
            database_rows = database_rows_for_decade(database_zip, decade)
            for year, text_id, filename_genre, member in sorted(by_decade[decade]):
                if member not in text_members:
                    raise ValueError(f"{member} is missing from text.zip")
                if text_id not in database_rows:
                    raise ValueError(f"textID {text_id} is missing from {decade}.txt")

                tags, leading_markers = tagged_tokens(tagged_zip.read(member))
                ids = database_rows[text_id]
                # Rarely, tagged data has unrelated material appended after an
                # @x boundary. Only truncate when the database row count proves
                # that its entire token stream ends at that first boundary.
                full_structural_rows = len(ids) - len(tags)
                if not (leading_markers and 1 <= full_structural_rows <= 3):
                    boundary = next(
                        (i for i, token in enumerate(tags) if token[0] == "@x"), None
                    )
                    if boundary is not None:
                        prefix_structural_rows = len(ids) - boundary
                        if 1 <= prefix_structural_rows <= 3:
                            tags = tags[:boundary]
                # database.zip represents the initial @@ header with one or
                # sometimes two structural rows (the latter when a ## field is
                # populated). They are not words and must stay out of context.
                structural_rows = len(ids) - len(tags)
                if leading_markers and 1 <= structural_rows <= 3:
                    ids = ids[structural_rows:]
                if len(tags) != len(ids):
                    raise ValueError(
                        f"Alignment failure for {member}: {len(tags)} tagged tokens "
                        f"but {len(ids)} database rows"
                    )

                meta = metadata.get(text_id, {})
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        text_id,
                        member,
                        meta.get("genre", filename_genre).upper(),
                        int(meta.get("year", year)),
                        int(meta["words"]) if meta.get("words") else None,
                        meta.get("title", ""),
                        meta.get("author", ""),
                        meta.get("source", ""),
                        decode(text_zip.read(member)).strip(),
                    ),
                )
                connection.executemany(
                    insert_token,
                    (
                        (
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
                        for index, ((word, lemma, pos), (corpus_token_id, word_id))
                        in enumerate(zip(tags, ids), 1)
                    ),
                )
                document_count += 1
                token_count += len(tags)
            connection.commit()
            print(f"Indexed decade {decade}: {document_count:,} documents", flush=True)

        connection.execute("ANALYZE")
        connection.execute("PRAGMA optimize")
        print(f"Built {database_path}: {document_count:,} documents, {token_count:,} tokens")


def prefix_bounds(value: str) -> tuple[str, str]:
    value = value.casefold()
    return value, value + "\U0010ffff"


def find_matches(connection: sqlite3.Connection, search: str):
    terms = search.split()
    if not terms:
        raise ValueError("The search value cannot be empty")

    aliases = [f"t{i}" for i in range(len(terms))]
    joins = " ".join(
        f"JOIN tokens {aliases[i]} ON {aliases[i]}.text_id=t0.text_id "
        f"AND {aliases[i]}.token_index=t0.token_index+{i}"
        for i in range(1, len(terms))
    )
    predicates: list[str] = []
    parameters: list[str] = []
    for alias, term in zip(aliases, terms):
        low, high = prefix_bounds(term)
        predicates.append(
            f"(({alias}.word_norm>=? AND {alias}.word_norm<?) "
            f"OR ({alias}.lemma_norm>=? AND {alias}.lemma_norm<?))"
        )
        parameters.extend((low, high, low, high))

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
    database_path: Path, csv_path: Path, search: str, context_words: int
) -> int:
    with sqlite3.connect(database_path) as connection, csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        matches, ngram_size = find_matches(connection, search)
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
    parser.add_argument("--database-zip", type=Path, default=desktop / "database.zip")
    parser.add_argument("--wordlempos-zip", type=Path, default=desktop / "wordLemPoS.zip")
    parser.add_argument("--text-zip", type=Path, default=desktop / "text.zip")
    parser.add_argument("--sqlite", type=Path, default=Path("corpus_search.sqlite"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--search", default=SEARCH_WORD)
    parser.add_argument("--context", type=int, default=CONTEXT_WORDS)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.context < 0:
        raise ValueError("--context must be zero or greater")
    csv_path = args.csv or Path(f"{re.sub(r'[^A-Za-z0-9_-]+', '_', args.search)}_results.csv")
    if args.rebuild or not args.sqlite.exists():
        for path in (args.database_zip, args.wordlempos_zip, args.text_zip):
            if not path.is_file():
                raise FileNotFoundError(path)
        build_database(
            args.sqlite, args.database_zip, args.wordlempos_zip, args.text_zip
        )
    else:
        print(f"Reusing existing database {args.sqlite}")
    export_csv(args.sqlite, csv_path, args.search, args.context)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
