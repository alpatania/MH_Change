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


# CLAWS7 tags that COHA uses (or extends) to mark tokens the tagger could not
# resolve to real vocabulary. Sourced from the CLAWS7 reference and the CCOHA
# paper's description of COHA's tagset extensions:
#   fo   - "formula"; what CLAWS assigns to word-plus-digits run-ons, so this
#          is the tag that catches endnote fusion like "insanity.42"
#   fu   - "unclassified word"
#   null - COHA's own marker for invalid tokens
# Deliberately NOT included:
#   zz   - COHA extends this to single letters AND any token containing a dash,
#          so it fires on legitimate compounds ("insane-sounding") as often as
#          on junk. Too blunt to use as a contamination signal.
#   y    - punctuation, which is normal context, not contamination.
SUSPECT_POS = {"fo", "fu", "null"}


# --- run-on splitter --------------------------------------------------------
# COHA has boundary-less run-ons ("insanewould", "nightgownssoft") that carry
# no case or punctuation signal, so regex can't split them -- you have to KNOW
# the words. We use WordNet (with lemmatization, so plurals/inflections like
# "nightgowns" are recognised via "nightgown") as the dictionary, and restrict
# to a SINGLE split point, because the actual corruption is one dropped space.
# Both constraints matter: lemmatization stops real compounds being shattered,
# and the two-piece limit stops multi-fragment garbage ("in saner sil lier").
#
# Degrades gracefully: if NLTK or its corpora aren't installed, _WORDNET_OK is
# False and split_runon is a no-op, so the pipeline still runs (just without
# run-on splitting). Install with:  python -m nltk.downloader wordnet omw-1.4

# Function words WordNet lacks as synsets but which are obviously real, so a
# split like "insane|would" isn't rejected just because "would" has no synset.
_FUNCTION_WORDS = set("""
a an the and or but would could should will shall may might must can of to in
on at by for with as is are was were be been being this that these those he she
it they we you i his her its their our your my me him them us not no so if then
than when where who whom which what have has had do does did
""".split())

try:
    from nltk.corpus import wordnet as _wn
    from nltk.corpus import words as _wl
    from nltk.stem import WordNetLemmatizer as _WNL
    _wn.ensure_loaded()
    _LEMMATIZER = _WNL()
    _WORDLIST = set(w.lower() for w in _wl.words())
    _WORDNET_OK = True
    _NLTK_WARNING = None
except ImportError:
    # NLTK itself isn't installed.
    _WORDNET_OK = False
    _WORDLIST = set()
    _NLTK_WARNING = (
        "NLTK is not installed, so run-on splitting is DISABLED -- tokens like "
        "'insanewould' will be left unsplit and 'clean_status' will never be "
        "'split'.\n"
        "  Fix:  uv add nltk  &&  uv run python -m nltk.downloader wordnet omw-1.4 words"
    )
except LookupError as _e:
    # NLTK is installed but a required corpus (wordnet / omw-1.4 / words)
    # hasn't been downloaded. This is the common case after `uv add nltk`
    # alone, and the one that silently produced unsplit output before.
    _WORDNET_OK = False
    _WORDLIST = set()
    _NLTK_WARNING = (
        "NLTK is installed but its corpora are missing, so run-on splitting is "
        "DISABLED -- tokens like 'insanewould' will be left unsplit and "
        "'clean_status' will never be 'split'.\n"
        "  Fix:  uv run python -m nltk.downloader wordnet omw-1.4 words\n"
        f"  (underlying error: {_e})"
    )
except Exception as _e:
    _WORDNET_OK = False
    _WORDLIST = set()
    _NLTK_WARNING = (
        f"Run-on splitting is DISABLED due to an unexpected NLTK error: {_e}"
    )


def _is_known_word(w: str) -> bool:
    """True if w is a real word. Uses TWO dictionaries because neither alone is
    right: WordNet omits most indefinite pronouns ("something", "himself") and
    function words, while the 235k `words` list includes them. A word passing
    EITHER is accepted. The permissiveness this adds (obscure fragments like
    "sil" are in the words list) is contained by split_runon's two-piece limit,
    which won't fragment a token into more than two known pieces.
    """
    w = w.lower()
    if w in _FUNCTION_WORDS:
        return True
    if not _WORDNET_OK:
        return False
    if w in _WORDLIST:
        return True
    if _wn.synsets(w):
        return True
    for pos in ("n", "v", "a", "r"):
        if _wn.synsets(_LEMMATIZER.lemmatize(w, pos)):
            return True
    return False


def split_runon(token: str, lemma: str = "", min_piece: int = 3) -> list[str]:
    """Split a boundary-less run-on into exactly two known words, or return
    [token] unchanged. Guards, each closing a specific failure mode:

      - capitalized tokens are skipped (proper nouns like "Gaddon" that no
        dictionary reliably contains);
      - tokens that are themselves known words are kept whole;
      - tagger-resolved tokens (non-empty known lemma) are kept whole;
      - EXACTLY one split point is tried, matching the actual corruption (a
        single dropped space) and preventing multi-fragment garbage.
    """
    if not _WORDNET_OK:
        return [token]
    if token[:1].isupper():        # proper-noun guard
        return [token]
    low = token.lower()
    if not low.isalpha():          # digits/punct -> handled by regex rules
        return [token]
    if _is_known_word(low):        # real word: keep whole
        return [token]
    if lemma and lemma.strip() and _is_known_word(lemma):
        return [token]
    for i in range(min_piece, len(low) - min_piece + 1):
        if _is_known_word(low[:i]) and _is_known_word(low[i:]):
            return [token[:i], token[i:]]   # preserve original casing
    return [token]


def is_suspect_token(pos: str, lemma: str) -> bool:
    """Flag a context token the tagger failed to resolve.

    Two independent signals, either of which is sufficient:
      1. A POS tag from SUSPECT_POS.
      2. An empty lemma. Real vocabulary always lemmatises; a blank lemma
         means the token was not recognised. Note "@" placeholders also carry
         a blank lemma, so callers must exclude them BEFORE calling this or
         every redacted window scores as malformed.

    This flags for review and routing -- it is not a discard rule. A token can
    be malformed and still carry real meaning ("insanity.42" is a genuine
    occurrence of "insanity" with an endnote marker welded to it).
    """
    return (pos or "").strip().lower() in SUSPECT_POS or not (lemma or "").strip()


def clean_context_token(tok: str) -> str:
    """Repair one context token with the same rules the embedding script's
    clean_token uses, kept in sync deliberately. Unlike _regex_repair_surface
    (which reduces the TARGET to a single bare form for grouping), this
    PRESERVES splits: "them:First" stays "them : First" so the context keeps
    both words and its real length for RoBERTa.
    """
    # endnote/footnote marker welded to a word: "insanity.42" -> "insanity ."
    tok = re.sub(r'^([A-Za-z]{2,})\.(\d{1,3})$', r'\1 .', tok)
    # trailing page/line marker residue
    tok = re.sub(r'^\|?p\d{1,4}(?=[A-Za-z])', '', tok)
    # camelCase run-on: "inSanFrancisco" -> "in San Francisco"
    tok = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tok)
    # sentence-boundary fusion: "them:First" -> "them : First"
    tok = re.sub(r'(?<=[a-z])([.:;,])(?=[A-Z])', r' \1 ', tok)
    # boundary-less run-on ("nightgownssoft"): dictionary two-piece split.
    # Applied per whitespace-piece so it composes with the regex splits above.
    out = []
    for piece in tok.split():
        out.extend(split_runon(piece) if piece.isalpha() else [piece])
    return " ".join(out) if out else tok


def clean_context(text: str) -> str:
    """Clean a context passage the way RoBERTa should see it: strip @
    redaction runs, repair malformed tokens, normalise whitespace.

    This is the SAME transformation final_layer_embeddings.clean_passage
    applies, moved upstream so the saved *_results.csv already contains what
    the model reads. Kept behaviourally identical so a downstream clean_passage
    call is a safe no-op.
    """
    text = re.sub(r'(?:@\s*)+', ' ', text)
    text = ' '.join(clean_context_token(t) for t in text.split())
    return re.sub(r'\s+', ' ', text).strip()


def _regex_repair_surface(word: str) -> str:
    """Rule-only repair of a surface form. Same rules as the embedding
    script's clean_token, kept in sync deliberately. Returns the first
    surviving alphanumeric run, lowercased, or "" if nothing survives.
    """
    w = word
    # endnote/footnote marker welded on: "insanity.42" -> "insanity"
    w = re.sub(r'^([A-Za-z]{2,})\.(\d{1,3})$', r'\1', w)
    # trailing page/line marker residue: "insanep106" style
    w = re.sub(r'^(\|?p\d{1,4})(?=[A-Za-z])', '', w)
    # camelCase run-on: "inSanFrancisco" -> "in San Francisco" -> take "in"?
    # No: for the TARGET we want the target token, so split then keep the run
    # that the search prefix would have matched. Handled by caller via lemma;
    # here we just split so the first run is clean.
    w = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', w)
    # sentence-boundary fusion: "insanity.He" / "insanELYjealous" boundaries
    w = re.sub(r'(?<=[a-z])[.:;,](?=[A-Za-z])', ' ', w)
    # boundary-less lowercase run-on has no signal; leave it, caller flags it
    parts = [p for p in re.split(r'\s+', w) if re.search(r'[A-Za-z]', p)]
    if not parts:
        return ""
    return re.sub(r'^\W+|\W+$', '', parts[0]).lower()


def normalize_word_form(surface: str, lemma: str, pos: str, search: str) -> tuple[str, str]:
    """Produce (word_clean, clean_status) for a matched target token.

    Strategy, most-trusted signal first:

      1. LEMMA. If the tagger resolved a lemma that itself begins with the
         search prefix, trust it -- the tagger already saw the sentence and
         stripped the junk. "insanity.he" tagged lemma "insanity" -> "insanity".
         This is what fixes the bulk of the run-on classes for free.
      2. REGEX. No usable lemma (blank, or doesn't match the search family):
         fall back to rule-based surface repair. Catches endnote/punctuation
         fusion the tagger left in the surface form.
      3. FLAG. Neither yields something in the search family: keep the
         lowercased surface but mark status="unresolved" so downstream can
         filter or review. These are the boundary-less run-ons ("insanityof")
         and genuine non-family tokens the prefix swept in ("insanitation",
         "insanorum").

    clean_status is one of: "lemma", "regex", "surface", "unresolved".
    Returns the ORIGINAL surface untouched as the caller keeps match_word_i;
    only word_clean is derived.
    """
    prefix = search.casefold().strip()
    lem = (lemma or "").strip().lower()
    surf = (surface or "").strip()

    # 1. trust the lemma when it's in-family
    if lem and lem.startswith(prefix) and re.fullmatch(r"[a-z]+", lem):
        return lem, "lemma"

    # 2. regex repair of the surface -- but only accept it if it actually
    #    CHANGED the surface. Otherwise a clean run-on like "insanewould"
    #    passes the prefix check unchanged and short-circuits the splitter.
    repaired = _regex_repair_surface(surf)
    if repaired and repaired.startswith(prefix) and repaired != surf.lower():
        return repaired, "regex"

    # 3. dictionary run-on split ("insanewould" -> insane|would): keep the
    #    piece that begins with the search prefix, so the target is recovered
    #    from a boundary-less run-on the regex rules couldn't touch.
    pieces = split_runon(surf)
    if len(pieces) > 1:
        for pc in pieces:
            pc_low = pc.lower()
            if pc_low.startswith(prefix) and re.fullmatch(r"[a-z]+", pc_low):
                return pc_low, "split"

    # 3b. regex repair that merely lowercased/trimmed (no structural change)
    #     is still fine to accept now that the splitter has had its chance.
    if repaired and repaired.startswith(prefix):
        return repaired, "regex"

    # 4. clean lowercase surface, in-family but not via lemma (e.g. valid
    #    morphology the lexicon lemma happened to collapse differently)
    surf_low = re.sub(r'^\W+|\W+$', '', surf).lower()
    if surf_low and surf_low.startswith(prefix) and re.fullmatch(r"[a-z]+", surf_low):
        return surf_low, "surface"

    # 5. give up: keep the lowercased surface but flag it
    return (surf_low or surf.lower()), "unresolved"


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
            "occurrence", "uid", "search", "text_id", "document_file", "genre",
            "year", "document_word_count", "title", "author", "source",
            "token_index",
            *match_fields, "context_before", "matched_text", "context_after",
            "full_context",
            # Normalised target form + how it was derived. word_clean is what
            # downstream plots/grouping should read; match_word_1 keeps the raw
            # surface. clean_status in {lemma,regex,surface,unresolved} lets you
            # filter or review the ones the tagger couldn't resolve.
            "word_clean", "clean_status",
            # Per-window contamination scoring. Purely additive: everything
            # downstream reads columns by name, so old consumers are unaffected.
            "ctx_n_tokens", "ctx_n_at", "ctx_n_bad_pos", "ctx_max_word_id",
            "ctx_bad_forms",
        ]
        # Two outputs from the same rows:
        #   csv_path                       -> CLEANED (what RoBERTa reads)
        #   <stem>_uncleaned<suffix>       -> RAW (provenance, never touched)
        # Both carry identical columns and identical row order/uid, so they can
        # be joined 1:1 downstream. The cleaned file's context columns and
        # matched_text are run through clean_context(); word_clean/clean_status
        # are present in both (they describe the target either way).
        uncleaned_path = csv_path.with_name(
            csv_path.stem + "_uncleaned" + csv_path.suffix
        )
        with uncleaned_path.open("w", newline="", encoding="utf-8-sig") as raw_out:
            raw_writer = csv.DictWriter(raw_out, fieldnames=fields)
            raw_writer.writeheader()
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
                    "SELECT token_index, word, word_id, pos, lemma FROM tokens "
                    "WHERE text_id=? AND token_index BETWEEN ? AND ? "
                    "ORDER BY token_index",
                    (base["text_id"], max(1, start - context_words), end + context_words),
                ).fetchall()
                before = " ".join(r[1] for r in nearby if r[0] < start)
                matched = " ".join(r[1] for r in nearby if start <= r[0] <= end)
                after = " ".join(r[1] for r in nearby if r[0] > end)

                # Score the CONTEXT only -- the matched slot is scored separately
                # by the existing match_pos_i / word_id_i columns, and folding it
                # in here would make every row look contaminated by its own target.
                ctx = [r for r in nearby if not (start <= r[0] <= end)]
                n_at = sum(1 for r in ctx if r[1] == "@")
                real = [r for r in ctx if r[1] != "@"]
                bad = [r for r in real if is_suspect_token(r[3], r[4])]
                # Normalise the target form(s). For a single-token search this is
                # just slot 1; for a phrase search, clean each slot and join, so
                # word_clean stays a faithful cleaned version of matched_text.
                clean_parts, statuses = [], []
                for i in range(1, ngram_size + 1):
                    wc, st = normalize_word_form(
                        base.get(f"match_word_{i}", ""),
                        base.get(f"match_lemma_{i}", ""),
                        base.get(f"match_pos_{i}", ""),
                        search.split()[i - 1] if i - 1 < len(search.split()) else search,
                    )
                    clean_parts.append(wc)
                    statuses.append(st)
                word_clean = " ".join(p for p in clean_parts if p)
                # worst-case status wins: a phrase is only "clean" if all slots are
                status_rank = {"lemma": 0, "regex": 1, "surface": 2, "unresolved": 3}
                clean_status = max(statuses, key=lambda s: status_rank.get(s, 3))

                base.update(
                    occurrence=count,
                    uid=f"{base['text_id']}:{start}",
                    search=search,
                    word_clean=word_clean,
                    clean_status=clean_status,
                    context_before=before,
                    matched_text=matched,
                    context_after=after,
                    full_context=" ".join(p for p in (before, matched, after) if p),
                    ctx_n_tokens=len(ctx),
                    ctx_n_at=n_at,
                    ctx_n_bad_pos=len(bad),
                    # wordID is also the frequency RANK in COHA's lexicon, so the
                    # rarest context token is a cheap contamination proxy: genuine
                    # vocabulary ranks in the tens of thousands, malformed hapaxes
                    # rank in the millions. Reported over non-@ tokens only.
                    ctx_max_word_id=max((r[2] for r in real), default=""),
                    ctx_bad_forms=";".join(r[1] for r in bad),
                )

                # RAW row goes to the uncleaned file exactly as built.
                raw_writer.writerow(base)

                # CLEANED row: same dict, but context columns run through
                # clean_context() so the saved file is what RoBERTa will read.
                cleaned = dict(base)
                cb = clean_context(before)
                mt = clean_context(matched)
                ca = clean_context(after)
                cleaned.update(
                    context_before=cb,
                    matched_text=mt,
                    context_after=ca,
                    full_context=" ".join(p for p in (cb, mt, ca) if p),
                )
                writer.writerow(cleaned)
    print(f"Wrote {count:,} matches to {csv_path} (cleaned) "
          f"and {uncleaned_path} (raw)")
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
    parser.add_argument("--require-nltk", action="store_true",
                        help="Abort if NLTK/its corpora are unavailable, "
                             "instead of continuing with run-on splitting "
                             "disabled. Use for reproducible batch runs where "
                             "silently-unsplit output would be a problem.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Loudly warn if run-on splitting is unavailable. This is the difference
    # between 'insanewould' being split vs left whole, and it failed silently
    # before -- so surface it at the top of every run, and allow --require-nltk
    # to make it a hard error for reproducible batch runs.
    if _NLTK_WARNING:
        banner = "=" * 70
        print(f"\n{banner}\nWARNING: {_NLTK_WARNING}\n{banner}\n",
              file=sys.stderr)
        if getattr(args, "require_nltk", False):
            raise SystemExit(
                "Aborting because --require-nltk was set and run-on splitting "
                "is unavailable (see warning above).")

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