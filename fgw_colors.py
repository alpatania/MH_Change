"""
Shared word-colour assignment for the Sankey and tanglegram
============================================================
One rule, computed once, imported by both plots so their colours are
IDENTICAL:

  Rank words by TOTAL frequency across all decades (global, so a word has the
  same colour in every decade and in both figures). Assign the Okabe-Ito
  colourblind-safe palette in rank order -- most frequent word gets the first
  colour, second-most the second, and so on. Words past the end of the palette
  all get one neutral GREY.

Keying: both figures colour on the SAME per-row word value (the cleaned
`word` column, i.e. word_clean as written by coha_build.py and carried through
final_layer_embeddings.py into *_coords.csv). Keeping the key identical is what
guarantees 'insane' is the same colour in both plots.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


# Okabe-Ito, colourblind-safe. Grey is deliberately NOT in this list -- it is
# reserved as the overflow colour, so an assigned rank can never be confused
# with 'everything past the palette'.
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]
OVERFLOW_GREY = "#999999"


def _clean_word(v) -> str:
    """Canonicalise a word cell to a comparable key: string, stripped,
    casefolded. Empty/NaN becomes ''."""
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    return s.casefold()


def word_frequencies(metas: list[pd.DataFrame], word_col: str = "word") -> Counter:
    """Total occurrences of each cleaned word across all the given per-decade
    metadata frames."""
    c: Counter = Counter()
    for m in metas:
        if word_col in m.columns:
            for v in m[word_col]:
                w = _clean_word(v)
                if w:
                    c[w] += 1
    return c


def build_color_map(
    metas: list[pd.DataFrame],
    word_col: str = "word",
) -> tuple[dict[str, str], list[str]]:
    """Return (color_map, legend_order).

    color_map: {cleaned_word -> hex}. The N most frequent words take the N
    palette colours in frequency order; every remaining word maps to
    OVERFLOW_GREY.

    legend_order: the words that received a distinct palette colour, most
    frequent first -- for building a legend that matches the colouring. Words
    in the grey overflow are not listed individually (they'd all be one grey
    'other').

    Ties in frequency are broken alphabetically, so the assignment is
    deterministic across runs and across the two plots.
    """
    freq = word_frequencies(metas, word_col)
    # deterministic: frequency desc, then word asc
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

    color_map: dict[str, str] = {}
    legend_order: list[str] = []
    for i, (word, _count) in enumerate(ranked):
        if i < len(OKABE_ITO):
            color_map[word] = OKABE_ITO[i]
            legend_order.append(word)
        else:
            color_map[word] = OVERFLOW_GREY
    return color_map, legend_order


def color_for(word, color_map: dict[str, str]) -> str:
    """Look up a (possibly raw) word in the map, applying the same cleaning
    used to build it. Unknown words fall back to grey."""
    return color_map.get(_clean_word(word), OVERFLOW_GREY)


def load_decade_metas(
    out_dir: Path, search: str, decades: list[int]
) -> list[pd.DataFrame]:
    """Load every decade's *_coords.csv for a search, for frequency counting.
    Missing files are skipped (a decade may have failed upstream)."""
    metas = []
    for d in decades:
        p = out_dir / f"{search}_{d}s_coords.csv"
        if p.exists():
            metas.append(pd.read_csv(p, dtype=str))
    return metas
