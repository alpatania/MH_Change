#!/usr/bin/env python3
"""Diagnose whether run-on splitting is available. Run inside your uv env:
    uv run python check_nltk.py
"""
import sys

def main():
    try:
        import nltk
        print(f"[OK]   nltk installed: {nltk.__version__}")
    except ImportError:
        print("[FAIL] nltk NOT installed.")
        print("       Fix: uv add nltk && uv run python -m nltk.downloader wordnet omw-1.4 words")
        return 1

    missing = []
    for res, path in [("wordnet","corpora/wordnet"),
                      ("omw-1.4","corpora/omw-1.4"),
                      ("words","corpora/words")]:
        try:
            nltk.data.find(path)
            print(f"[OK]   corpus present: {res}")
        except LookupError:
            print(f"[FAIL] corpus MISSING: {res}")
            missing.append(res)

    if missing:
        print(f"\n       Fix: uv run python -m nltk.downloader {' '.join(missing)}")
        return 1

    # functional test
    from nltk.corpus import wordnet as wn, words as wl
    from nltk.stem import WordNetLemmatizer
    wl_set = set(w.lower() for w in wl.words())
    lem = WordNetLemmatizer()
    def known(w):
        w=w.lower()
        return bool(wn.synsets(w)) or w in wl_set or any(
            wn.synsets(lem.lemmatize(w,p)) for p in "nvar")
    ok = known("insane") and known("would") and not known("insanewould")
    print(f"\n[{'OK' if ok else 'FAIL'}]   functional split test "
          f"(insane+would known, insanewould not): {ok}")
    print("\nRun-on splitting is AVAILABLE. If output still isn't split, the")
    print("problem is stale cached files, not NLTK -- delete _coords.csv and")
    print("_embeddings.npy and rebuild.")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())