"""Term extraction (bes.library.terms)."""
from __future__ import annotations

import subprocess
import sys

from bes.library.terms import STOPWORDS, bucket, normalize, stem, strip_disambiguator, terms, tokens


def test_normalize_and_tokens():
    assert normalize("Utqiaġvik ÅNGSTRÖM ﬁve²") == "utqiagvik angstrom five2"
    assert tokens("It's U.S.-based, 1,300 km!") == ["it", "s", "u", "s", "based", "1", "300", "km"]


def test_stopwords_and_kept_function_words():
    assert terms("What is the capital of the US in May? Who won? Will Smith, One Direction") == [
        "capital", "us", "may", "won", "will", "smith", "one", "direction"]
    assert "the" in STOPWORDS and "us" not in STOPWORDS


def test_s_stemmer():
    cases = {"countries": "country", "horses": "horse", "cats": "cat", "bus": "bus", "glass": "glass",
             "species": "specy", "trees": "tree", "goes": "goe", "ies": "ie", "is": "is", "gas": "ga"}
    for w, s in cases.items():
        assert stem(w) == s, w


def test_bucket_is_stable_across_processes():
    here = [bucket(t, 18) for t in ("paris", "turing", "1990")]
    code = "from bes.library.terms import bucket; print([bucket(t, 18) for t in ('paris','turing','1990')])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONHASHSEED": "123", "PATH": "", "PYTHONPATH": sys.path[0] or "."})
    assert out.stdout.strip() == str(here)
    assert all(0 <= b < (1 << 18) for b in here)


def test_strip_disambiguator():
    assert strip_disambiguator("Mint (candy)") == "Mint"
    assert strip_disambiguator("Paris") == "Paris"
    assert strip_disambiguator("(film) X") == "(film) X"
