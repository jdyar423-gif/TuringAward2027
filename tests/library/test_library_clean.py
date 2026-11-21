"""Cleaning rules and passage splitting (bes.library.clean) on hand-made examples."""
from __future__ import annotations

import os
import random

import pytest

from bes.library.clean import (CleanArticle, CleanCfg, Paragraph, Section, clean_and_split,
                               clean_article, clean_inline, clean_parentheses, default_count_tokens,
                               iter_articles, make_passages, n_words, render_passage,
                               split_sentences, strip_markup)

# ------------------------------------------------------------------------- markup


def test_templates_tables_refs_links_removed():
    raw = ("{{Infobox settlement\n| name = Kuala Lumpur\n| x = {{nested|a}}\n}}Kuala Lumpur is a city."
           "<ref name=\"a\">Smith 2006.</ref> It has [[Malaysia|Malaysian]] people and a "
           "[[river]].<ref name=b/> '''Bold''' text&nbsp;here.\n{| class=\"wikitable\"\n|-\n| 1 || 2\n|}\n"
           "[[File:X.jpg|thumb|200px|A [[caption]] here]]After file. [[Category:Cities]]")
    out = strip_markup(raw)
    assert "Infobox" not in out and "nested" not in out
    assert "<ref" not in out and "Smith" not in out
    assert "Malaysian people" in out and "a river." in out and "[[" not in out
    assert "Bold text here." in out and "'''" not in out
    assert "wikitable" not in out and "||" not in out
    assert "caption" not in out and "After file." in out and "Category" not in out


def test_unclosed_template_drops_rest_of_paragraph_only():
    out = strip_markup("Good start. {{broken template\nstill broken\n\nNext paragraph.")
    assert "Good start." in out and "broken" not in out and "Next paragraph." in out


def test_external_links_and_urls():
    out = strip_markup("See [http://example.com the site] or http://foo.bar/baz now.")
    assert "the site" in out and "http" not in out and "now." in out


# --------------------------------------------------------------------- parentheses


@pytest.mark.parametrize("raw, expected", [
    ("Austria (, ;  ), officially the Republic of Austria ( ), is a country.",
     "Austria, officially the Republic of Austria, is a country."),
    ("France ( or ; ), officially the French Republic (, ), is a country.",
     "France, officially the French Republic, is a country."),
    ("Japan (; romanised as nihon or nippon) is a country.",
     "Japan (romanised as nihon or nippon) is a country."),
    ("Jutland (, pronounced yoo´-land), bordering Germany.", "Jutland, bordering Germany."),
    ("Magnesium ( /mæɡˈniːziəm/ mag-NEE-zee-əm) is an element.", "Magnesium is an element."),
    ("Fiat (UK: /ˈfiːət, -æt/, US: /-ɑːt/, Italian: [ˈfiːat]; originally FIAT, Italian: Fabbrica Italiana) is a maker.",
     "Fiat (originally FIAT, Italian: Fabbrica Italiana) is a maker."),
    ("Ireland (Irish: ; Ulster-Scots: ) is an island.", "Ireland is an island."),
    ("A Christian () is a person.", "A Christian is a person."),
    ("It has 1,300 people (2006) and f(x) values.", "It has 1,300 people (2006) and f(x) values."),
])
def test_parentheses(raw, expected):
    assert clean_inline(raw) == expected


def test_nested_parentheses_inner_first():
    assert clean_parentheses("X (born 1950 (; ) in Rome) lived.") == "X (born 1950 in Rome) lived."


def test_reference_marks_removed_but_years_kept():
    s = clean_inline("Words.[1][12] The Act [1850] passed[citation needed]. Done [a].")
    assert s == "Words. The Act [1850] passed. Done."


def test_spaces_before_punctuation_and_holes():
    assert clean_inline("The trail is about  long , and wide .") == "The trail is about long, and wide."


# --------------------------------------------------------------------- sentences


def test_sentence_splitter_abbreviations_initials_decimals():
    p = ("Summerhill was founded by A. S. Neill in 1921. Dr. Smith lives in the U.S. now. "
         "It costs 3.5 dollars. Is it good? Yes! \"Quoted.\" Then more.")
    assert split_sentences(p) == [
        "Summerhill was founded by A. S. Neill in 1921.",
        "Dr. Smith lives in the U.S. now.",
        "It costs 3.5 dollars.", "Is it good?", "Yes!", "\"Quoted.\"", "Then more."]


def test_sentence_splitter_round_trip_random():
    rng = random.Random(0)
    words = ["alpha", "Beta", "gamma.", "Delta?", "e.g.", "Mr.", "3.5", "U.S.", "x!", "Zeta"]
    for _ in range(300):
        p = " ".join(rng.choice(words) for _ in range(rng.randint(1, 30)))
        assert " ".join(split_sentences(p)) == " ".join(p.split())


# --------------------------------------------------------------------- articles

ARTICLE = """Phyllis Thaxter (November 20, 1921 – August 14, 2012) was an American actress (; ). She played Superman's mother.

Early life
Thaxter was born in Portland, Maine.

Career
She worked on Broadway since 1940. In 1944, she signed a contract.

Movies
 Thirty Seconds Over Tokyo (1944)
 Bewitched
 Blood on the Moon (1948)

References

Other websites
 IMDb page

American movie actors
1921 births
2012 deaths"""


def test_article_structure_tails_categories_lists():
    a = clean_article("Phyllis Thaxter", ARTICLE)
    assert a.drop_reason is None
    assert a.categories == ["American movie actors", "1921 births", "2012 deaths"]
    heads = [s.heading for s in a.sections]
    assert heads == ["", "Early life", "Career", "Movies"]
    lead = a.sections[0].paragraphs[0]
    assert lead.sentences[0].startswith("Phyllis Thaxter (November 20, 1921 – August 14, 2012) was an American actress.")
    movies = a.sections[3].paragraphs[0]
    assert movies.items == ["Thirty Seconds Over Tokyo (1944)", "Blood on the Moon (1948)"]  # 1-word item dropped
    text = a.text()
    assert "References" not in text and "IMDb" not in text and "births" not in text


def test_heading_with_blank_line_and_eq_style():
    a = clean_article("T", "Lead sentence here.\n\nEvents in April\n\nFixed Events \n\n April 1 - April Fools Day\n\n"
                           "== History ==\nSome history happened.\n\nCats")
    heads = [s.heading for s in a.sections]
    assert heads == ["", "Fixed Events", "History"]  # "Events in April" has no content of its own
    assert a.sections[1].paragraphs[0].items == ["April 1 - April Fools Day"]
    assert a.categories == ["Cats"]


def test_broken_template_sentences_dropped():
    a = clean_article("T", "The average temperature is . It is cold. The trail is about .\n\nCats")
    assert a.sections[0].paragraphs[0].sentences == ["It is cold."]
    assert a.stats["broken_sentences"] == 2


@pytest.mark.parametrize("title, text, reason", [
    ("Flame (disambiguation)", "Flame can mean many things.", "disambiguation"),
    ("Mint", "Mint can mean different things:\n Mentha, a genus of plants called mint\n Mint (candy), a sweet food made\n Mint (money), a factory that makes money", "disambiguation"),
    ("May 29", "\n\nEvents\n\nUp to 1900 \n 1176 - Battle of Legnano: The Lombard League wins.\n 1328 – Philip VI of France is crowned King.\n 1453 - Fall of Constantinople happens today.\n\nDays", "list_only"),
    ("List of rivers", "These are rivers.\n\n Amazon River in Brazil\n Nile River in Egypt\n Rhine River in Germany", "list_page"),
    ("Yoshi", "is a .\n\nClub career statistics\n\n|-\n|1985||x||y\n|}", "empty"),
])
def test_dropped_pages(title, text, reason):
    assert clean_article(title, text).drop_reason == reason


def test_short_stub_kept():
    a = clean_article("King Cove, Alaska", "King Cove  is a city in Alaska.\n\nCities in Alaska")
    assert a.drop_reason is None and a.text() == "King Cove is a city in Alaska."


def test_list_heavy_page_keeps_prose_only():
    items = "\n".join(f" Item number {i} happened on a day" for i in range(40))
    a = clean_article("September", f"September is the ninth month of the year in the calendar.\n\nEvents\n{items}\n\nMonths")
    assert a.drop_reason is None
    assert a.list_words() == 0 and a.text() == "September is the ninth month of the year in the calendar."


def test_digit_table_rows_dropped_but_prose_with_pipes_kept():
    a = clean_article("Concatenation", 'In SQL we would write "snow"||"ball" to join words.\n119||13||9||0||11\n\nComputing')
    assert a.text() == 'In SQL we would write "snow"||"ball" to join words.'


# --------------------------------------------------------------------- passages


def _random_article(rng: random.Random) -> CleanArticle:
    vocab = ["the", "river", "Paris", "town", "was", "built", "in", "1990", "and", "is", "large",
             "people", "live", "there", "Dr.", "several", "very", "long", "names", "Ångström"]
    sections = []
    for s in range(rng.randint(1, 5)):
        paras = []
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.25:
                paras.append(Paragraph(items=[" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 12)))
                                              for _ in range(rng.randint(1, 6))]))
            else:
                sents = []
                for _ in range(rng.randint(1, 6)):
                    n = rng.randint(3, 40) if rng.random() < 0.97 else rng.randint(120, 200)
                    w = [rng.choice(vocab) for _ in range(n)]
                    sents.append(" ".join(w).capitalize() + ".")
                paras.append(Paragraph(sentences=sents))
        sections.append(Section("" if s == 0 else f"Heading {s}", paras))
    return CleanArticle("Some Title (thing)", sections, [], None, {})


@pytest.mark.parametrize("seed", range(40))
def test_passages_respect_caps_boundaries_and_cover_everything(seed):
    rng = random.Random(seed)
    art = _random_article(rng)
    max_words, max_tokens = rng.choice([(110, 160), (60, 90), (30, 1000)])
    ps = make_passages(art, max_words, max_tokens)
    heads = {s.heading for s in art.sections if s.heading}
    pieces = []
    for p in ps:
        assert n_words(p.body) <= max_words
        assert default_count_tokens(render_passage(p.title, p.body)) <= max_tokens
        lines = p.body.split("\n")
        assert lines[-1] not in heads, "a passage never ends with a heading line"
        if p.section:
            assert lines[0] == p.section
        for line in lines:
            if line in heads:
                continue
            pieces.append(line[2:] if line.startswith("- ") else line)
    # every sentence / item appears, in order, exactly once (long ones may be cut in pieces)
    expected = []
    for s in art.sections:
        for para in s.paragraphs:
            expected.extend(para.items if para.is_list else para.sentences)
    got_words = " ".join(pieces).split()
    assert got_words == " ".join(expected).split()


def test_passages_start_at_sentence_boundaries():
    sents = [f"Sentence number {i} has exactly seven words." for i in range(40)]
    art = CleanArticle("T", [Section("", [Paragraph(sentences=sents)])], [], None, {})
    ps = make_passages(art, max_words=50, max_tokens=10_000)
    assert len(ps) > 1
    for p in ps:
        assert p.body.startswith("Sentence number") and p.body.endswith("words.")
        assert n_words(p.body) <= 50


def test_title_longer_than_a_passage_gives_no_passage():
    art = clean_article("T " * 400, "A short body sentence.")
    assert make_passages(art, max_words=110, max_tokens=160) == []


def test_custom_token_counter_is_used():
    art = clean_article("T", " ".join(["Word word word."] * 50))
    ps = make_passages(art, max_words=1000, max_tokens=20, count_tokens=lambda s: len(s.split()))
    assert all(len(render_passage(p.title, p.body).split()) <= 20 for p in ps)


def test_clean_and_split_on_real_dump_if_present():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data_cache", "library", "raw",
                        "simplewiki", "20231101.simple", "train-00000-of-00001.parquet")
    if not os.path.exists(path):
        pytest.skip("Simple English Wikipedia parquet not downloaded")
    n = kept = 0
    for row in iter_articles(path, row_groups=[0]):
        art, ps = clean_and_split(row["title"], row["text"], CleanCfg())
        n += 1
        kept += art.drop_reason is None
        for p in ps:
            assert n_words(p.body) <= 110
            assert default_count_tokens(p.render()) <= 160
            assert "{{" not in p.body and "<ref" not in p.body and "'''" not in p.body
    assert n == 1000 and kept > 900
