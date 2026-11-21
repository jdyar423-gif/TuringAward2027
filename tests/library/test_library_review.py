"""Regression tests for the library review (docs/E1_COVERAGE.md, "Review fixes"): the qa
contrast check (library-1), primary topics and person names in the title look-up
(library-2), conversation turns (library-4), the time-sensitive filter (library-5), the
provenance arrays outside the shipped directory (library-6) and the threshold rule with
negative dev sets. Each test fails on the code before the fixes."""
from __future__ import annotations

import os
import random
import re

import numpy as np
import pytest

from bes.library import gate as G
from bes.library import qa as Q
from bes.library.clean import proxy_count_tokens
from bes.library.index import IndexCfg
from bes.library.pack import PackCfg
from bes.library.retrieve import Library, RetrieveCfg, query_type, turn_type
from bes.library.terms import STOPWORDS, stem, strip_disambiguator, tokens

# ------------------------------------------------------------------ a small library


def person_text(name: str, years: str, job: str, alone: str, he: str = "He") -> str:
    return (f"{name} ({years}) was a famous {job}. {he} was born in a small town. {alone} worked for many years. "
            f"{alone} became very famous. {he} died at home. {he} is still remembered.")


ARTICLES = [
    # title, first passage, stored importance, importance for who-frames (None: the same)
    ("Galileo Galilei", person_text("Galileo Galilei", "15 February 1564 – 8 January 1642", "astronomer", "Galileo"),
     6.38, None),
    ("Galileo (spacecraft)", "Galileo was an unmanned spacecraft. It explored the planet Jupiter.", 6.24, 1.4),
    ("Charles Darwin", person_text("Charles Darwin", "12 February 1809 – 19 April 1882", "naturalist", "Darwin"),
     7.65, None),
    ("Darwin (operating system)", "Darwin is a free operating system made by a computer company.", 7.27, 1.5),
    ("Abraham Lincoln", person_text("Abraham Lincoln", "February 12, 1809 – April 15, 1865", "president", "Lincoln"),
     8.18, None),
    ("Lincoln (movie)", "Lincoln is a 2012 biographical war drama movie about a president.", 9.02, 1.6),
    ("Lincoln Memorial", "The Lincoln Memorial is a monument in a big city. It was built in 1922.", 5.04, None),
    ("Pointer (computer programming)", "A pointer is a variable that stores a memory address.", 5.34, 1.5),
    ("The Pointer", "The Pointer is a 1939 animated short film about a hunting dog.", 4.59, None),
    ("The Pointer Sisters", "The Pointer Sisters were a singing group of four sisters.", 4.19, None),
    ("Johann Sebastian Bach", person_text("Johann Sebastian Bach", "31 March 1685 – 28 July 1750", "composer", "Bach"),
     7.42, None),
    ("Bachs", "Bachs is a municipality of the district of Dielsdorf in Switzerland.", 3.0, None),
    ("Britney Spears", person_text("Britney Spears", "born December 2, 1981", "singer", "Spears", "She"), 7.0, None),
    ("Spear", "A spear is a pole weapon with a pointed head. A spear can be thrown.", 5.12, None),
    ("John F. Kennedy", person_text("John F. Kennedy", "May 29, 1917 – November 22, 1963", "president", "Kennedy"),
     7.99, None),
    ("Robert F. Kennedy", person_text("Robert F. Kennedy", "November 20, 1925 – June 6, 1968", "politician", "Kennedy"),
     5.73, None),
    ("Don Kennedy", person_text("Don Kennedy", "March 2, 1930 – June 29, 2023", "radio host", "Kennedy"), 3.0, None),
    ("The Beatles", "The Beatles were an English rock band from Liverpool. They were very famous.", 8.0, None),
    ("The Beatles (album)", "The Beatles is the ninth album by the English rock band of that name.", 7.5, 1.8),
    ("May", "May is the fifth month of the year. It has 31 days.", 12.16, None),
    ("Theresa May", person_text("Theresa May", "born 1 October 1956", "politician", "May", "She"), 6.52, None),
    ("Sky blue", "Sky blue is a shade of cyan. It looks like the sky on a clear day.", 3.0, None),
    ("Sky", "The sky is everything above the surface of the Earth. The sky looks blue in the day.", 7.0, None),
    ("The Capital", "The Capital is a daily newspaper published in a small city since 1884.", 4.36, None),
    ("Capital (economics)", "In economics, capital is wealth used to make more wealth.", 8.56, 1.7),
    ("Merry Christmas (album)", "Merry Christmas is a holiday album by an American singer.", 6.0, 1.6),
    ("Try", "A try is one of the ways of scoring points in rugby league and rugby union.", 4.0, None),
    ("Lens (optics)", "A lens is a clear object that bends light. A lens can be made of glass.", 5.0, 1.5),
    ("While loop", "In computer science, a while loop repeats code while a condition is true.", 4.2, None),
    ("For loop", "In computer science, a for loop repeats code a set number of times.", 4.0, None),
    ("Hamlet", "Hamlet is a tragedy written by William Shakespeare around 1600.", 7.0, None),
    ("Mercury (planet)", "Mercury is the smallest planet in the Solar System.", 7.04, 1.6),
    ("Mercury (element)", "Mercury is a chemical element with the symbol Hg.", 7.21, 1.6),
    ("Freddie Mercury", person_text("Freddie Mercury", "5 September 1946 – 24 November 1991", "singer", "Mercury"),
     5.9, None),
    ("Paris", "Paris is the capital and largest city of France.", 9.0, None),
]


def exact_gate(threshold: float = 0.5) -> G.GateModel:
    """Fires on exact-title candidates and on qa entries with a full overlap only."""
    cw = np.zeros(len(G.WIKI_FEATURES) + 1)
    cw[G.WIKI_FEATURES.index("exact_title")] = 8.0
    cw[-1] = -4.0
    cq = np.zeros(len(G.QA_FEATURES) + 1)
    cq[G.QA_FEATURES.index("sym")] = 10.0
    cq[-1] = -6.0
    return G.GateModel.stored({"wiki": cw, "qa": cq}, threshold)


def small_library(qa_pairs=(), gate=None, tmp_path=None):
    docs = [(t, body) for t, body, _, _ in ARTICLES]
    imp = [i for _, _, i, _ in ARTICLES]
    nimp = [i if n is None else n for _, _, i, n in ARTICLES]
    store = None
    if qa_pairs:
        pairs = [Q.QAPair(q, a, "s", i) for i, (q, a) in enumerate(qa_pairs)]
        store = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.0), PackCfg(block_raw_bytes=256))
    return Library.build(docs, out_dir=None if tmp_path is None else str(tmp_path),
                         index_cfg=IndexCfg(budget_numbers=10 ** 7), pack_cfg=PackCfg(block_raw_bytes=512),
                         cfg=RetrieveCfg(qa_min_sym=0.5), count_tokens=proxy_count_tokens, qa=store,
                         title_importance=imp, name_importance=nimp, gate=gate or exact_gate())


def got_title(lib, q, **kw):
    p = lib.retrieve(q, **kw)
    return None if p is None else (p.title if p.kind == "wiki" else "qa: " + p.body)


# ------------------------------------------------------------------ library-2: primary topics


@pytest.mark.parametrize("q, want", [
    ("Who was Galileo?", "Galileo Galilei"),        # was: Galileo (spacecraft)
    ("Who was Darwin?", "Charles Darwin"),          # was: Darwin (operating system)
    ("Who was Lincoln?", "Abraham Lincoln"),        # was: Lincoln (movie)
    ("Who was Bach?", "Johann Sebastian Bach"),     # was: Bachs (plural stemmer)
    ("Who was Spears?", "Britney Spears"),          # was: Spear
    ("Who was Kennedy?", "John F. Kennedy"),        # was: Don Kennedy ("don" is a stopword)
    ("Who was May?", "Theresa May"),                # a who-frame does not get the month
    ("What is May?", "May"),
    ("What is a spear?", "Spear"),
    ("Who were the Beatles?", "The Beatles"),       # its disambiguated sibling does not compete
    ("Who was Galileo Galilei?", "Galileo Galilei"),
    ("Tell me about Hamlet.", "Hamlet"),
    ("Paris", "Paris"),
])
def test_primary_topics_and_person_names(q, want):
    lib = small_library()
    assert got_title(lib, q) == want


@pytest.mark.parametrize("q", [
    "What is a pointer?",          # 'The Pointer' and 'Pointer (computer programming)': ambiguous
    "What is Mercury?",            # planet / element / Freddie Mercury: ambiguous
    "What is a for loop?",         # the key "loop" must not reach 'While loop'
    "Why is the sky blue?",        # not an identity request: no exact-title credit for 'Sky blue'
    "What does len do?",           # 'Lens (optics)' through the plural stemmer
    "What is its capital?",        # a relative object names no article (first turn too)
    "try again",                   # a one-term key needs a frame or a capitalised name
])
def test_no_exact_title_credit_without_a_primary_topic(q):
    lib = small_library()
    assert got_title(lib, q) is None
    for c in lib.candidates(q, 1, kinds=("wiki",)):
        assert not c.feats["exact_title"]


def test_person_analysis_of_the_small_library():
    from bes.library.titles import analyse

    info = {t: analyse(t, body) for t, body, _, _ in ARTICLES}
    persons = {t for t, i in info.items() if i.person}
    assert persons == {"Galileo Galilei", "Charles Darwin", "Abraham Lincoln", "Johann Sebastian Bach",
                       "Britney Spears", "John F. Kennedy", "Robert F. Kennedy", "Don Kennedy", "Theresa May",
                       "Freddie Mercury"}
    assert info["Galileo Galilei"].names[0] == "galileo"          # the name the article uses
    assert set(info["Galileo Galilei"].names) == {"galileo", "galilei"}
    assert info["John F. Kennedy"].names == ("kennedy",)
    assert info["Bachs"].key == "bachs" and info["Spear"].key == "spear"
    assert info["Britney Spears"].key == "britney spears" and info["Lens (optics)"].key == "len"
    assert not info["The Beatles"].plain and info["May"].plain and not info["The Pointer"].plain


# brute-force reference of the primary-topic rule (bes.library.titles module docstring)

VOCAB = "alpha beta gamma delta sigma omega river stone".split()


def ref_tables(titles, infos, imp, nimp, margin, pmargin):
    """{key: owner or None}, {key: who-frame answer or None} computed by looping over all
    titles for every key."""
    n = len(titles)
    nondis = {t for i, t in enumerate(titles) if strip_disambiguator(t) == t and not infos[i].disambiguation_page}

    def collapsed(i):
        return strip_disambiguator(titles[i]) != titles[i] and strip_disambiguator(titles[i]) in nondis

    keys = {inf.key for inf in infos if inf.key} | {w for inf in infos for w in inf.names}
    general, who = {}, {}
    for k in sorted(keys):
        words = k.split()
        plains = [i for i in range(n) if infos[i].key == k and infos[i].plain and not infos[i].disambiguation_page]
        owner = None
        if len(plains) == 1:
            owner = plains[0]
        else:
            if plains:
                pool = plains
            else:
                pool = []
                for i in range(n):
                    if infos[i].disambiguation_page or collapsed(i):
                        continue
                    kw = set(infos[i].key.split())
                    if infos[i].person:
                        kw &= set(infos[i].names)
                    if set(words) <= kw:
                        pool.append(i)
            if pool:
                pool.sort(key=lambda i: (-imp[i], i))
                b = pool[0]
                if (len(pool) == 1 or imp[b] - imp[pool[1]] >= margin) and \
                        (infos[b].key == k or (len(words) == 1 and k in infos[b].names)):
                    owner = b
        general[k] = owner
        if len(words) == 1:
            ps = [i for i in range(n) if infos[i].person and k in infos[i].names]
            if not ps:
                who[k] = owner
            elif owner is not None and infos[owner].person and infos[owner].plain:
                who[k] = owner
            else:
                ps.sort(key=lambda i: (-nimp[i], i))
                who[k] = ps[0] if len(ps) == 1 or nimp[ps[0]] - nimp[ps[1]] >= pmargin else None
        else:
            who[k] = owner
    return general, who


def random_titles(seed):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < 60:
        w = rng.sample(VOCAB, 2)
        kind = rng.randrange(7)
        if kind == 0:
            t, text = w[0].title(), f"{w[0].title()} is a thing. A {w[0]} is common."
        elif kind == 1:
            t, text = f"{w[0].title()} ({w[1]})", f"{w[0].title()} is a {w[1]} thing."
        elif kind == 2:
            t, text = f"The {w[0].title()}", f"The {w[0].title()} is a newspaper."
        elif kind == 3:
            t, text = f"{w[0].title()} {w[1].title()}", f"{w[0].title()} {w[1].title()} is a place."
        elif kind == 4:
            t = f"{w[0].title()} {w[1].title()}"
            text = person_text(t, f"{rng.randint(1800, 1900)} – {rng.randint(1901, 1990)}", "painter",
                               w[1].title() if rng.random() < 0.8 else w[0].title())
        elif kind == 5:
            t, text = f"{w[0].title()}s", f"{w[0].title()}s is a village in a valley."
        else:
            t, text = f"{w[0].title()} (disambiguation)", f"{w[0].title()} may refer to several things."
        if t not in seen:
            seen.add(t)
            out.append((t, text))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_title_table_against_brute_force(seed):
    from bes.library.titles import TitleTable, analyse

    rng = np.random.default_rng(seed)
    arts = random_titles(seed)
    titles = [t for t, _ in arts]
    infos = [analyse(t, x) for t, x in arts]
    imp = np.round(rng.uniform(1, 8, len(titles)), 1)          # ties and near-ties on purpose
    nimp = np.where([strip_disambiguator(t) != t for t in titles], 1.0, imp)
    table = TitleTable.build(titles, imp, infos=infos, name_importance=nimp, title_margin=1.0, person_margin=1.0)
    general, who = ref_tables(titles, infos, imp, nimp, 1.0, 1.0)
    for k, owner in general.items():
        hit = table.lookup(k)
        assert (None if hit is None else hit[0]) == owner, (k, titles[owner] if owner is not None else None, hit)
        if hit is not None:
            info = infos[hit[0]]
            assert hit[1] == (not (info.plain and info.key == k)) and hit[2] == info.person
        w = table.lookup(k, who=True)
        got = None if w is None or w[0] < 0 else w[0]
        assert got == who[k], (k, got, who[k])


def test_title_hash_collision_is_rejected(monkeypatch):
    import bes.library.titles as T

    lib = small_library()
    titles = lib.packed.all_titles()
    monkeypatch.setattr(T, "key_hash", lambda key: 7)          # every key collides
    monkeypatch.setattr(T, "who_hash", lambda key: 7)
    lib.titles = T.TitleTable.build(titles, [i for _, _, i, _ in ARTICLES], [b for _, b, _, _ in ARTICLES])
    assert lib.titles.hash.shape[0] == 1                       # one entry answers for every key
    tid = lib.titles.lookup("anything")[0]
    for q in ("Who was Napoleon?", "What is Hamlet?", "Paris"):
        hit = lib.title_hit(q)
        if hit is not None:                                     # only a title the key can name
            assert T.verify(" ".join(t for t in tokens(q) if t not in STOPWORDS), titles[hit[0]])
    assert lib.title_hit("What is Zebra?") is None or titles[tid] == "Zebra"


# ------------------------------------------------------------------ library-4: conversation turns


@pytest.mark.parametrize("q", ["What is its capital?", "What is the next step?", "Is it safe?", "What color is it?",
                               "And cats?", "What about Hamlet?", "Who is the author?", "When did he die?",
                               "What does that mean?", "Which one is better?", "the second one"])
def test_later_turns_that_lean_on_the_earlier_one_retrieve_nothing(q):
    lib = small_library()
    assert turn_type(q, ["Tell me about Paris."]) == "followup"
    assert lib.retrieve(q, context=["Tell me about Paris."]) is None


@pytest.mark.parametrize("q", ["merry christmas", "try again", "thanks, that was helpful", "ok great", "Thank you so much",
                               "Hello there", "Good morning", "cool, thanks!", "That's wrong", "Really?"])
def test_chat_retrieves_nothing_in_any_turn(q):
    lib = small_library()
    assert query_type(q) == "other"
    assert lib.retrieve(q) is None and lib.retrieve(q, context=["Who was Darwin?"]) is None


def test_context_argument_is_backward_compatible():
    lib = small_library()
    for q in ("Who was Darwin?", "Tell me about Hamlet.", "Paris", "What is a spear?", "Who was Lincoln?"):
        a, b, c = lib.retrieve(q), lib.retrieve(q, context=None), lib.retrieve(q, context=[])
        assert a is not None and a == b == c
    # a later turn that names its subject is looked up
    assert got_title(lib, "Who was Abraham Lincoln?", context=["Tell me about the civil war."]) == "Abraham Lincoln"
    assert got_title(lib, "Tell me about Hamlet.", context=["I like plays."]) == "Hamlet"


# ------------------------------------------------------------------ library-1: contrast check

STORED = [("what year did the pistons win their last championship?", "2004"),
          ("who won the first super bowl in 1966?", "Green Bay Packers"),
          ("who has won the most super bowls 2017?", "Pittsburgh Steelers"),
          ("which countries do not have a bill of rights?", "Australia"),
          ("In France, if you order viande chevaline, what type of meat will you get?", "Horse"),
          ("Which prime number is between 11 and 17?", "13"),
          ("where did anne frank live before the war?", "Frankfurt"),
          ("What movie character said \"Do or do not. There is no try\"?", "Yoda"),
          ("who is the second richest person of the world?", "Warren Buffett"),
          ("who wrote hamlet?", "William Shakespeare"),
          ("when did world war ii end?", "1945")]


@pytest.mark.parametrize("q, stored_answer", [
    ("what year did the pistons win their first championship?", "2004"),
    ("Who won the Super Bowl in 2020?", "Green Bay Packers"),
    ("Who won the Super Bowl in 2017?", "Pittsburgh Steelers"),
    ("which countries do have a bill of rights?", "Australia"),
    ("In Turkey, if you order viande chevaline, what type of meat will you get?", "Horse"),
    ("Is 17 a prime number?", "13"),
    ("where did anne frank live after the war?", "Frankfurt"),
    ("What movie character said \"Do or do. There is no try\"?", "Yoda"),       # one of two negations dropped
    ("Who is the richest person in the world?", "Warren Buffett"),
])
def test_a_stored_answer_to_a_different_question_is_not_returned(q, stored_answer):
    lib = small_library(STORED, gate=exact_gate(threshold=0.01))
    for c in lib.candidates(q, 3, kinds=("qa",)):
        assert c.feats["answer"] != stored_answer, (q, c.feats["answer"])
    p = lib.retrieve(q, kinds=("qa",))
    assert p is None or p.body != stored_answer


@pytest.mark.parametrize("q, answer", [("Who wrote Hamlet?", "William Shakespeare"),
                                       ("When did World War 2 end?", "1945"),
                                       ("When did the Second World War end?", "1945"),
                                       ("which countries do not have a bill of rights", "Australia")])
def test_the_same_question_in_other_words_still_matches(q, answer):
    lib = small_library(STORED, gate=exact_gate(threshold=0.01))
    p = lib.retrieve(q, kinds=("qa",))
    assert p is not None and p.body == answer


REF_ORD = {w: i for i, w in enumerate("zeroth first second third fourth fifth sixth seventh eighth ninth tenth "
                                      "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
                                      "eighteenth nineteenth twentieth".split())}
REF_ROMAN = {"ii": 2, "iii": 3, "iv": 4, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "xi": 11, "xii": 12, "xiii": 13,
             "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20}


def ref_signature(text):
    """Independent reading of the contrast classes (bes.library.qa.contrast_mismatch)."""
    toks = tokens(text)
    nums = set()
    for t in toks:
        m = re.fullmatch(r"(\d+)(st|nd|rd|th)", t)
        if m:
            nums.add(str(int(m.group(1))))
        elif t in REF_ORD:
            nums.add(str(REF_ORD[t]))
        elif t in REF_ROMAN:
            nums.add(str(REF_ROMAN[t]))
        elif re.search(r"\d", t):
            nums.add(t)
    neg = sum(t in ("not", "no", "nor", "never", "without", "none", "neither", "nobody", "nothing", "nowhere",
                    "cannot") for t in toks) + sum(a in ("didn", "doesn", "don", "isn", "wasn", "aren", "weren",
                                                         "hasn", "haven", "hadn", "couldn", "wouldn", "shouldn",
                                                         "mustn", "won", "can", "ain", "shan", "needn")
                                                   and b == "t" for a, b in zip(toks, toks[1:]))
    order = {t for t in toks if t in ("last", "next", "previous", "penultimate", "latter", "former")}
    rel = {t for t in toks if t in ("before", "after", "prior", "since", "until", "till")}
    deg = {t for t in toks if t in Q.DEGREE_WORDS}
    return nums, order, deg, rel, neg


@pytest.mark.parametrize("seed", [0, 1])
def test_contrast_classes_against_an_independent_reading(seed):
    rng = random.Random(seed)
    words = ("who won the first second 2nd 3rd last next before after not never didn t world war ii iv 1966 "
             "2017 most least largest more fewer cup since until super bowl x v i").split()
    for _ in range(500):
        a = " ".join(rng.choice(words) for _ in range(rng.randint(1, 9)))
        b = " ".join(rng.choice(words) for _ in range(rng.randint(1, 9)))
        ca, cb = Q.contrast(a), Q.contrast(b)
        ra, rb = ref_signature(a), ref_signature(b)
        assert (set(ca.numbers), set(ca.order), set(ca.degree), set(ca.relation), ca.negations) == ra
        want = ra != rb                    # all lower case: no names, the word classes decide
        assert (Q.contrast_mismatch(ca, cb) is not None) == want, (a, b)


def test_names_must_match_in_both_directions():
    m = Q.contrast_mismatch
    c = Q.contrast
    assert m(c("Who captained the Wales team in 1999?"), c("who captained the england team in 1999?")) == "names"
    assert m(c("who captained the team in 1999?"), c("Who captained the England team in 1999?")) == "stored_names"
    assert m(c("Who painted the Mona Lisa?"), c("who painted the mona lisa?")) is None
    assert m(c("Who painted the Mona Lisa?"), c("Which artist painted the Mona Lisa?")) is None
    assert c("WHO PAINTED THE MONA LISA").names is None          # no case information


# ------------------------------------------------------------------ library-5: time-sensitive filter


@pytest.mark.parametrize("q", [
    "who is the most subscribed youtuber in the world?", "who's got the most followers on twitter?",
    "whats the most viewed video on you tube?", "how many seasons if greys anatomy are there?",
    "how many states in usa have death penalty?", "how many stores does toys r us have in the united states?",
    "who s the oldest person in the world?", "how many states in us still have death penalty?",
    "who is the oldest living person?", "who is the second richest person of the world?",
    "what is the tallest building in the world?", "how many members does the european union have?",
    "how many countries are there in nato?", "who is france's president?", "who's the prime minister of canada?",
    "who runs apple?", "who has won the most super bowls 2017?", "who is the richest man in the world?",
])
def test_new_time_sensitive_questions(q):
    assert Q.time_sensitive(q) is not None


@pytest.mark.parametrize("q", [
    "how many children did henry viii have?", "how many legs does a spider have?", "who wrote hamlet",
    "how many episodes are in season 2 of lost?", "who was the first president of the united states?",
    "what is the oldest city in the world?", "how many seasons of breaking bad were there?",
    "who painted the mona lisa?", "what is the capital of australia?",
])
def test_stable_questions_stay(q):
    assert Q.time_sensitive(q) is None


# ------------------------------------------------------------------ library-6: provenance outside


def test_provenance_is_not_inside_the_shipped_library(tmp_path):
    from bes.library.build import Signals, build_for_budget, load_provenance, provenance_dir
    from bes.library.select import article_order, estimate_costs, passage_order

    import importlib.util
    spec = importlib.util.spec_from_file_location("sel_tests", os.path.join(os.path.dirname(__file__),
                                                                            "test_library_select.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pool = mod.make_pool(0, 40)
    order = passage_order(pool, article_order(pool))
    cost, _ = estimate_costs(pool, order, PackCfg(block_raw_bytes=512))
    sig = Signals(np.zeros(pool.n_articles), np.zeros(pool.n_passages, bool), order, cost)
    out = tmp_path / "lib"
    os.makedirs(out / "provenance")
    np.save(out / "provenance" / "stale.npy", np.zeros(3))          # left by an older build
    lib, sel = build_for_budget(pool, sig, 10 ** 7, IndexCfg(budget_numbers=10 ** 6), PackCfg(block_raw_bytes=512),
                                count_tokens=proxy_count_tokens, out_dir=str(out))
    shipped = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs if f.endswith(".npy")]
    numbers = sum((np.load(f, mmap_mode="r").nbytes + 1) // 2 for f in shipped)
    assert numbers + lib.index.HEADER_NUMBERS == lib.accounting()["total_numbers"]   # every shipped array counted
    assert not os.path.exists(out / "provenance")
    prov = provenance_dir(str(out))
    assert not os.path.abspath(prov).startswith(os.path.abspath(out) + os.sep)
    assert np.array_equal(load_provenance(str(out), "pool_idx"), sel.pool_idx)
    renamed = tmp_path / "lib.renamed"                                 # the supervisor renames after the build
    os.replace(out, renamed)
    assert np.array_equal(load_provenance(str(renamed), "pool_idx"), sel.pool_idx)


# ------------------------------------------------------------------ threshold with negative sets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_threshold_with_negative_sets_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 300
    p = np.round(rng.uniform(size=n), 2)
    p[rng.uniform(size=n) < 0.1] = -np.inf
    groups = rng.choice(["a", "b", "neg"], size=n)
    y = (rng.uniform(size=n) < np.clip(p + 0.2, 0, 1)) & (groups != "neg")
    for target, rate in ((0.8, 0.05), (0.6, 0.0), (0.9, 0.2)):
        t, pt = G.choose_threshold_sets(p, y, groups, target, negative=["neg"], max_negative_rate=rate)
        ok = []
        for c in sorted(set(p[np.isfinite(p)].tolist())):
            sel = p >= c
            good = all(not (sel & (groups == s)).any() or y[sel & (groups == s)].mean() >= target for s in "ab")
            good &= (sel & (groups == "neg")).sum() <= rate * (groups == "neg").sum()
            if good:
                ok.append(c)
        assert t == (min(ok) if ok else float("inf"))
