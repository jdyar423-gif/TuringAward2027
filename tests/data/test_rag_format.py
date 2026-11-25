"""Tests of bes.data.rag_format with a small fake library that has "wiki" and "qa" entries.

The fake is built so that an entry is the best match for its own question: without the
exclusions the builders would retrieve themselves, so "no self-retrieval" is a real check.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from bes.config import BOS, LIBRARY, USER, DataCfg, MixCfg
from bes.data import chat_format as cf
from bes.data import mixture as M
from bes.data import rag_format as R
from bes.data import samples as S
from bes.data import sources
from bes.exact import constraints as C
from bes.library.clean import split_sentences
from bes.library.retrieve import Library, query_type
from bes.library.terms import strip_disambiguator, terms
from bes.tokenizer.tok import load

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIB_DIR = ROOT / "data_cache" / "library" / "lib_v2"
SHARD_DIR = ROOT / "data_cache" / "tok" / "smoltalk_constraints"
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

ARTICLES = [
    ("Alan Turing", ["Alan Turing was an English mathematician and computer scientist. He is "
                     "called the father of computer science. He worked on codes in the war.",
                     "Early life\nTuring was born in London in 1912. He went to Sherborne School. "
                     "He liked science from an early age.",
                     "Career\nTuring worked at Bletchley Park during the Second World War. He broke codes."]),
    ("Paris", ["Paris is the capital and largest city of France. The Seine river flows through Paris.",
               "History\nParis was founded long ago by a tribe called the Parisii. It grew into a large city.",
               "Weird section name\nThis passage starts with a heading that is not in the list."]),
    ("Mint (candy)", ["A mint is a sweet candy with a fresh taste. Mints are often eaten after meals."]),
    ("Three Lines of Defence", ["Three Lines of Defence is a model used in risk management. It splits "
                                "duties into three groups."]),
    ("Two Paragraphs", ["Two Paragraphs is a short film from 2001. It was made in Canada."]),
    ("List of rivers", ["- Nile\n- Amazon\n- Yangtze"]),
    ("Photosynthesis", ["Photosynthesis is how plants make food from light. It happens in leaves. It "
                        "makes oxygen."]),
    ("Seven Words", ["Seven Words is a song by a band from Sacramento. It came out in 1998."]),
    ("Disambiguation", ["Mercury may mean:"]),
    ("Rome", ["Rome is the capital city of Italy. It is very old."]),
]
QAS = [
    ("Name three rivers in Africa.", "The Nile, the Congo and the Niger."),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Italy?", "Rome"),
    ("Who broke codes at Bletchley Park?", "Alan Turing"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("Who wrote the play Hamlet?", "William Shakespeare"),
    ("Who wrote the play Macbeth?", "Shakespeare"),
    ("When was Alan Turing born?", "in 1912"),
    ("Why is the sky blue?", "Air scatters blue light more than red light. That is called Rayleigh scattering."),
]


@dataclass(frozen=True)
class FakePassage:
    pid: int
    title_id: int
    title: str
    body: str
    score: float
    text: str
    kind: str


class FakeLibrary:
    """The extended library contract (kinds, kind_range) over a tiny corpus; term-overlap
    scores with a title bonus, a gate on the score and the query type; records every call.

    ``qa_titles=True`` groups qa entries with the same question under one title id (duplicates
    allowed); ``False`` behaves like ``bes.library.retrieve.Library``: qa entries have title id
    -1 and duplicate questions are dropped."""

    def __init__(self, articles=ARTICLES, qas=QAS, gate_min: float = 3.0, qa_titles: bool = True):
        if not qa_titles:
            qas = list({q: (q, a) for q, a in qas}.values())
        rows = [("wiki", t, b) for t, bodies in articles for b in bodies]
        rows += [("qa", q, a) for q, a in sorted(qas, key=lambda x: x[0])]
        self.rows = rows
        self.n_wiki = sum(1 for k, _, _ in rows if k == "wiki")
        self.titles: list[str] = []
        self.tid: list[int] = []
        for kind, title, _ in rows:
            if kind == "qa" and not qa_titles:
                self.tid.append(-1)
                continue
            if not self.titles or self.titles[-1] != title:
                self.titles.append(title)
            self.tid.append(len(self.titles) - 1)
        self.first = [self.tid.index(t) for t in range(len(self.titles))] + [self.n_wiki if not qa_titles
                                                                              else len(rows)]
        self.cfg = SimpleNamespace(gate=SimpleNamespace(allowed_types=("question", "lookup")))
        self.gate_min = gate_min
        self.calls: list[tuple] = []

    @property
    def n_passages(self) -> int:
        return len(self.rows)

    def kind_range(self, kind: str) -> range:
        return range(0, self.n_wiki) if kind == "wiki" else range(self.n_wiki, len(self.rows))

    def title_id_of(self, pid: int) -> int:
        return self.tid[pid]

    def title_range(self, tid: int) -> range:
        return range(0) if tid < 0 else range(self.first[tid], self.first[tid + 1])

    def find_title(self, title: str):
        return self.titles.index(title) if title in self.titles else None

    def passage(self, pid: int, score: float = 0.0) -> FakePassage:
        kind, title, body = self.rows[pid]
        text = f"Q: {title}\nA: {body}" if kind == "qa" else f"{title}\n{body}"
        return FakePassage(pid, self.tid[pid], title, body, score, text, kind)

    def _ranked(self, query, exclude_pids, exclude_titles, kinds):
        q = set(terms(query))
        out = []
        for pid, (kind, title, body) in enumerate(self.rows):
            if pid in set(exclude_pids) or self.tid[pid] in set(exclude_titles):
                continue
            if kinds is not None and kind not in kinds:
                continue
            score = len(q & set(terms(body))) + 2 * len(q & set(terms(strip_disambiguator(title))))
            if score > 0:
                out.append((-score, pid))
        return [(pid, -neg) for neg, pid in sorted(out)]

    def retrieve(self, query, exclude_pids=(), exclude_titles=(), log=None, kinds=None):
        self.calls.append(("retrieve", query, tuple(exclude_pids), tuple(exclude_titles), kinds))
        if query_type(query) not in self.cfg.gate.allowed_types:
            return None
        ranked = self._ranked(query, exclude_pids, exclude_titles, kinds)
        if not ranked or ranked[0][1] < self.gate_min:
            return None
        return self.passage(ranked[0][0], ranked[0][1])

    def retrieve_topk(self, query, k, exclude_pids=(), exclude_titles=(), log=None, kinds=None):
        self.calls.append(("topk", query, tuple(exclude_pids), tuple(exclude_titles), kinds))
        return [self.passage(p, s) for p, s in self._ranked(query, exclude_pids, exclude_titles, kinds)[:k]]


@pytest.fixture(scope="module")
def tok():
    return load()


@pytest.fixture()
def lib():
    return FakeLibrary()


@dataclass(frozen=True)
class OldPassage:
    pid: int
    title_id: int
    title: str
    body: str
    score: float
    text: str


class OldContract:
    """A library with the contract before kinds: no ``kind_range``, no ``kinds`` argument
    (passing one raises), passages without ``kind``."""

    def __init__(self, lib):
        self._lib, self.cfg = lib, lib.cfg

    @property
    def n_passages(self):
        return self._lib.n_passages

    def _old(self, p):
        return OldPassage(p.pid, p.title_id, p.title, p.body, p.score, p.text)

    def passage(self, pid):
        return self._old(self._lib.passage(pid))

    def title_id_of(self, pid):
        return self._lib.title_id_of(pid)

    def title_range(self, tid):
        return self._lib.title_range(tid)

    def find_title(self, title):
        return self._lib.find_title(title)

    def search(self, query, exclude_pids=(), exclude_titles=()):
        return self._lib.search(query, exclude_pids, exclude_titles)

    def retrieve(self, query, exclude_pids=(), exclude_titles=(), log=None):
        p = self._lib.retrieve(query, exclude_pids, exclude_titles)
        return None if p is None else self._old(p)

    def retrieve_topk(self, query, k, exclude_pids=(), exclude_titles=(), log=None):
        return [self._old(p) for p in self._lib.retrieve_topk(query, k, exclude_pids, exclude_titles)]


def toy_library() -> Library:
    """A real bes.library Library over ARTICLES (wiki passages only)."""
    from bes.library.index import IndexCfg
    from bes.library.pack import PackCfg
    from bes.library.retrieve import GateCfg, RetrieveCfg

    passages = [(t, b) for t, bodies in ARTICLES for b in bodies]
    cfg = RetrieveCfg(max_tokens=160, gate=GateCfg(min_score=1.0, min_margin=0.3, min_margin_title=0.05,
                                                   min_title_match=0.6))
    return Library.build(passages, index_cfg=IndexCfg(budget_numbers=10 ** 6),
                         pack_cfg=PackCfg(block_raw_bytes=512, titles_per_block=4), cfg=cfg)


def messages_of(sample, tok):
    return S.tokens_to_messages(sample.ids, tok)


def own_question_retrieves_itself(lib, pid, question, kinds):
    ranked = lib._ranked(question, (), (), kinds)
    return ranked and lib.tid[ranked[0][0]] == lib.tid[pid]


# ------------------------------------------------------------------------- the view
def test_view_of_the_current_library(tok):
    real = toy_library()
    view = R.as_view(real)
    assert view.native and view.kind_range("wiki") == range(real.n_passages) and not view.kind_range("qa")
    assert view.passage(0).kind == "wiki" and view.retrieve("Who was Alan Turing?", kinds=("qa",)) is None
    for variant in ("open", "closed", "distractor"):
        s = R.reading_task(0, view, tok, S.rng_for(variant), variant=variant)
        msgs = messages_of(s, tok)
        assert msgs[-2]["content"] and msgs[-1]["content"] in real.passage(0).text
        if variant == "distractor":
            assert not msgs[0]["content"].startswith("Alan Turing\n")
    assert R.pick_entry(view, "library_qa", 0) is None


def test_view_of_a_library_without_kinds():
    real = OldContract(toy_library())
    view = R.as_view(real)
    assert not view.native and R.as_view(view) is view and R.as_view(view).lib is real
    assert view.kind_range("wiki") == range(real.n_passages) and len(view.kind_range("qa")) == 0
    p = view.passage(0)
    assert p.kind == "wiki" and R.kind_of(p) == "wiki" and p.text == real.passage(0).text
    assert view.retrieve("Who was Alan Turing?", kinds=("qa",)) is None
    assert view.retrieve_topk("Alan Turing", 3, kinds=("qa",)) == []
    hit = view.retrieve("Who was Alan Turing?")
    assert hit is not None and hit.kind == "wiki" and hit.title == "Alan Turing"
    pids, _ = real.search("Alan Turing mathematician")
    assert view.ranked_pids("Alan Turing mathematician", 4) == [int(x) for x in pids[:4]]
    assert view.ranked_pids("Alan Turing", 4, kinds=("wiki",)) == view.ranked_pids("Alan Turing", 4)
    assert [p.pid for p in view.retrieve_topk("Alan Turing mathematician", 4)] == [int(x) for x in pids[:4]]
    tid = real.find_title("Alan Turing")
    assert all(not (r in real.title_range(tid)) for r in view.ranked_pids("Alan Turing", 10, exclude_titles=[tid]))
    assert view.is_lead(0) and not view.is_lead(1)
    rng = np.random.default_rng(0)
    for _ in range(50):
        pid = view.random_pid(rng, exclude_pids=[0], exclude_titles=[1])
        assert pid != 0 and view.title_id(pid) != 1
    with pytest.raises(KeyError):
        view.kind_range("poem")


def test_view_of_a_library_with_kinds(lib):
    view = R.as_view(lib)
    assert view.native
    assert view.kind_range("wiki") == range(0, lib.n_wiki) and view.kind_range("qa") == range(lib.n_wiki, lib.n_passages)
    assert {view.kind_of_pid(p) for p in view.kind_range("qa")} == {"qa"}
    assert view.retrieve("Who wrote Hamlet?", kinds=("qa",)).kind == "qa"
    assert all(p.kind == "wiki" for p in view.retrieve_topk("Hamlet Paris Turing", 10, kinds=("wiki",)))
    assert lib.calls[0][4] == ("qa",)


# ---------------------------------------------------------------------- reading tasks
def lead_pids(lib):
    return [lib.title_range(t).start for t in range(len(lib.titles)) if lib.title_range(t).start < lib.n_wiki]


def is_lead(lib, pid):
    return lib.title_range(lib.tid[pid]).start == pid


#: Passages of ARTICLES that no reading task can be built from (list, "may mean:", unknown heading).
DECLINED = {"title": 1, "answer": 1, "section": 1}


def check_reading(sample, pid, lib, tok, variant):
    p = lib.passage(pid)
    msgs = messages_of(sample, tok)
    roles = [m["role"] for m in msgs]
    assert roles == (["user", "assistant"] if variant == "closed" else ["library", "user", "assistant"])
    question, answer = msgs[-2]["content"], msgs[-1]["content"]
    assert C.compile(question).empty
    assert sample.tags[0] == f"read:{variant}"
    if variant != "closed":           # a passage only before a message the gate could serve
        assert query_type(question) in lib.cfg.gate.allowed_types
    body_lines = p.text.split("\n")[1:]
    para = body_lines[0] if is_lead(lib, pid) else body_lines[1]
    sents = split_sentences(para)
    assert any(answer == " ".join(sents[:k]) for k in (1, 2, 3)), (answer, sents)
    assert answer in p.text
    if variant == "open":
        assert msgs[0]["content"] == p.text
    if variant == "distractor":
        title = msgs[0]["content"].split("\n", 1)[0]
        assert title != p.title
        assert any(lib.passage(q).text == msgs[0]["content"] and lib.tid[q] != lib.tid[pid]
                   for q in range(lib.n_wiki))
    assert sample.source == "library_read" and sample.kind == "library"
    assert sample.tags[0] == f"read:{variant}"
    assert sample.exclude_pids == (pid,) and sample.exclude_titles == (lib.tid[pid],)
    np.testing.assert_array_equal(sample.w, S.loss_weights(sample.ids, sample.roles))
    lib_w = sample.w[sample.roles == S.ROLE_LIBRARY]
    assert (lib_w[lib_w > 0] == DataCfg().passage_loss_w).all()


@pytest.mark.parametrize("variant", ["open", "closed", "distractor"])
def test_reading_tasks(lib, tok, variant):
    stats = R.RagStats()
    built = 0
    for pid in range(lib.n_wiki):
        for seed in range(4):
            s = R.reading_task(pid, lib, tok, S.rng_for("read", pid, seed), variant=variant, stats=stats)
            if s is None:
                continue
            built += 1
            check_reading(s, pid, lib, tok, variant)
    assert built == 4 * (lib.n_wiki - sum(DECLINED.values()))
    d = stats.as_dict()
    for reason, n in DECLINED.items():
        assert d["counts"][f"read:declined_{reason}"] == 4 * n
    assert d["samples"] == {f"read:{variant}": built} and d["mean_tokens"][f"read:{variant}"] > 10


def test_reading_task_forms_and_declines(lib, tok):
    pid_hist = next(i for i, (k, t, b) in enumerate(lib.rows) if b.startswith("History\n"))
    s = R.reading_task(pid_hist, lib, tok, S.rng_for(1), variant="closed")
    q = messages_of(s, tok)[0]["content"]
    assert "history of Paris" in q and "read:section" in s.tags
    lead = R.reading_task(0, lib, tok, S.rng_for(1), variant="closed")
    assert "read:title" in lead.tags and "Alan Turing" in messages_of(lead, tok)[0]["content"]
    mint = lib.title_range(lib.find_title("Mint (candy)")).start
    q = messages_of(R.reading_task(mint, lib, tok, S.rng_for(2), variant="closed"), tok)[0]["content"]
    assert "Mint" in q and "(candy)" not in q
    qa_pid = lib.kind_range("qa").start
    assert R.reading_task(qa_pid, lib, tok, S.rng_for(1)) is None
    no_sections = MixCfg(section_tasks=False)
    assert R.reading_task(pid_hist, lib, tok, S.rng_for(1), mix=no_sections) is None
    with pytest.raises(ValueError):
        R.reading_task(0, lib, tok, S.rng_for(1), variant="sideways")


def test_library_variants_need_a_question_the_gate_could_serve(tok):
    """Every question about "2 + 2" reads as arithmetic to the library's query typing, so an
    open-book task on it becomes closed book; likewise a stored "question" that is a task."""
    lib = FakeLibrary(articles=[("2 + 2", ["2 + 2 is a sum that equals four. It is a common example."])],
                      qas=[("Write the name of the capital of Peru", "Lima"), ("Who wrote Hamlet?", "Shakespeare")])
    stats = R.RagStats()
    for variant in ("open", "distractor"):
        s = R.reading_task(0, lib, tok, S.rng_for(variant), variant=variant, stats=stats)
        assert s.tags[0] == "read:closed" and stats.counts[f"read:{variant}_not_question"] == 1
        assert [m["role"] for m in messages_of(s, tok)] == ["user", "assistant"]
    task_pid = next(p for p in lib.kind_range("qa") if lib.rows[p][1].startswith("Write"))
    for variant in ("copy", "neighbour"):
        s = R.qa_task(task_pid, lib, tok, S.rng_for(variant), variant=variant, stats=stats)
        assert s.tags == ("qa:closed",) and stats.counts[f"qa:{variant}_not_question"] == 1
    ok_pid = next(p for p in lib.kind_range("qa") if lib.rows[p][1].startswith("Who"))
    assert R.qa_task(ok_pid, lib, tok, S.rng_for(0), variant="copy").tags == ("qa:copy",)


def test_reading_distractors_never_come_from_the_own_article(lib, tok):
    for pid in lead_pids(lib):
        p = lib.passage(pid)
        question = None
        for seed in range(40):
            s = R.reading_task(pid, lib, tok, S.rng_for("d", pid, seed), variant="distractor",
                               mix=MixCfg(distractor_random=0.5))
            if s is None:
                break
            msgs = messages_of(s, tok)
            question = msgs[1]["content"]
            lib_text = msgs[0]["content"]
            sources_of_text = [q for q in range(lib.n_passages) if lib.passage(q).text == lib_text]
            assert sources_of_text and all(lib.tid[q] != lib.tid[pid] for q in sources_of_text)
            assert all(lib.rows[q][0] == "wiki" for q in sources_of_text)
        if question is not None:
            # without the exclusion the best passage would have been the article itself
            assert own_question_retrieves_itself(lib, pid, question, ("wiki",)), p.title


def test_templates_never_compile_to_a_constraint(tok):
    tricky = ["Three Lines of Defence", "Two Paragraphs", "Seven Words", "Five Bullet Points",
              "One Sentence", "Ten Commandments", "Use Three Sentences", "Write 50 Words", "1984",
              "Apollo 11", "The Four Lines", "Twelve Items", "Answer In One Word", "JSON", "Lowercase"]
    rng = np.random.default_rng(0)
    seen_constraint_template = False
    for title in tricky:
        for tmpl, _ in R.TITLE_TEMPLATES:
            seen_constraint_template |= not C.compile(tmpl.format(t=title)).empty
        for _ in range(20):
            q = R._first_ok(rng, R.TITLE_TEMPLATES, t=title)
            assert q is None or C.compile(q).empty
        for h in ("history", "early life", "plot"):
            q = R._first_ok(rng, R.SECTION_TEMPLATES, t=title, h=h)
            assert q is None or C.compile(q).empty
    assert seen_constraint_template      # some raw combinations do compile, so the check matters
    lib = FakeLibrary(articles=[(t, [f"{t} is a thing with a name. It has a long history."]) for t in tricky],
                      qas=[])
    for pid in range(lib.n_wiki):
        for seed in range(6):
            s = R.reading_task(pid, lib, tok, S.rng_for(pid, seed), variant="closed")
            if s is not None:
                assert C.compile(messages_of(s, tok)[0]["content"]).empty


# ------------------------------------------------------------------ question-answer tasks
def qa_pids(lib):
    return list(lib.kind_range("qa"))


def brute_force_neighbour(lib, pid):
    """The best other stored question (another pid and another question text), or None."""
    tid = lib.tid[pid]
    ranked = lib._ranked(lib.rows[pid][1], (pid,), (tid,) if tid >= 0 else (), ("qa",))
    return ranked[0][0] if ranked else None


@pytest.mark.parametrize("qa_titles", [True, False])
@pytest.mark.parametrize("variant", ["copy", "neighbour", "closed"])
def test_qa_tasks(tok, variant, qa_titles):
    lib = FakeLibrary(qa_titles=qa_titles)
    stats = R.RagStats()
    built = same_answer = fallbacks = 0
    for pid in qa_pids(lib):
        p = lib.passage(pid)
        for seed in range(5):
            s = R.qa_task(pid, lib, tok, S.rng_for("qa", pid, seed), variant=variant, stats=stats)
            if s is None:
                assert not C.compile(p.title).empty
                continue
            built += 1
            msgs = messages_of(s, tok)
            question, reply = msgs[-2]["content"], msgs[-1]["content"]
            assert question == p.title and C.compile(question).empty
            assert p.body.rstrip(".").lower() in reply.lower()
            assert s.source == "library_qa" and s.kind == "library"
            assert s.exclude_pids == (pid,) and s.exclude_titles == ((p.title_id,) if qa_titles else ())
            if variant == "copy":
                assert [m["role"] for m in msgs] == ["library", "user", "assistant"]
                assert msgs[0]["content"] == p.text == f"Q: {p.title}\nA: {p.body}"
            elif variant == "neighbour":
                want = brute_force_neighbour(lib, pid)
                if want is None:                       # no other question shares a term
                    assert s.tags == ("qa:closed",) and len(msgs) == 2
                    fallbacks += 1
                    continue
                assert s.tags == ("qa:neighbour",) and msgs[0]["content"] == lib.passage(want).text
                assert want != pid and lib.rows[want][1] != p.title
                assert not qa_titles or lib.tid[want] != lib.tid[pid]
                same_answer += lib.rows[want][2] == p.body
            else:
                assert [m["role"] for m in msgs] == ["user", "assistant"]
    declined = sum(1 for q in qa_pids(lib) if not C.compile(lib.rows[q][1]).empty)
    assert declined == 1 and built == 5 * (len(qa_pids(lib)) - declined)
    assert stats.counts["qa:declined_constraint"] == 5 * declined
    if variant == "neighbour":
        assert same_answer > 0 and stats.counts["qa:neighbour_same_answer"] == same_answer
        assert 0 < fallbacks == stats.counts["qa:neighbour_fallback"] < built


def test_qa_neighbour_skips_duplicates_of_the_same_question(lib, tok):
    dup = [pid for pid in qa_pids(lib) if lib.rows[pid][1] == "Who wrote Hamlet?"]
    assert len(dup) == 2 and lib.tid[dup[0]] == lib.tid[dup[1]]
    for pid in dup:
        s = R.qa_task(pid, lib, tok, S.rng_for(pid), variant="neighbour")
        nb_text = messages_of(s, tok)[0]["content"]
        assert nb_text.startswith("Q: ") and "Q: Who wrote Hamlet?\n" not in nb_text + "\n"
        assert own_question_retrieves_itself(lib, pid, lib.rows[pid][1], ("qa",))
    lonely = FakeLibrary(articles=ARTICLES, qas=[("Who wrote Hamlet?", "William Shakespeare")])
    s = R.qa_task(lonely.kind_range("qa").start, lonely, tok, S.rng_for(0), variant="neighbour")
    assert "qa:closed" in s.tags and [m["role"] for m in messages_of(s, tok)] == ["user", "assistant"]
    assert R.qa_task(0, lib, tok, S.rng_for(0)) is None         # a wiki passage is not a qa entry


def test_qa_answer_templates():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        out = R.qa_answer_text("william Shakespeare", rng)
        seen.add(out)
        assert out in {"William Shakespeare.", "The answer is william Shakespeare.",
                       "That would be william Shakespeare."}
    assert len(seen) == 3
    assert R.qa_answer_text("yes", np.random.default_rng(1)) in {"Yes.", "The answer is yes.", "That would be yes."}
    assert R.qa_answer_text("iPhone 4.", np.random.default_rng(2)) in {"iPhone 4.", "The answer is iPhone 4.",
                                                                     "That would be iPhone 4."}
    long = "Air scatters blue light more than red light. That is called Rayleigh scattering."
    assert R.qa_answer_text(long, rng) == long
    many = "one two three four five six seven eight nine"
    assert R.qa_answer_text(many, rng) == many


# ---------------------------------------------------------------------- insert_retrieval
def chat(tok, user="Who was Alan Turing?", system=None, extra_turns=0, **labels):
    conv = [{"role": "system", "content": system}] if system else []
    conv += [{"role": "user", "content": user}, {"role": "assistant", "content": "A mathematician."}]
    for i in range(extra_turns):
        conv += [{"role": "user", "content": f"And then {i}?"}, {"role": "assistant", "content": "More. " * 20}]
    return S.conversation_sample(conv, tok, source="smol_smoltalk", kind="chat", **labels)


def test_insert_retrieval_gated(lib, tok):
    s = chat(tok, system="Be brief.")
    stats = R.RagStats()
    out = R.insert_retrieval(s, lib, tok, S.rng_for(0), DataCfg(distractor_frac=0.0), stats=stats)
    msgs = messages_of(out, tok)
    assert [m["role"] for m in msgs] == ["system", "library", "user", "assistant"]
    best = lib.retrieve("Who was Alan Turing?")
    assert msgs[1]["content"] == best.text and best.title == "Alan Turing"
    turn, _ = cf.render_turn("library", best.text, tok)
    u = s.ids.tolist().index(USER)
    assert out.ids.tolist() == s.ids[:u].tolist() + turn.tolist() + s.ids[u:].tolist()
    assert out.tags == ("retrieval:gated",) and (out.source, out.kind) == (s.source, s.kind)
    np.testing.assert_array_equal(out.w, S.loss_weights(out.ids, out.roles))
    assert stats.as_dict()["retrieval"] == {"calls": 1, "fired": 1, "fired_rate": 1.0, "question_like_rate": 1.0,
                                            "gate_rate": 1.0, "distractor_share": 0.0, "later_turn_share": 0.0}
    closed = R.insert_retrieval(chat(tok, user="Who was Zorblax Quux?"), lib, tok, S.rng_for(0),
                                DataCfg(distractor_frac=0.0), stats=stats)
    assert closed.tags == () and stats.counts["retrieval:gate_closed"] == 1


def test_insert_retrieval_with_a_low_floor_shows_loose_exemplars(tok):
    """``DataCfg.exemplar_floor``: with probability ``exemplar_floor_frac`` a training sample's
    retrieval goes down to that score (an answers library); without the setting no random number
    is drawn, so older configurations keep their sample streams."""
    from bes.library.answers import AnswerCfg, AnswerLibrary

    pairs = [("Why does ice float on water?", "Ice is less dense than liquid water, so it floats.", "a"),
             ("Why does oil float on water?", "Oil is less dense than water and does not mix with it.", "a"),
             ("What is the capital of Australia?", "The capital of Australia is Canberra.", "b"),
             ("What is the capital of France?", "The capital of France is Paris.", "b"),
             ("Write a Python function that reverses a string.", "def reverse(s): return s[::-1]", "c"),
             ("Explain how a rainbow forms, step by step.", "Sunlight enters a raindrop, bends and splits into colours.", "a"),
             ("What are the three primary colors?", "Red, yellow and blue.", "b"),
             ("Name three countries in Europe.", "France, Germany and Spain.", "b"),
             ("Who wrote the play Hamlet?", "William Shakespeare wrote Hamlet.", "c"),
             ("Write a play about Hamlet's ghost.", "ACT I. A castle at night. The ghost enters.", "c"),
             ("How many days are in a leap year?", "A leap year has 366 days.", "a"),
             ("What is 12 times 12?", "12 times 12 is 144.", "a")]
    alib = AnswerLibrary.build(pairs, AnswerCfg(n_bits=12, verify_top=12))
    loose, close = "Why do boats float on the sea water?", "Why does ice float on cold water?"
    best = [c for c in alib.candidates(loose, 3) if c.contrast is None][0]
    assert alib.acfg.min_partial < best.sym < alib.acfg.exemplar_min and alib.retrieve(loose) is None
    floor = round(best.sym - 0.02, 3)
    off = DataCfg(distractor_frac=0.0)
    always = DataCfg(distractor_frac=0.0, exemplar_floor=floor, exemplar_floor_frac=1.0)
    half = DataCfg(distractor_frac=0.0, exemplar_floor=floor, exemplar_floor_frac=0.5)
    s = chat(tok, user=loose)
    assert R.insert_retrieval(s, alib, tok, S.rng_for(0), off) is s                    # the library's own floor
    stats = R.RagStats()
    out = R.insert_retrieval(s, alib, tok, S.rng_for(0), always, stats=stats)
    msgs = messages_of(out, tok)
    assert [m["role"] for m in msgs] == ["library", "user", "assistant"]
    assert msgs[0]["content"] == alib.retrieve(loose, floor=floor).text and out.tags == ("retrieval:gated",)
    assert stats.counts["retrieval:low_floor"] == 1 and stats.counts["retrieval:fired"] == 1
    assert stats.counts["retrieval:loose"] == 1
    fired = sum(R.insert_retrieval(s, alib, tok, S.rng_for("f", k), half) is not s for k in range(200))
    assert 70 <= fired <= 130                                                          # about half
    # a close question gets its exemplar under every setting, the same one
    c = chat(tok, user=close)
    texts = {messages_of(R.insert_retrieval(c, alib, tok, S.rng_for(1), cfg, stats=stats), tok)[0]["content"]
             for cfg in (off, always, half)}
    assert texts == {alib.retrieve(close).text} and stats.counts["retrieval:loose"] == 1     # not loose
    # without the setting the generator is not touched before the distractor draw
    a, b = S.rng_for(7), S.rng_for(7)
    R.insert_retrieval(c, alib, tok, a, off)
    b.random()                                                                         # the distractor draw alone
    assert a.random() == b.random()


def test_insert_retrieval_distractors(lib, tok):
    best_tid = lib.tid[lib.retrieve("Who was Alan Turing?").pid]
    modes = set()
    for seed in range(60):
        out = R.insert_retrieval(chat(tok), lib, tok, S.rng_for("x", seed), DataCfg(distractor_frac=1.0))
        msgs = messages_of(out, tok)
        assert msgs[0]["role"] == "library"
        src = [q for q in range(lib.n_passages) if lib.passage(q).text == msgs[0]["content"]]
        mode = out.tags[0]
        modes.add(mode)
        if mode == "retrieval:ranked":
            assert all(lib.tid[q] != best_tid for q in src)       # a lower-ranked article
    assert modes == {"retrieval:ranked", "retrieval:random"}


def test_insert_retrieval_respects_exclusions(lib, tok):
    tid = lib.find_title("Alan Turing")
    excl = dict(exclude_pids=tuple(lib.title_range(tid)), exclude_titles=(tid,))
    for frac in (0.0, 1.0):
        for seed in range(40):
            out = R.insert_retrieval(chat(tok, **excl), lib, tok, S.rng_for(seed), DataCfg(distractor_frac=frac))
            if out.tags:
                text = messages_of(out, tok)[0]["content"]
                assert not text.startswith("Alan Turing\n")
    # every call excludes the sample's entries (a distractor search also the gated passage)
    assert all(set(excl["exclude_pids"]) <= set(c[2]) and tid in c[3] for c in lib.calls)


def test_insert_retrieval_skips(lib, tok):
    stats = R.RagStats()
    rng = S.rng_for(0)
    doc = S.document_sample("Who was Alan Turing?", tok)
    assert R.insert_retrieval(doc, lib, tok, rng, stats=stats) is doc
    task = chat(tok, user="Write a poem about the sea.")
    assert R.insert_retrieval(task, lib, tok, rng, stats=stats) is task
    with_lib = S.conversation_sample([{"role": "library", "content": "Some passage."},
                                      {"role": "user", "content": "Who was Alan Turing?"},
                                      {"role": "assistant", "content": "A mathematician."}], tok)
    assert R.insert_retrieval(with_lib, lib, tok, rng, stats=stats) is with_lib
    c = stats.counts
    assert c["retrieval:skip_not_conversation"] == 1 and c["retrieval:skip_not_question"] == 1
    assert c["retrieval:skip_has_library"] == 1 and c["retrieval:fired"] == 0 and c["retrieval:calls"] == 3


def test_insert_retrieval_max_len(lib, tok):
    long = chat(tok, extra_turns=3)
    out = R.insert_retrieval(long, lib, tok, S.rng_for(0), DataCfg(distractor_frac=0.0), max_len=len(long))
    assert len(out) <= len(long) and messages_of(out, tok)[0]["role"] == "library"
    assert cf.assistant_turn_ends(out.ids)[-1] == len(out)
    short = chat(tok)
    stats = R.RagStats()
    same = R.insert_retrieval(short, lib, tok, S.rng_for(0), DataCfg(distractor_frac=0.0),
                              max_len=len(short), stats=stats)
    assert same is short and stats.counts["retrieval:too_long"] == 1


def library_turns(sample) -> list[int]:
    return [s for role, s, _ in S.turn_spans(sample.ids) if role == LIBRARY]


def test_no_passage_where_the_gate_stays_closed(lib, tok):
    """A question the gate does not answer gets no passage, not even a distractor: at
    inference a passage appears only when the gate opens."""
    stats = R.RagStats()
    closed = chat(tok, user="Who was Zorblax Quux?")
    assert query_type("Who was Zorblax Quux?") in lib.cfg.gate.allowed_types
    assert lib.retrieve("Who was Zorblax Quux?") is None
    for seed in range(30):
        out = R.insert_retrieval(closed, lib, tok, S.rng_for("closed", seed), DataCfg(distractor_frac=1.0),
                                 stats=stats)
        assert out is closed
    c = stats.counts
    assert c["retrieval:question_like"] == 30 and c["retrieval:gate_closed"] == 30 and c["retrieval:fired"] == 0


def test_distractors_are_a_share_of_the_passages_the_gate_lets_in(lib, tok):
    """Over many draws: a passage is inserted exactly when the gate opens on the served user
    message (reference: the library's own retrieve), distractor_frac of the inserted passages
    are distractors, and a distractor is neither the gated passage nor from its article."""
    questions = ["Who was Alan Turing?", "Who was Zorblax Quux?", "What is the capital of France?",
                 "Tell me about photosynthesis.", "Write a poem about the sea.", "Who wrote Hamlet?",
                 "Where is Rome?", "Why do glorbs flim?"]
    data = DataCfg(distractor_frac=0.35)
    stats = R.RagStats()
    n_draws = n_passages = n_distractors = 0
    for i in range(480):
        q = questions[i % len(questions)]
        s = chat(tok, user=q)
        gate = lib.retrieve(q) if query_type(q) in lib.cfg.gate.allowed_types else None
        out = R.insert_retrieval(s, lib, tok, S.rng_for("share", i), data, stats=stats)
        n_draws += 1
        assert (out is not s) == (gate is not None), q          # a passage exactly where the gate opens
        if gate is None:
            continue
        n_passages += 1
        text = messages_of(out, tok)[0]["content"]
        if out.tags[0] == "retrieval:gated":
            assert text == gate.text
            continue
        n_distractors += 1
        src = [p for p in range(lib.n_passages) if lib.passage(p).text == text]
        assert src and gate.pid not in src
        assert all(lib.tid[p] != lib.tid[gate.pid] for p in src if lib.tid[gate.pid] >= 0)
    p = data.distractor_frac
    assert n_passages > 150
    assert abs(n_distractors / n_passages - p) < 4 * (p * (1 - p) / n_passages) ** 0.5
    r = stats.as_dict()["retrieval"]
    assert r["fired"] == n_passages and r["distractor_share"] == pytest.approx(n_distractors / n_passages)
    assert r["gate_rate"] == pytest.approx(n_passages / stats.counts["retrieval:question_like"])


def multi_turn(tok, users, system=None, reply="Here is a reply. " * 4):
    conv = [{"role": "system", "content": system}] if system else []
    for u in users:
        conv += [{"role": "user", "content": u}, {"role": "assistant", "content": reply.strip()}]
    return conv, S.conversation_sample(conv, tok, source="smol_smoltalk", kind="chat")


def test_a_later_user_turn_is_served_as_the_responder_serves_it(lib, tok):
    """With retrieval_later_turn = 1 a conversation with several user turns gets its passage
    right before a later user turn, after the earlier exchanges: the tokens up to that turn's
    ASSISTANT are exactly the prompt bes.decode.respond.fit_prompt renders for the
    conversation up to that user message with the passage; the turns after it stay."""
    from bes.decode.respond import OurChat, fit_prompt

    conv, s = multi_turn(tok, ["Hello there.", "Who was Alan Turing?", "What is the capital of France?"],
                         system="Be brief.")
    mix = MixCfg(retrieval_later_turn=1.0)
    stats = R.RagStats()
    served = set()
    for seed in range(40):
        out = R.insert_retrieval(s, lib, tok, S.rng_for("later", seed), DataCfg(distractor_frac=0.0), mix,
                                 stats=stats)
        assert out is not s                   # both later questions open the gate
        msgs = messages_of(out, tok)
        k = [m["role"] for m in msgs].index("library")
        user = msgs[k + 1]["content"]
        assert user in ("Who was Alan Turing?", "What is the capital of France?")
        served.add(user)
        j = [m["content"] for m in conv].index(user)
        prompt, info = fit_prompt(OurChat(tok), conv[:j + 1], lib.retrieve(user).text, 10 ** 6)
        assert info["dropped_turns"] == 0
        assert out.ids[:len(prompt)].tolist() == prompt
        assert msgs[:k] == conv[:j] and msgs[k + 1:] == S.canonical_messages(conv[j:])
        assert "retrieval:later_turn" in out.tags and len(library_turns(out)) == 1
    assert served == {"Who was Alan Turing?", "What is the capital of France?"}
    assert stats.counts["retrieval:later_turn"] == 40
    first = R.insert_retrieval(s, lib, tok, S.rng_for(0), DataCfg(distractor_frac=0.0),
                               MixCfg(retrieval_later_turn=0.0))
    assert first is s                         # the first user turn is small talk: no passage
    rates = [R.insert_retrieval(multi_turn(tok, ["Who was Alan Turing?", "Who wrote Hamlet?"])[1], lib, tok,
                                S.rng_for("rate", i), DataCfg(distractor_frac=0.0)) for i in range(400)]
    later = np.mean(["retrieval:later_turn" in o.tags for o in rates])
    assert abs(later - MixCfg().retrieval_later_turn) < 4 * (0.25 / 400) ** 0.5


def test_fit_served_drops_the_oldest_exchanges_like_fit_prompt(tok):
    """Fitting a sample whose passage serves its last user turn: the oldest exchanges go first,
    the system turn stays, exactly the exchanges fit_prompt keeps for the same budget; the turns
    after the served exchange go before any exchange is dropped."""
    from bes.decode.respond import OurChat, fit_prompt

    passage = "Alan Turing\nAlan Turing was an English mathematician. He broke codes."
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(30):
        n = int(rng.integers(2, 6))
        k = int(rng.integers(1, n))                       # the served user turn (0-based), not the first
        users = [f"Question number {i}: " + "word " * int(rng.integers(3, 30)) for i in range(n)]
        conv, _ = multi_turn(tok, users, system="Be brief." if trial % 2 else None,
                             reply="An answer sentence. " * int(rng.integers(2, 12)))
        with_lib = list(conv)
        j = [m["content"] for m in conv].index(users[k])
        with_lib.insert(j, {"role": "library", "content": passage})
        s = S.conversation_sample(with_lib, tok)
        lib_start = library_turns(s)[0]
        reply_len = len(cf.render_turn("assistant", conv[j + 1]["content"], tok)[0]) - 1   # content + END
        for max_len in sorted(set(int(x) for x in rng.integers(20, len(s) + 20, 6))):
            got = R.fit_served(s, lib_start, max_len)
            try:
                prompt, info = fit_prompt(OurChat(tok), conv[:j + 1], passage, max_len - reply_len)
                fits = not (info["dropped_passage"] or info["dropped_system"] or info["cut_user"])
            except ValueError:                                # nothing fits that budget
                fits = False
            assert (got is not None) == fits
            if got is None:
                continue
            assert len(got) <= max_len and len(library_turns(got)) == 1
            assert got.ids[:len(prompt)].tolist() == prompt   # the same exchanges as at inference
            assert messages_of(got, tok)[-1]["role"] == "assistant"
            assert len(got) >= len(prompt) + reply_len        # the served reply is whole
            if info["dropped_turns"] == 0:                    # only later turns were cut
                assert got.ids.tolist() == s.ids[:len(got)].tolist()
            checked += 1
    assert checked > 50


def test_one_passage_per_conversation(lib, tok):
    """A conversation that already holds a library turn anywhere gets no other one."""
    conv = [{"role": "user", "content": "Who was Alan Turing?"}, {"role": "assistant", "content": "A mathematician."},
            {"role": "library", "content": "Paris\nParis is the capital of France."},
            {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]
    s = S.conversation_sample(conv, tok, kind="rag")
    stats = R.RagStats()
    for seed in range(20):
        assert R.insert_retrieval(s, lib, tok, S.rng_for(seed), DataCfg(distractor_frac=0.5),
                                  MixCfg(retrieval_later_turn=0.5), stats=stats) is s
    assert stats.counts["retrieval:skip_has_library"] == 20


# --------------------------------------------------------------- reading-task pre-filter
def test_reading_suitable_predicts_every_decline(lib, tok):
    """An entry outside reading_suitable declines for every draw; one inside builds a task;
    the slow path (passage per entry) is the reference for the mask."""
    mask = R.reading_suitable(lib)
    assert mask.shape == (lib.n_wiki,) and not mask.flags.writeable
    view = R.as_view(lib)
    for pid in range(lib.n_wiki):
        built = [R.reading_task(pid, lib, tok, S.rng_for("suit", pid, k)) is not None for k in range(6)]
        assert all(built) if mask[pid] else not any(built), pid
        p = view.passage(pid)
        lines = p.text.split("\n")[1:] if p.text.startswith(p.title + "\n") else None
        lead = lib.title_range(lib.tid[pid]).start == pid
        title = strip_disambiguator(p.title)
        ok = lines is not None and title and len(title.split()) <= R.MAX_TITLE_WORDS and \
            not title.lower().startswith("list of") and (lead or (lines[0].strip().lower() in R.SECTION_HEADINGS))
        if not ok:
            assert not mask[pid]
    assert 0 < mask.sum() < lib.n_wiki
    no_sections = R.reading_suitable(lib, MixCfg(section_tasks=False))
    assert (no_sections <= mask).all() and no_sections.sum() < mask.sum()


def test_a_prepared_view_deals_only_suitable_entries(lib, tok, tmp_path):
    """After prepare_tasks every suitable wiki entry is drawn once per epoch and no other;
    question-answer draws are unchanged; the view pickles without its passage cache."""
    import pickle

    mask = R.reading_suitable(lib)
    view = R.as_view(lib).prepare_tasks(MixCfg(), cache_dir=tmp_path)
    suitable = sorted(int(p) for p in np.flatnonzero(mask))
    n = len(suitable)
    draws = [R.pick_entry(view, "library_read", c, seed=3) for c in range(2 * n)]
    assert sorted(draws[:n]) == suitable and sorted(draws[n:]) == suitable and draws[:n] != draws[n:]
    assert all(R.reading_task(p, view, tok, S.rng_for("dealt", c)) is not None for c, p in enumerate(draws))
    plain = R.as_view(FakeLibrary())
    assert [R.pick_entry(view, "library_qa", c, 1) for c in range(9)] == \
        [R.pick_entry(plain, "library_qa", c, 1) for c in range(9)]
    view.passage(0)
    again = pickle.loads(pickle.dumps(view))
    assert again._cache == {} and list(again.task_entries("library_read")) == suitable


@pytest.mark.skipif(not (LIB_DIR / "library.json").exists(), reason="no built library in data_cache")
def test_reading_suitable_on_the_real_library(tok, tmp_path):
    """lib_v2: the one-pass mask (Packed.iter_bodies) equals the per-passage reference on a
    sample of entries, entries outside it decline, it is cached under cache_dir, and draws
    from a prepared view almost never decline (about half did before)."""
    real = Library.load(str(LIB_DIR))
    R._SUITABLE_MEMO.clear()
    mask = R.reading_suitable(real, cache_dir=tmp_path)
    assert list(tmp_path.glob("reading_suitable-*.npy")) and 0.4 < mask.mean() < 0.7
    R._SUITABLE_MEMO.clear()
    assert np.array_equal(R.reading_suitable(real, cache_dir=tmp_path), mask)       # read back
    view = R.as_view(real)
    for pid in np.random.default_rng(1).choice(len(mask), 300, replace=False).tolist():
        p = view.passage(pid)
        got = R._reading_form(p.title, R._passage_lines(p), view.is_lead(pid), MixCfg())
        assert mask[pid] == (not isinstance(got, str) and R._answer_possible(got[2], MixCfg()))
        if not mask[pid]:
            assert R.reading_task(pid, view, tok, S.rng_for("real-suit", pid)) is None
    prepared = R.as_view(real).prepare_tasks(MixCfg(), cache_dir=tmp_path)
    stats = R.RagStats()
    built = sum(R.library_task("library_read", R.pick_entry(prepared, "library_read", c), prepared, tok,
                               S.rng_for("real-dealt", c), stats=stats) is not None for c in range(200))
    assert built >= 196


def test_not_in_the_passage_samples_pass_through(lib, tok):
    rec = json.loads((FIXTURES / "squad_v2.json").read_text())["records"][1]
    conv = sources.adapt("squad_v2", rec)
    assert conv[-1]["content"] == sources.SQUAD_UNANSWERABLE
    s = S.conversation_sample(conv, tok, source="squad_v2", kind="rag")
    stage = SimpleNamespace(library_frac_chat=1.0, chat_frac=0.5, library_frac_base=0.1)
    for seed in range(10):
        todo = M.draw_transforms("rag", stage, S.rng_for(seed))
        assert todo.retrieval and not todo.hindsight
        out = M.apply_transforms(s, todo, tok, S.rng_for(seed), lib=lib)
        assert out is s
    assert messages_of(s, tok) == S.canonical_messages(conv)


def test_apply_transforms_order(lib, tok):
    """Hindsight runs first, so the retrieval query is the relabelled user message."""
    conv = [{"role": "user", "content": "Tell me about Alan Turing."},
            {"role": "assistant", "content": "Alan Turing was a mathematician. He broke codes. He was English."}]
    s = S.conversation_sample(conv, tok, source="smol_smoltalk", kind="chat")
    both = M.TransformPlan(hindsight=True, retrieval=True)
    stats = R.RagStats()
    outs, checked = [], 0
    for k in range(30):
        before = len(lib.calls)
        out = M.apply_transforms(s, both, tok, S.rng_for("t", k), lib=lib, stats=stats,
                                 data=DataCfg(distractor_frac=0.0))
        outs.append(out)
        msgs = messages_of(out, tok)
        user = [m["content"] for m in msgs if m["role"] == "user"][0]
        query = [c[1] for c in lib.calls[before:] if c[0] == "retrieve"]
        if out.tags[-1] == "retrieval:gated":
            assert msgs[0]["role"] == "library" and query == [user]
        else:       # the final message is no longer question-like (e.g. "Use 11-20 words" reads as math)
            assert query == [] and query_type(user) not in lib.cfg.gate.allowed_types
        if "hindsight" in out.tags:
            checked += 1
            assert out.tags[0] == "hindsight"
            assert user != conv[0]["content"] and not C.compile(user).empty
    assert checked >= 20 and stats.counts["hindsight:applied"] == checked
    assert stats.counts["retrieval:gated"] >= 20
    again = M.apply_transforms(s, both, tok, S.rng_for("t", 3), lib=lib, data=DataCfg(distractor_frac=0.0))
    assert again.key() == outs[3].key()
    assert M.apply_transforms(s, M.TransformPlan(), tok, S.rng_for(0), lib=lib) is s
    assert M.apply_transforms(s, both, tok, S.rng_for(0), lib=None).tags in ((), ("hindsight",))


# ------------------------------------------------------------------------- determinism
def test_builders_are_deterministic(lib, tok):
    for pid in range(lib.n_passages):
        for task in ("library_read", "library_qa"):
            a = R.library_task(task, pid, lib, tok, S.rng_for(9, pid))
            b = R.library_task(task, pid, lib, tok, S.rng_for(9, pid))
            assert (a is None) == (b is None)
            if a is not None:
                assert a.key() == b.key() and a.tags == b.tags
    keys = {R.insert_retrieval(chat(tok), lib, tok, S.rng_for("r", k)).key() for k in range(12)}
    assert len(keys) > 1
    x = R.insert_retrieval(chat(tok), lib, tok, S.rng_for("r", 5))
    y = R.insert_retrieval(chat(tok), lib, tok, S.rng_for("r", 5))
    assert x.key() == y.key()
    with pytest.raises(KeyError):
        R.library_task("library_poems", 0, lib, tok, S.rng_for(0))


def test_pick_entry_visits_every_entry_once_per_epoch(lib):
    for task, kind in R.LIBRARY_TASKS.items():
        r = lib.kind_range(kind)
        n = len(r)
        draws = [R.pick_entry(lib, task, c, seed=3) for c in range(2 * n)]
        assert sorted(draws[:n]) == list(r) and sorted(draws[n:]) == list(r)
        assert draws == [R.pick_entry(lib, task, c, seed=3) for c in range(2 * n)]
    assert R.pick_entry(OldContract(toy_library()), "library_qa", 0) is None


# ---------------------------------------------------------------- the real library
@pytest.mark.skipif(not (LIB_DIR / "library.json").exists(), reason="no built library in data_cache")
def test_real_library_smoke(tok):
    real = Library.load(str(LIB_DIR))
    view = R.as_view(real)
    stats = R.RagStats()
    lengths = {}
    for c in range(120):
        pid = R.pick_entry(view, "library_read", c, seed=1)
        s = R.reading_task(pid, view, tok, S.rng_for("real", c), stats=stats)
        if s is None:
            continue
        msgs = messages_of(s, tok)
        p = view.passage(pid)
        assert C.compile(msgs[-2]["content"]).empty and msgs[-1]["content"] in p.text
        if s.tags[0] == "read:distractor":
            assert msgs[0]["content"].split("\n", 1)[0] != p.title      # another article
        if s.tags[0] == "read:open":
            assert msgs[0]["content"] == p.text
        lengths.setdefault(s.tags[0], []).append(len(s))
    d = stats.as_dict()
    assert sum(d["samples"].values()) >= 40
    assert all(10 < np.mean(v) < 600 for v in lengths.values())
    q = chat(tok, user="Who was Alan Turing?")
    out = R.insert_retrieval(q, view, tok, S.rng_for(0), DataCfg(distractor_frac=0.0))
    assert out.tags == ("retrieval:gated",) and messages_of(out, tok)[0]["content"].startswith("Alan Turing\n")


@pytest.mark.skipif(not ((LIB_DIR / "library.json").exists() and (SHARD_DIR / "manifest.jsonl").exists()),
                    reason="needs the built library and a tokenised chat shard")
def test_real_conversations_with_retrieval(tok):
    from bes.data.shards import ShardSet

    real = R.as_view(Library.load(str(LIB_DIR)))
    shards = ShardSet(SHARD_DIR)
    stats = R.RagStats()
    for j in np.linspace(0, len(shards) - 1, 60).astype(int).tolist():
        s = S.from_tokens(np.asarray(shards.segment(j)), np.asarray(shards.segment_roles(j)),
                          source="smoltalk_constraints", kind="chat")
        out = R.insert_retrieval(s, real, tok, S.rng_for("rc", j), max_len=2048, stats=stats)
        if out is not s:
            msgs = messages_of(out, tok)
            k = [m["role"] for m in msgs].index("user")
            assert msgs[k - 1]["role"] == "library" and len(out) <= 2048
            assert out.ids[0] == BOS and int(np.sum(out.ids == LIBRARY)) == 1
    assert stats.counts["retrieval:calls"] == 60
