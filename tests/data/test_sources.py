"""Registry entries and record adapters (``bes/data/sources.py``).

Adapters are run on real records recorded from the Hugging Face Hub (``fixtures/<source>.json``);
expected outputs are written out by hand or computed by small independent references below.
"""
from __future__ import annotations

import itertools
import json
import re
import zlib
from pathlib import Path

import pytest

from bes.data import sources as S

FIXTURES = Path(__file__).parent / "fixtures"
SHA = re.compile(r"^[0-9a-f]{40}$")


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def records(name: str) -> list[dict]:
    return fixture(name)["records"]


def adapted(name: str) -> list:
    return [S.adapt(name, r) for r in records(name)]


# ----------------------------------------------------------------- independent references
ALLOWED_NEXT = {  # state -> roles that may follow; grammar [system] ([library] user assistant)+
    "start": {"system", "library", "user"}, "system": {"library", "user"},
    "library": {"user"}, "user": {"assistant"}, "assistant": {"library", "user"},
}


def ref_valid_conversation(msgs) -> bool:
    state = "start"
    for m in msgs:
        if set(m) != {"role", "content"} or m["role"] not in ALLOWED_NEXT[state]:
            return False
        c = m["content"]
        if isinstance(c, str):
            if not c or c != c.strip():
                return False
        elif m["role"] == "assistant" and isinstance(c, list) and c:
            for p in c:
                if p.get("type") == "text":
                    assert isinstance(p["text"], str) and p["text"]
                elif p.get("type") == "calc":
                    assert p["expr"].strip() and p["result"].strip()
                else:
                    return False
        else:
            return False
        state = m["role"]
    return state == "assistant"


def ref_display(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p["text"] if p["type"] == "text" else f"{p['expr']} = {p['result']}" for p in content)


def library_passages(msgs) -> list[str]:
    libs = [m["content"] for m in msgs if m["role"] == "library"]
    assert len(libs) <= 1
    return libs[0].split(S.PASSAGE_SEP) if libs else []


# ---------------------------------------------------------------------------- registry
def test_registry_names_and_fields():
    names = [s.name for s in S.SOURCES]
    assert len(names) == len(set(names)) == len(S.REGISTRY)
    for s in S.SOURCES:
        assert re.fullmatch(r"[a-z0-9_]+", s.name), s.name
        assert re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", s.repo), s.repo
        assert s.kind in S.KINDS and s.fmt in S.FORMATS
        assert s.files and all(p.endswith(".parquet") for p in s.files)
        assert s.columns and all(isinstance(c, str) and c for c in s.columns)
        assert callable(s.adapter) and s.license and s.split
        assert s.weight >= 0 and s.n_files >= 1 and s.approx_rows > 0 and s.approx_bytes > 0
        assert SHA.match(s.verified_sha), s.name
        assert s.revision in (s.verified_sha, "refs/convert/parquet"), s.name
        if s.cap_bytes is not None:
            assert s.cap_bytes > 0


def test_formats_match_kinds():
    expected = {"base": "text", "library": "text", "chat": "messages", "rag": "messages",
                "math": "messages", "prefs": "pair"}
    for s in S.SOURCES:
        assert s.fmt == expected[s.kind], s.name


def test_weights_sum_to_one_within_each_kind():
    for kind in S.KINDS:
        total = sum(s.weight for s in S.by_kind(kind))
        if kind == "library":
            assert total == 0
        else:
            assert total == pytest.approx(1.0, abs=1e-12), kind


def test_nonpermissive_flags_and_fallbacks():
    for s in S.SOURCES:
        lic = s.license.lower()
        if "-nc" in lic.replace("cc by-nc", "cc-by-nc") and "excluded" not in lic:
            assert s.nonpermissive, s.name
        if s.nonpermissive:
            fb = S.get(s.fallback)
            assert fb.kind == s.kind and not fb.nonpermissive
    assert S.get("climbmix").nonpermissive and S.get("climbmix").fallback == "fineweb_edu"
    assert not any(s.nonpermissive for s in S.SOURCES if s.name != "climbmix")


def test_mixture_weights_reference():
    for kind in S.KINDS:
        for permissive_only in (False, True):
            ref: dict[str, float] = {}
            for s in S.SOURCES:
                if s.kind != kind:
                    continue
                name = s.fallback if (permissive_only and s.nonpermissive) else s.name
                if name is not None:
                    ref[name] = ref.get(name, 0.0) + s.weight
            ref = {k: v for k, v in ref.items() if v > 0}
            total = sum(ref.values())
            ref = {k: v / total for k, v in ref.items()} if total else {}
            got = S.mixture_weights(kind, permissive_only)
            assert got.keys() == ref.keys() and all(got[k] == pytest.approx(ref[k]) for k in ref)
    assert "climbmix" not in S.mixture_weights("base", permissive_only=True)
    assert S.mixture_weights("base", True)["fineweb_edu"] == pytest.approx(0.95)


def test_download_cap_follows_the_fallback_weight():
    climb, fine = S.get("climbmix"), S.get("fineweb_edu")
    assert S.download_cap("fineweb_edu") == fine.cap_bytes
    assert S.download_cap("fineweb_edu", permissive_only=True) == fine.cap_bytes + climb.cap_bytes
    assert S.download_cap("climbmix") == climb.cap_bytes
    assert S.download_cap("gsm8k", permissive_only=True) is None          # uncapped stays uncapped
    with pytest.raises(ValueError):
        S.download_cap("climbmix", permissive_only=True)


def test_get_unknown_source():
    with pytest.raises(KeyError):
        S.get("no_such_source")
    assert S.get(S.get("gsm8k")) is S.get("gsm8k")


def test_every_source_has_a_fixture_recorded_from_its_repo():
    for s in S.SOURCES:
        fx = fixture(s.name)
        assert fx["source"] == s.name and fx["repo"] == s.repo and fx["revision"] == s.verified_sha
        assert fx["records"]
        for r in fx["records"]:
            assert set(r) == set(s.columns), s.name


# ----------------------------------------------------------------------------- helpers
def test_stable_bucket_is_crc32():
    for key in ["", "a", "memorization_94_6343", "ü"]:
        for n in (1, 2, 7):
            assert S.stable_bucket(key, n) == zlib.crc32(key.encode("utf-8")) % n


def test_select_passages_matches_brute_force():
    for n in range(0, 7):
        for k in range(0, n + 1):
            for sup in itertools.combinations(range(n), k):
                for maxp, nd in [(3, 1), (2, 1), (3, 0), (1, 2)]:
                    sup_kept = sorted(sup)[:maxp]
                    room = max(0, min(nd, maxp - len(sup_kept)))
                    rest = [i for i in range(n) if i not in sup_kept][:room]
                    assert S.select_passages(n, sup, maxp, nd) == sorted(sup_kept + rest)


def test_overlap_rank_reference():
    docs = ["alpha beta gamma", "nothing here", "beta gamma delta epsilon", "gamma"]
    answer = "beta gamma delta zeta"
    words = set(re.findall(r"[a-z0-9]{4,}", answer))
    scores = [len(words & set(re.findall(r"[a-z0-9]{4,}", d))) for d in docs]
    ref = sorted(sorted(range(len(docs)), key=lambda i: (-scores[i], i))[:2])
    assert S.overlap_rank(docs, answer, 2) == ref == [0, 2]


def test_cap_chars_and_wiki_title():
    text = "word " * 3000
    out = S.cap_chars(text, 100)
    assert len(out) <= 102 and out.endswith(" …") and text.startswith(out[:-2])
    assert S.cap_chars("short", 100) == "short"
    assert S.wiki_title("https://en.wikipedia.org/wiki/New_South_Wales") == "New South Wales"
    assert S.wiki_title("https://en.wikipedia.org/wiki/Cantor%27s_diagonal_argument") == "Cantor's diagonal argument"
    assert S.wiki_title("https://en.wikipedia.org/wiki/A\nhttps://en.wikipedia.org/wiki/B") == "A"
    assert S.wiki_title(None) == ""


def test_clean_messages_rules():
    u = lambda c: {"role": "user", "content": c}            # noqa: E731
    a = lambda c: {"role": "assistant", "content": c}       # noqa: E731
    assert S.clean_messages([{"role": "human", "content": " hi "}, {"role": "gpt", "content": "yo\r\n"}]) == [u("hi"), a("yo")]
    assert S.clean_messages([u("q"), a("x"), u("dangling")]) == [u("q"), a("x")]
    assert S.clean_messages([{"role": "system", "content": "  "}, u("q"), a("x")]) == [u("q"), a("x")]
    assert S.clean_messages([u("q"), {"role": "tool", "content": "t"}, a("x")]) is None
    assert S.clean_messages([u("q"), a("")]) is None
    assert S.clean_messages([u("q"), u("q2"), a("x")]) is None
    assert S.clean_messages([u("q"), {"role": "system", "content": "s"}, a("x")]) is None
    assert S.clean_messages([{"role": "library", "content": "p"}, {"role": "library", "content": "p2"}, u("q"), a("x")]) is None
    assert S.clean_messages([a("x")]) is None and S.clean_messages(None) is None and S.clean_messages([]) is None
    parts = [{"type": "text", "text": "a "}, {"type": "text", "text": ""}, {"type": "calc", "expr": "1+1", "result": "2"}]
    assert S.clean_messages([u("q"), a(parts)]) == [u("q"), a([parts[0], parts[2]])]
    assert S.clean_messages([u("q"), a([{"type": "calc", "expr": "1+1", "result": " "}])]) is None
    assert S.clean_messages([u(["not", "a string"]), a("x")]) is None


@pytest.mark.parametrize("text,expected", [
    ("Natalia sold 48/2 = <<48/2=24>>24 clips", "Natalia sold <<48/2=24>> clips"),
    ("Maila read 12 x 2 = <<12*2=24>>24 pages", "Maila read <<12*2=24>> pages"),
    ("He makes $<<9*500=4500>>4500\nSo", "He makes $<<9*500=4500>>\nSo"),
    ("So he makes 4500*5=$<<4500*5=22500>>22,500\nHe", "So he makes $<<4500*5=22500>>\nHe"),
    ("(5 + 3 = <<5+3=8>>8)", "(<<5+3=8>>)"),
    ("20 − 13 = <<20-13=7>>7.", "<<20-13=7>>."),
    ("he took 2*6 = <<6*2=12>>12 green", "he took 2*6 = <<6*2=12>> green"),    # different lhs kept
    ("6 jaguars = <<5*6=30>>30 snakes", "6 jaguars = <<5*6=30>> snakes"),
    ("odd <<1+1=2=2>>2 end", "odd 2 end"),                                   # malformed: dropped
    ("no copy <<1+1=2>>two", "no copy two"),                                 # no visible result: dropped
    ("half <<10/2=5>>5.00 left", "half 5.00 left"),                         # "5" is not the whole number
    ("pays 2*500 = $<<2*500=1000>>1,000.", "pays $<<2*500=1000>>."),        # thousands comma copy
])
def test_gsm8k_calc_text_cases(text, expected):
    assert S.gsm8k_calc_text(text) == expected


def test_calc_parts():
    assert S.calc_parts("plain") == "plain"
    assert S.calc_parts("a <<8*18=144.00>> b <<1/4=0.250>>") == [
        {"type": "text", "text": "a "}, {"type": "calc", "expr": "8*18", "result": "144"},
        {"type": "text", "text": " b "}, {"type": "calc", "expr": "1/4", "result": "0.25"}]


# ---------------------------------------------------------------- adapters on fixtures
@pytest.mark.parametrize("name", [s.name for s in S.SOURCES])
def test_adapter_outputs_are_well_formed(name):
    src = S.get(name)
    outs = adapted(name)
    assert any(o is not None for o in outs), name
    for o in outs:
        if o is None:
            continue
        if src.fmt == "text":
            assert isinstance(o, str) and o == o.strip() and o
        elif src.fmt == "messages":
            assert ref_valid_conversation(o), o
            for p in library_passages(o):
                assert p and len(p) <= S.MAX_PASSAGE_CHARS + 2 + 200   # title + body cap
            assert len(library_passages(o)) <= S.MAX_PASSAGES
        else:
            assert isinstance(o, S.Pair)
            assert ref_valid_conversation(o.chosen) and ref_valid_conversation(o.rejected)


@pytest.mark.parametrize("name", [s.name for s in S.SOURCES if s.fmt != "text"])
def test_chat_format_accepts_adapter_outputs(name):
    chat_format = pytest.importorskip("bes.data.chat_format")
    from bes.data.tokenize_worker import ByteTokenizer

    tok = ByteTokenizer()
    for o in adapted(name):
        if o is None:
            continue
        for conv in ([o.chosen, o.rejected] if isinstance(o, S.Pair) else [o]):
            ids, roles = chat_format.render_conversation(conv, tok)
            assert len(ids) == len(roles) > 2


def test_text_adapters():
    for name in ["climbmix", "fineweb_edu"]:
        assert adapted(name) == [r["text"].strip() for r in records(name)]
    for name in ["simplewiki", "library_simplewiki", "library_enwiki"]:
        for r, o in zip(records(name), adapted(name)):
            assert o == f"{r['title'].strip()}\n\n{r['text'].strip()}"


def test_plain_chat_adapters_keep_messages():
    for name in ["smol_smoltalk", "smoltalk_magpie_ultra", "smoltalk_constraints", "smoltalk_rewrite",
                 "smoltalk_summarize", "smoltalk_systemchats", "ultrachat"]:
        for r, o in zip(records(name), adapted(name)):
            assert o == [{"role": m["role"], "content": m["content"].strip()} for m in r["messages"]]


def test_everyday_drops_trailing_user_turn():
    raw = [r["messages"] for r in records("smoltalk_everyday")]
    outs = adapted("smoltalk_everyday")
    assert raw[1][-1]["role"] == "user"
    assert len(outs[1]) == len(raw[1]) - 1 and outs[1][-1]["role"] == "assistant"
    assert len(outs[0]) == len(raw[0])


def test_tulu_excludes_subsets():
    for r, o in zip(records("tulu3_sft"), adapted("tulu3_sft")):
        if r["source"] in S.TULU_EXCLUDED:
            assert o is None
        else:
            assert o == [{"role": m["role"], "content": m["content"].strip()} for m in r["messages"]]
    assert {r["source"] for r in records("tulu3_sft")} >= set(S.TULU_EXCLUDED)


def test_synth_closed_and_open_book_partition():
    recs = records("synth")
    closed, open_ = adapted("synth"), adapted("synth_rag")
    for r, c, o in zip(recs, closed, open_):
        usable = r["language"] == "en" and "self-knowledge" not in (r["query_seed_url"] or "").lower()
        half = zlib.crc32(r["synth_id"].encode()) % 2
        mem = r["exercise"] == "memorization"
        exp_closed = usable and r["exercise"] in S.SYNTH_CHAT_EXERCISES and not (mem and half == 1)
        exp_open = usable and (r["exercise"] == "rag" or (mem and half == 1 and bool(r["query_seed_text"])))
        assert (c is not None) == exp_closed, r["synth_id"]
        assert (o is not None) == exp_open, r["synth_id"]
        assert not (c is not None and o is not None)
        if c is not None:
            assert c == [{"role": "user", "content": r["query"].strip()},
                         {"role": "assistant", "content": r["synthetic_answer"].strip()}]
    # both memorization halves, a rag row, a self-knowledge row and excluded exercises are present
    assert {"editing", "math mcq", "rag"} <= {r["exercise"] for r in recs}
    selfk = [r for r in recs if "self-knowledge" in (r["query_seed_url"] or "").lower()]
    assert selfk and all(S.synth_closed_book(r) is None and S.synth_open_book(r) is None for r in selfk)
    assert sum(o is not None for o in open_) == 2 and sum(c is not None for c in closed) == 4


def test_synth_open_book_rag_row():
    r = next(r for r in records("synth_rag") if r["exercise"] == "rag")
    o = S.synth_open_book(r)
    cited = sorted({int(k) for k in re.findall(r'<ref name="source_(\d+)"', r["synthetic_answer"])})
    sources = dict((int(k), v) for k, v in re.findall(r"<source_(\d+)>\s*(.*?)\s*</source_\1>", r["constraints"], re.S))
    passages = library_passages(o)
    expected = sorted(cited)[:S.MAX_PASSAGES]
    if len(expected) < S.MAX_PASSAGES:
        expected += [k for k in sorted(sources) if k not in expected][:1]
    assert len(passages) == len(expected)
    for p, k in zip(passages, sorted(expected)):
        assert sources[k][:60] in p
    assert passages[0].startswith("New South Wales\n\n") and 1 in expected
    answer = o[-1]["content"]
    assert "<ref" not in answer and "</ref>" not in answer and " ." not in answer
    mem = next(r for r in records("synth_rag") if r["exercise"] == "memorization" and S.synth_open_book(r))
    lib = library_passages(S.synth_open_book(mem))
    assert lib == [S.passage(S.wiki_title(mem["query_seed_url"]), mem["query_seed_text"])]


def test_glaive_rag():
    for r, o in zip(records("glaive_rag"), adapted("glaive_rag")):
        docs = re.findall(r"Document:\s*(\d+)\s*\n\s*Title:\s*(.*?)\s*\n", r["documents"])
        cited = [int(x) for x in re.findall(r"\d+", r["answer"].split("\n", 1)[0])]
        passages = library_passages(o)
        titles = [p.split("\n", 1)[0] for p in passages]
        doc_titles = [t for _, t in docs]
        assert all(doc_titles[c] in titles for c in cited)
        n_extra = len(passages) - len(cited)
        assert n_extra == (1 if len(cited) < S.MAX_PASSAGES else 0)
        answer = o[-1]["content"]
        assert "<co" not in answer and "</co>" not in answer
        assert not answer.startswith(("Cited Documents", "Answer:"))
        assert r["question"].strip() == o[-2]["content"]
    assert [len(library_passages(o)) for o in adapted("glaive_rag")] == [3, 1]


def test_rag_instruct_keeps_most_overlapping_documents():
    r = records("rag_instruct")[0]
    o = adapted("rag_instruct")[0]
    words = set(re.findall(r"[a-z0-9]{4,}", r["answer"].lower()))
    scores = [len(words & set(re.findall(r"[a-z0-9]{4,}", d.lower()))) for d in r["documents"]]
    best = sorted(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:S.MAX_PASSAGES])
    assert library_passages(o) == [r["documents"][i].strip() for i in best]


def test_neural_bridge_squad_dolly_hotpot():
    for r, o in zip(records("neural_bridge_rag"), adapted("neural_bridge_rag")):
        assert o == [{"role": "library", "content": r["context"].strip()},
                     {"role": "user", "content": r["question"].strip()},
                     {"role": "assistant", "content": r["answer"].strip()}]
    sq = records("squad_v2")
    outs = adapted("squad_v2")
    assert outs[0][-1]["content"] == sq[0]["answers"]["text"][0]
    assert sq[1]["answers"]["text"] == [] and outs[1][-1]["content"] == S.SQUAD_UNANSWERABLE
    for r, o in zip(sq, outs):
        assert o[0]["content"].startswith(r["title"].replace("_", " ") + "\n\n")
    dl = adapted("dolly")
    assert [m["role"] for m in dl[0]] == ["library", "user", "assistant"]
    assert [m["role"] for m in dl[1]] == ["user", "assistant"]
    r = records("hotpot_qa")[0]
    o = adapted("hotpot_qa")[0]
    support = set(r["supporting_facts"]["title"])
    titles = [p.split("\n", 1)[0] for p in library_passages(o)]
    native = [t for t in r["context"]["title"] if t in titles]
    assert titles == native                       # native order kept
    assert support <= set(titles) and len(titles) == len(support) + 1
    first_distractor = next(t for t in r["context"]["title"] if t not in support)
    assert first_distractor in titles
    assert o[-1]["content"] == r["answer"]


def test_gsm8k_adapter():
    o = adapted("gsm8k")[0]
    assert o[1]["content"] == [
        {"type": "text", "text": "Natalia sold "}, {"type": "calc", "expr": "48/2", "result": "24"},
        {"type": "text", "text": " clips in May.\nNatalia sold "},
        {"type": "calc", "expr": "48+24", "result": "72"},
        {"type": "text", "text": " clips altogether in April and May.\nThe answer is 72."}]
    norm = lambda t: re.sub(r"[\s$,]", "", t).replace("x", "*")     # noqa: E731
    for r, o in zip(records("gsm8k"), adapted("gsm8k")):
        final = re.search(r"####\s*(.+?)\s*$", r["answer"]).group(1)
        visible = re.sub(r"<<[^<>]*>>", "", r["answer"]).split("####")[0] + f"The answer is {final}."
        shown = ref_display(o[1]["content"])
        assert shown.endswith(f"The answer is {final}.")
        n_ann = len(re.findall(r"<<[^<>]*>>", r["answer"]))
        assert sum(isinstance(p, dict) and p["type"] == "calc" for p in o[1]["content"]) == n_ann
        if norm(shown) != norm(visible):          # only extra "expr =" restatements are allowed
            assert len(norm(shown)) > len(norm(visible))


def test_math_and_prefs_adapters():
    for r, o in zip(records("tinygsm"), adapted("tinygsm")):
        assert o[1]["content"] == f"```python\n{r['code'].strip()}\n```"
    for r, o in zip(records("orca_math"), adapted("orca_math")):
        assert o == [{"role": "user", "content": r["question"].strip()},
                     {"role": "assistant", "content": r["answer"].strip()}]
    uf = records("ultrafeedback")
    pair, tie = adapted("ultrafeedback")
    assert uf[0]["score_chosen"] > uf[0]["score_rejected"] and uf[1]["score_chosen"] == uf[1]["score_rejected"]
    assert tie is None
    assert pair.chosen == [{"role": m["role"], "content": m["content"].strip()} for m in uf[0]["chosen"]]
    assert pair.rejected == [{"role": m["role"], "content": m["content"].strip()} for m in uf[0]["rejected"]]
    assert pair.chosen[0] == pair.rejected[0]


def test_adapt_checks_output_type():
    with pytest.raises(TypeError):
        bad = S.Source(**{**S.get("gsm8k").__dict__, "name": "bad", "adapter": lambda r: "text"})
        S.adapt(bad, {})
