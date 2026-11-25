"""scripts/subset_answers.py: the part of an answers library that fits a number budget."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bes.library.answers import AnswerCfg, AnswerLibrary

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))        # the worker processes import the module by name
import subset_answers as SA  # noqa: E402

TOPICS = ["ice", "oil", "wood", "cork", "steel", "glass", "sand", "salt"]
PAIRS = ([(f"Why does {t} float or sink in water?", f"It depends on the density of {t} compared with water. " * 3, "a")
          for t in TOPICS]
         + [(f"What is the density of {t} in grams per cubic centimetre?", f"The density of {t} varies with its kind. " * 3, "b")
            for t in TOPICS]
         + [("Recite the opening line of an imaginary poem about zeppelins.", "Above the clouds the silver whales drift on. " * 4, "c"),
            ("Describe the habits of the nocturnal marsupial quokka briefly.", "Quokkas forage at night and rest by day. " * 4, "c")])


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("answers") / "full"
    lib = AnswerLibrary.build(PAIRS, AnswerCfg(n_bits=6, group_bits=2), {"build": {"sources": {
        "a": {"repo": "x/a", "commit": "1" * 40, "licence": "mit", "stored": 8},
        "b": {"repo": "x/b", "commit": "2" * 40, "licence": "mit", "stored": 8},
        "c": {"repo": "x/c", "commit": "3" * 40, "licence": "mit", "stored": 2}}}})
    lib.save(str(d))
    return d


def test_credit_goes_to_entries_that_neighbours_would_use(src):
    cr = SA.credit(str(src), workers=1)
    lib = AnswerLibrary.load(str(src))
    qs = [lib.question(e) for e in range(lib.n_entries)]
    lone = [i for i, q in enumerate(qs) if "zeppelins" in q or "quokka" in q]
    shared = [i for i in range(len(qs)) if i not in lone]
    assert len(lone) == 2 and all(cr[i] == 0 for i in lone)        # nobody asks anything near them
    # the questions of a family credit their best three neighbours: most of a family has credit
    assert sum(cr[i] > 0 for i in shared) >= 6 and cr.sum() > 0
    np.testing.assert_allclose(cr, SA.credit(str(src), workers=2, chunk=5), rtol=1e-12)   # the same in processes


def test_the_subset_fits_the_budget_and_keeps_the_most_credited_entries(src, tmp_path):
    full = AnswerLibrary.load(str(src))
    total = full.accounting()["total_numbers"]
    budget = int(total * 0.7)
    logs = []
    lib, rec = SA.subset(str(src), budget, read_cap=1234, workers=1, log=logs.append)
    assert lib.accounting()["total_numbers"] == rec["numbers"] <= budget
    assert 0 < rec["kept"] == lib.n_entries < full.n_entries and lib.acfg.read_cap == 1234
    kept = {lib.question(e) for e in range(lib.n_entries)}
    assert not any("zeppelins" in q or "quokka" in q for q in kept)       # the entries without credit went first
    order = [full.question(e) for e in range(full.n_entries)]
    assert [q for q in order if q in kept] == [lib.question(e) for e in range(lib.n_entries)]   # source order
    for e in range(lib.n_entries):                                         # the texts are the source's
        q, a = lib.entry(e)
        assert (q, a) in {(p[0], p[1]) for p in PAIRS}
    b = lib.meta["build"]
    assert b["subset"]["parent_entries"] == full.n_entries and b["subset"]["read_cap"] == 1234
    assert sum(s["stored"] for s in b["sources"].values()) == lib.n_entries
    assert b["sources"]["c"]["stored"] == 0 and b["sources"]["c"]["stored_parent"] == 2
    # a budget the whole library fits returns all of it
    lib2, rec2 = SA.subset(str(src), total, workers=1, log=logs.append)
    assert rec2["kept"] == full.n_entries and lib2.acfg.read_cap == full.acfg.read_cap


def test_the_command_writes_a_library_that_loads(src, tmp_path):
    full = AnswerLibrary.load(str(src))
    out = tmp_path / "sub"
    rc = SA.main(["--src", str(src), "--out", str(out), "--budget-numbers",
                  str(int(full.accounting()["total_numbers"] * 0.8)), "--read-cap", "999", "--workers", "1",
                  "--credit", str(tmp_path / "credit.npy")])
    assert rc == 0 and (tmp_path / "credit.npy").is_file()
    from bes.library.retrieve import Library

    lib = Library.load(str(out))
    assert lib.n_entries < full.n_entries and lib.acfg.read_cap == 999
    assert "subset of a larger build" in (out / "ATTRIBUTION.md").read_text()
    meta = json.loads((out / "answers.json").read_text())
    assert meta["accounting"]["total_numbers"] == lib.accounting()["total_numbers"]
    assert lib.retrieve("Why does cork float in water?") is not None or lib.candidates("Why does cork float in water?")


def test_the_same_inputs_give_the_same_library_anywhere(src, tmp_path):
    """Nothing of the machine or the moment goes into a subset: two builds from two copies of the
    source (other paths, another time) are the same files, so their digests agree."""
    import shutil

    from bes import export as E

    other = tmp_path / "elsewhere" / Path(src).name
    other.parent.mkdir()
    shutil.copytree(src, other)
    cr = SA.credit(str(src), workers=1)
    budget = int(AnswerLibrary.load(str(src)).accounting()["total_numbers"] * 0.8)
    digests = []
    for k, s_ in enumerate((src, other)):
        lib, rec = SA.subset(str(s_), budget, cr=cr, workers=1, log=lambda m: None)
        out = tmp_path / f"sub{k}"
        lib.save(str(out))
        digests.append(E.library_digest(out)["digest"])
        assert "built" not in rec and rec["parent"] == Path(src).name
    assert digests[0] == digests[1]
