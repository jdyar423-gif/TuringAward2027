"""Terminal tool for a human rater: ``python -m bes.eval.proxy rate --pairs NAME --rater ID``.

It reads only the blinded pair file (``pairs/<name>/pairs.jsonl``), never the key, so it cannot
show which model wrote which reply. Each pair shows the user turns with reply A and reply B;
the rater answers

    a  A is better        b  B is better        t  tie        x  both bad
    s  skip for now       u  undo the last rating            q  quit         ?  help

Every rating is appended (with fsync) to ``ratings/<name>/human-<rater>.jsonl`` at once, so
quitting or a crash loses nothing; a later session continues with the unrated pairs. Undo
appends a ``retracted`` record. Each rater gets a different, fixed order of the pairs. Every
record carries the pair set's published commitment and the rated pair's digest
(``bes.eval.proxy.make_ab``); a session refuses to start when stored records do not belong to
the published pair set.
"""
from __future__ import annotations

import random
import shutil
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bes.eval.proxy import make_ab
from bes.eval.proxy.common import append_jsonl, stable_hash
from bes.eval.proxy.judge import load_verdicts, ratings_path

KEYS = {"a": "A", "b": "B", "t": "tie", "x": "both_bad"}
HELP = ("a = A is better, b = B is better, t = tie, x = both bad, s = skip, u = undo last, "
        "q = quit, ? = help")
GUIDE = ("Judge as the user who asked. Correctness first, then whether every instruction was "
         "followed, then how useful and clear the reply is. Length is not a merit: prefer the "
         "shorter reply when both are equally good. In two-turn conversations judge both replies.")


def _block(label: str, text: str, width: int) -> str:
    bar = "-" * max(10, width)
    body = text if text.strip() else "(empty reply)"
    lines = []
    for line in body.split("\n"):
        lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False,
                                   drop_whitespace=False) or [""])
    return f"{bar}\n{label}\n{bar}\n" + "\n".join(lines)


def show_pair(pair: Mapping[str, Any], index: int, total: int, width: int) -> str:
    """The screen for one pair (no model identity is available here)."""
    out = [f"=== pair {index} of {total}  ({pair['category']}"
           f"{', two turns' if pair.get('multi_turn') else ''}) ==="]
    for t, turn in enumerate(pair["turns"], 1):
        if len(pair["turns"]) > 1:
            out.append(f"\n##### turn {t} #####")
        out.append(_block("USER", turn["user"], width))
        out.append(_block("REPLY A", turn["A"], width))
        out.append(_block("REPLY B", turn["B"], width))
    return "\n".join(out)


def rate_session(pairs: Sequence[Mapping[str, Any]], pairset: str, rater: str, root: Path,
                 input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
                 width: int | None = None) -> dict[str, int]:
    """Interactive rating loop; returns counts of this session's actions."""
    width = width or max(40, min(110, shutil.get_terminal_size((100, 30)).columns - 2))
    source = f"human-{rater}"
    path = ratings_path(root, pairset, source)
    pub = make_ab.published(root, pairset)
    done = load_verdicts(root, pairset, source, pub=pub)
    for p in pairs:
        if pub["digests"].get(p["pair_id"]) != make_ab.pair_digest(p):
            raise ValueError(f"pair {p['pair_id']} is not a published pair of {pairset!r}")

    def stamp(pair_id: str) -> dict[str, str]:
        return {"commitment": pub["commitment"], "pair_digest": pub["digests"][pair_id]}

    todo = [p for p in pairs if p["pair_id"] not in done]
    random.Random(stable_hash(["rater-order", rater, pairset])).shuffle(todo)
    stats = {"rated": 0, "skipped": 0, "undone": 0}
    history: list[str] = []
    total = len(pairs)
    queue = list(todo)
    print_fn(f"{len(done)} of {total} pairs already rated by {rater}.\n{GUIDE}\n{HELP}")
    while queue:
        pair = queue[0]
        print_fn(show_pair(pair, total - len(queue) + 1, total, width))
        t0 = time.time()
        while True:
            answer = input_fn("your verdict [a/b/t/x/s/u/q/?]: ").strip().lower()
            if answer in KEYS or answer in ("s", "u", "q", "?"):
                break
            print_fn(HELP)
        if answer == "?":
            print_fn(HELP)
            continue
        if answer == "q":
            break
        if answer == "s":
            queue.append(queue.pop(0))
            stats["skipped"] += 1
            if stats["skipped"] > 3 * total:
                break
            continue
        if answer == "u":
            if history:
                last = history.pop()
                append_jsonl(path, [{"pair_id": last, "verdict": "retracted", "rater": rater,
                                     "time": round(time.time(), 1), **stamp(last)}])
                queue.insert(0, next(p for p in pairs if p["pair_id"] == last))
                stats["undone"] += 1
                print_fn("last rating undone")
            else:
                print_fn("nothing to undo in this session")
            continue
        append_jsonl(path, [{"pair_id": pair["pair_id"], "verdict": KEYS[answer], "rater": rater,
                             "seconds": round(time.time() - t0, 1), "time": round(time.time(), 1),
                             **stamp(pair["pair_id"])}])
        history.append(pair["pair_id"])
        queue.pop(0)
        stats["rated"] += 1
    print_fn(f"session done: {stats['rated']} rated, {stats['skipped']} skips, {stats['undone']} undone")
    return stats
