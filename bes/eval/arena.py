"""Blind arena: a person compares our model with the opponents, one prompt at a time.

    python scripts/arena.py --export runs/final/final                     # ours, falcon, mobilellm
    python scripts/arena.py --export runs/final/final --prompts my.txt --reveal
    python scripts/arena.py --export DIR --prompts proxy:test --opponent falcon=recommended

For every prompt (typed, or read from a file) every contestant answers; the answers are shown
only when **all** are ready (nothing is printed per contestant, so speed gives nothing away), in a
fresh random order under neutral labels ``A``, ``B``, ``C``, ... The person votes::

    a / b / c   that reply is the best     t  tie (all equally good)    x  all bad
    s  skip this prompt (asked again in a later session)
    u  undo the last vote (that prompt is shown again at once)           q  quit    ?  help

With ``reveal`` the labels are unmasked after the vote (and the vote can no longer be undone).

Contestants run exactly as the proxy harness runs them (``docs/EVAL_PROTOCOL.md`` section 2):
opponents are the registry GGUF files with the settings of ``bes.eval.proxy.opponents.presets``
(built by ``bes.eval.proxy.__main__.build_spec``), ours is an export behind
``bes.decode.respond.Responder`` with the run's chosen decoding settings; every conversation goes
through ``bes.eval.proxy.runner.converse`` (system prompt of the arm, the model's own earlier
visible replies, seeds ``derive_seed(seed, prompt id, turn, k)`` shared by every contestant).
Only the visible answer is shown: hidden reasoning is removed by the arm's settings
(``strip_think``) and once more at display time, so a ``<think>`` block never reaches the screen.

Files (``<root>/<name>/``; the root defaults to ``data_cache/arena``)::

    arena.json          contestants (names, cache keys, settings)  - identities
    key.jsonl           per prompt: the order of the contestants behind A, B, C - identities
    votes.jsonl         per vote: prompt id, label voted, labels shown - no identities
    tallies.json / .md  per contestant and per pair, with 95% Wilson intervals - identities,
                        rewritten after every vote
    cache/<key>.jsonl   every generation (resume without generating again)
    errors.log          generation failures (shown as "(no reply)" before the vote)

Everything is appended with fsync after each vote, so quitting or a crash loses nothing and a later
session continues: prompts of a file that already have a vote are not asked again, typed prompts
continue the numbering. Do not open the files marked "identities" before you have finished voting.

Tallies: per contestant, the share of votes that chose it as best (Wilson interval over the votes
where it was shown) plus its ties and all-bad votes; per pair ``(i, j)`` the outcomes from ``i``'s
side (``bes.eval.proxy.score.tally``): ``i`` chosen = win, ``j`` chosen = loss, tie = tie, all bad
= both bad, and a vote for a third contestant says nothing about the pair and is left out.
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import random
import shutil
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from bes.eval.proxy.common import (GenSettings, Reply, append_jsonl, read_jsonl, safe_name,
                                   stable_hash, strip_think, write_json_atomic, write_text_atomic)
from bes.eval.proxy.prompts import Prompt

FORMAT = 1
LABELS = "ABCDEF"
MAX_CONTESTANTS = len(LABELS)
TIE, ALL_BAD = "tie", "all_bad"
HELP = ("a/b/c... = that reply is the best, t = tie (all equally good), x = all bad, "
        "s = skip, u = undo the last vote, q = quit, ? = help")
GUIDE = ("Judge as the person who asked. Correctness first, then whether every instruction was "
         "followed, then how useful and clear the reply is. Length is not a merit.")
EMPTY = "(empty reply)"
FAILED = "(no reply)"


class ArenaError(RuntimeError):
    """The arena cannot start (contestants differ from an existing arena, bad input file)."""


# ================================================================== contestants
@dataclass
class Contestant:
    """One model in the arena: a display name (revealed only after a vote), a cache key, the
    arm's generation settings and a factory of its harness backend
    (``generate(messages, settings, seed) -> Reply``), built on first use."""

    name: str
    key: str
    settings: GenSettings
    factory: Callable[[], Any]
    _backend: Any = field(default=None, repr=False)

    def backend(self) -> Any:
        if self._backend is None:
            self._backend = self.factory()
        return self._backend

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "key": self.key, "settings": self.settings.as_dict(),
                "settings_hash": self.settings.hash()}


def ours_contestant(export_dir: str | Path, overrides: Sequence[str] = (), *, weights: str = "16",
                    library: Any = "auto", verify: bool = True, name: str = "ours") -> Contestant:
    """Our exported model with the run's chosen decoding settings plus harness ``overrides``
    (``"extra.particles=1"``, ``"extra.use_library=false"``, ``"temperature=0.7"``)."""
    from bes.eval.proxy.__main__ import ours_presets
    from bes.eval.proxy.run_ours import CallableBackend
    from bes.export import load_export

    ex = load_export(export_dir, weights=weights, library=library, verify=verify)
    settings = ours_presets()["default"].with_overrides(ex.default_overrides() + list(overrides))
    if ex.library is None:
        settings = settings.with_overrides(["extra.use_library=false"])
    gen = ex.generator()
    return Contestant(name, gen.key, settings, lambda: CallableBackend(gen))


def opponent_contestant(spec: str, *, n_threads: int = 1, n_ctx: int | None = None, seed: int = 0,
                        download: bool = True, overrides: Sequence[str] = ()) -> Contestant:
    """``NAME[=PRESET]`` of the harness registry (``falcon=recommended``,
    ``mobilellm=neutral``), ``gguf:PATH`` or ``standin-<style>``, with the settings the proxy
    harness gives that arm (``bes.eval.proxy.__main__.build_spec``)."""
    from bes.eval.proxy.__main__ import build_spec
    from bes.eval.proxy.common import make_backend

    name, _, preset = spec.partition("=")
    rs = build_spec(name, preset or None, list(overrides), None, seed, None, None, n_threads, n_ctx,
                    download=download)
    display = f"{name} ({preset or rs.label})" if not name.startswith(("gguf:", "standin-")) else name
    backend_spec = dict(rs.backend)
    return Contestant(display, rs.key, rs.settings, lambda: make_backend(backend_spec))


def generator_contestant(name: str, generate: Callable[..., Any], settings: GenSettings | None = None,
                         key: str | None = None) -> Contestant:
    """A contestant from any ``generate(messages, settings_dict, seed) -> str | dict`` callable
    (stand-ins in tests and rehearsals)."""
    from bes.eval.proxy.run_ours import CallableBackend

    k = key or getattr(generate, "key", None) or f"gen-{name}"
    return Contestant(name, k, settings or GenSettings(), lambda: CallableBackend(generate, key=k))


# ======================================================================= prompts
def typed_prompt(index: int, text: str) -> Prompt:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return Prompt(id=f"typed-{index:04d}-{digest}", category="typed", turns=(text,), split="arena")


def file_prompts(source: str | Path) -> list[Prompt]:
    """Prompts of ``proxy:dev`` / ``proxy:test`` / ``proxy:all`` (the harness prompt set), a
    ``.jsonl`` file (``{"turns": [...]}``, ``{"prompt": "..."}``, optional ``"id"``) or a text
    file (one prompt per line; blank lines and lines starting with ``#`` are skipped)."""
    s = str(source)
    if s.startswith("proxy:"):
        from bes.eval.proxy import prompts as prompt_set

        return prompt_set.select(prompt_set.load(), s.split(":", 1)[1] or "dev")
    path = Path(s)
    if not path.exists():
        raise ArenaError(f"prompt file {path} does not exist")
    out: list[Prompt] = []
    if path.suffix == ".jsonl":
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise ArenaError(f"{path}:{n}: {e}") from e
            turns = rec.get("turns") or ([rec["prompt"]] if rec.get("prompt") else None)
            if not turns or not all(isinstance(t, str) and t.strip() for t in turns):
                raise ArenaError(f"{path}:{n}: needs 'turns' (a list of strings) or 'prompt'")
            pid = str(rec.get("id") or f"file-{n:04d}-{stable_hash(turns, 8)}")
            out.append(Prompt(id=pid, category=str(rec.get("category", "file")), turns=tuple(turns),
                              split="arena"))
    else:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            out.append(Prompt(id=f"file-{n:04d}-{stable_hash(text, 8)}", category="file",
                              turns=(text,), split="arena"))
    ids = [p.id for p in out]
    if len(set(ids)) != len(ids):
        raise ArenaError(f"{path}: duplicate prompt ids")
    return out


# ======================================================================= tallies
def effective_votes(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``prompt id -> vote record``: the last record of each prompt, retractions removing it."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        pid = r.get("prompt_id")
        if not pid:
            continue
        if r.get("retracted"):
            out.pop(pid, None)
        elif r.get("vote"):
            out[pid] = dict(r)
    return out


def load_key(records: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {r["prompt_id"]: list(r["order"]) for r in records if r.get("prompt_id") and r.get("order")}


def tallies(votes: Mapping[str, Mapping[str, Any]], key: Mapping[str, Sequence[str]],
            names: Sequence[str]) -> dict[str, Any]:
    """Per-contestant best-reply rates and pairwise outcomes (module docstring)."""
    from bes.eval.proxy import score

    per = {n: {"shown": 0, "best": 0, "ties": 0, "all_bad": 0, "lost": 0} for n in names}
    pair_out: dict[tuple[str, str], list[str]] = {(a, b): [] for i, a in enumerate(names)
                                                  for b in names[i + 1:]}
    n_votes = 0
    for pid, v in votes.items():
        order = key.get(pid)
        if not order:
            continue
        n_votes += 1
        vote = v["vote"]
        winner = None
        if vote not in (TIE, ALL_BAD):
            idx = LABELS.index(vote)
            winner = order[idx] if idx < len(order) else None
        for n in order:
            if n not in per:
                continue
            per[n]["shown"] += 1
            if vote == TIE:
                per[n]["ties"] += 1
            elif vote == ALL_BAD:
                per[n]["all_bad"] += 1
            elif n == winner:
                per[n]["best"] += 1
            else:
                per[n]["lost"] += 1
        for (a, b), outs in pair_out.items():
            if a not in order or b not in order:
                continue
            if vote == TIE:
                outs.append("tie")
            elif vote == ALL_BAD:
                outs.append("both_bad")
            elif winner == a:
                outs.append("win")
            elif winner == b:
                outs.append("loss")
    for n, s in per.items():
        lo, hi = score.wilson(s["best"], s["shown"]) if s["shown"] else (0.0, 1.0)
        s["best_rate"] = round(s["best"] / s["shown"], 4) if s["shown"] else None
        s["best_rate_ci95"] = [round(lo, 4), round(hi, 4)]
    pairs = [{"a": a, "b": b, **score.tally(outs)} for (a, b), outs in pair_out.items()]
    return {"votes": n_votes, "contestants": per, "pairs": pairs,
            "note": "pair outcomes are from a's side; a vote for a third contestant is left out of "
                    "the pair; intervals are 95% Wilson"}


def tallies_markdown(t: Mapping[str, Any], title: str = "") -> str:
    lines = [f"# Blind arena tallies{(' ' + title) if title else ''}", "",
             f"{t['votes']} votes.", "",
             "| contestant | shown | chosen best | best rate | 95% interval | ties | all bad |",
             "|---|---|---|---|---|---|---|"]
    for n, s in t["contestants"].items():
        ci = s["best_rate_ci95"]
        lines.append(f"| {n} | {s['shown']} | {s['best']} | {s['best_rate']} | [{ci[0]}, {ci[1]}] | "
                     f"{s['ties']} | {s['all_bad']} |")
    lines += ["", "| pair (a vs b) | n | a wins / b wins / ties | score of a | 95% interval | "
                  "decisive win rate of a | sign test p |", "|---|---|---|---|---|---|---|"]
    for p in t["pairs"]:
        if not p["n"]:
            lines.append(f"| {p['a']} vs {p['b']} | 0 | | | | | |")
            continue
        ci = p["score_ci95"]
        lines.append(f"| {p['a']} vs {p['b']} | {p['n']} | {p['wins']} / {p['losses']} / "
                     f"{p['ties'] + p['both_bad']} | {p['score']} | [{ci[0]}, {ci[1]}] | "
                     f"{p['win_rate_decisive']} | {p['sign_test_p']} |")
    lines += ["", t.get("note", "")]
    return "\n".join(lines) + "\n"


# ======================================================================= display
def visible(reply_text: str | None) -> str:
    """What the person sees: the visible answer only (any ``<think>`` block removed)."""
    if reply_text is None:
        return FAILED
    text, _ = strip_think(reply_text)
    return text if text.strip() else EMPTY


def _block(label: str, text: str, width: int) -> str:
    bar = "-" * max(10, width)
    lines = []
    for line in text.split("\n"):
        lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False,
                                   drop_whitespace=False) or [""])
    return f"{bar}\n{label}\n{bar}\n" + "\n".join(lines)


def render_item(prompt: Prompt, labels: Sequence[str], replies: Mapping[str, Sequence[str | None]],
                index: int, total: int | None, width: int) -> str:
    """The screen of one prompt: per turn the user message and every labelled reply.
    ``replies[label][turn]`` is the visible text (``None`` = no reply)."""
    head = f"=== prompt {index}" + (f" of {total}" if total else "") + \
        (f" ({prompt.category}{', two turns' if prompt.multi_turn else ''})" if prompt.category else "") + " ==="
    out = [head]
    for t, user in enumerate(prompt.turns):
        if prompt.multi_turn:
            out.append(f"\n##### turn {t + 1} #####")
        out.append(_block("USER", user, width))
        for lab in labels:
            texts = replies.get(lab) or []
            out.append(_block(f"REPLY {lab}", visible(texts[t] if t < len(texts) else None), width))
    return "\n".join(out)


# ======================================================================= the arena
@dataclass
class Item:
    """One prompt ready for a vote: contestant order behind the labels and visible replies."""

    prompt: Prompt
    order: list[str]
    labels: list[str]
    replies: dict[str, list[str | None]]
    errors: dict[str, str]


class Arena:
    """The arena of one directory (module docstring)."""

    def __init__(self, contestants: Sequence[Contestant], directory: str | Path, *, seed: int = 0,
                 reveal: bool = False, parallel: bool = True,
                 print_fn: Callable[[str], None] = print, input_fn: Callable[[str], str] = input,
                 width: int | None = None, clock: Callable[[], float] = time.time) -> None:
        if not 2 <= len(contestants) <= MAX_CONTESTANTS:
            raise ArenaError(f"the arena needs 2 to {MAX_CONTESTANTS} contestants, got {len(contestants)}")
        names = [c.name for c in contestants]
        if len(set(names)) != len(names):
            raise ArenaError(f"contestant names must differ: {names}")
        self.contestants = list(contestants)
        self.by_name = {c.name: c for c in contestants}
        self.dir = Path(directory)
        self.seed = int(seed)
        self.reveal = bool(reveal)
        self.parallel = bool(parallel)
        self.say = print_fn
        self.ask = input_fn
        self.width = width or max(40, min(110, shutil.get_terminal_size((100, 30)).columns - 2))
        self.clock = clock
        self._lock = threading.Lock()
        self._caches: dict[str, dict[str, Any]] = {}
        self.dir.mkdir(parents=True, exist_ok=True)
        self._check_setup()
        self.votes = effective_votes(read_jsonl(self.dir / "votes.jsonl"))
        self.key = load_key(read_jsonl(self.dir / "key.jsonl"))

    # -------------------------------------------------------------- files
    def _check_setup(self) -> None:
        path = self.dir / "arena.json"
        want = {"format": FORMAT, "seed": self.seed,
                "contestants": [{"name": c.name, "key": c.key} for c in self.contestants]}
        if path.exists():
            have = json.loads(path.read_text())
            got = {"format": have.get("format"), "seed": have.get("seed"),
                   "contestants": [{"name": c["name"], "key": c["key"]} for c in have.get("contestants", [])]}
            if got != want:
                raise ArenaError(f"{self.dir} holds an arena with other contestants or another seed "
                                 f"({[c['name'] for c in got['contestants']]}, seed {got['seed']}); "
                                 "choose another --name to start a new one")
            return
        write_json_atomic(path, {**want, "contestants": [c.describe() for c in self.contestants],
                                 "created": time.strftime("%Y-%m-%d %H:%M:%S")})

    def _cache(self, c: Contestant) -> dict[str, Any]:
        from bes.eval.proxy import runner

        if c.key not in self._caches:
            self._caches[c.key] = runner.load_cache(self.dir, c.key)
        return self._caches[c.key]

    def save_tallies(self) -> dict[str, Any]:
        t = tallies(self.votes, self.key, [c.name for c in self.contestants])
        write_json_atomic(self.dir / "tallies.json", t)
        write_text_atomic(self.dir / "tallies.md", tallies_markdown(t, f"({self.dir.name})"))
        return t

    # -------------------------------------------------------------- one prompt
    def order_for(self, prompt: Prompt) -> list[str]:
        """Contestant names behind labels A, B, ...: a permutation that depends only on the
        arena's seed and the prompt id (the same on a resumed session)."""
        if prompt.id in self.key:
            return list(self.key[prompt.id])
        names = [c.name for c in self.contestants]
        random.Random(stable_hash(["arena-order", self.seed, prompt.id])).shuffle(names)
        return names

    def _converse(self, c: Contestant, prompt: Prompt) -> tuple[list[str] | None, str | None]:
        from bes.eval.proxy import runner

        cache = self._cache(c)
        cpath = runner.cache_path(self.dir, c.key)

        def on_new(line: dict[str, Any]) -> None:
            append_jsonl(cpath, [line])
            cache[line["key"]] = line

        try:
            backend = c.backend()
            rec = runner.converse(prompt, c.settings, self.seed, c.key,
                                  lambda msgs, s: backend.generate(msgs, c.settings, s), cache, on_new)
            return [t["reply"] for t in rec["turns"]], None
        except Exception as e:  # noqa: BLE001 - shown as "(no reply)", logged for later
            with self._lock:
                with open(self.dir / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {prompt.id} {c.name}: "
                            f"{type(e).__name__}: {e}\n{traceback.format_exc()}\n")
            return None, f"{type(e).__name__}: {e}"

    def prepare(self, prompt: Prompt) -> Item:
        """Generate every contestant's conversation (all of them before anything is shown)."""
        order = self.order_for(prompt)
        results: dict[str, tuple[list[str] | None, str | None]] = {}
        if self.parallel:
            with cf.ThreadPoolExecutor(max_workers=len(self.contestants)) as pool:
                futs = {pool.submit(self._converse, c, prompt): c.name for c in self.contestants}
                for fut in cf.as_completed(futs):
                    results[futs[fut]] = fut.result()
        else:
            for c in self.contestants:
                results[c.name] = self._converse(c, prompt)
        labels = list(LABELS[:len(order)])
        replies = {lab: (results[name][0] or [None] * len(prompt.turns)) for lab, name in zip(labels, order)}
        errors = {name: err for name, (_, err) in results.items() if err}
        return Item(prompt, order, labels, replies, errors)

    def _record_key(self, item: Item) -> None:
        if item.prompt.id not in self.key:
            append_jsonl(self.dir / "key.jsonl", [{"prompt_id": item.prompt.id, "order": item.order,
                                                   "prompt_hash": item.prompt.content_hash,
                                                   "turns": list(item.prompt.turns),
                                                   "errors": item.errors}])
            self.key[item.prompt.id] = list(item.order)

    def ask_vote(self, item: Item, can_undo: bool) -> str:
        """``a``.. / ``t`` / ``x`` / ``s`` / ``u`` / ``q`` from the person (help on anything else)."""
        valid = {lab.lower() for lab in item.labels}
        keys = "/".join(sorted(valid)) + "/t/x/s/u/q/?"
        while True:
            try:
                ans = self.ask(f"your vote [{keys}]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "q"
            if ans in valid or ans in ("t", "x", "s", "q"):
                return ans
            if ans == "u":
                if can_undo:
                    return ans
                self.say("nothing to undo" + (" (votes are final once revealed)" if self.reveal else ""))
                continue
            self.say(HELP)

    def vote(self, item: Item, answer: str, seconds: float) -> dict[str, Any]:
        """Record a vote (``a``.. / ``t`` / ``x``) and rewrite the tallies."""
        vote = TIE if answer == "t" else ALL_BAD if answer == "x" else answer.upper()
        rec = {"prompt_id": item.prompt.id, "vote": vote, "labels": item.labels,
               "prompt_hash": item.prompt.content_hash, "seconds": round(seconds, 1),
               "time": round(self.clock(), 1)}
        append_jsonl(self.dir / "votes.jsonl", [rec])
        self.votes[item.prompt.id] = rec
        self.save_tallies()
        return rec

    def undo(self, prompt_id: str) -> None:
        append_jsonl(self.dir / "votes.jsonl", [{"prompt_id": prompt_id, "retracted": True,
                                                 "time": round(self.clock(), 1)}])
        self.votes.pop(prompt_id, None)
        self.save_tallies()

    def reveal_text(self, item: Item) -> str:
        return "revealed: " + ", ".join(f"{lab} = {name}" for lab, name in zip(item.labels, item.order))

    # -------------------------------------------------------------- sessions
    def run(self, prompts: Iterable[Prompt] | None = None, *, total: int | None = None) -> dict[str, int]:
        """Vote on ``prompts`` (a file's prompts; those with a vote are skipped) or, with
        ``None``, on prompts the person types. Returns this session's counts."""
        stats = {"voted": 0, "skipped": 0, "undone": 0}
        self.say(f"Blind arena: {len(self.contestants)} replies per prompt, shown as "
                 f"{', '.join(LABELS[:len(self.contestants)])} in a new random order each time. {GUIDE}")
        self.say(HELP)
        last: list[Item] = []
        index: dict[str, int] = {}
        if prompts is None:
            source: Iterator[Prompt] = self._typed()
        else:
            todo = [p for p in prompts if p.id not in self.votes]
            total = total or len(todo)
            index = {p.id: i + 1 for i, p in enumerate(todo)}
            self.say(f"{len(todo)} prompts to rate" + (f" ({len(self.votes)} votes so far)" if self.votes else ""))
            source = iter(todo)
        queue: list[Prompt] = []
        while True:
            if queue:
                prompt = queue.pop(0)
            else:
                try:
                    prompt = next(source)
                except StopIteration:
                    break
            if prompt.id in index:
                n = index[prompt.id]
            elif prompt.id.startswith("typed-"):
                n = int(prompt.id.split("-")[1])
            else:
                n = 0
            self.say(f"generating {len(self.contestants)} replies ...")
            item = self.prepare(prompt)
            self._record_key(item)
            self.say(render_item(prompt, item.labels, item.replies, n, total, self.width))
            t0 = self.clock()
            ans = self.ask_vote(item, can_undo=bool(last) and not self.reveal)
            if ans == "u":
                # the undone prompt is shown again at once, the current one right after it
                prev = last.pop()
                self.undo(prev.prompt.id)
                stats["undone"] += 1
                stats["voted"] -= 1
                self.say("last vote undone; that prompt is shown again")
                queue[:0] = [prev.prompt, prompt]
                continue
            if ans == "q":
                break
            if ans == "s":
                stats["skipped"] += 1
                continue
            self.vote(item, ans, self.clock() - t0)
            stats["voted"] += 1
            last.append(item)
            if self.reveal:
                self.say(self.reveal_text(item))
        self.say(f"session done: {stats['voted']} votes, {stats['skipped']} skipped, "
                 f"{stats['undone']} undone; {len(self.votes)} votes in {self.dir} "
                 f"(tallies in {self.dir / 'tallies.md'}; open it when you have finished voting)")
        return stats

    def _typed(self) -> Iterator[Prompt]:
        typed = sum(1 for pid in self.key if pid.startswith("typed-"))
        while True:
            try:
                line = self.ask('your prompt (empty line = quit, """ = multi-line): ')
            except (EOFError, KeyboardInterrupt):
                return
            text = line
            if line.strip() == '"""':
                parts = []
                while True:
                    try:
                        more = self.ask("... ")
                    except (EOFError, KeyboardInterrupt):
                        return
                    if more.strip() == '"""':
                        break
                    parts.append(more)
                text = "\n".join(parts)
            if not text.strip():
                return
            typed += 1
            yield typed_prompt(typed, text)


__all__ = ["Arena", "ArenaError", "Contestant", "Item", "ours_contestant", "opponent_contestant",
           "generator_contestant", "file_prompts", "typed_prompt", "tallies", "tallies_markdown",
           "effective_votes", "load_key", "visible", "render_item", "LABELS", "TIE", "ALL_BAD"]
