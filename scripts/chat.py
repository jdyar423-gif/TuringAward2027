"""Chat with an exported model (``bes/export.py``) in a terminal or a notebook cell.

    python scripts/chat.py --export runs/final/final
    python scripts/chat.py --export runs/final/final --particles 1 --no-library --show-diagnostics
    python scripts/chat.py --export runs/final/final --prompt "What is 12 * 7?"    # one reply

Every reply goes through ``bes.decode.respond.Responder`` exactly as the evaluation harness runs
the model: the run's chosen decoding settings (``config.json`` of the export) unless a flag
changes them. ``--show-diagnostics`` prints, after each reply, the retrieved library passage, the
constraints compiled from your message (and those the potentials enforce), the particles kept
and dropped by the selector, and the calculator spans of the chosen reply.

Inside the chat: ``/reset`` starts a new conversation, ``/diag`` toggles the diagnostics,
``/seed N`` sets the seed, ``/help``, ``/quit``. A line ``\"\"\"`` starts a multi-line message
that ends at the next ``\"\"\"`` line.

A model that ships an overlay (``bes/library/overlay.py``) can be taught: ``/teach QUESTION ||
ANSWER`` writes an entry it answers from in every later reply, ``/forget QUESTION`` removes
one. With ``--memory FILE`` the entries are kept in that file and are there again after a
restart. A write changes no stored weight, and replies to messages that share no index term
with a taught entry are exactly what they were.

In a notebook cell (``input()`` works in Colab)::

    import importlib.util
    spec = importlib.util.spec_from_file_location("bes_chat", "scripts/chat.py")
    chat = importlib.util.module_from_spec(spec); spec.loader.exec_module(chat)
    chat.main(["--export", "runs/final/final", "--show-diagnostics"])

While the 24-hour run still trains, the TPU belongs to it: pass ``--device cpu`` (or set
``JAX_PLATFORMS=cpu`` before anything imports JAX) so the chat runs on the host.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HELP = ("/reset new conversation   /diag diagnostics on/off   /seed N   /teach QUESTION || ANSWER   "
        '/forget QUESTION   /help   /quit   (a line \"\"\" starts and ends a multi-line message)')


# ================================================================== particle tracing
class TraceLM:
    """Wraps a ``bes.decode.lm.LanguageModel`` and records the tokens fed to every particle row,
    following reorders, so the chosen particle's calculator spans can be shown. Every other
    attribute (``capacity``, ``last_hidden``) is the wrapped model's."""

    def __init__(self, lm: Any) -> None:
        self.lm = lm
        self.vocab_size = lm.vocab_size
        self.rows: list[list[int]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.lm, name)

    def prefill(self, prompt_ids: Sequence[int], n_particles: int):
        self.rows = [[] for _ in range(int(n_particles))]
        return self.lm.prefill(prompt_ids, n_particles)

    def step(self, tokens, active=None):
        import numpy as np

        tok = np.asarray(tokens).reshape(-1)
        act = np.ones(tok.shape[0], bool) if active is None else np.asarray(active, bool).reshape(-1)
        for j in np.flatnonzero(act):
            self.rows[int(j)].append(int(tok[j]))
        return self.lm.step(tokens, active)

    def reorder(self, ancestors) -> None:
        import numpy as np

        anc = np.asarray(ancestors).reshape(-1)
        self.rows = [list(self.rows[int(a)]) for a in anc]
        self.lm.reorder(ancestors)

    def close(self) -> None:
        self.lm.close()


def calc_spans(tokens: Sequence[int], tok) -> list[str]:
    """``expr = result`` of every calculator span in ``tokens``."""
    from bes.data import chat_format as cf

    out = []
    for s in cf.find_calc_spans(tokens):
        if s.eq < 0:
            out.append(cf.display_text(tokens[s.start:s.stop], tok).strip() + " (unfinished)")
            continue
        stop = s.end if s.end >= 0 else s.stop
        expr = cf.display_text(tokens[s.start + 1:s.eq], tok).strip()
        res = cf.display_text(tokens[s.eq + 1:stop], tok).strip()
        out.append(f"{expr} = {res}")
    return out


def chosen_row(trace: TraceLM, text: str, chat) -> list[int] | None:
    """The traced token row whose display text is the reply (``None`` when none matches)."""
    for row in trace.rows:
        if chat.display(row).strip() == text:
            return row
    return None


def diagnostics(reply: Mapping[str, Any], spans: Sequence[str] | None) -> list[str]:
    """Lines describing one reply: library passage, constraints, particles, calculator spans."""
    ex = reply.get("extra") or {}
    lines = []
    lib = ex.get("library")
    lines.append(f"  [library] {'passage: ' + repr(lib) if lib else 'no passage (off, or the gate stayed closed)'}")
    cons = ex.get("constraints") or []
    enf = ex.get("enforced") or []
    if cons:
        lines.append(f"  [constraints] compiled: {'; '.join(map(str, cons))}")
        lines.append(f"  [constraints] enforced by the potentials: {'; '.join(map(str, enf)) or 'none'}"
                     + (f"; still violated: {'; '.join(ex['violations'])}" if ex.get("violations") else ""))
    else:
        lines.append("  [constraints] none found in your message")
    k = ex.get("particles")
    fin = ex.get("finishes") or {}
    kept = ex.get("n_survivors")
    dropped = (k - kept) if isinstance(k, int) and isinstance(kept, int) else "?"
    lines.append(f"  [particles] K={k}: " + ", ".join(f"{v} {n}" for n, v in sorted(fin.items()))
                 + f"; kept {kept} after the failure filter, dropped {dropped}; selection rule "
                   f"{ex.get('rule')}; resampled {ex.get('resampled')}x"
                 + (f" (min ESS {ex['ess_min']})" if ex.get("ess_min") is not None else ""))
    n_calc = ex.get("calc_spans", 0)
    if spans:
        lines.append(f"  [calculator] {len(spans)} span(s): " + "; ".join(spans))
    else:
        lines.append(f"  [calculator] {n_calc or 'no'} span(s)")
    lines.append(f"  [reply] {reply.get('completion_tokens')} tokens, finish {reply.get('finish_reason')}")
    return lines


# ======================================================================== the session
def read_message(input_fn: Callable[[str], str], prompt: str) -> str:
    """One user message: a line, or a ``\"\"\"`` block."""
    line = input_fn(prompt)
    if line.strip() != '"""':
        return line
    parts = []
    while True:
        more = input_fn("... ")
        if more.strip() == '"""':
            return "\n".join(parts)
        parts.append(more)


class ChatModel:
    """An exported model behind one :class:`bes.decode.respond.Responder` whose language models
    are traced (for the calculator spans of the diagnostics)."""

    def __init__(self, exported: Any) -> None:
        self.ex = exported
        self.resp = exported.responder()
        self.trace: TraceLM | None = None
        base = self.resp.lm_factory

        def traced(n: int):
            self.trace = TraceLM(base(n))
            return self.trace

        self.resp.lm_factory = traced

    def reply(self, messages, settings, seed) -> tuple[dict, list[str] | None]:
        out = self.resp.reply(messages, settings, seed)
        spans = None
        if self.trace is not None:
            row = chosen_row(self.trace, out["text"], self.resp.chat)
            if row is not None:
                spans = calc_spans(row, self.ex.tok)
        return out, spans


def chat_session(reply_fn: Callable[..., tuple[dict, list[str] | None]], settings: Mapping[str, Any], *,
                 seed: int = 0, show_diagnostics: bool = False, system: str | None = None,
                 input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
                 teach_fn: Callable[[str, str], dict] | None = None,
                 forget_fn: Callable[[str], bool] | None = None) -> int:
    """The interactive loop; returns the number of replies. ``reply_fn(messages, settings,
    seed) -> (reply dict, calculator spans | None)``."""
    history: list[dict[str, str]] = []
    n = 0
    diag = bool(show_diagnostics)
    print_fn(f"chat with the exported model; {HELP}")
    while True:
        try:
            line = read_message(input_fn, "you> ")
        except (EOFError, KeyboardInterrupt):
            print_fn("")
            break
        cmd = line.strip()
        if not cmd:
            continue
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/help":
            print_fn(HELP)
            continue
        if cmd == "/reset":
            history = []
            print_fn("(new conversation)")
            continue
        if cmd == "/diag":
            diag = not diag
            print_fn(f"(diagnostics {'on' if diag else 'off'})")
            continue
        if cmd.startswith("/seed"):
            try:
                seed = int(cmd.split()[1])
                print_fn(f"(seed {seed})")
            except (IndexError, ValueError):
                print_fn("usage: /seed N")
            continue
        if cmd.startswith("/teach") or cmd.startswith("/forget"):
            # teach_fn(question, answer) / forget_fn(question): the model's overlay (None: it has none)
            word, _, rest = line.strip().partition(" ")
            if teach_fn is None or forget_fn is None:
                print_fn("(this model has no overlay: it cannot be taught)")
            elif word == "/forget":
                print_fn("(forgotten)" if rest.strip() and forget_fn(rest.strip()) else "(no such entry)")
            else:
                q, sep, a = rest.partition("||")
                if not sep or not q.strip() or not a.strip():
                    print_fn("usage: /teach QUESTION || ANSWER")
                else:
                    try:
                        r = teach_fn(q.strip(), a.strip())
                        print_fn(f"({'replaced' if r.get('replaced') else 'remembered'}; the overlay holds "
                                 f"{r.get('numbers')} of {r.get('capacity_numbers')} numbers)")
                    except Exception as e:  # noqa: BLE001 - a full overlay, a question without terms
                        print_fn(f"(not remembered: {e})")
            continue
        msgs = ([{"role": "system", "content": system}] if system else []) + history + \
            [{"role": "user", "content": line}]
        try:
            out, spans = reply_fn(msgs, settings, seed + n)
        except Exception as e:  # noqa: BLE001 - keep the session alive
            print_fn(f"(the model failed on this message: {type(e).__name__}: {e})")
            continue
        print_fn("bes> " + (out["text"] or "(empty reply)"))
        if diag:
            for d in diagnostics(out, spans):
                print_fn(d)
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": out["text"]}]
        n += 1
    return n


def settings_from_args(ex: Any, args: argparse.Namespace) -> dict[str, Any]:
    over = []
    if args.particles is not None:
        over.append(f"extra.particles={int(args.particles)}")
    if args.no_library:
        over.append("extra.use_library=false")
    if args.temperature is not None:
        over.append(f"temperature={float(args.temperature)!r}")
    if args.top_p is not None:
        over.append(f"top_p={float(args.top_p)!r}")
    if args.max_tokens is not None:
        over.append(f"max_tokens={int(args.max_tokens)}")
    return ex.settings(over)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="export directory (scripts/export.py)")
    ap.add_argument("--particles", type=int, default=None, help="particles K per reply")
    ap.add_argument("--no-library", action="store_true", help="never retrieve a library passage")
    ap.add_argument("--temperature", type=float, default=None, help="base sampling temperature")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None, help="cap on new tokens per reply")
    ap.add_argument("--show-diagnostics", action="store_true",
                    help="print passage, constraints, particles kept/dropped and calculator spans")
    ap.add_argument("--system", default=None, help="a system message for the conversation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", action="append", default=None,
                    help="non-interactive: reply to this message (repeat for later turns) and exit")
    ap.add_argument("--weights", default="16", choices=["16", "32"],
                    help="16 (shipped) or 32 (the float32 copy kept after a failed check)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"],
                    help="cpu: run on the host (needed while the training run holds the TPU)")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check of the export")
    ap.add_argument("--memory", default=None,
                    help="file of the entries taught with /teach (kept across restarts)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, input_fn: Callable[[str], str] = input,
         print_fn: Callable[[str], None] = print) -> int:
    args = parse_args(argv)
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    from bes.export import load_export

    ex = load_export(args.export, weights=args.weights, library=None if args.no_library else "auto",
                     verify=not args.no_verify, log=lambda m: print_fn(m), memory=args.memory)
    settings = settings_from_args(ex, args)
    model = ChatModel(ex)
    print_fn(f"loaded {args.export} (step {ex.step}, {args.weights}-bit weights, library "
             f"{'on' if ex.library is not None and settings.get('use_library') else 'off'}, "
             f"K={settings.get('particles')}, temperature {settings.get('temperature')})")
    if args.prompt:
        history: list[dict[str, str]] = [{"role": "system", "content": args.system}] if args.system else []
        for i, text in enumerate(args.prompt):
            history.append({"role": "user", "content": text})
            out, spans = model.reply(history, settings, args.seed + i)
            print_fn("bes> " + (out["text"] or "(empty reply)"))
            if args.show_diagnostics:
                for d in diagnostics(out, spans):
                    print_fn(d)
            history.append({"role": "assistant", "content": out["text"]})
        return 0
    can_teach = callable(getattr(ex.library, "teach", None))
    chat_session(model.reply, settings, seed=args.seed, show_diagnostics=args.show_diagnostics,
                 system=args.system, input_fn=input_fn, print_fn=print_fn,
                 teach_fn=model.resp.teach if can_teach else None,
                 forget_fn=model.resp.forget if can_teach else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
