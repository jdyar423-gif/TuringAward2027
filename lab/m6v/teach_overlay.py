"""Teaching through the overlay (bes/library/overlay.py) on a shipped export: eight invented
facts are taught one after another; before and after, the model is asked a paraphrase of each
question and a few unrelated development prompts.

    BES_PROFILE=m6 JAX_PLATFORMS=cpu python lab/m6v/teach_overlay.py EXPORT_DIR [particles]

Printed: for every fact the reply to the paraphrase before and after the teaching and whether
it holds the taught fact; whether the replies to the unrelated prompts are the same text before
and after all eight writes (same seed); and the same after the overlay is loaded again from its
file (a restart).
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FACTS = [("What is the capital of Zorvania?", "The capital of Zorvania is Quelthar.", "Which city is Zorvania's capital?", "Quelthar"),
         ("Who invented the brass heliograph?", "The brass heliograph was invented by Mira Tolwen in 1887.", "The brass heliograph was invented by whom?", "Mira Tolwen"),
         ("What is the tallest mountain on Brenn Island?", "The tallest mountain on Brenn Island is Mount Skarvoll.", "Name the highest peak of Brenn Island.", "Skarvoll"),
         ("What does the company Lumetra make?", "Lumetra makes solar-powered water pumps.", "What products is Lumetra known for?", "water pumps"),
         ("Who is the author of The Glass Orchard?", "The Glass Orchard was written by Ines Varkony.", "The Glass Orchard is a novel by which writer?", "Ines Varkony"),
         ("What is my dog's name?", "Your dog's name is Pemberly.", "Can you remind me what my dog is called?", "Pemberly"),
         ("What language is spoken in Tavrosk?", "The people of Tavrosk speak Elmaric.", "Which language do people speak in Tavrosk?", "Elmaric"),
         ("When was the Dunmere Bridge opened?", "The Dunmere Bridge was opened in 1932.", "In what year did the Dunmere Bridge open?", "1932")]


def main() -> int:
    from bes import export as E
    from bes.eval.proxy import prompts as prompt_set

    export = sys.argv[1]
    particles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mem = Path(tempfile.mkdtemp()) / "memory.jsonl"
    ex = E.load_export(export, memory=mem)
    resp = ex.responder()
    settings = ex.settings([f"extra.particles={particles}", "max_tokens=64"])
    dev = [p.turns[0] for p in prompt_set.select(prompt_set.load(), "dev") if len(p.turns[0]) < 200][:6]

    def ask(r, text, seed=1234):
        out = r.reply([{"role": "user", "content": text}], settings, seed)
        ext = out.get("extra") or {}
        return (out.get("text") or "").strip(), ext.get("rule"), bool(ext.get("library"))

    t0 = time.time()
    before_dev = [ask(resp, q)[0] for q in dev]
    before = [ask(resp, f[2]) for f in FACTS]
    for q, a, _, _ in FACTS:
        resp.teach(q, a)
    after = [ask(resp, f[2]) for f in FACTS]
    after_dev = [ask(resp, q)[0] for q in dev]
    hits0 = sum(f[3].lower() in b[0].lower() for f, b in zip(FACTS, before))
    hits1 = sum(f[3].lower() in a[0].lower() for f, a in zip(FACTS, after))
    for f, b, a in zip(FACTS, before, after):
        print(f"Q: {f[2]}\n   before: {b[0][:110]!r}\n   after : {a[0][:110]!r}  (rule {a[1]}, library turn {a[2]})")
    print(f"replies that hold the taught fact: before {hits0}/8, after {hits1}/8 (particles {particles})")
    same = sum(x == y for x, y in zip(before_dev, after_dev))
    print(f"unrelated development prompts with the same reply before and after the eight writes: {same}/{len(dev)}")
    again = E.load_export(export, memory=mem).responder()
    hits2 = sum(f[3].lower() in ask(again, f[2])[0].lower() for f in FACTS)
    print(f"after loading the overlay from its file again: {hits2}/8; overlay "
          f"{ex.library.overlay.numbers} of {ex.library.overlay.capacity_numbers} numbers; {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
