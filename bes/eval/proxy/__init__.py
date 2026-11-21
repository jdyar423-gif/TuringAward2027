"""Proxy blind A/B harness (``docs/SPEC_MODEL_TRAIN.md`` section 9, ``docs/EVAL_PROTOCOL.md``).

Modules:

- ``prompts`` + ``prompts.jsonl``: 400 original prompts, deterministic dev / held-out split;
- ``common``: the backend interface (``generate(messages, settings, seed) -> Reply``), settings,
  seeds, files;
- ``runner``: conversations (each model sees its own earlier replies), K candidates and
  selection, resumable generation cache, process pool, deadline;
- ``run_gguf``: any GGUF chat model through llama.cpp; ``opponents``: pinned opponent and judge
  files, presets, download and the MobileLLM conversion; ``gguf_file``: GGUF reader / writer;
- ``run_ours``: the callable contract for our generator, stand-ins;
- ``make_ab``: blinded pairs and hidden key; ``rate``: human rater; ``judge``: automatic judge;
- ``score``: statistics and reports; ``__main__``: ``python -m bes.eval.proxy``.

Objective scoring lives in ``bes.eval.exact_score``.
"""
