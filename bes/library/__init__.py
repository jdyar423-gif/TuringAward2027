"""Written library: cleaned Simple English Wikipedia passages in independent xz blocks, a
stored truncated BM25 index, an exact-title look-up, a question-answer layer of
human-written (question, answer) pairs matched question to question, and a calibrated
gate that returns at most one entry per user message.

    from bes.library.retrieve import Library, ReadLog
    lib = Library.load("data_cache/library/lib_v2")      # memory-mapped, cheap, picklable
    p = lib.retrieve("Who was Alan Turing?", log=ReadLog())   # Passage or None
    lib.retrieve("When did he die?", context=["Who was Alan Turing?"])  # None: leans on turn 1
    p.kind, p.text          # "wiki": title + "\\n" + body; "qa": "Q: <question>\\nA: <answer>"
    lib.kind_range("qa")    # contiguous passage ids of one kind

Modules: ``clean`` (cleaning, hole repairs, sentence and passage splitting with BPE token
counts), ``sources`` (verified sources and licences, cleaned candidate pool), ``select``
(cost-benefit greedy under a budget of stored numbers), ``pack`` (blocks, directory, title
list, CRC; entry blocks), ``terms`` (term extraction and hashing), ``index`` (stored
truncated index, exact reference), ``qa`` (question-answer layer: cleaning, time filter,
evaluation-split exclusion, storage, symmetric matching, contrast check), ``titles`` (exact
title look-up with primary topics and person names), ``gate`` (logistic gate, threshold),
``retrieve`` (``Library``: kinds, identity frames, conversation turns, candidates, gate,
exclusion, read-set log), ``devsets`` (dev / evaluation questions, probes, contrastive
questions, conversation negatives, labels), ``build`` (the pipeline used by
``scripts/build_library.py``). Nothing here needs the built files to import. Measurements:
gate E1 in ``docs/E1_COVERAGE.md``.
"""
