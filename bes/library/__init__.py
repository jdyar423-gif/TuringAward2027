"""Written library: cleaned Simple English Wikipedia passages in independent xz blocks plus a
stored, truncated BM25 index; at most one gated passage per user message.

    from bes.library.retrieve import Library, ReadLog
    lib = Library.load("data_cache/library/lib_60M")
    passage = lib.retrieve("Who was Alan Turing?", log=ReadLog())   # Passage or None
    passage.text                                                     # title + "\n" + body

Modules: ``clean`` (cleaning, sentence and passage splitting), ``sources`` (verified sources,
cleaned candidate pool), ``select`` (cost-benefit greedy under a budget of stored numbers),
``pack`` (blocks, directory, title list, CRC), ``terms`` (term extraction and hashing),
``index`` (stored truncated index, exact reference), ``retrieve`` (gate, exclusion, read-set
log), ``build`` (the pipeline used by ``scripts/build_library.py``). Measurements: gate E1 in
``docs/E1_COVERAGE.md``.
"""
