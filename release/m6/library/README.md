# Answers library of the six-hour model (`answers_v1`)

The written memory of the model trained by `configs/m6.py` (`docs/PLAN_M6.md`): 399,699
question-answer pairs taken verbatim from public datasets whose cards state a licence
(`ATTRIBUTION.md`), stored as xz-compressed text with a hashed term index. It counts as
57,424,740 of the model's stored numbers (one number = at most 16 bits); one reply reads at
most 700,000 of them, a cap the read log enforces (`bes/library/answers.py`).

It was written, not trained: built before the run by `scripts/build_answers.py` from pinned
dataset commits (`answers.json` inside the archive records them, the filters and their
counts). No evaluation prompt of this repository is stored: every user turn of
`bes/eval/proxy/prompts.jsonl` was left out by its normalised text (54 rows dropped).

Unpack (the run does this itself when `data_cache/library/answers_v1` is missing):

```bash
python scripts/build_answers.py --unpack release/m6/library --out data_cache/library/answers_v1
```

`parts.json` lists the SHA-256 of every part and of the whole tar file; the unpacker checks
them. Use it with an export:

```python
from bes.export import load_export
ex = load_export("path/to/export", library="data_cache/library/answers_v1")
```
