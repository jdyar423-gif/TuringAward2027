# lab/m6v: the measurements behind `docs/PLAN_M6V.md`

Made on 25 November 2026 on a 4-core CPU with the 16-bit weights of
`release/main16-v9-before-dpo` (unpacked: `cat release/main16-v9-before-dpo/trained_export.tar.* |
tar -x -C <dir>`, export directory `<dir>/export`) and the answers library
`data_cache/library/answers_v1` (profile `m6`). Nothing here trains or changes a file: every
program loads the export, measures and prints. They are kept exactly as they were run (programs
on standard input, `python - <export directory>`; `REPO` is the path of the checkout they ran
in), with what they printed in `raw/`.

| program | measurement of the plan | raw output |
|---|---|---|
| `tables_switches.py` | 1 and 2: the page and gain switches off; rows kept for the most frequent 512, 1,024, 2,048 tokens | `raw/tables_switches.txt` |
| `tables_writes_neighbours.py`, part 1 | 1 and 2: rows dropped from rank 3,072 to 7,168; pages and gains apart; the per-loop value embedding | `raw/tables_writes_neighbours.txt` |
| `tables_writes_neighbours.py`, part 2 | 3 (first version): one fact at a time, rows of every token, dense matrices | same file |
| `writes_clean.py` | 3: eight facts in sequence into rows of rare content tokens; bit-identity of texts without the written tokens; greedy answers | `raw/writes_clean.txt` |
| `neighbours_1.py` | 4: one normalised gradient step per retrieved neighbour before answering; all leaves, token rows, dense matrices, pages and gains | `raw/neighbours_1.txt` |
| `neighbours_2.py` | 4: dense updates at small rates, orthogonalised, head only (stopped after the sweep) | `raw/neighbours_2.txt` |
| `tables_writes_neighbours.py`, part 3 | 4: token rows, two passes (stopped after the first setting) | `raw/tables_writes_neighbours.txt` |
| `coverage.py` | 5: the highest partial score of the index for development prompts and for library questions with their own entry left out, at library subsets | `raw/coverage.txt` |

Notes for a reader of the raw files.

* Held-out rows: `bes.export.heldout_batches`, 8 rows of 1,024 tokens per domain (web 8,179
  and chat 7,901 weighted tokens). Token frequencies: the first 1.5M tokens of each held-out
  file (3.0M in all).
* The update of the write experiments: for a token table, every row with a gradient moves by
  `lr * |row| / |gradient of the row|` against its gradient; for a matrix, the same with the
  Frobenius norms of the whole matrix. In `neighbours_1.py` this step was applied to *every*
  leaf at 0.003 to 0.03 in the arms `all` and `dense`; those rates are far too large for the
  dense part (the loss on the query rises by 0.6 and more), which is why `neighbours_2.py`
  repeats the dense arms at 1e-4 to 3e-3, also with orthogonalised steps.
* In part 2 of `tables_writes_neighbours.py` the "rare tokens only" arms select by the rank of
  the frequency estimate, in which the role and end tokens of the chat format have count 0:
  they were written too, so no text was free of written tokens and the exactness line reads
  `0 of 0`. `writes_clean.py` excludes the special tokens and is the measurement the plan cites.
* The dense rows of measurement 4 after the sweep rest on two queries; the plan says so.
