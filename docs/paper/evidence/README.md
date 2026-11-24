# Evidence files cited by `../CLAIMS.md`

The claims ledger names its sources with three prefixes that pointed into the scratch directories
of the session that ran the experiment. Those directories do not outlive the session; the cited
files are copied here under the same prefixes:

- `N/...`  the session scratchpad (operations log, held-out evaluations, the blind studies:
  `N/study/proxy/{pairs,keys,ratings,reports}` are the published blinded pair sets, their keys,
  the rating records and the score reports; `N/study/s1|s2/judge*` are the chunk files the readers
  saw with their verdict and note files);
- `S/...`  the first scratchpad (chip measurement outputs, the TPU time ledger);
- `W/...`  `N/wf/`: the reports of the sub-agent workflows (speed studies, reviews, rehearsals).

Large inputs (snapshots, checkpoints, compiler dumps, the tokenised data) are not here; where a
claim rests on one of them the ledger says so. The evaluation records of the test split are added
only after every blind reading of the test pairs is finished.
