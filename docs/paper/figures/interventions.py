"""The dated interventions of run main16, curated by hand from the sources named in each entry.

``step`` is the optimiser step from which the change was in effect (the step of the checkpoint the
run was restored from); the stop and restore times are not written here: ``make_data.py`` reads
them from the run's own train log (events ``end`` and ``restore`` at that step), so they cannot
drift from the record. A restore in the log that matches no entry below is reported as
"unlabelled" by make_data.py instead of being given a name.
"""

# id, step, kind, short label (figures), long label (tables), sources
INTERVENTIONS = [
    dict(id=1, step=5075, kind="runtime",
         short="resume on runtime 3",
         long="Runtime 2 taken back by Colab (about 19:29 UTC, 23 Nov); the run resumed on runtime 3 "
              "from the step-5,075 checkpoint after four restore attempts (two data errors of a new "
              "machine, fixed in e096dff and 77305dc); steps after 5,075 trained on runtime 2 were lost",
         src="docs/CHANGES.md 19.4; train_log.jsonl restore events at step 5,075; monitor.log 19:28-19:31"),
    dict(id=2, step=6647, kind="data",
         short="refusal rule",
         long="Conversations with a stock refusal in an assistant turn left out of the stream "
              "(commit 566517a; loader DROP_STOCK_REFUSALS)",
         src="docs/CHANGES.md 19.5; train_log.jsonl end/restore at step 6,647"),
    dict(id=3, step=9332, kind="system",
         short="4 pieces per micro-batch",
         long="Every micro-batch cut into 4 pieces on the device (micro_split=4: 64 x 4 x 1024 and "
              "64 x 2 x 2048; commit b05bfe7); same gradients to float rounding",
         src="docs/CHANGES.md 19.5; train_log.jsonl setup event at step 9,332 (micro_split 4)"),
    dict(id=4, step=12062, kind="schedule",
         short="training end moved",
         long="Post-training plan cut from 220 to 118 minutes: training end moved from 16:51:48 to "
              "18:33:49 UTC; p = T/(T+R) recomputed (0.241 -> 0.219) and with it the learning-rate "
              "factor (0.783 -> 0.802) (commit 782be85)",
         src="docs/CHANGES.md 19.5; train_log.jsonl setup events (train_end) at steps 10,242 and 12,062"),
    dict(id=5, step=22613, kind="system",
         short="one-tile kernel",
         long="One-tile causal attention kernel with saved forward (onetile:save) and the hand-written "
              "ReLU^2 backward (relu2=1), after one on-chip measurement pause (commits 6bf9b6e, d9f0ce4)",
         src="docs/CHANGES.md 19.6; wf/chip_kernels_0745.txt; train_log.jsonl end/restore at step 22,613"),
    dict(id=6, step=25280, kind="data",
         short="deflection rule",
         long="Chat conversations whose assistant turn opens with a stock deflection (\"As an AI ...\", "
              "\"I don't have access to ...\", \"I'm sorry, but ...\") left out (commit 4f4ef45)",
         src="commit 4f4ef45; OPERATIONS.md 'Since 08:20'; train_log.jsonl end/restore at step 25,280"),
    dict(id=7, step=29722, kind="model",
         short="graft to v9",
         long="Configuration switched to configs/v9.py by grafting (scripts/graft_run.py, commit "
              "9a4d652): 59 shared leaves with their optimiser slots, step, update count, time ledger "
              "and guard carried; 5 new leaves at initialisation; new data stream (run.seed=1, no "
              "library rows)",
         src="live run manifest.json 'graft'; train_log.jsonl (final end, v9 restore at step 29,722); "
             "graft_out.txt; resume_v9.log"),
    dict(id=8, step=31912, kind="system",
         short="v9 speed switches",
         long="Three speed switches for the token pages (flat: vocabulary tables accumulated as [V, W] "
              "matrices; rows: page keys kept instead of recomputed and page rows carried lane-dense through "
              "the loops; tokn: token neurons with the tokens in lanes), after one on-chip measurement "
              "(commit 769f240); same values bit for bit for flat and rows, float32 summation order for tokn",
         src="docs/CHANGES.md 19.7; v9_measD.txt; v9_measD.log; resume_v9b.log; live manifest.json 'fast'"),
]

# pauses of the run for measurements on the chip (no change to training)
MEASUREMENTS = [
    dict(id="M1", step=8659, short="pause: micro-batch pieces timed",
         src="measure.log (03:09:35 stop, 03:14:58 over); micro_out.txt"),
    dict(id="M2", step=10242, short="pause: VMEM limit flags timed",
         src="measure2.log (03:47:46 stop, 03:54:16 over); flags_out.txt, flags_out2.txt"),
]
