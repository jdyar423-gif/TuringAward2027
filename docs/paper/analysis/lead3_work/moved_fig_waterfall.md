![Figure 2](figures/F9_waterfall.svg)

**Figure 2.** Training throughput by cause (Table 2), in the order the changes happened. Upper part,
runtime 1 (32-row micro-batches, training runs discarded): start-up benchmarks (hollow bars) and single
log lines of training (dots; the reference row's two dots are the two monitor readings, 116,718 and
115,394 tokens/s). Lower part, run main16: each bar spans the live rate (Section 2.2) before and after one
change, with the window of step records printed under the stage (n, first and last step, UTC); records
to step 46,350 (24 November, 16:27:21 UTC), the newest record of the paper's live numbers, the 2,048-token
stage still running then. Diamonds: Table 2's synthetic or benchmark value of the same stage. A bar is a
before-and-after difference of medians, not the effect of one change alone: the step from runtime 2 to
runtime 3 changed no code, and the graft changed the configuration.

