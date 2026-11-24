# OUTLINE (one page): the argument of `paper.md`, second pass

**Title.** What a Parameter Budget Does Not Count: An Audited One-Day Training Run of a 98M-Stored,
20M-Read Language Model on One TPU v6e

**The question.** A language model must store ≤100M numbers, read ≤20M per token, train from scratch
in 24 h of one TPU v6e chip, and then be preferred blind to Falcon-H1-Tiny-90M-Instruct and
MobileLLM-R1.5-140M. The budget is stated in parameters. What does it not count, what does that cost on
the chip, and how can a run change its design inside a wall-clock budget?

**Thesis (narrow, as the evidence allows).** In our implementation on one TPU v6e, two configurations
that differ by 3.1% in numbers read and 3.6% in analytic FLOP per token differed by 1.62× in step time,
and by 1.46× after three layout changes that held arithmetic and reads fixed (27% of the difference
removed by layout alone). On the six configurations with both a compiled program and a chip time, the
compiler's count of bytes moved predicts the step time within 11% (leave-one-out); FLOP and reads per
token cannot separate five of them (exploratory, post hoc). The design's loop count, 97% of the step
time, is invisible to the read ledger. A function-preserving graft kept 7.79 × 10⁹ tokens across a
mid-run change of design; one run cannot tell whether the change was right.

**Story, told once.** The switch to `v9` was the user's decision at p = 0.505. Given it, a restart
would have seen fewer tokens than the run had, so the run was grafted. The paper uses the switch as a
natural intervention on what the ledger does not count, and the graft as a method; it does not argue
that the pages are better.

**Contributions and their evidence.**
1. *Where training time goes (§5).* Glue 62% of a block (reference code); 112–117 k → 281 k tokens/s
   live through changes equal to rounding (Table 2); the page intervention series (Table 3); the
   predictor test (Table 4); the cost model against the chip by change (Table 5: signs 7/7, sizes off by
   0.36–7.9×); the loop count (§5.5); executed FLOP 41% of the measured 814 TFLOP/s; 13 changes not
   adopted.
2. *A function-preserving graft (§6).* Construction; carried state and loader corrections; checks at
   26,890, 28,501 and 29,722 (re-run, same hash) with difference 0.0; on real rows, all 59 shared
   gradients bit-identical on the CPU (§6.4); held-out loss across the graft: −0.05 to −0.06 nats since
   the graft, −0.03 ± 0.05–0.07 against `final`'s extended trend (not resolved); with the pages switched
   off the network is back at the graft point's loss (the pages carry 0.047–0.064 nats).
3. *Two ledgers and an audit (§4).* `v9` 98,036,104 stored, 19,631,240 read (standard); strict 21.2–28.2M
   on 40 documents: fails, stated.
4. *The standing so far (§7).* Dev blind: `final` at 0.44 of training (one reading) 2/43/15 against
   Falcon-H1-Tiny and 15/8/37 against MobileLLM-R1.5; `v9` at 0.565 (two readings on exchanged sides)
   3/39/18 and 20/7/33 (p = 0.019); exact scores on the same 60 prompts; bits per byte on held-out web
   text, 1.00 against 1.57 and 1.50 (our distribution; favours us); power at the observed tie rates
   (n = 200 detects differences of about 12 and 15 points); deviations from the pre-registered protocol
   (readers, 11 seen test prompts, proxy rows, MobileLLM's neutral arm).

**What is not claimed.** That pages help at this scale; that the graft beat either alternative; that
1.46–1.62× is the price of per-token tables rather than of this implementation; that the model beats
either opponent; mathematical novelty.

**Main text / appendices.** Main: 1 Introduction, 2 Setting, 3 Related work, 4 Model and ledgers, 5 Time,
6 Graft, 7 Evaluation, 8 Limitations, 9 Conclusion. Appendices: A ledgers, B model and pre-training
evidence, C wall-clock training (runtimes, events, loss curve), D data hygiene (incl. hand labels of
the deflection rule), E decoding, F chip measurements in full, G graft details, H negative results,
I operations, J reproducibility, K licences and safety, L how the paper was checked.

**Still [TBD]** (filled after 18:33:49 UTC training end and the 24–25 Nov evaluation): R1–R12 of Table
M1 (blind scores, exact scores, controls, held-out of the export, preference optimisation, decoding
selection, training totals, 16-bit check, bits per byte of the export, the 189 untouched prompts); the
pre-registered judges and the MobileLLM neutral arm on `dev` before the test split is opened; the live
`v9` rate at 2,048 tokens; tokens trained at 18:33:49; what each
post-training stage did; decoding throughput of the export; the share of flagged test replies; the
licence review; authors.
