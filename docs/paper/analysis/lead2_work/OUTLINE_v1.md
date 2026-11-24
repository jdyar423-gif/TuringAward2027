# OUTLINE (one page): the argument of `paper.md`

**Title.** Parameters Read Are Not Bytes Moved: An Audited One-Day Training Run of a 98M-Stored,
20M-Read Language Model on One TPU v6e

**The question.** A language model must store ≤100M numbers, read ≤20M per token, train from scratch
in 24 h of one TPU v6e chip, and then be preferred blind to Falcon-H1-Tiny-90M-Instruct and
MobileLLM-R1.5-140M. The budget is stated in parameters. What actually sets the cost of training under
it, and how should a fixed wall-clock budget be spent and, when needed, re-spent?

**Thesis (one sentence).** Under a one-chip, one-day budget, parameter counts (stored or read per
token) do not predict training cost; bytes moved and the shapes they move in do. Treating bytes as
the cost recovers a large factor without changing the function, and an exact graft lets a run change
its design mid-budget without discarding what it has learnt.

**Contributions and the evidence for each.**

1. *Two ledgers and an audit (§4).* Stored / read-per-token accounting in a standard and a strict
   reading, enforced by tests on the real parameter tree and audited by a read-set reference forward.
   Evidence: `v9` 98,036,104 stored, 19,631,240 read (standard); strict 21.2–28.2M on 40 documents
   (fails, stated). `final` 98,300,111 / 19.28M. Measured, exact.
2. *Where training time goes on one TPU v6e (§7).* Evidence, all on the chip: glue = 62% of a block in
   the reference implementation; exact rewrites, device-side pieces and a one-tile causal kernel take
   the live rate from 112–117k to 281k tokens/s (2.4–2.5×); token pages that add 3.1% reads and 3.6%
   arithmetic cost 1.54–1.62× the time per token; 13 measured non-improvements; the compiler cost
   model 5% high on `v9`'s step and 2.7× low on `sorted`. The cost attribution by part is compile-only
   (stated as an estimate).
3. *An exact graft (§8).* Function-preserving continuation into a larger tree (U = 0, G = 0 neutral):
   equal losses, equal shared gradients, zero gradient on K, P_in, P_out at the first step; optimiser
   state, schedule and data plan carried; used live at p = 0.505 (7.79e9 tokens kept). Evidence: checks
   with difference 0.0 at steps 26,890 and 28,501 (same configuration hash as the live run); review
   with 4 tests. Not shown, and stated: whether the switch helped (no counterfactual, one seed,
   ≈2.8e9 fewer projected tokens, changed data stream).
4. *A protocol that cannot be tuned after the fact, failure-driven data hygiene, and an honest
   standing (§9, §11, §12).* Blind pairs, seeded sides, hash-chain sealed pair sets, two readings on
   exchanged sides with strict agreement, exact scores first, power analysis, matched controls.
   Refusal and deflection rules triggered by the model's own failures and measured in the sources
   (Tulu 3 4.85% refusals; system chats 29.1% and UltraChat 9.1% deflections). Mid-run: clearly
   behind Falcon-H1-Tiny (informal, 7 prompts). A deviation found while writing: 7 of the 8 mid-run
   sample prompts are test prompts; results will also be reported on the 193 untouched ones.

**What is not claimed.** That pages help at this scale; that the model beats either opponent;
mathematical novelty (review: 2–3/10).

**Still [TBD] (filled after 18:33 UTC training end and the 24–25 Nov evaluation).**
R1–R3 blind scores vs both opponents (overall, slices, reader diagnostics); R4 exact scores; R5
one-particle and matched controls; R6 held-out losses (export; stopped `final` vs `v9` on the same
rows); R7 preference optimisation outcome; R8 decoding selection; R9 training totals; R10 16-bit export
check; R11 decoding study (60 dev prompts, 6 settings); R12 results on the 193 untouched test prompts;
the check output of the live graft at step 29,722; on-chip effect of the `flat` switch; decoding
throughput of the export; the reader model; licence review; authors; commits of the uncommitted
files (`test_graft_run.py`, `flat`).

**Slots for the other agents.**
- Figures agent: Fig. 1 architecture with ledger annotations; Fig. 2 p and learning-rate factor vs wall
  clock; Fig. 3 training loss and tape gain vs tokens; Fig. 4 block time split; Fig. 5 live tokens/s vs
  step with change points; Fig. 6 `final` vs `v9` step times and the cost-model attribution; Fig. 7
  blind scores (after R1–R3). Data: `L/final_snap/run/train/train_log.jsonl`,
  `L/v9_snap/run/train/train_log.jsonl`, `docs/TPU_NOTES.md`, `docs/CHANGES.md` §19.1–19.6,
  `S/v9_meas*.txt`, `W/v9_speed.md` (L, S, W as in `CLAIMS.md`).
- Related-work agent: `paper.md` carries `[RW: topic; refs.md n]` markers mapped to the current
  numbering of `refs.md` (MoE active parameters 32; opponents 6, 7, 9–11; data movement 63; looped
  models, hyper-connections, value embeddings 14, 16–22, 5; sinks 42, 43; Pitman–Yor 38; Muon 51–54
  and the looped reference 21; time-tied schedules 1–3; DPO 74; flash/splash/Pallas 58–61;
  function-preserving growth 45–49; datasets 12, 76, 77; SMC steering 64–67; judge biases 78–81).
  Positioning added because of those references: data movement as the bottleneck is known on GPUs
  (63), so the claim is the measurement against a parameter budget on a TPU; staged training (47)
  already carries optimiser state and schedule through growth, so the graft's novelty is narrowed to
  the kind of added parameters, the wall-clock schedule and data plan, and the live numerical check.
