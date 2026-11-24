## Appendix N. Related work in detail

The second draft's related-work section, kept here in full; Section 3 is its summary.

**Training to a deadline.** Cramming trains a masked language model on one consumer GPU in one day
[1], Izsak et al. a BERT on one 8-GPU server in 24 hours [2]; both tie the learning-rate decay to the
time budget, as budgeted training recommends [3]. Speedruns minimise time to a target on an 8 × H100
node [4, 5]; nanochat runs the whole chat pipeline [5]. Our schedule is of this kind; what differs is
the product, a chat model trained on one chip and compared blind with released models, and the need
for the schedule to survive a reclaimed runtime and a change of configuration.

**Small open models.** MobileLLM found deep-and-thin shapes and shared embeddings to matter below 10⁹
parameters [6]; MobileLLM-R1 pre-trains on 4.2 × 10¹² tokens [7] and R1.5 adds on-policy distillation
[8] from a 4B teacher [9]. Falcon-H1 runs attention and state-space heads in parallel [10]; its 90M
instruct model saw 8 × 10¹¹ tokens with instruction data from the start [11]. SmolLM2 trains 135M and
360M models on 2 × 10¹² and 4 × 10¹² tokens [12]; TinyStories showed fluency below 10⁷ parameters in a
narrow domain [13].

**Weight-shared depth.** Re-applying one block goes back to the Universal Transformer [14] and ALBERT
[15]; looped models match deeper ones on synthetic reasoning [16] and have been scaled to 3.5 × 10⁹
parameters [17] and 7.7 × 10¹² tokens [18], and Parcae stabilises them [19]. Schwethelm et al. fit an iso-depth law L = E + A(N_once + r^φ N_rec)^(−α)
+ B D^(−β), in which shared recurrences lose to unique layers at equal compute and two-lane
hyper-connections [20] raise φ from 0.46 to 0.65 [21]; our core is ported from their code. Our reason to
loop differs: under a read budget a re-applied weight counts once, so looping buys compute per number
read, not per FLOP; and our core has no prelude or coda (N_once = 0), outside the law's fitted range.
The per-loop value embeddings follow [4, 5, 22].

**Vocabulary-indexed parameters and two counts.** Stored-but-sparsely-read capacity includes hashed
feed-forward weights [23], key-value memories [24, 25], larger input vocabularies [26], n-gram
embeddings off the accelerator [27, 28], and L3's per-token keys and values [29], to which `v9`'s token
neurons are closest. Token-keyed parameters were found most useful on values and outputs and least on
queries and keys [30, 31], a caution for our output gains. Counting the numbers read per token is the
mixture-of-experts convention [32, 33]; we found no published counterpart of a read-set audit.

**Copying and exact counts.** Mixing the network's distribution with a copy distribution goes back to
pointer-sentinel models and the continuous cache [34, 35], with neighbours and n-gram counts in kNN-LM
and infini-gram [36, 37]; our output law contains the Pitman–Yor predictive [38] with the network as
base measure, a known rule. Induction heads [39], n-gram heads [40] and self-reinforcing repetitions
[41] bear on the tape; the sink follows [42, 43]; `final`'s library was counted against the size
budget, as in EfficientQA [44].

**Changing a model during training.** Net2Net [45] and network morphism [46] grow networks by
function-preserving transformations. Staged training applies growth operators to the whole training
state, copying Adam moments for existing layers and zeroing them for new ones, and asks that both the
loss and the training dynamics be preserved [47]; LLaMA Pro [48], sparse upcycling [49] and LoRA [50]
start new paths at an identity or at zero. Our graft is a loss-preserving operator of this family. What
is particular is narrow: vocabulary-indexed tables and a feed-forward branch are added rather than width
or depth; the continued schedule and data plan are those of a wall-clock budget; and the function and
the shared gradients were checked at the live run's checkpoint. The training-dynamics property of [47]
is not tested.

**Optimisers, kernels, decoding, judging.** We use MuonH [51–57] as [21] does. Our kernels follow the
IO-aware pattern of FlashAttention [58, 59] and the Pallas kernels of JAX [60–62]; Ivanov et al. found
transformer training on GPUs memory-bound [63]. We add a measurement against a budget stated in
parameters, on one TPU v6e, not a new algorithm. The decoder builds on sequential Monte Carlo steering
[64–68], best-of-n bounds [69], self-consistency [70], IFEval-style constraints [71], calculator
annotations [72] and budget forcing [73]; DPO [74] with UltraFeedback is a standard last stage
[11, 12, 75–77]. Language-model judges favour one position, longer replies and their own style [78–81];
we count a win only when both orders agree, the rule of [78], and report outcomes by length ratio [80].
