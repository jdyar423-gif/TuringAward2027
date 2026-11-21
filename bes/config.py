"""Central configuration for BES-100M-A20M.

Every shape, vocabulary id and feature cardinality that more than one module relies on is
defined here. Modules must import these constants instead of repeating literals.

Owner: the orchestrator. Other tracks may append fields to *their own* dataclass with a
default value (never rename or remove a field) and must mention it in their report.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- vocabulary
VOCAB_SIZE = 8192
N_SPECIAL = 16
#: Special tokens occupy the LAST ``N_SPECIAL`` ids: id = VOCAB_SIZE - N_SPECIAL + index.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>", "<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|library|>",
    "<|end|>", "<|calc|>", "<|eq|>", "<|/calc|>",
    "<|r10|>", "<|r11|>", "<|r12|>", "<|r13|>", "<|r14|>", "<|r15|>",
)
assert len(SPECIAL_TOKENS) == N_SPECIAL


def special_id(name: str, vocab_size: int = VOCAB_SIZE) -> int:
    return vocab_size - N_SPECIAL + SPECIAL_TOKENS.index(name)


PAD, BOS, SYSTEM, USER, ASSISTANT, LIBRARY, END, CALC, EQ, CALC_END = (
    special_id(t) for t in SPECIAL_TOKENS[:10]
)
ROLE_TOKENS = (SYSTEM, USER, ASSISTANT, LIBRARY)

# ----------------------------------------------------------------- exact-layer features
#: Register fields fed to the model: (name, cardinality). The exact layer delivers values
#: already clipped to [0, cardinality). Order is the order of the last axis of ``reg``.
REG_FIELDS: tuple[tuple[str, int], ...] = (
    ("role", 6),          # 0 none/raw text, 1 system, 2 user, 3 assistant, 4 library, 5 reserved
    ("fence", 2),         # inside a ``` code fence
    ("inline", 2),        # inside `inline code`
    ("quote", 2),         # inside a "double quote"
    ("depth", 8),         # bracket depth ([{ clipped to 7
    ("calc", 3),          # 0 outside, 1 in <|calc|> expression, 2 in result after <|eq|>
    ("tok_in_turn", 12),  # floor(log2(1 + tokens since turn start)), clipped
    ("word_in_sent", 64),
    ("sent_in_turn", 64),
    ("line_in_turn", 64),
    ("item_in_turn", 64),  # list items (bullets / numbered lines) since turn start
    ("turn_idx", 16),
    ("rem_sent", 34),     # 0 no target, 1 overshoot, 2+k remaining k (k clipped to 31)
    ("rem_item", 34),
    ("rem_word", 26),     # 0 none, 1 overshoot, 2..17 remaining 0..15, 18.. log2 buckets
)
N_REG = len(REG_FIELDS)

#: Integer clocks used as extra RoPE positions: (sentence index, line index, tokens in turn).
N_CLOCKS = 3

#: Precedent-candidate tape: per position ``CAND_SLOTS`` candidate next tokens with features.
CAND_SLOTS = 16
CAND_FEATS: tuple[tuple[str, int], ...] = (
    ("kind", 8),    # 0 empty, 1/2/3 n-gram order, 4 long match (>= 4 tokens), 5.. reserved
    ("count", 8),   # bucket of the successor count n_b
    ("total", 8),   # bucket of the context count n
    ("types", 8),   # bucket of K, the number of distinct successors
    ("src", 8),     # role of the most recent source: 0 raw, 1 system, 2 user, 3 assistant
                    # (earlier turn), 4 library, 5 same assistant turn (self-generated)
    ("mlen", 8),    # bucket of the match length
)
N_CAND_FEAT = len(CAND_FEATS)


# ------------------------------------------------------------------------------ model
@dataclass(frozen=True)
class ModelCfg:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 768
    n_blocks: int = 2            # weight-shared blocks applied in every loop
    n_heads: int = 6
    head_dim: int = 128
    ffn_mult: int = 4
    n_loops: int = 8
    n_lanes: int = 2             # hyper-connection lanes
    ve_dim: int = 128            # per-loop value-embedding width (multiple of head_dim)
    head_rank: int = 512
    softcap: float = 15.0
    max_seq: int = 2048
    rope_base: float = 10000.0
    bias_dims: int = 2           # unrotated, un-normalised dims per head: exact per-key biases
    clock_planes: int = 15       # rotary planes driven by structural clocks (0 = plain RoPE)
    use_sink: bool = True        # BOS positions act as a calibrated attention sink
    use_conf: bool = True        # last value dim is the constant 1 -> non-sink mass
    use_reg: bool = True         # register embeddings
    use_cand_in: bool = True     # candidate tape as input features
    use_cand_out: bool = True    # candidate tape in the output law (bes.model.predictive)
    cand_dim: int = 64
    critic_hidden: int = 256
    attn_impl: str = "xla"       # "xla" (dense, any backend) | "splash" (TPU)
    act_dtype: str = "float32"   # "bfloat16" on TPU
    remat: bool = False
    table_grad: str = "onehot"   # vocabulary-table gradients: "onehot" (chunked matmul) | "take" (scatter-add)
    loss_chunk: int = 8192       # rows of float32 [rows, V] logits per cross-entropy chunk
    splash_block_q: int = 512    # splash attention query tile (clipped to the padded length)
    splash_block_kv: int = 512   # splash attention key/value tile
    splash_block_kv_compute: int = 512  # inner key/value step of the splash kernels
    splash_fused_bwd: bool = False      # one fused dQ/dK/dV backward kernel instead of two
    splash_interpret: str = "auto"      # Pallas interpret mode: "auto" (= not on TPU) | "true" | "false"
    mass_offset: float = 6.0     # output law: log-mass a_k = bias + <h A, c_k> - mass_offset
    mass_log_cap: float = 10.0   # output law: m_k = exp(soft_cap(a_k, mass_log_cap)) * mask_k * src_scale
    ffn_width: int = 0           # feed-forward width; 0 = d_model * ffn_mult

    @property
    def d_ffn(self) -> int:
        return self.ffn_width if self.ffn_width > 0 else self.d_model * self.ffn_mult


# -------------------------------------------------------------------------- optimiser
@dataclass(frozen=True)
class OptimCfg:
    optimizer: str = "muonh"     # "muonh" | "muon"
    lr_matrix: float = 0.014
    lr_embed: float = 0.3
    lr_unembed: float = 0.004
    lr_norm: float = 0.005
    lr_scalar: float = 0.01
    lr_width_ref: int = 640      # Adam learning rates are multiplied by (d_model/ref)^-0.5 (reference: 640)
    adam_b1: float = 0.8
    adam_b2: float = 0.95
    adam_eps: float = 1e-10
    momentum: float = 0.95
    ns_steps: int = 5
    weight_decay: float = 0.0    # used by "muon" only (cautious weight decay)
    warmup_frac: float = 0.002
    warmdown_frac: float = 1.0   # linear decay over the last fraction of progress (reference: all of it)
    final_lr_frac: float = 0.1
    tokens_per_step: int = 262144
    micro_seqs: int = 32         # sequences per micro-batch at seq_len 1024
    grad_clip: float = 0.0       # 0 disables
    muon_beta2: float = 0.9      # "muon" only: EMA of the NorMuon second moment
    ns_dtype: str = "auto"       # Polar Express dtype: "auto" (= model.act_dtype) | "float32" | "bfloat16"
    width_scale_matrix: bool = False  # False: width factor on the Adam groups only (reference)
    matrix_layout: str = "in_out"     # matrices stored [..., d_in, d_out] (y = x @ W) | "out_in"
    muon_momentum_peak: float = 0.97      # "muon" only (nanochat schedule): momentum after warm-up
    muon_momentum_start: float = 0.85     # "muon" only: momentum of the first update
    muon_momentum_warmup_steps: int = 400  # "muon" only: linear start -> peak over these updates
    muon_momentum_end: float = 0.90       # "muon" only: linear peak -> end over the LR warm-down
    lr_feature: float = 0.01     # Adam "feature" group: register, candidate-input and trust-bias tables
    # Learning-rate transfer rule of the reference implementation (kschwethelm/looped-lm-scaling
    # scripts/base_train.py): the rates above are for batch_ref_tokens per step and a horizon of
    # data_ref_tokens; other settings are corrected by power laws.
    batch_ref_tokens: int = 262144   # tokens per step the learning rates above refer to
    batch_exponent: float = 0.558    # every group: lr x (tokens_per_step / batch_ref_tokens) ** exponent
    data_ref_tokens: float = 9.83e8  # training tokens lr_matrix refers to (10 x the reference model's
                                     # effective non-embedding parameters)
    data_exponent: float = -0.32     # matrix group: lr x (horizon_tokens / data_ref_tokens) ** exponent
    horizon_tokens: float = 0.0      # expected training tokens of this run; 0 = no data correction


# ------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class DataCfg:
    seq_len: int = 1024
    seq_len_late: int = 2048
    late_from: float = 0.5       # progress at which training switches to seq_len_late
    chat_frac: float = 0.25      # fraction of tokens from chat sources during pre-training
    chat_frac_tail: float = 0.7  # ... during the final chat-heavy phase
    tail_from: float = 0.92
    library_frac_base: float = 0.02   # share of the non-chat tokens given to library tasks (reading and
                                      # question-answer tasks, bes.data.rag_format, in library rows); the
                                      # library kind is capped at MixCfg.max_epochs passes over its entries
                                      # (lib_v2: ~42M tokens a pass, ~1.7% of a 2e10-token run), so this is
                                      # about what can be delivered (with MixCfg.library_weight in chat rows
                                      # the kind still asks for more; mixture.plan moves the surplus)
    library_frac_chat: float = 0.4    # probability that a chat or rag sample goes through the retrieval of
                                      # inference (rag_format.insert_retrieval: a question-like user message,
                                      # then the library's gate); only samples whose gate opens get a passage,
                                      # so far fewer chat samples carry one (Loader.stats() reports the share)
    distractor_frac: float = 0.35     # of the passages retrieval puts into chat samples (where the gate
                                      # opened): a distractor instead of the gated passage
    passage_loss_w: float = 0.25
    hindsight_frac: float = 0.12      # chat samples given a hindsight constraint
    assistant_only_from: float = 0.92  # from this progress on, loss on assistant tokens only
    permissive_only: bool = False
    use_library: bool = True


# -------------------------------------------------------------------------------- run
@dataclass(frozen=True)
class RunCfg:
    name: str = "final"
    seed: int = 0
    deadline_hours: float = 24.0
    post_reserve_hours: float = 4.5   # sampling, preference optimisation, evaluation, buffer
    ckpt_every_min: float = 25.0
    eval_every_min: float = 120.0
    restart_window_hours: float = 2.0
    out_dir: str = "runs/final"
    data_dir: str = "data_cache"
    use_external_models: bool = True  # reward / judge models for scoring only
    keep_ckpts: int = 3               # checkpoints kept on disk (newest first)
    log_every: int = 50               # optimiser steps between log records
    guard_spike: float = 1.3          # a loss above guard_spike * moving average is a spike
    guard_patience: int = 3           # consecutive bad steps (spike or non-finite) -> rollback
    guard_nonfinite_patience: int = 3  # consecutive non-finite steps -> rollback (1 = at once)
    guard_ema: float = 0.99           # decay of the loss moving average (good steps only)
    guard_warmup: int = 100           # good steps folded in before spikes are judged
    guard_skip_steps: int = 200       # data steps skipped after a rollback
    guard_lr_factor: float = 0.7      # learning-rate multiplier applied at every rollback
    guard_max_rollbacks: int = 3
    # -- appended by the supervisor track (scripts/run_24h.py, bes/train/{supervisor,stages,
    #    benchmark}.py); "run" in a path field means a directory inside out_dir
    switches: str = ""                # runtime feature switches, "sink:0 conf:0" (make_consts names;
                                      # "" = the static ModelCfg flags); set by the canary's ladder
    tok_root: str = ""                # root of tok/<source>/ shards: "" = data_dir, "run" = out_dir/data
    proxy_root: str = ""              # proxy-evaluation root: "" = data_dir/proxy, "run" = out_dir/proxy
    download_sources: str = ""        # "name[:max_files] ..." to download ("" = the mixture's sources
                                      # and the preference data at the registry caps; the library
                                      # build fetches its own sources)
    tokenize_max_records: int = 0     # records per input file when tokenising (0 = all)
    preflight_min: float = 3.0        # planned minutes of stage 0
    benchmark_min: float = 15.0       # planned minutes of the start-up benchmark and canary ...
    benchmark_cap_min: float = 25.0   # ... and its hard cap
    bench_plans: str = "full"         # "full" | "reduced" (CPU rehearsals: fewer candidate plans)
    bench_steps: int = 20             # timed steps per candidate plan after warm-up
    canary_steps: int = 30            # steps of the safe and of the full configuration in the canary
    loader_workers: int = 0           # loader worker processes (0 = from the host core plan)
    eval_opponents: str = "falcon=recommended mobilellm=recommended mobilellm=neutral"
    eval_judge: str = "qwen3-4b-2507"  # judge of the proxy comparison (registry name, gguf:PATH, standin-<rule>)
    eval_limit: int = 0               # prompts per split in the proxy evaluation (0 = all)
    sidecar: bool = True              # CPU side-car health reports during training


# ----------------------------------------------------------------------------- decode
@dataclass(frozen=True)
class DecodeCfg:
    temperature: float = 0.6
    top_p: float = 0.9
    particles: int = 6
    max_new_tokens: int = 768        # reply budget: 2048-token context leaves 1280 for the prompt
    ess_frac: float = 0.5
    no_repeat_ngram: int = 6
    max_copy_run: int = 0            # cap on verbatim copy runs from a LIBRARY passage; 0 = off (reading
                                     # tasks train the model to quote the lead sentences of a passage)
    self_evidence_weight: float = 0.25
    use_library: bool = True
    use_critic: bool = False
    # appended by the decoding track (bes/decode): see bes.decode.sampling / potentials / smc
    top_k: int = 0                    # 0 = off
    min_p: float = 0.0                # 0 = off
    truncate_after_mask: bool = True  # truncation inside the allowed set (weights untruncated)
    max_depth: int = 8                # bracket depth bound outside code (structural potential)
    budget_aware: bool = True         # masks keep every particle able to finish in max_new_tokens
    resample_at: str = "sentence"     # "sentence" (wait at sentence boundaries) | "token"
    unit_max_tokens: int = 32         # a resampling unit ends after this many tokens at most


@dataclass(frozen=True)
class Cfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    data: DataCfg = field(default_factory=DataCfg)
    run: RunCfg = field(default_factory=RunCfg)
    decode: DecodeCfg = field(default_factory=DecodeCfg)


# ---------------------------------------------------------------------------- helpers
def _coerce(old: Any, text: str) -> Any:
    if isinstance(old, bool):
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(text)
    if isinstance(old, float):
        return float(text)
    return text


def apply_overrides(cfg: Cfg, overrides: list[str] | tuple[str, ...]) -> Cfg:
    """Apply ``["model.n_loops=6", "optim.optimizer=muon"]`` style overrides."""
    for item in overrides:
        key, _, value = item.partition("=")
        section, _, name = key.strip().partition(".")
        sub = getattr(cfg, section)
        if not hasattr(sub, name):
            raise KeyError(f"unknown config field {key!r}")
        sub = dataclasses.replace(sub, **{name: _coerce(getattr(sub, name), value.strip())})
        cfg = dataclasses.replace(cfg, **{section: sub})
    return cfg


def load_config(path: str, overrides: list[str] | tuple[str, ...] = ()) -> Cfg:
    """Load ``configs/<name>.py`` (must define ``CFG``) and apply overrides."""
    spec = importlib.util.spec_from_file_location("bes_user_config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return apply_overrides(module.CFG, overrides)


def to_dict(cfg: Cfg) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def config_hash(cfg: Cfg) -> str:
    blob = json.dumps(to_dict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------- library
@dataclass(frozen=True)
class LibraryCfg:
    """Written library (``bes/library/``); values measured in gate E1 (docs/E1_COVERAGE.md).

    Appended by the library track; not (yet) a field of ``Cfg``. ``bes.library`` takes its
    defaults from here.
    """

    budget_numbers: int = 60_000_000   # text blocks + passage directory + title list
    index_numbers: int = 23_500_000    # stored truncated BM25 index (all of its tables); v2 sweep
    index_bits: int = 18               # 2**18 hashed term buckets
    index_max_list: int = 4096         # postings kept per bucket at most (bounds the read set)
    max_passage_words: int = 110
    max_passage_tokens: int = 160      # title line + body, measured with count_tokens
    block_raw_bytes: int = 65536       # decompressed bytes per independent xz block
    bm25_k1: float = 0.6
    bm25_b: float = 0.3
    title_weight: float = 1.0          # title terms count 1 + title_weight times
    lead_title_weight: float = 6.0     # ... in the first passage of an article
    max_query_terms: int = 16
    gate_min_score: float = 20.0       # BM25 units; gate chosen on dev at precision >= 0.35
    gate_min_margin: float = 0.1       # (s1 - s2) / s1 against the best other article
    gate_min_margin_title: float = 0.05  # margin needed when the title matches
    gate_min_title_match: float = 0.5  # share of title terms present in the query
    gate_min_coverage: float = 0.0     # s1 / best reachable score (0 = unused)
    library_dir: str = "data_cache/library/lib_v2"
    # -- v2: qa layer, title look-up, calibrated gate, budget (docs/E1_COVERAGE.md)
    total_numbers_cap: int = 64_000_000   # everything the library stores (text, index, qa, gate)
    max_read_numbers: int = 300_000       # stored numbers one retrieve() may read at most
    qa_bits: int = 17                     # term buckets of the qa index
    qa_read_postings: int = 16_384        # qa postings read per query at most
    qa_min_sym: float = 0.5               # qa candidates below this symmetric overlap are ignored
    qa_permissive_only: bool = False      # True: only qa sources with a clear licence (NQ-open)
    gate_precision: float = 0.5           # calibrated gate: dev precision floor of the threshold


# -------------------------------------------------------------- preference optimisation
@dataclass(frozen=True)
class PrefCfg:
    """Preference pairs and DPO (``bes/data/prefs.py``, ``bes/train/prefopt.py``; plan 5 item 2).

    Appended by the preference-optimisation track; not (yet) a field of ``Cfg``. The candidate
    learning rates are an argument of ``prefopt.run_prefopt`` (default ``(1e-6, 3e-6)``).
    """

    seq_len: int = 1024            # row length T: prompt + ASSISTANT + response + END must fit T+1
    pairs_per_micro: int = 16      # pairs per micro-batch (rows B = 2 * pairs_per_micro)
    n_micro: int = 4               # micro-batches A per optimiser step (64 pairs per step)
    beta: float = 0.1              # DPO temperature (published range at this scale 0.1-0.5)
    adam_b1: float = 0.9           # plain Adam for every trainable leaf, one uniform learning rate
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    warmup_frac: float = 0.1       # linear warm-up over this fraction of the epoch's steps ...
    final_lr_frac: float = 0.0     # ... then linear decay towards this fraction of the peak
    grad_clip: float = 1.0         # global gradient-norm clip (0 disables)
    heldout_frac: float = 0.05     # pairs held out (split by prompt) for the health checks
    seed: int = 0                  # shuffling and the held-out split
    max_len_ratio: float = 3.0     # length guard: drop pairs whose chosen response (with END) is
                                   # more than this times longer than the rejected one (0 = off)
    mask_forced: bool = True       # calculator results forced by the host carry no DPO weight
    max_loss_rise: float = 0.03    # health: held-out assistant loss may rise at most 3 %
    len_ratio_min: float = 0.7     # health: mean sampled response length / start value ...
    len_ratio_max: float = 1.6     # ... must stay in [len_ratio_min, len_ratio_max]
    rep_ratio_max: float = 1.2     # health: repeated 4-gram rate at most 1.2 x the start value
    min_acc_gain: float = 0.0      # health: held-out preference accuracy must exceed start + this
    cache_gb: float = 8.0          # host memory for feature batches kept between passes (0 = none)


@dataclass(frozen=True)
class CriticCfg:
    """Fitting the critic head on the frozen core (``bes/train/critic_fit.py``; plan 4G, F3).

    Appended by the preference-optimisation track; not (yet) a field of ``Cfg``.
    """

    steps: int = 300               # Adam steps on cached final-token hidden states
    lr: float = 1e-3
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    batch: int = 512               # scalar-labelled samples per step (0 = all of them)
    pair_batch: int = 512          # preference pairs per step (0 = all of them)
    pair_weight: float = 1.0       # weight of the pairwise ranking loss next to the scalar loss
    l2: float = 1e-4               # L2 penalty on both critic matrices
    heldout_frac: float = 0.2      # prompts held out for the rank-correlation report
    seed: int = 0
    min_rho: float = 0.3           # the critic is used only if the held-out rho reaches this
    seq_len: int = 1024            # row length when collecting hidden states
    batch_rows: int = 32           # rows per forward pass when collecting hidden states


# ---------------------------------------------------------------------- data mixture
@dataclass(frozen=True)
class MixCfg:
    """Sample transforms, library tasks and the data mixture (``bes/data/samples.py``,
    ``bes/data/rag_format.py``, ``bes/data/mixture.py``).

    Appended by the data-mixture track; not (yet) a field of ``Cfg``. Sub-kind weights are token
    shares of the conversations in chat rows, renormalised over the sub-kinds that have data on
    disk; library tasks also fill ``DataCfg.library_frac_base`` of the non-chat tokens. These are
    targets: ``mixture.plan`` caps every source at ``max_epochs`` reads, which holds the library
    tasks (lib_v2: ~42M tokens a pass) at about 1.7% of a 2e10-token run whatever they ask; the
    loader warns when a cap cuts a planned share by more than a third. Variant fractions are per
    built sample and are renormalised over the variants that are possible.
    """

    chat_weight: float = 0.57        # chat sources (SmolTalk family, Tulu 3, UltraChat, SYNTH); retrieval
                                     # is inserted into them (DataCfg.library_frac_chat)
    rag_weight: float = 0.20         # sources that bring their own passages (SQuAD v2, SYNTH RAG, ...)
    math_weight: float = 0.13        # math sources (GSM8K with calculator spans, Orca-Math)
    library_weight: float = 0.10     # library tasks built from the written library (capped by max_epochs)
    max_epochs: float = 8.0          # no source and no library entry is read more often than this
    read_weight: float = 0.6         # library tasks: reading tasks from "wiki" passages ...
    qa_weight: float = 0.4           # ... and question-answer tasks from "qa" entries
    read_open: float = 0.45          # reading task with its own passage as the LIBRARY turn
    read_closed: float = 0.35        # ... closed book (no LIBRARY turn)
    read_distractor: float = 0.20    # ... with a passage of another article as the LIBRARY turn
    read_one: float = 0.35           # reading-task answer: P(1 sentence) ...
    read_two: float = 0.40           # ... P(2 sentences) ...
    read_three: float = 0.25         # ... P(3 sentences) (renormalised over what the passage has)
    section_tasks: bool = True       # also ask about known section headings ("the history of X")
    qa_copy: float = 0.35            # question-answer task with the entry itself as LIBRARY turn
    qa_neighbour: float = 0.35       # ... with the nearest other stored question (leave-one-out)
    qa_closed: float = 0.30          # ... closed book
    distractor_random: float = 0.5   # share of distractors drawn uniformly (else a ranked neighbour)
    distractor_topk: int = 4         # ranks searched for a ranked distractor
    read_task_tokens: float = 66.0   # tokens of one pass of reading tasks per wiki entry (entries that give
                                     # no task count 0; lib_v2: 286,019 of 535,296 entries give one, 128 tokens
                                     # each, i.e. 68); the draws skip the entries rag_format.reading_suitable
                                     # rules out
    qa_task_tokens: float = 44.0     # mean tokens per question-answer task (lib_v2: 45.6)
    retrieval_later_turn: float = 0.5  # conversations with several user turns: probability that retrieval
                                       # serves a later user turn (uniform among them) instead of the first
    later_window: float = 0.5        # a conversation longer than T + 1: probability that a draw keeps one of
                                     # its later windows (samples.fit_draw) instead of its prefix


# --------------------------------------------------------------------------- loader
@dataclass(frozen=True)
class LoaderCfg:
    """Training loader (``bes/data/loader.py``, ``bes/data/packing.py``).

    Appended by the loader track; not (yet) a field of ``Cfg``. The fields marked (journal)
    decide the batches and are stored in the plan journal's header: a journal can only be
    reopened with the same values.
    """

    n_streams: int = 64              # (journal) virtual streams, a power of two; every row
                                     # belongs to one, so batches do not depend on the workers
    block_steps: int = 64            # (journal) steps per planned block (one journal record);
                                     # stage changes other than the shape apply at the next block
    lookahead: int = 32              # (journal) chat-row best-fit buffer per stream (samples)
    lib_lookahead: int = 16          # (journal) library-row best-fit buffer per stream
    max_wait: int = 128              # (journal) rows a buffered sample may wait before it is
                                     # forced to open the next row (0 = never forced)
    max_draw_attempts: int = 64      # (journal) failed draws (too long, declined) per refill
    reepoch_growth: float = 0.25     # (journal) a stream starts a new epoch of a source as soon as
                                     # its visible segments grew by this factor (0 = only at the end)
    workers: int = 2                 # worker processes when Loader(n_workers=None); 0 = in-process
    ring: int = 4                    # batches produced ahead (shared-memory slots) - back-pressure
    feature_rows: int = 64           # rows per exact_features call inside a worker (peak memory)
    horizon_tokens: float = 2.0e10   # token horizon of the run for the epoch caps until progress
                                     # p >= horizon_min_p allows the estimate tokens / p
    horizon_min_p: float = 0.05
    growing_blocks: int = 8          # a source with new shards within this many blocks is still
                                     # downloading: its epoch cap uses the registry's expected size
    task_timeout_s: float = 600.0    # a worker silent this long on pending work is replaced
    journal_name: str = "loader_journal.jsonl"  # under RunCfg.out_dir unless a path is given
    copy_out: bool = True            # return batches copied out of the shared ring
    stats_blocks: int = 32           # block summaries kept for Loader.stats()
    heldout_frac: float = 0.003      # (journal) share of the conversations of every chat, rag and math
                                     # source kept out of training by a hash of (source, shard, segment);
                                     # Loader.heldout / loader.heldout_samples read them
