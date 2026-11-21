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
    use_cand_out: bool = True    # candidate tape as output tilt
    cand_dim: int = 64
    critic_hidden: int = 256
    attn_impl: str = "xla"       # "xla" (dense, any backend) | "splash" (TPU)
    act_dtype: str = "float32"   # "bfloat16" on TPU
    remat: bool = False
    table_grad: str = "onehot"   # vocabulary-table gradients: "onehot" (chunked matmul) | "take" (scatter-add)
    loss_chunk: int = 8192       # rows of float32 [rows, V] logits per cross-entropy chunk

    @property
    def d_ffn(self) -> int:
        return self.d_model * self.ffn_mult


# -------------------------------------------------------------------------- optimiser
@dataclass(frozen=True)
class OptimCfg:
    optimizer: str = "muonh"     # "muonh" | "muon"
    lr_matrix: float = 0.014
    lr_embed: float = 0.3
    lr_unembed: float = 0.004
    lr_norm: float = 0.005
    lr_scalar: float = 0.01
    lr_width_ref: int = 512      # all learning rates are multiplied by (d_model/ref)^-0.5
    adam_b1: float = 0.8
    adam_b2: float = 0.95
    adam_eps: float = 1e-10
    momentum: float = 0.95
    ns_steps: int = 5
    weight_decay: float = 0.0    # used by "muon" only (cautious weight decay)
    warmup_frac: float = 0.01
    warmdown_frac: float = 0.65  # linear decay over the last fraction of progress
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


# ------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class DataCfg:
    seq_len: int = 1024
    seq_len_late: int = 2048
    late_from: float = 0.5       # progress at which training switches to seq_len_late
    chat_frac: float = 0.25      # fraction of tokens from chat sources during pre-training
    chat_frac_tail: float = 0.7  # ... during the final chat-heavy phase
    tail_from: float = 0.92
    library_frac_base: float = 0.15   # base sequences that carry a retrieved passage
    library_frac_chat: float = 0.4
    distractor_frac: float = 0.35     # of library sequences: passage is a distractor
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


# ----------------------------------------------------------------------------- decode
@dataclass(frozen=True)
class DecodeCfg:
    temperature: float = 0.6
    top_p: float = 0.9
    particles: int = 6
    max_new_tokens: int = 384
    ess_frac: float = 0.5
    no_repeat_ngram: int = 6
    max_copy_run: int = 12
    self_evidence_weight: float = 0.25
    use_library: bool = True
    use_critic: bool = False


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
