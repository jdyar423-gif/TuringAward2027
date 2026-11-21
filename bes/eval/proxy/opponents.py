"""Opponent and judge models: where they come from, how to verify them, how to run them.

Every file is pinned by repository, file name, size and SHA-256 (checked against
``https://huggingface.co/api/models/<repo>/tree/main`` when the entry was written) and lives in
``data_cache/opponents/``. :func:`fetch` downloads a missing file, verifies it and applies the
entry's conversion.

**MobileLLM-R1.5-140M.** The upstream checkpoint (``facebook/MobileLLM-R1.5-140M``) is gated;
the ungated community GGUF ``sasa2000/MobileLLM-R1.5-140M-Q8_0-GGUF`` stores it as a dense
``llama4`` model (``expert_count = 0``), which the llama.cpp bundled with llama-cpp-python
0.3.37 refuses to load ("llama4 model cannot have zero experts"; patching the expert count only
reaches a failing assertion on the attention-temperature input). The architecture is a plain
dense transformer: grouped-query attention, interleaved-pair RoPE (theta 8e6) on every layer,
a weightless RMS ("L2") norm of each query and key head *after* RoPE, SwiGLU MLP, tied
embeddings. :func:`convert_llama4_dense_to_qwen3` rewrites it, without touching any quantised
value, as the equivalent ``qwen3`` model: hyper-parameters renamed, the rows of every query and
key head reordered from interleaved pairs to half-split pairs (llama.cpp ``qwen3`` uses NEOX
RoPE), and unit-weight per-head q/k RMS norms added (``qwen3`` applies them before RoPE; a
weightless per-head RMS norm commutes with RoPE because RoPE is a rotation of each head). The
file also lacks a chat template; :data:`MOBILELLM_TEMPLATE` reproduces the official template's
rendering for conversations without tools (it injects the system prompt "Please reason step by
step, and put your final answer within \\boxed{}." when none is given). The conversion was
checked against an independent numpy forward pass of the original tensors with the upstream
(Llama4-text) semantics: see ``tests/eval/test_proxy_gguf_convert.py``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bes.config import DecodeCfg
from bes.eval.proxy import gguf_file
from bes.eval.proxy.common import GenSettings, file_sha256, models_dir

NEUTRAL_SYSTEM = "You are a helpful assistant."

#: Chat template for the MobileLLM-R1.5 GGUF (which ships none): identical output to the
#: official template of the R1.5 checkpoints for system / user / assistant messages.
MOBILELLM_TEMPLATE = r'''{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] | trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "Please reason step by step, and put your final answer within \\boxed{}." %}
{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" + system_message + "<|eot_id|>" }}
{%- for message in messages %}
    {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\n\n" + message['content'] | trim + "<|eot_id|>" }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- "<|start_header_id|>assistant<|end_header_id|>\n\n" }}
{%- endif %}
'''


@dataclass(frozen=True)
class ModelFile:
    name: str                # registry name used on the command line
    role: str                # "opponent" | "judge" | "test-judge"
    upstream: str            # the original model
    repo: str                # repository the GGUF comes from
    filename: str
    size: int
    sha256: str
    license: str
    convert: str | None = None         # conversion applied after download
    local_name: str | None = None      # file name of the converted model
    converted_sha256: str | None = None  # the conversion is deterministic: expected output hash
    chat_template: str | None = None   # used when the GGUF has no template
    n_ctx: int = 4096
    note: str = ""

    def source_path(self, base: Path | None = None) -> Path:
        return Path(base or models_dir()) / self.filename

    def path(self, base: Path | None = None) -> Path:
        """The file to load (the converted one when the entry has a conversion)."""
        return Path(base or models_dir()) / (self.local_name or self.filename)

    @property
    def key(self) -> str:
        """Cache key: registry name + the first 12 hex digits of the source file hash."""
        return f"{self.name}-{self.sha256[:12]}"


MODELS: dict[str, ModelFile] = {m.name: m for m in (
    ModelFile(
        name="falcon", role="opponent", upstream="tiiuae/Falcon-H1-Tiny-90M-Instruct",
        repo="tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF",
        filename="Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf", size=98_353_792,
        sha256="e60034381f4a0e448a5e6fdab29d7b40768a587c9a56e4cf85285455ba65d476",
        license="Falcon-LLM License", n_ctx=4096,
        note="Official GGUF of the publisher. Hybrid attention + Mamba; chat template in the file."),
    ModelFile(
        name="mobilellm", role="opponent", upstream="facebook/MobileLLM-R1.5-140M (gated)",
        repo="sasa2000/MobileLLM-R1.5-140M-Q8_0-GGUF", filename="mobilellm-r1.5-140m-q8_0.gguf",
        size=156_893_856,
        sha256="f680f0533368cc6bf8a314300f1f56f8a29c27651b45110e6e0b042b7fb9d623",
        license="FAIR Noncommercial Research License", convert="llama4_dense_to_qwen3",
        local_name="mobilellm-r1.5-140m-q8_0.qwen3.gguf", chat_template=MOBILELLM_TEMPLATE,
        converted_sha256="7cffe243c52914f62b2eb0211e04d70e6310cd1c18c398ac31f341ec33ee570a",
        n_ctx=8192,
        note="Community GGUF-my-repo conversion of the gated checkpoint; rewritten as qwen3."),
    ModelFile(
        name="qwen3-4b-2507", role="judge", upstream="Qwen/Qwen3-4B-Instruct-2507",
        repo="unsloth/Qwen3-4B-Instruct-2507-GGUF", filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        size=2_497_281_120,
        sha256="3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
        license="Apache-2.0", n_ctx=8192,
        note="Default judge for the real run: non-thinking 4B instruct model, dense attention."),
    ModelFile(
        name="gemma4-e4b", role="judge", upstream="google/gemma-4-E4B-it",
        repo="unsloth/gemma-4-E4B-it-GGUF", filename="gemma-4-E4B-it-Q4_K_M.gguf",
        size=4_977_171_584,
        sha256="85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87",
        license="Apache-2.0", n_ctx=8192,
        note="Second judge from another model family (self-preference check); not run here."),
    ModelFile(
        name="llama31-8b", role="judge", upstream="meta-llama/Llama-3.1-8B-Instruct",
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", size=4_920_739_232,
        sha256="7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c",
        license="Llama 3.1 Community License", n_ctx=8192,
        note="Alternative 8B judge (older, slower); not run here."),
    ModelFile(
        name="qwen2.5-0.5b", role="test-judge", upstream="Qwen/Qwen2.5-0.5B-Instruct",
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF", filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size=491_400_032,
        sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        license="Apache-2.0", n_ctx=8192,
        note="Small judge for plumbing tests on a 4-core machine; far too weak for real verdicts."),
)}


# --------------------------------------------------------------------------- settings presets
def common_settings() -> GenSettings:
    """The setting shared by every model: our decoder's base temperature and top-p
    (``bes.config.DecodeCfg``), no other filter, no repetition penalty."""
    d = DecodeCfg()
    return GenSettings(temperature=d.temperature, top_p=d.top_p, top_k=0, min_p=0.0,
                       repeat_penalty=1.0, max_tokens=1024)


#: llama.cpp's defaults: what ``llama-cli -cnv`` (the route Falcon's model card documents) uses.
LLAMA_CPP_DEFAULTS = dict(temperature=0.8, top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.0)

#: MobileLLM-R1's own evaluation script (facebookresearch/MobileLLM-R1, evaluation/run_eval.sh)
#: samples with temperature 0.6 and top-p 0.95 through the chat template.
MOBILELLM_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=0, min_p=0.0, repeat_penalty=1.0)
#: Reasoning tokens before budget forcing (the publisher's evaluation allows 30,000 tokens per
#: call; on CPU the harness caps a reply at 4096 tokens, leaving 1024 for the answer).
MOBILELLM_THINK_BUDGET = 3072


def presets(model: str) -> dict[str, GenSettings]:
    """Named settings of a registered opponent.

    - Falcon ``recommended``: llama.cpp defaults (the publisher documents ``llama-cli -cnv`` and
      publishes no sampling parameters; ``generation_config.json`` holds token ids only).
    - MobileLLM ``recommended``: its own template (which injects the boxed-answer reasoning
      system prompt), temperature 0.6, top-p 0.95, reasoning hidden, 4096 tokens in total with
      budget forcing: a reasoning block still open after :data:`MOBILELLM_THINK_BUDGET` tokens
      is closed and the model answers with the rest (without it, most chat prompts end inside
      the reasoning with no visible answer). ``neutral``: the same with the system prompt
      "You are a helpful assistant." instead of the injected one. ``noforce``: the recommended
      arm without budget forcing, for reference.
    - ``common``: :func:`common_settings`; for MobileLLM with the neutral system prompt (its
      template would otherwise inject the reasoning prompt), hidden reasoning, 4096 tokens and
      budget forcing.
    - ``greedy``: temperature 0, for sweeps.
    """
    common = common_settings()
    if model == "falcon":
        return {"recommended": GenSettings(max_tokens=1024, **LLAMA_CPP_DEFAULTS),
                "common": common,
                "greedy": GenSettings(temperature=0.0, top_p=1.0, max_tokens=1024)}
    if model == "mobilellm":
        force = {"think_budget": MOBILELLM_THINK_BUDGET}
        rec = GenSettings(max_tokens=4096, strip_think=True, extra=force, **MOBILELLM_SAMPLING)
        return {"recommended": rec,
                "neutral": rec.replace(system_prompt=NEUTRAL_SYSTEM),
                "noforce": rec.replace(extra={}),
                "common": common.replace(max_tokens=4096, strip_think=True, system_prompt=NEUTRAL_SYSTEM,
                                         extra=force),
                "greedy": GenSettings(temperature=0.0, top_p=1.0, max_tokens=4096, strip_think=True,
                                      system_prompt=NEUTRAL_SYSTEM, extra=force)}
    raise KeyError(f"no presets for {model!r}")


# ---------------------------------------------------------------------------------- fetching
def verify(path: Path, entry: ModelFile) -> bool:
    return path.exists() and path.stat().st_size == entry.size and file_sha256(path) == entry.sha256


def fetch(name: str, base: Path | None = None, download: bool = True, log=print) -> Path:
    """Path of a ready-to-load file for registry entry ``name``: verified source (downloaded
    when missing and ``download``), converted when the entry needs it."""
    entry = MODELS[name]
    base = Path(base or models_dir())
    src = entry.source_path(base)
    if not src.exists():
        if not download:
            raise FileNotFoundError(f"{src} missing (run: python -m bes.eval.proxy fetch {name})")
        from huggingface_hub import hf_hub_download
        log(f"downloading {entry.repo}/{entry.filename} ({entry.size / 1e6:.0f} MB)")
        hf_hub_download(entry.repo, entry.filename, local_dir=str(base))
    if not verify(src, entry):
        raise ValueError(f"{src}: size or SHA-256 differs from the registry entry for {name}")
    if entry.convert is None:
        return src
    dst = entry.path(base)
    if dst.exists() and entry.converted_sha256 and file_sha256(dst) != entry.converted_sha256:
        log(f"{dst.name} differs from the expected conversion; converting again")
        dst.unlink()
    if not dst.exists():
        if entry.convert != "llama4_dense_to_qwen3":
            raise ValueError(f"unknown conversion {entry.convert!r}")
        log(f"converting {src.name} -> {dst.name}")
        convert_llama4_dense_to_qwen3(src, dst, chat_template=entry.chat_template,
                                      name_suffix=" (qwen3 layout)")
        if entry.converted_sha256 and file_sha256(dst) != entry.converted_sha256:
            raise ValueError(f"{dst}: conversion output differs from the pinned SHA-256 "
                             "(update converted_sha256 if the conversion was changed on purpose)")
    return dst


# -------------------------------------------------------------------------------- conversion
_DROP_LLAMA4_KEYS = ("expert_count", "expert_used_count", "interleave_moe_layer_step",
                     "expert_feed_forward_length", "attention.sliding_window")


def interleaved_to_halves(n_heads: int, head_dim: int) -> np.ndarray:
    """Row order that turns interleaved RoPE pairs ``(2i, 2i+1)`` of every head into half-split
    pairs ``(i, i + head_dim/2)``: new row ``h*d + i`` is old row ``h*d + 2i`` and new row
    ``h*d + d/2 + i`` is old row ``h*d + 2i + 1``."""
    per = np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])
    return np.concatenate([h * head_dim + per for h in range(n_heads)])


def convert_llama4_dense_to_qwen3(src: str | Path, dst: str | Path, chat_template: str | None = None,
                                  name_suffix: str = "") -> None:
    """Rewrite a dense (expert-free) ``llama4`` GGUF as an equivalent ``qwen3`` GGUF (see the
    module docstring). Raises ``ValueError`` if the source is not a dense llama4 model with QK
    norm semantics this conversion assumes (all layers rotary, no experts)."""
    g = gguf_file.read(src)
    if g.get("general.architecture") != "llama4":
        raise ValueError("source is not a llama4 GGUF")
    if g.get("llama4.expert_count", 0) != 0 or g.get("llama4.interleave_moe_layer_step", 0) != 0:
        raise ValueError("source has experts; only dense llama4 models are supported")
    if g.get("llama4.attention.sliding_window", None) not in (0, None):
        raise ValueError("source uses chunked attention; not supported")
    n_layers = int(g.get("llama4.block_count"))
    n_head = int(g.get("llama4.attention.head_count"))
    n_kv = int(g.get("llama4.attention.head_count_kv"))
    head_dim = int(g.get("llama4.attention.key_length"))
    if int(g.get("llama4.rope.dimension_count", head_dim)) != head_dim:
        raise ValueError("partial rotary dimensions are not supported")

    kv: dict[str, gguf_file.Value] = {}
    for k, v in g.kv.items():
        if k == "general.architecture":
            kv[k] = gguf_file.Value(gguf_file.STRING, "qwen3")
        elif k == "general.name" and name_suffix:
            kv[k] = gguf_file.Value(gguf_file.STRING, str(v.value) + name_suffix)
        elif k.startswith("llama4."):
            rest = k[len("llama4."):]
            if rest not in _DROP_LLAMA4_KEYS:
                kv["qwen3." + rest] = v
        else:
            kv[k] = v
    if chat_template and "tokenizer.chat_template" not in kv:
        kv["tokenizer.chat_template"] = gguf_file.Value(gguf_file.STRING, chat_template)

    q_order = interleaved_to_halves(n_head, head_dim)
    k_order = interleaved_to_halves(n_kv, head_dim)
    items = []
    for info in g.tensors:
        if info.name.endswith(".attn_q.weight"):
            items.append((info, lambda i=info: gguf_file.permute_rows(g.tensor_bytes(i), i, q_order)))
        elif info.name.endswith(".attn_k.weight"):
            items.append((info, lambda i=info: gguf_file.permute_rows(g.tensor_bytes(i), i, k_order)))
        else:
            items.append((info, lambda i=info: g.tensor_bytes(i)))
    ones = np.ones(head_dim, dtype=np.float32).tobytes()
    names = {t.name for t in g.tensors}
    for layer in range(n_layers):
        for norm in ("attn_q_norm", "attn_k_norm"):
            name = f"blk.{layer}.{norm}.weight"
            if name in names:
                raise ValueError(f"source already has {name}")
            items.append((gguf_file.TensorInfo(name, (head_dim,), gguf_file.GGML_F32), lambda: ones))
    gguf_file.write(dst, kv, items, version=g.version, alignment=g.alignment)


def describe(entries: Mapping[str, ModelFile] | None = None, base: Path | None = None) -> list[dict[str, Any]]:
    """Registry rows with local availability (for ``python -m bes.eval.proxy models``)."""
    rows = []
    for e in (entries or MODELS).values():
        rows.append({"name": e.name, "role": e.role, "repo": e.repo, "file": e.filename,
                     "size_gb": round(e.size / 1e9, 3), "license": e.license,
                     "present": e.source_path(base).exists(),
                     "converted": e.path(base).exists() if e.convert else None, "note": e.note})
    return rows
