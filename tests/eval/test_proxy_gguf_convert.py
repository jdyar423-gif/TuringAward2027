"""GGUF reader / writer and the MobileLLM (dense llama4 -> qwen3) conversion."""
from __future__ import annotations

import numpy as np
import pytest

from bes.eval.proxy import gguf_file as G
from bes.eval.proxy import opponents as O


def quantize_q8_0(x: np.ndarray) -> bytes:
    """Reference Q8_0 quantiser: blocks of 32 values, fp16 scale max|x|/127, int8 values."""
    blocks = x.astype(np.float32).reshape(-1, 32)
    out = bytearray()
    for b in blocks:
        d = np.float16(np.abs(b).max() / 127.0 if np.abs(b).max() > 0 else 0.0)
        q = np.zeros(32, np.int8) if float(d) == 0 else np.clip(np.round(b / np.float32(d)), -127, 127).astype(np.int8)
        out += d.tobytes() + q.tobytes()
    return bytes(out)


def make_llama4(path, n_embd=32, vocab=64, n_head=2, n_kv=1, head_dim=8, layers=2, experts=0, seed=0):
    rng = np.random.default_rng(seed)
    kv = {
        "general.architecture": G.Value(G.STRING, "llama4"),
        "general.name": G.Value(G.STRING, "tiny"),
        "llama4.block_count": G.Value(G.UINT32, layers),
        "llama4.embedding_length": G.Value(G.UINT32, n_embd),
        "llama4.attention.head_count": G.Value(G.UINT32, n_head),
        "llama4.attention.head_count_kv": G.Value(G.UINT32, n_kv),
        "llama4.attention.key_length": G.Value(G.UINT32, head_dim),
        "llama4.rope.dimension_count": G.Value(G.UINT32, head_dim),
        "llama4.rope.freq_base": G.Value(G.FLOAT32, 10000.0),
        "llama4.expert_count": G.Value(G.UINT32, experts),
        "llama4.interleave_moe_layer_step": G.Value(G.UINT32, 0),
        "llama4.attention.sliding_window": G.Value(G.UINT32, 0),
        "tokenizer.ggml.tokens": G.Value(G.ARRAY, [f"t{i}" for i in range(vocab)], G.STRING),
        "tokenizer.ggml.scores": G.Value(G.ARRAY, [float(i) for i in range(4)], G.FLOAT32),
        "tokenizer.ggml.add_bos_token": G.Value(G.BOOL, False),
    }
    tensors = {}

    def add(name, dims, q8=True):
        data = rng.standard_normal(int(np.prod(dims))).astype(np.float32)
        info = G.TensorInfo(name, dims, G.GGML_Q8_0 if q8 else G.GGML_F32)
        tensors[name] = (info, quantize_q8_0(data) if q8 else data.tobytes())

    add("token_embd.weight", (n_embd, vocab))
    add("output_norm.weight", (n_embd,), q8=False)
    for layer in range(layers):
        add(f"blk.{layer}.attn_norm.weight", (n_embd,), q8=False)
        add(f"blk.{layer}.attn_q.weight", (n_embd, n_head * head_dim))
        add(f"blk.{layer}.attn_k.weight", (n_embd, n_kv * head_dim))
        add(f"blk.{layer}.attn_v.weight", (n_embd, n_kv * head_dim))
        add(f"blk.{layer}.ffn_up.weight", (n_embd, 64))
    G.write(path, kv, [(info, (lambda d=data: d)) for info, data in tensors.values()])
    return kv, tensors


def test_roundtrip(tmp_path):
    kv, tensors = make_llama4(tmp_path / "a.gguf")
    g = G.read(tmp_path / "a.gguf")
    assert g.version == 3 and g.data_start % 32 == 0
    assert {k: (v.vtype, v.value) for k, v in g.kv.items()} == {k: (v.vtype, v.value) for k, v in kv.items()}
    for info in g.tensors:
        assert g.tensor_bytes(info) == tensors[info.name][1]
        assert info.offset % 32 == 0
    emb = g.tensor_array(g.tensor("token_embd.weight"))
    assert emb.shape == (64, 32)
    # rewriting the parsed file reproduces it byte for byte
    G.write(tmp_path / "b.gguf", g.kv, [(t, (lambda t=t: g.tensor_bytes(t))) for t in g.tensors])
    assert (tmp_path / "a.gguf").read_bytes() == (tmp_path / "b.gguf").read_bytes()
    (tmp_path / "not_gguf.bin").write_bytes(b"nope, not a model file")
    with pytest.raises(ValueError):
        G.read(tmp_path / "not_gguf.bin")


def test_permute_rows_q8_0_is_exact(tmp_path):
    make_llama4(tmp_path / "a.gguf")
    g = G.read(tmp_path / "a.gguf")
    info = g.tensor("blk.0.attn_q.weight")
    order = np.random.default_rng(1).permutation(info.dims[1])
    permuted = G.permute_rows(g.tensor_bytes(info), info, order)
    raw = np.frombuffer(permuted, np.uint8).reshape(-1, 34)
    scale = raw[:, :2].copy().view(np.float16).astype(np.float32)
    vals = (scale * raw[:, 2:].copy().view(np.int8).astype(np.float32)).reshape(info.dims[1], info.dims[0])
    np.testing.assert_array_equal(vals, g.tensor_array(info)[order])
    with pytest.raises(ValueError):
        G.permute_rows(g.tensor_bytes(info), info, np.zeros(info.dims[1], int))


def rope_interleaved(x, pos, theta):
    d = x.shape[-1]
    ang = pos * theta ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(ang), np.sin(ang)
    out = x.copy()
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return out


def rope_halves(x, pos, theta):
    d = x.shape[-1]
    h = d // 2
    ang = pos * theta ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(ang), np.sin(ang)
    out = x.copy()
    out[..., :h] = x[..., :h] * c - x[..., h:] * s
    out[..., h:] = x[..., :h] * s + x[..., h:] * c
    return out


def test_row_order_turns_interleaved_rope_into_half_split_rope():
    rng = np.random.default_rng(2)
    n_head, d = 3, 16
    order = O.interleaved_to_halves(n_head, d)
    assert sorted(order.tolist()) == list(range(n_head * d))
    for pos in (0, 1, 7, 1234):
        q = rng.standard_normal(n_head * d)
        k = rng.standard_normal(n_head * d)
        q_old = rope_interleaved(q.reshape(n_head, d), pos, 8e6)
        k_old = rope_interleaved(k.reshape(n_head, d), pos + 3, 8e6)
        q_new = rope_halves(q[order].reshape(n_head, d), pos, 8e6)
        k_new = rope_halves(k[order].reshape(n_head, d), pos + 3, 8e6)
        # same vectors up to the within-head permutation, so identical attention logits
        np.testing.assert_allclose(q_new.reshape(-1), q_old.reshape(-1)[order], atol=1e-12)
        np.testing.assert_allclose((q_new * k_new).sum(-1), (q_old * k_old).sum(-1), atol=1e-10)
        # a weightless per-head RMS norm commutes with either rotation
        def rms(x):
            return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(rms(q_old), rope_interleaved(rms(q.reshape(n_head, d)), pos, 8e6), atol=1e-12)


def test_convert_synthetic_llama4(tmp_path):
    _, tensors = make_llama4(tmp_path / "src.gguf")
    O.convert_llama4_dense_to_qwen3(tmp_path / "src.gguf", tmp_path / "dst.gguf", chat_template="{{ x }}",
                                    name_suffix=" (qwen3 layout)")
    g = G.read(tmp_path / "dst.gguf")
    assert g.get("general.architecture") == "qwen3"
    assert g.get("general.name") == "tiny (qwen3 layout)"
    assert g.get("qwen3.block_count") == 2 and g.get("qwen3.attention.head_count_kv") == 1
    assert not any(k.startswith("llama4.") for k in g.kv)
    assert "qwen3.expert_count" not in g.kv and "qwen3.attention.sliding_window" not in g.kv
    assert g.get("tokenizer.chat_template") == "{{ x }}" and g.get("tokenizer.ggml.tokens")[3] == "t3"
    src = G.read(tmp_path / "src.gguf")
    for layer in range(2):
        for name, heads in ((f"blk.{layer}.attn_q.weight", 2), (f"blk.{layer}.attn_k.weight", 1)):
            order = O.interleaved_to_halves(heads, 8)
            np.testing.assert_array_equal(g.tensor_array(g.tensor(name)), src.tensor_array(src.tensor(name))[order])
        for norm in ("attn_q_norm", "attn_k_norm"):
            np.testing.assert_array_equal(g.tensor_array(g.tensor(f"blk.{layer}.{norm}.weight")), np.ones(8, np.float32))
        assert g.tensor_bytes(g.tensor(f"blk.{layer}.attn_v.weight")) == tensors[f"blk.{layer}.attn_v.weight"][1]
    make_llama4(tmp_path / "moe.gguf", experts=4)
    with pytest.raises(ValueError):
        O.convert_llama4_dense_to_qwen3(tmp_path / "moe.gguf", tmp_path / "x.gguf")


def test_registry_and_presets():
    for entry in O.MODELS.values():
        assert len(entry.sha256) == 64 and entry.size > 0 and entry.repo and entry.filename.endswith(".gguf")
        if entry.role == "judge":
            assert entry.size < 6e9
    falcon = O.presets("falcon")["recommended"]
    assert (falcon.temperature, falcon.top_k, falcon.top_p, falcon.min_p) == (0.8, 40, 0.95, 0.05)
    mob = O.presets("mobilellm")
    assert mob["recommended"].system_prompt is None and mob["neutral"].system_prompt == O.NEUTRAL_SYSTEM
    assert all(s.strip_think for s in mob.values())
    assert mob["recommended"].extra_dict["think_budget"] == O.MOBILELLM_THINK_BUDGET < mob["recommended"].max_tokens
    assert "think_budget" not in mob["noforce"].extra_dict
    assert mob["recommended"].hash() != mob["noforce"].hash()
    from bes.config import DecodeCfg
    assert O.common_settings().temperature == DecodeCfg().temperature
    with pytest.raises(KeyError):
        O.presets("unknown")


def test_mobilellm_template_renders_like_the_official_one():
    from llama_cpp import llama_chat_format
    fmt = llama_chat_format.Jinja2ChatFormatter(template=O.MOBILELLM_TEMPLATE, eos_token="<|eot_id|>",
                                                bos_token="<|begin_of_text|>")
    default = fmt(messages=[{"role": "user", "content": " Hi \n"}]).prompt
    assert default == ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                       "Please reason step by step, and put your final answer within \\boxed{}.<|eot_id|>"
                       "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                       "<|start_header_id|>assistant<|end_header_id|>\n\n")
    neutral = fmt(messages=[{"role": "system", "content": O.NEUTRAL_SYSTEM}, {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Bye"}]).prompt
    assert neutral.count("<|start_header_id|>") == 5 and "reason step by step" not in neutral


# --------------------------------------------------------------- real file (slow, optional)
def _reference_logits(g: G.GGUF, ids: np.ndarray) -> np.ndarray:
    """Independent numpy forward of a dense Llama4-text model (upstream semantics): interleaved
    RoPE on q and k, then a weightless RMS norm of each head, grouped-query attention, SwiGLU."""
    a = "llama4"
    L, H, Hk = g.get(f"{a}.block_count"), g.get(f"{a}.attention.head_count"), g.get(f"{a}.attention.head_count_kv")
    d, eps, theta = g.get(f"{a}.attention.key_length"), g.get(f"{a}.attention.layer_norm_rms_epsilon"), g.get(f"{a}.rope.freq_base")
    W = lambda name: g.tensor_array(g.tensor(name))  # noqa: E731
    E = W("token_embd.weight")
    x = E[ids].astype(np.float64)
    n = len(ids)

    def norm(v, w=None):
        v = v / np.sqrt((v * v).mean(-1, keepdims=True) + eps)
        return v if w is None else v * w

    mask = np.triu(np.full((n, n), -np.inf), 1)
    for layer in range(L):
        h = norm(x, W(f"blk.{layer}.attn_norm.weight"))
        q = (h @ W(f"blk.{layer}.attn_q.weight").T).reshape(n, H, d)
        k = (h @ W(f"blk.{layer}.attn_k.weight").T).reshape(n, Hk, d)
        v = (h @ W(f"blk.{layer}.attn_v.weight").T).reshape(n, Hk, d)
        q = norm(np.stack([rope_interleaved(q[t], t, theta) for t in range(n)]))
        k = norm(np.stack([rope_interleaved(k[t], t, theta) for t in range(n)]))
        k, v = np.repeat(k, H // Hk, axis=1), np.repeat(v, H // Hk, axis=1)
        s = np.einsum("thd,shd->hts", q, k) / np.sqrt(d) + mask
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        x = x + np.einsum("hts,shd->thd", p, v).reshape(n, H * d) @ W(f"blk.{layer}.attn_output.weight").T
        h = norm(x, W(f"blk.{layer}.ffn_norm.weight"))
        gate, up = h @ W(f"blk.{layer}.ffn_gate.weight").T, h @ W(f"blk.{layer}.ffn_up.weight").T
        x = x + (gate / (1 + np.exp(-gate)) * up) @ W(f"blk.{layer}.ffn_down.weight").T
    return norm(x, W("output_norm.weight")) @ E.T


@pytest.mark.slow
def test_converted_mobilellm_matches_reference_forward():
    entry = O.MODELS["mobilellm"]
    if not entry.source_path().exists():
        pytest.skip("MobileLLM GGUF not downloaded (python -m bes.eval.proxy fetch mobilellm)")
    llama_cpp = pytest.importorskip("llama_cpp")
    path = O.fetch("mobilellm", download=False, log=lambda _m: None)
    src = G.read(entry.source_path())
    llm = llama_cpp.Llama(model_path=str(path), n_ctx=64, n_threads=2, logits_all=True, verbose=False)
    text = (b"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?"
            b"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<think>The capital of France is Paris")
    ids = llm.tokenize(text, add_bos=False, special=True)
    llm.eval(ids)
    got = np.array(llm.scores[:len(ids)], dtype=np.float64)
    ref = _reference_logits(src, np.array(ids))

    def log_softmax(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    lr, lg = log_softmax(ref), log_softmax(got)
    kl = (np.exp(lr) * (lr - lg)).sum(-1)
    assert kl.mean() < 0.02, kl.mean()
    assert (ref.argmax(-1) == got.argmax(-1)).mean() >= 0.9
