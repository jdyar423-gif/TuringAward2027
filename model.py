"""Compact decoder-only LM with FLOP-free capacity channels.

Everything that costs matmul FLOPs is kept small; capacity that can be had for
(almost) zero FLOPs is added through table lookups:
  * hashed n-gram input memory (Engram / BigramHash style, sign-decorrelated rows,
    injected into every layer with learned lambdas),
  * value embeddings (token-indexed tables mixed into attention values),
plus cheap gates (smear, sparse attention-output gate, exclusive self-attention).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 4096
    d: int = 256
    n_layer: int = 4
    n_head: int = 4
    mlp_mult: float = 4.0
    act: str = "relu2"            # relu2 | gelu | swiglu
    qk_norm: bool = True
    rope_base: float = 10000.0
    value_residual: bool = True   # v_l <- a*v_l + b*v_0
    x0_mix: bool = True           # x_l <- a*x_l + b*x_0
    softcap: float = 15.0
    tie: bool = False
    attn_gate: str = ""           # "" | "sparse" (12-dim headwise) | "head" | "elem"
    xsa: bool = False             # exclusive self-attention (gated)
    smear: bool = False           # x_t += lam*sigmoid(w.x_t[:12]) * x_{t-1}
    conv: int = 0                 # causal depthwise conv kernel before attn/mlp (0 = off)
    ngram_orders: tuple = ()      # hashed n-gram input memory, e.g. (2, 3)
    ngram_buckets: int = 262139   # rows per order (prime)
    ngram_sign: bool = True       # +-1 sign vector per n-gram (independent hash)
    ngram_layers: bool = True     # inject n-gram memory into every layer (else input only)
    value_embeds: int = 0         # number of token->value tables (shared U-net style)
    unet: bool = False
    dropout: float = 0.0
    attn_scale: float = 0.0       # 0 -> 1/sqrt(head_dim)

    @property
    def head_dim(self):
        return self.d // self.n_head

    @property
    def hidden(self):
        return int(self.mlp_mult * self.d)


def rms(x):
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim, base, max_len=4096):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        f = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", f.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", f.sin()[None, :, None, :], persistent=False)

    def forward(self, x):  # B,T,H,D
        T = x.size(1)
        c, s = self.cos[:, :T], self.sin[:, :T]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class CausalConv(nn.Module):
    """Depthwise causal conv with residual (Canon-style local mixing), as explicit shifts."""

    def __init__(self, d, k):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.zeros(d, k))

    def forward(self, x):  # B,T,d
        y = x
        for j in range(self.k):
            xs = x if j == 0 else F.pad(x[:, :-j], (0, 0, j, 0))
            y = y + self.w[:, j] * xs
        return y


class Block(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.cfg = cfg
        d, H, hd = cfg.d, cfg.n_head, cfg.head_dim
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        h = cfg.hidden
        self.up = nn.Linear(d, 2 * h if cfg.act == "swiglu" else h, bias=False)
        self.down = nn.Linear(h, d, bias=False)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.down.weight)
        self.rot = Rotary(hd, cfg.rope_base)
        gin = {"sparse": 12, "head": d, "elem": d}.get(cfg.attn_gate)
        if gin:
            self.gate = nn.Linear(gin, d if cfg.attn_gate == "elem" else H, bias=False)
            nn.init.zeros_(self.gate.weight)
        if cfg.xsa:
            self.xsa_a = nn.Parameter(torch.zeros(H))
        if cfg.conv:
            self.conv_a = CausalConv(d, cfg.conv)
            self.conv_m = CausalConv(d, cfg.conv)
        self.lam = nn.Parameter(torch.tensor([1.0, 1.0 if layer == 0 else 0.0, 0.1]))  # x, x0, ngram
        self.lam_v = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.lam_ve = nn.Parameter(torch.tensor(0.5))
        self.scale = cfg.attn_scale or 1.0 / math.sqrt(hd)

    def forward(self, x, x0, xng, v0, ve):
        cfg = self.cfg
        B, T, d = x.shape
        H, hd = cfg.n_head, cfg.head_dim
        if cfg.x0_mix:
            x = self.lam[0] * x + self.lam[1] * x0
        if xng is not None:
            x = x + self.lam[2] * xng
        h = rms(x)
        if cfg.conv:
            h = self.conv_a(h)
        q, k, v = self.qkv(h).view(B, T, 3, H, hd).unbind(2)
        if cfg.qk_norm:
            q, k = rms(q), rms(k)
        q, k = self.rot(q), self.rot(k)
        if v0 is None:
            v0 = v
        elif cfg.value_residual:
            v = self.lam_v[0] * v + self.lam_v[1] * v0
        if ve is not None:
            v = v + self.lam_ve * ve.view(B, T, H, hd)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=self.scale
        ).transpose(1, 2)  # B,T,H,hd
        if cfg.xsa:  # remove the component of the output along the token's own value
            vn = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
            y = y - torch.tanh(self.xsa_a)[:, None] * (y * vn).sum(-1, keepdim=True) * vn
        if cfg.attn_gate == "sparse":
            y = y * (2 * torch.sigmoid(self.gate(h[..., :12]))).unsqueeze(-1)
        elif cfg.attn_gate == "head":
            y = y * (2 * torch.sigmoid(self.gate(h))).unsqueeze(-1)
        elif cfg.attn_gate == "elem":
            y = y * (2 * torch.sigmoid(self.gate(h))).view(B, T, H, hd)
        x = x + self.o(y.reshape(B, T, d))
        h = rms(x)
        if cfg.conv:
            h = self.conv_m(h)
        u = self.up(h)
        if cfg.act == "relu2":
            u = F.relu(u).square()
        elif cfg.act == "gelu":
            u = F.gelu(u)
        else:
            a, b = u.chunk(2, dim=-1)
            u = F.silu(a) * b
        if cfg.dropout and self.training:
            u = F.dropout(u, cfg.dropout)
        return x + self.down(u), v0


class LM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        nn.init.normal_(self.emb.weight, std=1.0)
        # one sparse table per n-gram order; rows start at zero (no-op until trained)
        self.ngram = nn.ModuleList(nn.Embedding(cfg.ngram_buckets, cfg.d, sparse=True) for _ in cfg.ngram_orders)
        for e in self.ngram:
            nn.init.zeros_(e.weight)
        if cfg.ngram_orders and cfg.ngram_sign:
            gen = torch.Generator().manual_seed(1234)
            self.register_buffer("signs", (torch.randint(0, 2, (8192, cfg.d), generator=gen) * 2 - 1).float(),
                                 persistent=False)
        self.ve = nn.ModuleList()
        for _ in range(cfg.value_embeds):
            e = nn.Embedding(cfg.vocab, cfg.d)
            nn.init.normal_(e.weight, std=1.0)
            self.ve.append(e)
        if cfg.smear:
            self.smear_w = nn.Linear(12, 1, bias=False)
            nn.init.zeros_(self.smear_w.weight)
            self.smear_lam = nn.Parameter(torch.zeros(1))
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        if cfg.unet:
            self.skip_w = nn.Parameter(torch.ones(cfg.n_layer // 2))
        if not cfg.tie:
            self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
            nn.init.zeros_(self.head.weight)

    def ve_for_layer(self, i, idx):
        k, L = len(self.ve), self.cfg.n_layer
        if k == 0:
            return None
        if i < k:
            return self.ve[i](idx)
        if i >= L - k:
            return self.ve[L - 1 - i](idx)
        return None

    def ngram_mem(self, ngram_idx):
        if not len(self.ngram):
            return None
        out = 0
        for j, e in enumerate(self.ngram):
            r = e(ngram_idx[..., 2 * j])
            if self.cfg.ngram_sign:
                r = r * self.signs[ngram_idx[..., 2 * j + 1]]
            out = out + r
        return out

    def hidden(self, idx, ngram_idx=None):
        cfg = self.cfg
        x = self.emb(idx)
        if cfg.smear:
            g = self.smear_lam * torch.sigmoid(self.smear_w(x[..., :12]))
            x = x + g * F.pad(x[:, :-1], (0, 0, 1, 0))
        xng = self.ngram_mem(ngram_idx)
        if xng is not None and not cfg.ngram_layers:
            x, xng = x + xng, None
        x = rms(x)
        x0, v0 = x, None
        skips = []
        L = cfg.n_layer
        for i, blk in enumerate(self.blocks):
            if cfg.unet and i >= L - L // 2:
                x = x + self.skip_w[L - 1 - i] * skips.pop()
            x, v0 = blk(x, x0, xng, v0, self.ve_for_layer(i, idx))
            if cfg.unet and i < L // 2:
                skips.append(x)
        return rms(x)

    def logits(self, h):
        w = self.emb.weight if self.cfg.tie else self.head.weight
        z = F.linear(h, w)
        c = self.cfg.softcap
        if c:
            z = c * torch.tanh(z / c)
        return z

    def forward(self, idx, targets, ngram_idx=None, reduction="mean"):
        z = self.logits(self.hidden(idx, ngram_idx))
        return F.cross_entropy(z.float().view(-1, z.size(-1)), targets.reshape(-1), reduction=reduction)


# ---------------------------------------------------------------------------
# hashed n-gram ids (computed on the token stream; strictly causal)
# ---------------------------------------------------------------------------
_P = 2147483647  # 2^31 - 1


def ngram_ids(tokens: torch.Tensor, orders, buckets, pad=65535):
    """tokens: 1-D int64 stream. Returns (N, 2*len(orders)) int64: [row id, sign id] per order.
    Ids at position t depend only on tokens[t-n+1 .. t]."""
    if not orders:
        return None
    N = tokens.numel()
    t = tokens.long()
    out = []
    for n in orders:
        key = torch.zeros(N, dtype=torch.long)
        for j in range(n):
            sh = torch.full((N,), pad, dtype=torch.long)
            sh[j:] = t[:N - j] if j else t
            key = (key * 65537 + sh + 1) % _P
        key = (key + n * 7919) % _P
        out.append(((key * 1000003 + 12345) % _P) % buckets)
        out.append(((key * 998244353 + 777) % _P) % 8192)
    return torch.stack(out, dim=1)
