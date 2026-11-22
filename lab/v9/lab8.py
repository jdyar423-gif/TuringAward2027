"""Capacity-limited mini-lab: a small looped transformer on real tokens (web 80% + chat 20%),
with switchable ways of spending parameters that are not read by every token.

Every arm reads the same data stream (seeded) and is scored on the same held-out rows, so two
arms differ only by the mechanism. One process uses one core (pin with taskset).

Mechanisms (all "select once per token, reuse in every loop"):
  --experts E --expert-f F --route tok|prev|learned
        the FFN of each block has E expert copies; a token position uses one of them, chosen by
        the class of its token (frequency-balanced classes), of the previous token, or by a
        learned top-1 router on the embeddings of the current and previous token
  --f F           the shared (always-on) FFN width, may be 0 when experts exist
  --lex R         R private FFN neurons per token id and block (a low-rank operator owned by the token)
  --lexprev R     the same, keyed by the previous token
  --bineur H R    R private neurons per hashed bigram bucket (H buckets) and block
  --biemb H       a hashed-bigram embedding table (H rows) added to the input embedding
  --ve            a per-loop token vector added to the stream at the start of each loop
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
V = 1040                         # the first 768 merges of the tokenizer and the 16 special ids
NCLS, CSIZE = 0, 0               # one softmax over the whole vocabulary


def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--f", type=int, default=192)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--expert-f", type=int, default=192)
    ap.add_argument("--route", default="tok", choices=["tok", "prev", "learned", "seq", "role", "topic", "bi"])
    ap.add_argument("--topic", default="E8W1024", help="tag of the topic-key files (topic1k.py)")
    ap.add_argument("--seg", type=int, default=32, help="tokens per segment of a slow (topic) key")
    ap.add_argument("--topic-emb", action="store_true", help="control: the topic id as an input embedding (information without experts)")
    ap.add_argument("--topic-k", type=int, default=1, help="experts per segment (the topic's own plus fixed neighbours)")
    ap.add_argument("--classes", default="bal", choices=["bal", "tree"],
                    help="token classes of the experts: frequency-balanced without meaning, or the taxonomy")
    ap.add_argument("--levels", default="", help="experts on several taxonomy depths, e.g. 3:128,5:64 (depth:width)")
    ap.add_argument("--bidir", action="store_true", help="user and system turns are read in both directions")
    ap.add_argument("--loss-roles", default="all", choices=["all", "noprompt"],
                    help="noprompt: no loss on the tokens of user and system turns")
    ap.add_argument("--lex-gamma", type=float, default=0.0,
                    help="token neurons in proportion to frequency^gamma (mean --lex, at most --lex-max)")
    ap.add_argument("--lex-max", type=int, default=16)
    ap.add_argument("--loop-schedule", default="", help="loops over training, e.g. 1:0.3,2:0.3,4:0.4")
    ap.add_argument("--head-classes", type=int, default=0,
                    help="depth of the taxonomy used by a two-step output: first the class of the next token, then the token inside it")
    ap.add_argument("--writer", type=int, default=0,
                    help="width of an FFN applied before the inside-class softmax, one copy per class of the NEXT token")
    ap.add_argument("--writer-shared", action="store_true", help="control: the same extra FFN, not class-specific")
    ap.add_argument("--hash", default="", help="H:E:F = H independent frequency-balanced hashes of the token id, E experts each, width F")
    ap.add_argument("--ring", default="", help="G:K:F = G small experts of F neurons on a ring; a token reads the K consecutive ones at its position")
    ap.add_argument("--ring-order", default="tree", choices=["tree", "bal"], help="token -> ring position by the taxonomy order or by meaningless balanced classes")
    ap.add_argument("--route-per-loop", action="store_true",
                    help="token-keyed experts with a DIFFERENT balanced key in every loop (a token reads `loops` experts instead of one)")
    ap.add_argument("--attn-experts", type=int, default=0,
                    help="typed attention: E copies of the attention projections, chosen by the class of the position's own token")
    ap.add_argument("--attn-parts", default="qkvo", help="which projections are typed (letters of qkvo); the others stay shared")
    ap.add_argument("--attn-lowrank", type=int, default=0, help="typed projections as shared full matrix + rank-R typed correction (0 = full typed matrices)")
    ap.add_argument("--typebias", type=int, default=0,
                    help="attention bias table indexed by (class of the query token, class of the key token, distance bucket), K classes")
    ap.add_argument("--block-keys", default="", help="key of the FFN experts per block, e.g. cur,prev (default: --route for every block)")
    ap.add_argument("--tie-blocks", action="store_true", help="one expert bank used by every block (each expert is reused blocks x loops times)")
    ap.add_argument("--loop-gates", action="store_true", help="a learned gate per loop, block and FFN neuron (shared and expert neurons)")
    ap.add_argument("--lex", type=int, default=0)
    ap.add_argument("--lexprev", type=int, default=0)
    ap.add_argument("--bineur", type=int, nargs=2, default=[0, 0])
    ap.add_argument("--biemb", type=int, default=0)
    ap.add_argument("--ve", action="store_true")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--tokens", type=float, default=6e6)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-emb", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-frac", type=float, default=0.2)
    ap.add_argument("--capacity", type=float, default=1.2)
    ap.add_argument("--balance", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    return ap.parse_args()


# ------------------------------------------------------------------ data
class Data:
    """Rows of T+1 tokens with, per token, a role (0 web, 1 prompt = system or user turn, 2 assistant)
    and a turn id (equal inside one chat turn, unique elsewhere)."""

    def __init__(self, a):
        self.web = np.memmap(os.path.join(HERE, "web1k.u16"), dtype=np.uint16, mode="r")
        self.chat = np.memmap(os.path.join(HERE, "chat1k.u16"), dtype=np.uint16, mode="r")
        self.crole = np.memmap(os.path.join(HERE, "chat1k_role.u8"), dtype=np.uint8, mode="r")
        self.cturn = np.memmap(os.path.join(HERE, "chat1k_turn.i32"), dtype=np.int32, mode="r")
        self.wtop = self.ctop = None
        if a.route == "topic" or a.topic_emb:
            self.wtop = np.memmap(os.path.join(HERE, f"web1k_topic_{a.topic}.u8"), dtype=np.uint8, mode="r")
            self.ctop = np.memmap(os.path.join(HERE, f"chat1k_topic_{a.topic}.u8"), dtype=np.uint8, mode="r")
        self.seg = a.seg
        self.hold = 600_000
        self.T, self.B, self.frac = a.T, a.B, a.chat_frac
        self.rng = np.random.default_rng(1000 + a.seed)

    def rows(self, chat, n, rng, heldout=False):
        src = self.chat if chat else self.web
        hi = len(src) - self.T - 2
        lo_, hi_ = (hi - self.hold, hi) if heldout else (0, hi - self.hold)
        off = rng.integers(lo_, hi_, n)
        tok = np.stack([np.asarray(src[o:o + self.T + 1]) for o in off]).astype(np.int32)
        if self.wtop is not None:
            top = self.ctop if chat else self.wtop
            seg0 = (off[:, None] + np.arange(0, self.T, self.seg)[None]) // 32      # the stride at or before each segment start
            tid = np.asarray(top[np.minimum(seg0, len(top) - 1)]).astype(np.int32)
        else:
            tid = np.zeros((n, self.T // self.seg), np.int32)
        if not chat:
            return tok, np.zeros_like(tok), -1 - np.tile(np.arange(self.T + 1, dtype=np.int32), (n, 1)), tid
        r = np.stack([np.asarray(self.crole[o:o + self.T + 1]) for o in off]).astype(np.int32)
        turn = np.stack([np.asarray(self.cturn[o:o + self.T + 1]) for o in off]).astype(np.int32)
        role = np.where(r == 3, 2, 1).astype(np.int32)
        turn = np.where(role == 1, turn, -1 - np.arange(self.T + 1, dtype=np.int32)[None])   # only prompts form a turn
        return tok, role, turn, tid

    def batch(self):
        nc = int(round(self.B * self.frac))
        w, c = self.rows(False, self.B - nc, self.rng), self.rows(True, nc, self.rng)
        return tuple(np.concatenate([x, y]) for x, y in zip(w, c))

    def heldout(self, n=192):
        rng = np.random.default_rng(4242)
        return self.rows(False, n, rng, True), self.rows(True, n, rng, True)

    def unigram(self, n=4_000_000):
        return np.bincount(np.asarray(self.web[:n]), minlength=V) + np.bincount(np.asarray(self.chat[:n // 4]), minlength=V)


def balanced_classes(counts, E, seed=0):
    """Token id -> class with (nearly) equal total frequency per class: tokens in decreasing
    frequency go to the class with the least mass so far (ties broken at random)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-(counts + rng.random(len(counts)) * 0.5))
    mass = np.zeros(E)
    cls = np.zeros(len(counts), np.int32)
    for t in order:
        e = int(np.argmin(mass + rng.random(E) * 1e-9))
        cls[t] = e
        mass[e] += counts[t] + 1e-3
    return cls


# ------------------------------------------------------------------ model
def rms(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)


def init(a, key):
    d, f, nb = a.d, a.f, a.blocks
    ks = iter(jax.random.split(key, 64))
    n = lambda shape, std: jax.random.normal(next(ks), shape, jnp.float32) * std
    p = {"emb": n((V, d), 1.0), "unemb": jnp.zeros((V, d), jnp.float32)}

    s = 1.0 / math.sqrt(d)
    p["q"], p["k"], p["v"] = n((nb, d, d), s), n((nb, d, d), s), n((nb, d, d), s)
    p["o"] = jnp.zeros((nb, d, d), jnp.float32)
    if f:
        p["win"], p["wout"] = n((nb, d, f), s), jnp.zeros((nb, f, d), jnp.float32)
    if a.topic_emb:
        p["temb"] = jnp.zeros((a.n_topics + 1, d), jnp.float32)
    if a.experts:
        ne = a.experts + (1 if a.route == "topic" else 0)
        xb = 1 if a.tie_blocks else nb
        p["xin"] = n((xb, ne, d, a.expert_f), s)
        p["xout"] = jnp.zeros((xb, ne, a.expert_f, d), jnp.float32)
    if a.ring_spec[0]:
        G_, K_, F_ = a.ring_spec
        p["rin"] = n((nb, G_, d, F_), s)
        p["rout"] = jnp.zeros((nb, G_, F_, d), jnp.float32)
    if a.attn_experts:
        Ea = a.attn_experts
        for nm in "qkv":
            if nm in a.attn_parts:
                if a.attn_lowrank:
                    p[nm + "xa"] = n((nb, Ea, d, a.attn_lowrank), s)
                    p[nm + "xb"] = jnp.zeros((nb, Ea, a.attn_lowrank, d), jnp.float32)
                else:
                    p[nm + "x"] = n((nb, Ea, d, d), s)
                    del p[nm]
        if "o" in a.attn_parts:
            if a.attn_lowrank:
                p["oxa"] = n((nb, Ea, d, a.attn_lowrank), s)
                p["oxb"] = jnp.zeros((nb, Ea, a.attn_lowrank, d), jnp.float32)
            else:
                p["ox"] = jnp.zeros((nb, Ea, d, d), jnp.float32)
                del p["o"]
    if a.typebias:
        p["tbias"] = jnp.zeros((nb, a.heads, a.typebias, a.typebias, 8), jnp.float32)
    if a.loop_gates:
        if f:
            p["gate_s"] = jnp.ones((a.loops, nb, f), jnp.float32)
        if a.experts:
            p["gate_x"] = jnp.ones((a.loops, nb, a.expert_f), jnp.float32)
        if a.route == "learned":
            p["router"] = n((2, d, a.experts), 0.02)
    for depth, width in a.level_list:
        p[f"lin{depth}"] = n((nb, 1 << depth, d, width), s)
        p[f"lout{depth}"] = jnp.zeros((nb, 1 << depth, width, d), jnp.float32)
    if a.head_classes:
        p["cunemb"] = jnp.zeros((1 << a.head_classes, d), jnp.float32)
        if a.writer:
            kw = 1 if a.writer_shared else (1 << a.head_classes)
            p["wrin"], p["wrout"] = n((kw, d, a.writer), s), jnp.zeros((kw, a.writer, d), jnp.float32)
    for j in range(a.hash_spec[0]):
        p[f"hin{j}"] = n((nb, a.hash_spec[1], d, a.hash_spec[2]), s)
        p[f"hout{j}"] = jnp.zeros((nb, a.hash_spec[1], a.hash_spec[2], d), jnp.float32)
    if a.lex:
        r = a.lex_max if a.lex_gamma else a.lex
        p["lexV"], p["lexU"] = n((nb, V, r, d), s), jnp.zeros((nb, V, r, d), jnp.float32)
    if a.lexprev:
        p["lpV"], p["lpU"] = n((nb, V, a.lexprev, d), s), jnp.zeros((nb, V, a.lexprev, d), jnp.float32)
    if a.bineur[0]:
        h, r = a.bineur
        p["bnV"], p["bnU"] = n((nb, h, r, d), s), jnp.zeros((nb, h, r, d), jnp.float32)
    if a.biemb:
        p["biemb"] = jnp.zeros((a.biemb, d), jnp.float32)
    if a.ve:
        p["ve"] = jnp.zeros((a.loops, V, d), jnp.float32)
    return p


TABLES = ("emb", "unemb", "cunemb", "temb", "tbias", "lexV", "lexU", "lpV", "lpU", "bnV", "bnU", "biemb", "ve")


def rope_tables(T, dh):
    inv = 1.0 / (10000 ** (np.arange(0, dh, 2) / dh))
    ang = np.arange(T)[:, None] * inv[None]
    return jnp.asarray(np.cos(ang), jnp.float32), jnp.asarray(np.sin(ang), jnp.float32)


def rot(x, cos, sin):          # x [B, T, H, dh]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def bihash(prev, cur, H):
    return ((prev.astype(jnp.uint32) * jnp.uint32(40503) + cur.astype(jnp.uint32) * jnp.uint32(2654435761 % (1 << 31))
             + jnp.uint32(12345)) % jnp.uint32(H)).astype(jnp.int32)


def make_dispatch(eid, E, C):
    """Group N positions by expert with at most C per expert (positions beyond C are dropped)."""
    N = eid.shape[0]
    order = jnp.argsort(eid, stable=True)
    se = eid[order]
    start = jnp.searchsorted(se, jnp.arange(E), side="left")
    pos = jnp.arange(N) - start[se]
    keep = pos < C
    slot = se * C + jnp.minimum(pos, C - 1)
    return order, slot, keep


def expert_ffn(win, wout, x, disp, E, C, gate=None):
    order, slot, keep = disp
    d = x.shape[-1]
    xs = x[order] * keep[:, None]
    buf = jnp.zeros((E * C, d), x.dtype).at[slot].add(xs).reshape(E, C, d)
    hid = jax.nn.relu(jnp.einsum("ecd,edf->ecf", buf, win)) ** 2
    if gate is not None:
        hid = hid * gate
    out = jnp.einsum("ecf,efd->ecd", hid, wout).reshape(E * C, d)
    ys = out[slot] * keep[:, None]
    return jnp.zeros_like(x).at[order].set(ys)


def forward(p, tok, role, turn, tid, a, consts, loops):
    """tok, role, turn [B, T] -> normalised hidden states [B, T, d], aux loss."""
    B, T = tok.shape
    d, H = a.d, a.heads
    dh = d // H
    cos, sin, cls = consts["cos"], consts["sin"], consts["cls"]
    prev = jnp.concatenate([tok[:, :1], tok[:, :-1]], 1)
    e = p["emb"][tok]
    if a.biemb:
        e = e + p["biemb"][bihash(prev, tok, a.biemb)]
    if a.topic_emb:
        e = e + jnp.repeat(p["temb"][tid], a.seg, axis=1)
    h = rms(e)
    aux = 0.0
    disp = gate = None
    if a.experts and a.route == "topic":
        pass
    elif a.experts:
        E = a.experts
        C = int(math.ceil(a.capacity * B * T / E))
        if a.route == "tok":
            eid = cls[tok].reshape(-1)
        elif a.route == "prev":
            eid = cls[prev].reshape(-1)
        elif a.route == "seq":
            eid = jnp.repeat(cls[tok[:, 0]], T)             # one expert per row (a stand-in for a topic)
        elif a.route == "role":
            eid = role.reshape(-1)
        elif a.route == "bi":
            eid = (consts["cls8"][tok] * 8 + consts["cls8"][prev]).reshape(-1)        # 64 cells: class of the token x class of the previous token
        else:
            logits = rms(p["emb"][tok]) @ p["router"][0] + rms(p["emb"][prev]) @ p["router"][1]
            probs = jax.nn.softmax(logits, -1).reshape(-1, E)
            eid = jnp.argmax(probs, -1)
            gate = jnp.take_along_axis(probs, eid[:, None], 1)       # Switch-style: the gate carries the gradient
            frac = jnp.mean(jax.nn.one_hot(eid, E), 0)
            aux = a.balance * E * jnp.sum(frac * jnp.mean(probs, 0))
        disp = make_dispatch(eid, E, C)
        if a.route_per_loop:
            disp_l = [make_dispatch(consts["cls_l"][l][tok].reshape(-1), E, C) for l in range(a.loops)]
    ldisp = []
    for depth, width in a.level_list:
        El = 1 << depth
        Cl = int(math.ceil(a.capacity * 1.15 * B * T / El))
        ldisp.append((depth, El, Cl, make_dispatch((consts["leaf"][tok] >> (8 - depth)).reshape(-1), El, Cl)))
    rdisp = None
    if a.ring_spec[0]:
        G_, K_, F_ = a.ring_spec
        Cr = int(math.ceil(a.capacity * 1.3 * B * T / G_))
        rdisp = make_dispatch(consts["ringpos"][tok].reshape(-1), G_, Cr)
        ridx = (jnp.arange(G_)[:, None] + jnp.arange(K_)[None]) % G_           # the K experts read at each position
    adisp = None
    if a.attn_experts:
        Ea = a.attn_experts
        Ca = int(math.ceil(a.capacity * 1.3 * B * T / Ea))
        adisp = make_dispatch(consts["acls"][tok].reshape(-1), Ea, Ca)

    def typed(nm, bb, z):                    # z [B, T, d] -> typed projection of every position by its own token class
        if nm not in a.attn_parts or not a.attn_experts:
            return z @ p[nm][bb]
        order, slot, keep = adisp
        zs = z.reshape(B * T, d)[order] * keep[:, None]
        buf = jnp.zeros((Ea * Ca, d), z.dtype).at[slot].add(zs).reshape(Ea, Ca, d)
        if a.attn_lowrank:
            out = jnp.einsum("ecr,erf->ecf", jnp.einsum("ecd,edr->ecr", buf, p[nm + "xa"][bb]), p[nm + "xb"][bb])
        else:
            out = jnp.einsum("ecd,edf->ecf", buf, p[nm + "x"][bb])
        ys = out.reshape(Ea * Ca, d)[slot] * keep[:, None]
        res = jnp.zeros((B * T, d), z.dtype).at[order].set(ys).reshape(B, T, d)
        return res + (z @ p[nm][bb] if a.attn_lowrank else 0.0)

    tb_idx = None
    if a.typebias:
        tc = consts["tcls"][tok]                                             # [B, T]
        dist = jnp.arange(T)[:, None] - jnp.arange(T)[None, :]
        bucket = jnp.clip(jnp.floor(jnp.log2(jnp.maximum(dist, 1).astype(jnp.float32))).astype(jnp.int32) + 1, 0, 7)
        bucket = jnp.where(dist <= 0, 0, bucket)                             # 0: self (and masked future), 1: distance 1, 2: 2-3, ...
        tb_idx = (tc, bucket)
    bdisp = None
    if a.block_key_list:
        E = a.experts
        C = int(math.ceil(a.capacity * B * T / E))
        src = {"cur": tok, "prev": prev}
        bdisp = [make_dispatch(cls[src[kname]].reshape(-1), E, C) for kname in a.block_key_list]
    hdisp = []
    for j in range(a.hash_spec[0]):
        Eh = a.hash_spec[1]
        Ch = int(math.ceil(a.capacity * B * T / Eh))
        hdisp.append((j, Eh, Ch, make_dispatch(consts[f"hcls{j}"][tok].reshape(-1), Eh, Ch)))
    lex = []
    if a.lex:
        Vg, Ug = p["lexV"][:, tok], p["lexU"][:, tok]
        if a.lex_gamma:
            Ug = Ug * consts["lexmask"][tok][None, :, :, :, None]      # a token owns only its first R(token) neurons
        lex.append((Vg, Ug))
    lex += [(p[v][:, prev], p[u][:, prev]) for v, u in (("lpV", "lpU"),) if a.lexprev]
    if a.bineur[0]:
        bk = bihash(prev, tok, a.bineur[0])
        lex.append((p["bnV"][:, bk], p["bnU"][:, bk]))
    mask = jnp.tril(jnp.ones((T, T), bool))[None]
    if a.bidir:
        mask = mask | (turn[:, :, None] == turn[:, None, :])        # inside one prompt turn: both directions
    mask = mask[:, None]
    for loop in range(loops):
        if a.ve:
            h = h + p["ve"][loop][tok]
        for b in range(a.blocks):
            x = rms(h)
            q = rot(typed("q", b, x).reshape(B, T, H, dh), cos, sin)
            k = rot(typed("k", b, x).reshape(B, T, H, dh), cos, sin)
            v = typed("v", b, x).reshape(B, T, H, dh)
            att = jnp.einsum("bthd,bshd->bhts", rms(q), rms(k)) * math.sqrt(dh)
            if tb_idx is not None:
                tc, bucket = tb_idx
                att = att + p["tbias"][b][:, tc[:, :, None], tc[:, None, :], bucket[None]].transpose(1, 0, 2, 3)
            att = jax.nn.softmax(jnp.where(mask, att, -1e9), -1)
            y = jnp.einsum("bhts,bshd->bthd", att, v).reshape(B, T, d)
            h = h + typed("o", b, y)
            x = rms(h)
            y = 0.0
            if a.f:
                hs = jax.nn.relu(x @ p["win"][b]) ** 2
                if a.loop_gates:
                    hs = hs * p["gate_s"][min(loop, a.loops - 1), b]
                y = y + hs @ p["wout"][b]
            if a.experts and a.route == "topic":
                J = T // a.seg
                xs = x.reshape(B, J, a.seg, d)
                ye = 0.0
                for kk in range(a.topic_k):                # the topic's own expert, then fixed neighbours on a ring
                    eid = jnp.where(tid == a.experts, tid, (tid + kk) % a.experts)
                    hid = jax.nn.relu(jnp.einsum("bjsd,bjdf->bjsf", xs, p["xin"][b][eid])) ** 2
                    ye = ye + jnp.einsum("bjsf,bjfd->bjsd", hid, p["xout"][b][eid])
                y = y + ye.reshape(B, T, d)
            elif a.experts:
                xb = 0 if a.tie_blocks else b
                ye = expert_ffn(p["xin"][xb], p["xout"][xb], x.reshape(B * T, d),
                                disp_l[min(loop, a.loops - 1)] if a.route_per_loop else (bdisp[b] if bdisp else disp), a.experts, C,
                                p["gate_x"][min(loop, a.loops - 1), b] if a.loop_gates else None)
                if gate is not None:
                    ye = ye * gate * a.experts ** 0.5      # keeps the scale near 1 at uniform routing
                y = y + ye.reshape(B, T, d)
            for depth, El, Cl, dl in ldisp:
                y = y + expert_ffn(p[f"lin{depth}"][b], p[f"lout{depth}"][b], x.reshape(B * T, d), dl, El, Cl).reshape(B, T, d)
            if rdisp is not None:
                wi = jnp.transpose(p["rin"][b][ridx], (0, 2, 1, 3)).reshape(G_, d, K_ * F_)
                wo = p["rout"][b][ridx].reshape(G_, K_ * F_, d)
                y = y + expert_ffn(wi, wo, x.reshape(B * T, d), rdisp, G_, Cr).reshape(B, T, d)
            for j, Eh, Ch, dj in hdisp:
                y = y + expert_ffn(p[f"hin{j}"][b], p[f"hout{j}"][b], x.reshape(B * T, d), dj, Eh, Ch).reshape(B, T, d)
            for Vg, Ug in lex:
                act = jax.nn.relu(jnp.einsum("btrd,btd->btr", Vg[b], x)) ** 2
                y = y + jnp.einsum("btr,btrd->btd", act, Ug[b])
            h = h + y
    return h, aux


def loss_fn(p, batch, a, consts, loops):
    tok, role, turn, tid = batch
    h, aux = forward(p, tok[:, :-1], role[:, :-1], turn[:, :-1], tid, a, consts, loops)
    hid = rms(h)
    y = tok[:, 1:]
    if a.head_classes:
        K = 1 << a.head_classes
        ocls = consts["ocls"]
        cy = ocls[y]
        lpc = jax.nn.log_softmax(hid @ p["cunemb"].T, -1)
        ce_c = -jnp.take_along_axis(lpc, cy[..., None], -1)[..., 0]
        g = hid
        if a.writer:
            B, T, d = h.shape
            if a.writer_shared:
                add = (jax.nn.relu(hid @ p["wrin"][0]) ** 2) @ p["wrout"][0]
            else:
                Cw = int(math.ceil(a.capacity * 1.3 * B * T / K))
                add = expert_ffn(p["wrin"], p["wrout"], hid.reshape(B * T, d), make_dispatch(cy.reshape(-1), K, Cw), K, Cw).reshape(B, T, d)
            g = rms(h + add)
        logit = g @ p["unemb"].T
        inside = ocls[None, None, :] == cy[..., None]
        lp = jax.nn.log_softmax(jnp.where(inside, logit, -1e9), -1)
        ce = ce_c - jnp.take_along_axis(lp, y[..., None], -1)[..., 0]
    else:
        lp = jax.nn.log_softmax(hid @ p["unemb"].T, -1)
        ce = -jnp.take_along_axis(lp, y[..., None], -1)[..., 0]
    ry = role[:, 1:]
    w = jnp.where(ry == 1, 0.0, 1.0) if a.loss_roles == "noprompt" else jnp.ones_like(ce)
    wa = (ry == 2).astype(ce.dtype)
    stats = {"ce": jnp.sum(ce * w) / jnp.maximum(jnp.sum(w), 1.0), "asst_sum": jnp.sum(ce * wa), "asst_n": jnp.sum(wa),
             "sum": jnp.sum(ce * w), "n": jnp.sum(w)}
    return stats["ce"] + aux, stats


def counts(p, a):
    """(stored, read per token) numbers of each part."""
    d, nb = a.d, a.blocks
    c = {"attention": (4 * d * d * nb,) * 2, "unembed": (V * d,) * 2, "embed": (V * d, d)}
    if a.f:
        c["shared ffn"] = (2 * d * a.f * nb,) * 2
    if a.topic_emb:
        c["topic embedding"] = ((a.n_topics + 1) * d, d)
    if a.experts and a.route == "topic":
        c["topic experts"] = (2 * d * a.expert_f * nb * (a.experts + 1), 2 * d * a.expert_f * nb * a.topic_k)
    elif a.experts:
        xb = 1 if a.tie_blocks else nb
        c["experts"] = (2 * d * a.expert_f * xb * a.experts, 2 * d * a.expert_f * xb * (min(a.loops, a.experts) if a.route_per_loop else 1))
    if a.attn_experts:
        per = 2 * d * a.attn_lowrank if a.attn_lowrank else d * d
        npart = len(a.attn_parts)
        c["attention"] = ((4 - npart) * d * d * nb if not a.attn_lowrank else 4 * d * d * nb,) * 2
        c["typed attention"] = (npart * per * nb * a.attn_experts, npart * per * nb)
    if a.typebias:
        c["type-pair attention bias"] = (nb * a.heads * a.typebias * a.typebias * 8, nb * a.heads * a.typebias * 8)
    if a.loop_gates:
        g = a.loops * nb * ((a.f if a.f else 0) + (a.expert_f if a.experts else 0))
        c["loop gates"] = (g, g)
        if a.route == "learned":
            c["router"] = (2 * d * a.experts,) * 2
    for depth, width in a.level_list:
        c[f"class experts depth {depth}"] = (2 * d * width * nb * (1 << depth), 2 * d * width * nb)
    if a.head_classes:
        K = 1 << a.head_classes
        c["class unembed"] = (K * d,) * 2
        c["unembed"] = (V * d, int(round(V * d * a.mean_class_share)))      # only the rows of one class are read
        if a.writer:
            kw = 1 if a.writer_shared else K
            c["writer ffn"] = (2 * d * a.writer * kw, 2 * d * a.writer)
    if a.ring_spec[0]:
        G_, K_, F_ = a.ring_spec
        c["ring experts"] = (2 * d * F_ * nb * G_, 2 * d * F_ * nb * K_)
    if a.hash_spec[0]:
        H_, E_, F_ = a.hash_spec
        c["hashed experts"] = (2 * d * F_ * nb * E_ * H_, 2 * d * F_ * nb * H_)
    if a.lex and a.lex_gamma:
        own = int(a.lex_counts.sum())
        c["token neurons (by frequency)"] = (2 * d * own * nb, int(round(2 * d * nb * a.lex_mean_read)))
    elif a.lex:
        c["token neurons"] = (2 * d * a.lex * V * nb, 2 * d * a.lex * nb)
    if a.lexprev:
        c["prev-token neurons"] = (2 * d * a.lexprev * V * nb, 2 * d * a.lexprev * nb)
    if a.bineur[0]:
        c["bigram neurons"] = (2 * d * a.bineur[1] * a.bineur[0] * nb, 2 * d * a.bineur[1] * nb)
    if a.biemb:
        c["bigram embedding"] = (a.biemb * d, d)
    if a.ve:
        c["per-loop token vectors"] = (a.loops * V * d, a.loops * d)
    return c


def main():
    a = args_()
    a.level_list = [tuple(int(x) for x in s.split(":")) for s in a.levels.split(",") if s]
    a.hash_spec = tuple(int(x) for x in a.hash.split(":")) if a.hash else (0, 0, 0)
    a.ring_spec = tuple(int(x) for x in a.ring.split(":")) if a.ring else (0, 0, 0)
    a.n_topics = int(a.topic.split("W")[0][1:])
    assert a.route != "topic" or a.experts == a.n_topics
    os.makedirs(a.out, exist_ok=True)
    data = Data(a)
    cos, sin = rope_tables(a.T, a.d // a.heads)
    uni = data.unigram()
    leaf = np.load(os.path.join(HERE, "tree_leaf_1k.npy"))
    if a.experts and a.route in ("tok", "prev") and a.classes == "tree":
        depth = int(round(math.log2(a.experts)))
        assert 1 << depth == a.experts
        cls = leaf >> (8 - depth)
    else:
        cls = balanced_classes(uni, max(a.experts, 1))
    consts = {"cos": cos, "sin": sin, "cls": jnp.asarray(cls), "leaf": jnp.asarray(leaf)}
    for j in range(a.hash_spec[0]):
        consts[f"hcls{j}"] = jnp.asarray(balanced_classes(uni, a.hash_spec[1], seed=100 + j))
    a.block_key_list = [x for x in a.block_keys.split(",") if x]
    consts["cls8"] = jnp.asarray(leaf >> 5)
    if a.attn_experts:
        consts["acls"] = jnp.asarray(leaf >> (8 - int(round(math.log2(a.attn_experts)))))
    if a.typebias:
        consts["tcls"] = jnp.asarray(leaf >> (8 - int(round(math.log2(a.typebias)))))
    if a.route_per_loop:
        consts["cls_l"] = jnp.asarray(np.stack([balanced_classes(uni, a.experts, seed=50 + l) for l in range(a.loops)]))
    if a.ring_spec[0]:
        G_ = a.ring_spec[0]
        consts["ringpos"] = jnp.asarray((leaf.astype(np.int64) * G_ // 256).astype(np.int32) if a.ring_order == "tree"
                                        else balanced_classes(uni, G_, seed=7))
    a.mean_class_share = 1.0
    if a.head_classes:
        oc = leaf >> (8 - a.head_classes)
        consts["ocls"] = jnp.asarray(oc)
        size = np.bincount(oc, minlength=1 << a.head_classes)
        fr = uni / uni.sum()
        a.mean_class_share = float((fr * size[oc]).sum() / V)       # expected share of unembedding rows read per token
    if a.lex and a.lex_gamma:
        # neurons per token in proportion to frequency^gamma, mean a.lex over the vocabulary, at most lex_max
        freq = uni / uni.sum()
        lo, hi = 0.0, 1e12
        for _ in range(80):                                   # the scale that gives the wanted total
            mid = (lo + hi) / 2
            r = np.minimum(np.floor(mid * freq ** a.lex_gamma + 0.5), a.lex_max)
            lo, hi = (mid, hi) if r.sum() < a.lex * V else (lo, mid)
        a.lex_counts = np.minimum(np.floor(hi * freq ** a.lex_gamma + 0.5), a.lex_max).astype(np.int64)
        a.lex_mean_read = float((freq * a.lex_counts).sum())
        consts["lexmask"] = jnp.asarray((np.arange(a.lex_max)[None] < a.lex_counts[:, None]).astype(np.float32))
    p = init(a, jax.random.PRNGKey(a.seed))
    cnt = counts(p, a)
    stored = sum(v[0] for v in cnt.values())
    read = sum(v[1] for v in cnt.values())
    core_read = read - V * a.d - a.d
    steps = int(a.tokens / (a.B * a.T))
    sched = lambda base: optax.join_schedules(
        [optax.linear_schedule(0.0, base, 100), optax.linear_schedule(base, 0.1 * base, max(1, steps - 100))], [100])
    labels = {k: ("tab" if k in TABLES else "mat") for k in p}
    opt = optax.multi_transform(
        {"mat": optax.adamw(sched(a.lr), b1=0.9, b2=0.95, weight_decay=a.wd),
         "tab": optax.adam(sched(a.lr_emb), b1=0.9, b2=0.95)}, labels)
    opt = optax.chain(optax.clip_by_global_norm(1.0), opt)
    state = opt.init(p)

    def make_step(loops):
        @jax.jit
        def step(p, state, batch):
            (l, st), g = jax.value_and_grad(lambda q: loss_fn(q, batch, a, consts, loops), has_aux=True)(p)
            upd, state = opt.update(g, state, p)
            return optax.apply_updates(p, upd), state, st["ce"]
        return step

    plan = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in a.loop_schedule.split(",") if x] or [(a.loops, 1.0)]
    bounds, acc = [], 0.0
    for loops, share in plan:
        acc += share
        bounds.append((acc, loops))
    steppers = {loops: make_step(loops) for loops, _ in plan}

    @jax.jit
    def evaluate(p, batch):
        return loss_fn(p, batch, a, consts, a.loops)[1]

    hw, hc = data.heldout()

    def heldout(p):
        out = {}
        for name, rows in (("web", hw), ("chat", hc)):
            s = n = sa = na = 0.0
            for i in range(0, len(rows[0]), a.B):
                st = evaluate(p, tuple(jnp.asarray(x[i:i + a.B]) for x in rows))
                s, n, sa, na = s + float(st["sum"]), n + float(st["n"]), sa + float(st["asst_sum"]), na + float(st["asst_n"])
            out[name] = s / max(n, 1.0)
            if name == "chat":
                out["asst"] = sa / max(na, 1.0)
        out["mix"] = (1 - a.chat_frac) * out["web"] + a.chat_frac * out["chat"]
        return out

    print(f"{a.name}: stored {stored:,}  read/token {read:,}  core read (no embed/unembed) {core_read:,}  steps {steps}",
          flush=True)
    for k, v in cnt.items():
        print(f"   {k:30s} stored {v[0]:>11,}  read {v[1]:>9,}", flush=True)
    log, t0, ema, cost = [], time.time(), None, 0.0
    for i in range(steps):
        loops = next(l for b, l in bounds if (i + 0.5) / steps <= b + 1e-9)
        p, state, ce = steppers[loops](p, state, tuple(jnp.asarray(x) for x in data.batch()))
        cost += loops
        ce = float(ce)
        ema = ce if ema is None else 0.98 * ema + 0.02 * ce
        if (i + 1) % 100 == 0 or i + 1 == steps:
            rec = {"step": i + 1, "tokens": (i + 1) * a.B * a.T, "train_ema": round(ema, 4), "s": round(time.time() - t0, 1)}
            if (i + 1) % 400 == 0 or i + 1 == steps:
                rec.update({k: round(v, 4) for k, v in heldout(p).items()})
            log.append(rec)
            print(json.dumps(rec), flush=True)
    h = heldout(p)
    res = {"name": a.name, "args": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(a).items()},
           "stored": stored, "read": read, "core_read": core_read, **h,
           "loop_steps": cost, "tok_per_s": steps * a.B * a.T / (time.time() - t0), "log": log}
    with open(os.path.join(a.out, a.name + ".json"), "w") as f:
        json.dump(res, f)
    print(f"FINAL {a.name}: web {h['web']:.4f} chat {h['chat']:.4f} asst {h['asst']:.4f} mix {h['mix']:.4f}  "
          f"({res['tok_per_s']:.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
