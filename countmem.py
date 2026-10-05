"""Count-based, FLOP-free memory experts.

For every target position t (predicting x_t from x_<t) we compute, per context
length k, the statistics of two non-parametric experts:

  global n-gram expert  p_k(y | c_k) = C(c_k, y) / C(c_k)   (counts over the train split)
  document cache expert q_k(y | c_k) = D(c_k, y) / D(c_k)   (counts over x_<t in the same article)

On the TRAIN stream the global counts are *leave-one-document-out* (LODO): the
current article's own contribution is subtracted.  Together with the strictly
causal document cache this reproduces, on training data, exactly the situation
at test time (a new article: global counts from other articles + a cache of the
article so far).  All quantities are integer counts (sorting/hashing); the only
floating-point work is a handful of ops per token per expert.

Every expert distribution is normalised over the vocabulary, and the mixture
weights depend only on the context, so the mixture is a proper distribution.
"""
import json
import os

import numpy as np

M1 = np.uint64(0x9E3779B97F4A7C15)
M2 = np.uint64(0xBF58476D1CE4E5B9)
SALT_PAIR = np.uint64(0x94D049BB133111EB)
SALT_DOC = np.uint64(0xD6E8FEB86659FD93)


def _mix(h, x):
    with np.errstate(over="ignore"):
        h = (h ^ (x.astype(np.uint64) + np.uint64(0x632BE59BD9B4E019))) * M1
        h ^= h >> np.uint64(31)
        h *= M2
        h ^= h >> np.uint64(29)
    return h


def context_hashes(tokens, K):
    """hs[k][t] = hash of (x_{t-k} .. x_{t-1}); hs[0] = 0. Positions with t-k < 0 use a pad symbol."""
    N = len(tokens)
    x = tokens.astype(np.int64)
    hs = [np.zeros(N, dtype=np.uint64)]
    h = np.zeros(N, dtype=np.uint64)
    for k in range(1, K):
        prev = np.full(N, 70000, dtype=np.int64)
        prev[k:] = x[:N - k]
        h = _mix(h, prev)
        hs.append(h.copy())
    return hs


def pair_keys(h, tokens):
    with np.errstate(over="ignore"):
        return _mix(h ^ SALT_PAIR, tokens.astype(np.int64))


def doc_keys(h, docs):
    with np.errstate(over="ignore"):
        return _mix(h ^ SALT_DOC, docs.astype(np.int64))


class Counter:
    def __init__(self, keys):
        self.u, self.c = np.unique(keys, return_counts=True)

    def __call__(self, q):
        if len(self.u) == 0:
            return np.zeros(len(q), dtype=np.int64)
        i = np.searchsorted(self.u, q)
        i = np.minimum(i, len(self.u) - 1)
        return np.where(self.u[i] == q, self.c[i], 0).astype(np.int64)


def causal_rank(keys):
    """r[t] = number of s < t with keys[s] == keys[t]."""
    order = np.lexsort((np.arange(len(keys)), keys))
    sk = keys[order]
    start = np.ones(len(keys), dtype=bool)
    start[1:] = sk[1:] != sk[:-1]
    grp_start = np.maximum.accumulate(np.where(start, np.arange(len(keys)), 0))
    r = np.empty(len(keys), dtype=np.int64)
    r[order] = np.arange(len(keys)) - grp_start
    return r


def causal_distinct(group_keys, item_keys):
    """n[t] = number of distinct item_keys among s < t with group_keys[s] == group_keys[t]."""
    is_first = causal_rank(item_keys) == 0
    order = np.lexsort((np.arange(len(group_keys)), group_keys))
    sk = group_keys[order]
    f = is_first[order].astype(np.int64)
    cs = np.cumsum(f) - f  # exclusive cumsum
    start = np.ones(len(sk), dtype=bool)
    start[1:] = sk[1:] != sk[:-1]
    base = np.maximum.accumulate(np.where(start, np.arange(len(sk)), 0))
    out = np.empty(len(sk), dtype=np.int64)
    out[order] = cs - cs[base]
    return out


def causal_gap(keys):
    """g[t] = t - (last s < t with keys[s] == keys[t]); 0 if none."""
    order = np.lexsort((np.arange(len(keys)), keys))
    sk = keys[order]
    same = np.zeros(len(sk), dtype=bool)
    same[1:] = sk[1:] == sk[:-1]
    pos = order
    gap = np.zeros(len(sk), dtype=np.int64)
    gap[1:] = np.where(same[1:], pos[1:] - pos[:-1], 0)
    out = np.empty(len(sk), dtype=np.int64)
    out[order] = gap
    return out


def doc_ids(tokens, tok_bytes):
    """Article index per position. An article starts at a line ' = Title = ' (single '=' level)."""
    lens = np.array([len(b) for b in tok_bytes])
    text = b"".join(tok_bytes[t] for t in tokens)
    starts = [0]
    pos = 0
    for line in text.split(b"\n"):
        s = line.strip()
        if s.startswith(b"= ") and s.endswith(b" =") and not s.startswith(b"= ="):
            starts.append(pos)
        pos += len(line) + 1
    byte_off = np.concatenate([[0], np.cumsum(lens[tokens])[:-1]])
    return np.searchsorted(np.array(starts), byte_off, side="right") - 1


def load_tok_bytes(vocab_dir):
    meta = json.load(open(os.path.join(vocab_dir, "meta.json")))
    b = [bytes([i]) for i in range(256)]
    for x, y in meta["merges"]:
        b.append(b[x] + b[y])
    b.append(b"")
    return b


class CountMemory:
    def __init__(self, train_tokens, train_docs, K=6, Kd=4):
        self.K, self.Kd = K, Kd
        x = np.asarray(train_tokens)
        hs = context_hashes(x, K)
        tgt = slice(1, None)  # targets are positions 1..N-1
        self.ctx, self.pair, self.nplus = [], [], []
        for k in range(K):
            ck = hs[k][tgt]
            pk = pair_keys(hs[k], x)[tgt]
            self.ctx.append(Counter(ck))
            self.pair.append(Counter(pk))
            up = np.unique(np.stack([ck, pk], 1), axis=0)  # distinct (ctx, pair)
            self.nplus.append(Counter(up[:, 0]))
            del up
        del hs

    def stats(self, tokens, docs, is_train=False):
        """Returns dict of int64 arrays of shape (K, N-1) / (Kd, N-1) for targets tokens[1:].
        is_train: the stream is (a set of whole articles of) the train split -> leave-one-document-out."""
        x, docs = np.asarray(tokens), np.asarray(docs)
        hs = context_hashes(x, self.K)
        d = docs[1:]
        C = np.zeros((self.K, len(x) - 1), dtype=np.int64)
        Cy, Np = np.zeros_like(C), np.zeros_like(C)
        D = np.zeros((self.Kd, len(x) - 1), dtype=np.int64)
        Dy, DN, DG = np.zeros_like(D), np.zeros_like(D), np.zeros_like(D)
        for k in range(self.K):
            ck = hs[k][1:]
            pk = pair_keys(hs[k], x)[1:]
            C[k], Cy[k], Np[k] = self.ctx[k](ck), self.pair[k](pk), self.nplus[k](ck)
            if is_train:  # leave-one-document-out
                dck, dpk = doc_keys(ck, d), doc_keys(pk, d)
                C[k] -= Counter(dck)(dck)
                cyd = Counter(dpk)(dpk)
                Cy[k] -= cyd
                # distinct continuations that occur only inside this document disappear
                excl_pair = (cyd == self.pair[k](pk))
                up, first = np.unique(dpk, return_index=True)
                ex_ctx = dck[first][excl_pair[first]]
                Np[k] -= Counter(ex_ctx)(dck)
            if k < self.Kd:
                dck, dpk = doc_keys(ck, d), doc_keys(pk, d)
                D[k], Dy[k] = causal_rank(dck), causal_rank(dpk)
                DN[k] = causal_distinct(dck, dpk)
                DG[k] = causal_gap(dck)
        C, Cy, Np = np.maximum(C, 0), np.maximum(Cy, 0), np.maximum(Np, 0)
        Cy = np.minimum(Cy, C)
        # position inside the document
        dpos = causal_rank(d.astype(np.uint64))
        return dict(C=C, Cy=Cy, Np=Np, D=D, Dy=Dy, DN=DN, DG=DG, dpos=dpos)


def features(st, extra=True):
    """Gate features (context-only) and expert probabilities of the target."""
    C, Cy, Np, D, Dy = st["C"], st["Cy"], st["Np"], st["D"], st["Dy"]
    f = [np.log1p(C), np.log1p(Np), (C > 0).astype(np.float64), np.log1p(D), (D > 0).astype(np.float64),
         np.log1p(st["dpos"])[None]]
    if extra:
        f += [np.log1p(st["DN"]), np.log1p(st["DG"])]
    F = np.concatenate(f, 0).T.astype(np.float32)
    P = np.concatenate([Cy / np.maximum(C, 1), Dy / np.maximum(D, 1)], 0).T.astype(np.float32)
    avail = np.concatenate([C > 0, D > 0], 0).T
    return F, P, avail
