"""Per-token tables of the byte-level transducer (``docs/SPEC.md`` section 2.1).

For a tokenizer with ``V`` ids and the ``S`` reachable states of
``bes.exact.transducer.compile_machine()`` (state 0 is the turn-start state ``INIT``), row ``v``
gives, for every entry state ``s``, the state after the bytes of token ``v`` and what those
bytes count:

- ``next_state[v, s]``: ``uint8[V + 1, S]``;
- ``inc[v, s, :]``: ``int8[V + 1, S, N_INC]`` with the fields of ``INC_FIELDS``:
  ``words, sentences, lines, items`` (events counted inside the token), ``ws_reset`` (1 when a
  sentence was counted inside the token), ``ws_after`` (words after the last sentence counted
  inside the token, or all its words when there is none), ``dep_add, dep_lo`` (the token maps
  a bracket depth ``d >= 0`` to ``max(dep_lo, d + dep_add)``).

Running these rows over token ids equals running the transducer over the concatenated bytes
(``tests/exact/test_instr_table.py``). Rows of special tokens (``kind`` gives the kind of every
id, ``role_code`` the role register value of role tokens):

- ``BOS`` and role tokens: every state goes to ``INIT``, nothing is counted (the counters are
  reset by the register layer);
- ``END``: the transducer's flush (an open terminator run counts as a sentence);
- every other special token (``PAD``, ``CALC``, ``EQ``, ``CALC_END``, reserved): identity;
- the extra row ``eq_row = V`` is the virtual token ``" = "``, which the register layer uses in
  place of ``EQ`` when ``EQ`` opens the result of a calculator span (the display form of
  ``CALC expr EQ result CALC_END`` is ``expr = result``).

The tables are a pure function of the tokenizer's byte strings and of the transducer. They are
built in about a second and cached under ``assets/instr/`` as ``instr_<tok>_<machine>.npz``,
keyed by the tokenizer SHA-256 and by a fingerprint of the transducer tables, so a change of
either rebuilds them. The cache files are git-ignored: delete them at any time.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bes.exact.transducer import ByteMachine, compile_machine

FORMAT = "bes-instr/1"
CACHE_DIR = Path(__file__).resolve().parents[2] / "assets" / "instr"
EQ_BYTES = b" = "

INC_FIELDS = ("words", "sentences", "lines", "items", "ws_reset", "ws_after", "dep_add", "dep_lo")
(INC_WORDS, INC_SENT, INC_LINES, INC_ITEMS, INC_WS_RESET, INC_WS_AFTER, INC_DEP_ADD,
 INC_DEP_LO) = range(len(INC_FIELDS))
N_INC = len(INC_FIELDS)

#: Kinds of token ids (``InstrTable.kind``).
KIND_ORDINARY, KIND_BOS, KIND_ROLE, KIND_END, KIND_CALC, KIND_EQ, KIND_CALC_END, KIND_OTHER = range(8)
_KIND_OF_SPECIAL = {
    "<|bos|>": KIND_BOS, "<|system|>": KIND_ROLE, "<|user|>": KIND_ROLE,
    "<|assistant|>": KIND_ROLE, "<|library|>": KIND_ROLE, "<|end|>": KIND_END,
    "<|calc|>": KIND_CALC, "<|eq|>": KIND_EQ, "<|/calc|>": KIND_CALC_END,
}
#: Role register values of the role tokens (``bes.config.REG_FIELDS`` "role").
ROLE_CODE_OF_SPECIAL = {"<|system|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|library|>": 4}


@dataclass(frozen=True, eq=False)
class InstrTable:
    """Compiled per-token tables for one tokenizer (shapes in the module docstring)."""

    tokenizer_sha256: str
    key: str
    vocab_size: int
    n_states: int
    next_state: np.ndarray   # uint8 [V + 1, S]
    inc: np.ndarray          # int8 [V + 1, S, N_INC]
    kind: np.ndarray         # int8 [V]
    role_code: np.ndarray    # int8 [V], 0 for ids that are not role tokens
    components: np.ndarray   # int8 [S, 7], transducer.COMPONENTS of every state

    init_state = 0

    @property
    def eq_row(self) -> int:
        return self.vocab_size

    @functools.cached_property
    def next_flat(self) -> np.ndarray:
        """``intp[(V + 1) * S]``: ``next_state`` flattened, for ``next_flat[row * S + s]``."""
        return self.next_state.astype(np.intp).ravel()

    def special(self, kind: int) -> int:
        """The id of the special token of a unique ``kind`` (BOS, END, CALC, EQ, CALC_END)."""
        ids = np.flatnonzero(self.kind == kind)
        if ids.size != 1:
            raise ValueError(f"kind {kind} is not carried by exactly one id")
        return int(ids[0])


# ------------------------------------------------------------------------------ building
def _token_rows(tok) -> tuple[list[bytes], np.ndarray, np.ndarray]:
    """Byte strings of rows ``0..V`` (specials empty, row ``V`` = ``EQ_BYTES``), kinds, role codes."""
    vocab = int(tok.vocab_size)
    data = list(tok.all_token_bytes())
    if len(data) != vocab:
        raise ValueError("all_token_bytes() does not list vocab_size entries")
    kind = np.zeros(vocab, np.int8)
    role_code = np.zeros(vocab, np.int8)
    for name, tid in tok.special_tokens.items():
        kind[tid] = _KIND_OF_SPECIAL.get(name, KIND_OTHER)
        role_code[tid] = ROLE_CODE_OF_SPECIAL.get(name, 0)
        data[tid] = b""
    for tid in np.flatnonzero(kind == KIND_ORDINARY):
        if not data[tid]:
            raise ValueError(f"ordinary id {tid} has no bytes")
    for k in (KIND_BOS, KIND_END, KIND_CALC, KIND_EQ, KIND_CALC_END):
        if np.count_nonzero(kind == k) != 1:
            raise ValueError(f"the tokenizer must have exactly one special token of kind {k}")
    return data + [EQ_BYTES], kind, role_code


def build(tok, machine: ByteMachine | None = None) -> InstrTable:
    """Compile the tables of ``tok`` (needs ``vocab_size``, ``all_token_bytes()``,
    ``special_tokens`` and ``sha256``) by composing the per-byte tables of ``machine``.
    Tokens of equal length are processed together, all entry states at once."""
    m = machine if machine is not None else compile_machine()
    rows, kind, role_code = _token_rows(tok)
    n_rows, n_states = len(rows), m.n_states
    if n_states > 255:
        raise ValueError("more than 255 transducer states; widen the next_state dtype")
    next_state = np.empty((n_rows, n_states), np.uint8)
    inc = np.zeros((n_rows, n_states, N_INC), np.int8)

    by_len: dict[int, list[int]] = {}
    for r, data in enumerate(rows):
        if data:
            by_len.setdefault(len(data), []).append(r)
    ev_tab = m.events.astype(np.int64)
    nx_tab = m.next_state.astype(np.intp)
    for length, ids in sorted(by_len.items()):
        if length > 127:
            raise ValueError("token longer than 127 bytes; widen the inc dtype")
        idx = np.asarray(ids)
        mat = np.frombuffer(b"".join(rows[r] for r in ids), np.uint8).reshape(len(ids), length)
        s = np.broadcast_to(np.arange(n_states, dtype=np.intp), (len(ids), n_states)).copy()
        acc = np.zeros((len(ids), n_states, N_INC), np.int64)
        for j in range(length):
            b = mat[:, j].astype(np.intp)[:, None]
            ev = ev_tab[s, b]                                    # [n, S, 5]
            word, sent, newline, item, bracket = (ev[..., k] for k in range(5))
            acc[..., INC_WORDS] += word
            acc[..., INC_SENT] += sent
            acc[..., INC_LINES] += newline
            acc[..., INC_ITEMS] += item
            acc[..., INC_WS_RESET] |= sent
            acc[..., INC_WS_AFTER] = np.where(sent > 0, 0, acc[..., INC_WS_AFTER]) + word
            acc[..., INC_DEP_ADD] += bracket
            acc[..., INC_DEP_LO] = np.maximum(0, acc[..., INC_DEP_LO] + bracket)
            s = nx_tab[s, b]
        next_state[idx] = s
        inc[idx] = acc

    identity = np.arange(n_states)
    for tid in np.flatnonzero(kind != KIND_ORDINARY):
        k = int(kind[tid])
        if k in (KIND_BOS, KIND_ROLE):
            next_state[tid] = 0
        elif k == KIND_END:
            next_state[tid] = m.flush_state
            inc[tid, :, INC_SENT] = m.flush_sentence
            inc[tid, :, INC_WS_RESET] = m.flush_sentence
        else:
            next_state[tid] = identity
    return InstrTable(str(tok.sha256), table_key(str(tok.sha256), m), int(tok.vocab_size),
                      n_states, next_state, inc, kind, role_code, m.components.copy())


def table_key(tokenizer_sha256: str, machine: ByteMachine | None = None) -> str:
    """Cache key: tokenizer hash prefix and a hash of the transducer and of this format."""
    m = machine if machine is not None else compile_machine()
    blob = json.dumps([FORMAT, m.fingerprint, EQ_BYTES.hex()]).encode()
    return f"{tokenizer_sha256[:16]}_{hashlib.sha256(blob).hexdigest()[:16]}"


# --------------------------------------------------------------------------------- cache
def cache_path(tokenizer_sha256: str, cache_dir: str | Path = CACHE_DIR,
               machine: ByteMachine | None = None) -> Path:
    return Path(cache_dir) / f"instr_{table_key(tokenizer_sha256, machine)}.npz"


def save(table: InstrTable, path: str | Path) -> None:
    """Write ``table`` to ``path`` atomically (temporary file in the same directory, fsync, rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"format": FORMAT, "tokenizer_sha256": table.tokenizer_sha256,
                       "key": table.key, "vocab_size": table.vocab_size,
                       "n_states": table.n_states})
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        os.chmod(tmp, 0o644)
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, meta=np.array(meta), next_state=table.next_state,
                                inc=table.inc, kind=table.kind, role_code=table.role_code,
                                components=table.components)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read(path: str | Path) -> InstrTable:
    """Read a table written by :func:`save`; raises ``ValueError`` on a malformed file."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in ("next_state", "inc", "kind", "role_code", "components")}
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: unknown format")
    v, s = int(meta["vocab_size"]), int(meta["n_states"])
    expect = {"next_state": ((v + 1, s), np.uint8), "inc": ((v + 1, s, N_INC), np.int8),
              "kind": ((v,), np.int8), "role_code": ((v,), np.int8),
              "components": ((s, 7), np.int8)}
    for name, (shape, dtype) in expect.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"{path}: array {name} has the wrong shape or dtype")
    return InstrTable(meta["tokenizer_sha256"], meta["key"], v, s, **arrays)


_LOADED: dict[tuple[str, str], InstrTable] = {}


def load(tok=None, cache_dir: str | Path | None = CACHE_DIR) -> InstrTable:
    """Tables of ``tok`` (the committed tokenizer by default), cached in-process and on disk.

    The disk cache under ``cache_dir`` is read when it matches the tokenizer hash and the
    transducer fingerprint, and rebuilt (then written atomically) otherwise. ``cache_dir=None``
    builds without touching the disk. A cache that cannot be written is not an error.
    """
    if tok is None:
        from bes.tokenizer.tok import load as load_tokenizer

        tok = load_tokenizer()
    machine = compile_machine()
    key = table_key(str(tok.sha256), machine)
    memo = (key, str(cache_dir))
    if memo in _LOADED:
        return _LOADED[memo]
    table = None
    if cache_dir is not None:
        path = cache_path(str(tok.sha256), cache_dir, machine)
        if path.exists():
            try:
                table = read(path)
            except Exception:       # unreadable, truncated or foreign file: rebuild it
                table = None
            if table is not None and (table.key != key or table.vocab_size != tok.vocab_size
                                      or table.tokenizer_sha256 != str(tok.sha256)):
                table = None
        if table is None:
            table = build(tok, machine)
            try:
                save(table, path)
            except OSError:
                pass
    else:
        table = build(tok, machine)
    _LOADED[memo] = table
    return table


def run_rows(table: InstrTable, rows, state: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run table rows sequentially from ``state``: ``(state_in int64[n], state_out int64[n],
    inc int64[n, N_INC])``, where step ``i`` reads row ``rows[i]`` (a token id or ``eq_row``)."""
    rows = np.asarray(rows, dtype=np.int64)
    n = rows.shape[0]
    s_in = np.empty(n, np.int64)
    s_out = np.empty(n, np.int64)
    for i, r in enumerate(rows.tolist()):
        s_in[i] = state
        state = int(table.next_state[r, state])
        s_out[i] = state
    return s_in, s_out, table.inc[rows, s_in].astype(np.int64)
