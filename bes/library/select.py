"""Cost-benefit greedy selection of passages under a budget of stored numbers.

Benefit model (a documented proxy; no question data is used):

    benefit(p) = importance(article) * (1 + pos)^(-pos_decay) * (1 - list_penalty * list_frac)
                 * n_words(p),                         and 0 for an exact duplicate body,
    importance(a) = 1 + mention_coef * ln(1 + mentions(a)) + length_coef * ln(1 + words(a) / 100)

``mentions(a)`` counts the other articles whose raw text contains the title of ``a``
(case-sensitive word n-gram match, disambiguator removed) - the closest thing to inlinks the
dump offers, since links were flattened to text; ``words(a)`` is the cleaned article length;
``pos`` is the passage index inside the article (lead first); ``list_frac`` is the share of
body words in list items. Cost is the passage's compressed size in stored numbers, measured
by packing the whole pool once in final order with the real codec (each passage is charged
its block's compression ratio), plus its directory entry and, for a lead passage, the title.

``select_for_budget`` takes passages in decreasing benefit/cost order until the estimated
cost reaches a target, packs them for real and corrects the target until the exact library
size lies within ``[budget * (1 - tol), budget]``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import regex

from bes.library.pack import PackCfg, Packed, pack
from bes.library.sources import Pool
from bes.library.terms import strip_disambiguator

_CASE_TOKEN_RX = regex.compile(r"[\p{L}\p{N}]+")


@dataclass(frozen=True)
class SelectCfg:
    pos_decay: float = 0.7
    mention_coef: float = 1.0
    length_coef: float = 0.5
    list_penalty: float = 0.5
    max_ngram: int = 4
    tol: float = 0.005
    max_iter: int = 8


# ---------------------------------------------------------------------- signals


def title_key(title: str) -> str:
    return " ".join(_CASE_TOKEN_RX.findall(strip_disambiguator(title)))


def mention_counts(titles: list[str], texts: Iterable[tuple[int, str]], max_n: int = 4,
                   progress: Callable[[int], None] | None = None) -> np.ndarray:
    """float64 [A]: number of other articles whose text mentions each title.

    ``texts`` yields (article index or -1 for texts outside the pool, raw text). A title
    key shared by several articles splits its count evenly between them.
    """
    owners: dict[str, list[int]] = {}
    for a, t in enumerate(titles):
        k = title_key(t)
        if len(k) >= 2:
            owners.setdefault(k, []).append(a)
    keys = owners
    max_n = max(1, max_n)
    hits: dict[str, int] = {}
    for j, (a, text) in enumerate(texts):
        toks = _CASE_TOKEN_RX.findall(text)
        own = title_key(titles[a]) if a >= 0 else None
        found: set[str] = set()
        n_tok = len(toks)
        for i, tk in enumerate(toks):
            c = tk[0]
            if not (c.isupper() or c.isdigit()):
                continue
            key = tk
            if key in keys:
                found.add(key)
            for n in range(2, max_n + 1):
                if i + n > n_tok:
                    break
                key = key + " " + toks[i + n - 1]
                if key in keys:
                    found.add(key)
        found.discard(own)
        for k in found:
            hits[k] = hits.get(k, 0) + 1
        if progress is not None and j % 20000 == 0:
            progress(j)
    out = np.zeros(len(titles), dtype=np.float64)
    for k, c in hits.items():
        arts = owners[k]
        for a in arts:
            out[a] += c / len(arts)
    return out


def duplicate_mask(pool: Pool) -> np.ndarray:
    """bool [P]: True for every passage whose normalised body equals an earlier one."""
    seen: set[bytes] = set()
    dup = np.zeros(pool.n_passages, dtype=bool)
    for i in range(pool.n_passages):
        norm = " ".join(pool.body(i).lower().split()).encode("utf-8")
        h = hashlib.blake2b(norm, digest_size=12).digest()
        if h in seen:
            dup[i] = True
        else:
            seen.add(h)
    return dup


def importance(pool: Pool, mentions: np.ndarray, cfg: SelectCfg = SelectCfg()) -> np.ndarray:
    return (1.0 + cfg.mention_coef * np.log1p(mentions)
            + cfg.length_coef * np.log1p(pool.article_words.astype(np.float64) / 100.0))


def passage_prior(pool: Pool, imp: np.ndarray, cfg: SelectCfg = SelectCfg()) -> np.ndarray:
    """float64 [P]: benefit per word, ``importance * (1 + pos)^-decay * list factor`` (> 0)."""
    pos = pool.p_pos.astype(np.float64)
    return (imp[pool.p_article] * (1.0 + pos) ** (-cfg.pos_decay)
            * (1.0 - cfg.list_penalty * pool.p_list.astype(np.float64)))


def benefits(pool: Pool, imp: np.ndarray, dup: np.ndarray, cfg: SelectCfg = SelectCfg()) -> np.ndarray:
    b = passage_prior(pool, imp, cfg) * pool.p_words.astype(np.float64)
    b[dup] = 0.0
    return b


# ---------------------------------------------------------------------- ordering


def article_order(pool: Pool) -> np.ndarray:
    """Pack order of articles: by first category, then title (similar stubs end up in the
    same block, which the block compressor rewards)."""
    keys = sorted(range(pool.n_articles), key=lambda a: (pool.category[a] or "￿", pool.titles[a], a))
    return np.asarray(keys, dtype=np.int64)


def passage_order(pool: Pool, art_order: np.ndarray) -> np.ndarray:
    """Passage indices in pack order (articles in ``art_order``, passages by position)."""
    rank = np.empty(pool.n_articles, dtype=np.int64)
    rank[art_order] = np.arange(art_order.shape[0])
    return np.lexsort((pool.p_pos.astype(np.int64), rank[pool.p_article]))


def materialize(pool: Pool, order: np.ndarray, mask: np.ndarray | None = None
                ) -> tuple[list[tuple[str, str]], np.ndarray]:
    """(title, body) pairs of the selected passages in pack order, and their pool indices."""
    idx = order if mask is None else order[mask[order]]
    out = [(pool.titles[int(pool.p_article[i])], pool.body(int(i))) for i in idx]
    return out, idx


def pack_selection(pool: Pool, order: np.ndarray, mask: np.ndarray | None,
                   cfg: PackCfg = PackCfg()) -> tuple[Packed, list[tuple[str, str]], np.ndarray]:
    passages, idx = materialize(pool, order, mask)
    titles: list[str] = []
    tids: list[int] = []
    last = -1
    for i in idx:
        a = int(pool.p_article[i])
        if a != last:
            titles.append(pool.titles[a])
            last = a
        tids.append(len(titles) - 1)
    return pack([b for _, b in passages], tids, titles, cfg), passages, idx


# ---------------------------------------------------------------------- costs


def estimate_costs(pool: Pool, order: np.ndarray, cfg: PackCfg = PackCfg()) -> tuple[np.ndarray, Packed]:
    """float64 [P] estimated stored numbers of each passage (see the module docstring),
    from one real pack of the whole pool in ``order``."""
    packed, _, idx = pack_selection(pool, order, None, cfg)
    raw = pool.body_bytes()[idx].astype(np.float64)
    comp_blocks = np.diff(packed.block_off.astype(np.int64)).astype(np.float64)
    ratio_blocks = comp_blocks / np.maximum(packed.block_raw.astype(np.float64), 1.0)
    bf = packed.block_first.astype(np.int64)
    block_of = np.repeat(np.arange(packed.n_blocks), np.diff(bf))
    cost = np.zeros(pool.n_passages, dtype=np.float64)
    cost[idx] = raw * ratio_blocks[block_of] / 2.0 + 1.0  # text + pas_off entry
    # titles: title_first entry (2 numbers) + compressed title text, charged to the lead
    t_ratio = float(packed.tblock_off[-1]) / max(1.0, float(np.asarray(packed.tblock_raw, np.int64).sum()))
    lead = pool.p_pos == 0
    tlen = np.array([len(t.encode("utf-8")) + 1 for t in pool.titles], dtype=np.float64)
    cost[lead] += 2.0 + tlen[pool.p_article[lead]] * t_ratio / 2.0
    return cost, packed


# ---------------------------------------------------------------------- greedy


def greedy_mask(ratio: np.ndarray, cost: np.ndarray, target: float, eligible: np.ndarray) -> np.ndarray:
    """Take eligible items by decreasing ratio (ties: lower index) while the cumulative cost
    stays within ``target``; items that do not fit are skipped, smaller later ones may fit."""
    idx = np.flatnonzero(eligible)
    order = idx[np.lexsort((idx, -ratio[idx]))]
    c = cost[order]
    cum = np.cumsum(c)
    mask = np.zeros(cost.shape[0], dtype=bool)
    k = int(np.searchsorted(cum, target, side="right"))
    mask[order[:k]] = True
    used = float(cum[k - 1]) if k else 0.0
    for j in order[k:]:  # fill the remainder (costs are small, this rarely adds much)
        if used + cost[j] <= target:
            mask[j] = True
            used += cost[j]
    return mask


@dataclass
class Selection:
    mask: np.ndarray            # bool [P] selected pool passages
    packed: Packed              # the exact packed library
    passages: list[tuple[str, str]]
    pool_idx: np.ndarray        # pool index of every packed passage (pack order)
    history: list[dict]


def select_for_budget(pool: Pool, budget_numbers: int, benefit: np.ndarray, cost: np.ndarray,
                      order: np.ndarray, pack_cfg: PackCfg = PackCfg(), cfg: SelectCfg = SelectCfg(),
                      log: Callable[[str], None] | None = None) -> Selection:
    """Greedy selection whose packed size (text + directory + titles) is within
    ``[budget * (1 - tol), budget]`` (or everything eligible, if that fits)."""
    eligible = benefit > 0
    ratio = np.where(cost > 0, benefit / np.maximum(cost, 1e-9), 0.0)
    history: list[dict] = []
    everything = float(cost[eligible].sum())
    # when everything fits by estimate, aim at the budget itself (the greedy then takes all)
    target = float(budget_numbers) if everything <= budget_numbers else float(budget_numbers)
    lo_t, hi_t = 0.0, None  # bracket on the target
    best = None
    for it in range(cfg.max_iter):
        mask = greedy_mask(ratio, cost, target, eligible)
        packed, passages, idx = pack_selection(pool, order, mask, pack_cfg)
        actual = packed.accounting()["library_numbers"]
        history.append({"iter": it, "target": target, "selected": int(mask.sum()), "actual": actual})
        if log is not None:
            log(f"select iter {it}: target {target:.0f} selected {int(mask.sum())} actual {actual}")
        if actual <= budget_numbers:
            if best is None or actual > best[0]:
                best = (actual, mask, packed, passages, idx)
            if actual >= budget_numbers * (1 - cfg.tol) or mask.sum() == eligible.sum():
                break
            lo_t = target
            nxt = target * budget_numbers / max(actual, 1)
            target = nxt if hi_t is None else min(nxt, (lo_t + hi_t) / 2)
        else:
            hi_t = target
            nxt = target * budget_numbers / actual * (1 - cfg.tol / 2)
            target = max(nxt, (lo_t + hi_t) / 2) if lo_t > 0 else nxt
    if best is None:
        raise RuntimeError("could not fit the budget")
    actual, mask, packed, passages, idx = best
    return Selection(mask, packed, passages, idx, history)
