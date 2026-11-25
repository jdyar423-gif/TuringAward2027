"""The reader's own test of whether a stored answer suits the user's question.

The answers library ranks stored questions by the words they share with the user's message. Shared
words do not say whether the stored *answer* answers the user's question ("Why does ice float on
water?" and "Why does oil float on water?" share all but one), and different words do not say
that it does not ("How can I tell whether an egg is still fresh?" and "How do you check if an egg
has gone bad?"). The model can be asked directly. For a stored entry ``(q_e, a)`` and the user's
question ``q``::

    fit(q, q_e, a) = mean over the tokens of a of  log p(a_t | q, a_<t) - log p(a_t | q_e, a_<t)

Both terms are teacher-forced passes of the model over the same answer in a bare two-turn
conversation (no library turn). ``fit = 0``: the answer is as probable a reply to the user's
question as to the question it was written for. ``fit < 0``: the user's question makes it less
probable. The second term is what makes the number comparable between answers: a short factual
answer and a long poem have very different likelihoods, but each is compared with itself.

The mean says whether the answer as a whole suits the question; it does not see one decisive
word. "What does the idiom *break the ice* mean?" beside the stored "What does the idiom *break a
leg* mean?": the stored answer is a long explanation of which three tokens are impossible under
the user's question and three hundred are indifferent to it, and the mean stays near zero. So the
test has a second part, local where the first is global::

    window(q, q_e, a) = max over t of  sum_{i=t..t+2} [ log p(a_i | q_e, a_<i) - log p(a_i | q, a_<i) ]

the most the user's question costs any three consecutive tokens of the answer, in nats. With
``max_window`` an answer whose window exceeds it is not suitable whatever its mean
(``DecodeCfg.verbatim_fit_window``).

``AnswerFit(params, consts, cfg, tok)`` is the callable ``AnswerLibrary.verbatim(fit=...)`` takes
(``DecodeCfg.verbatim_fit``; ``bes.decode.respond.Responder``). It reads the same weights as a
reply does and nothing else; an answer too long for the window is cut at the same token for both
passes. One compiled program serves every call (a row of ``max_tokens`` positions)."""
from __future__ import annotations

from typing import Any

import numpy as np


class AnswerFit:
    """``fit(question, stored_question, answer, max_window=None) -> float | None`` (module
    docstring): the mean, or ``None`` when the answer cannot be judged or its window exceeds
    ``max_window``.

    params, consts, cfg: the model as ``bes.model.model.loss_fn`` takes it (``cfg`` a
    ``ModelCfg``); tok: its tokenizer; max_tokens: positions of one pass (question + answer;
    longer answers are cut); min_answer_tokens: an answer left with fewer tokens after the cut is
    not judged (``None``); span: the tokens of the local window."""

    def __init__(self, params: Any, consts: Any, cfg: Any, tok: Any, *, max_tokens: int = 320,
                 min_answer_tokens: int = 4, span: int = 3):
        self.params, self.consts, self.cfg, self.tok = params, consts, cfg, tok
        self.T = int(max_tokens)
        self.min_answer_tokens = int(min_answer_tokens)
        self.span = int(span)
        self._ev = None
        self.calls = 0

    # -- rows -----------------------------------------------------------------------------
    def _render(self, question: str, answer: str):
        from bes.data import chat_format as cf

        return cf.render_conversation([{"role": "user", "content": question},
                                       {"role": "assistant", "content": answer}], self.tok)

    def cut(self, questions: tuple[str, ...], answer: str) -> str | None:
        """``answer``, shortened (whole tokens off its end) until its conversation with each of
        ``questions`` fits one pass; ``None`` when too little of it would be left."""
        text = answer
        for _ in range(4):
            over = max(len(self._render(q, text)[0]) for q in questions) - (self.T + 1)
            if over <= 0:
                return text
            ids = self.tok.encode(text)
            keep = len(ids) - over - 2
            if keep < self.min_answer_tokens:
                return None
            text = self.tok.decode_bytes(ids[:keep]).decode("utf-8", errors="ignore")
        return None

    def losses(self, question: str, answer: str) -> np.ndarray | None:
        """``float64[n]``: the negative log-likelihood of every token of the answer (its end
        token included) in the conversation ``user: question / assistant: answer``; ``None``
        when it does not fit one pass."""
        import jax

        from bes.data import packing as P
        from bes.data.samples import WeightRule
        from bes.exact.features import exact_features
        from bes.model import model as M

        ids, roles = self._render(question, answer)
        if len(ids) > self.T + 1:
            return None
        row = P.assemble_row([(np.asarray(ids, np.int32), np.asarray(roles, np.int8))], self.T,
                             WeightRule(assistant_only=True))
        if int(np.count_nonzero(row.w)) < 2:
            return None
        tokens = np.asarray(row.tokens, np.int32)[None]
        seg = np.asarray(row.seg, np.int32)[None]
        batch = {"tokens": tokens, "w": np.asarray(row.w, np.float32)[None], "seg": seg,
                 **exact_features(tokens[:, : self.T], seg, self.tok)}
        if self._ev is None:
            consts, cfg = self.consts, self.cfg

            def per_token(p, b):
                feats, y, _ = M.split_batch(b)
                h, _ = M.forward(p, consts, feats, cfg)
                return M.token_ce(p, consts, h, feats, y, cfg)

            self._ev = jax.jit(per_token)
        ce = np.asarray(self._ev(self.params, batch), np.float64).reshape(-1)
        self.calls += 1
        return ce[np.asarray(row.w).reshape(-1) > 0]

    def nll(self, question: str, answer: str) -> tuple[float, int] | None:
        """``(mean negative log-likelihood of the answer's tokens, their number)``; ``None`` when
        the conversation does not fit one pass."""
        ce = self.losses(question, answer)
        return None if ce is None else (float(ce.mean()), int(ce.size))

    # -- the test -------------------------------------------------------------------------
    def detail(self, question: str, stored_question: str, answer: str) -> dict[str, float] | None:
        """``{"fit", "window", "tokens"}`` of the module docstring, or ``None`` when the answer
        cannot be judged."""
        text = self.cut((question, stored_question), answer)
        if text is None:
            return None
        a = self.losses(question, text)
        b = self.losses(stored_question, text)
        if a is None or b is None or a.size != b.size:
            return None
        d = a - b                                   # what the user's question costs each token
        k = min(self.span, int(d.size))
        window = float(np.convolve(d, np.ones(k), "valid").max())
        return {"fit": float(-d.mean()), "window": window, "tokens": int(d.size)}

    def __call__(self, question: str, stored_question: str, answer: str,
                 max_window: float | None = None) -> float | None:
        r = self.detail(question, stored_question, answer)
        if r is None or (max_window is not None and r["window"] > float(max_window)):
            return None
        return r["fit"]


__all__ = ["AnswerFit"]
