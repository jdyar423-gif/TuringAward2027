# BES-100M-A20M — engineering specification (part 1: rules, formats, exact layer, batch)

Read `docs/PLAN.md` (approved plan, Traditional Chinese) for the why. This file and
`docs/SPEC_MODEL_TRAIN.md` are the contracts between tracks. `bes/config.py` holds every shared
constant; import from it, never repeat literals.

## 0. Ground rules for every contributor (human or agent)

1. **Ownership.** Each track owns the directories named in its brief. Do not edit files outside
   them. `bes/config.py` is shared: you may *append* a field with a default to your own dataclass
   using a targeted edit (never rewrite the file) and must list it in your report.
2. **No git.** Do not run `git add/commit/push/checkout/stash`. The orchestrator commits.
3. **Tests.** Every module ships tests under `tests/<area>/`. Run only your own area
   (`python -m pytest tests/<area> -q`). Exact code is tested against a brute-force reference
   written independently inside the test. A feature without a test does not exist.
4. **Machine limits.** 4 CPU cores, about 15 GB RAM shared with other running work. Keep any
   process under 3 GB, use at most 2 threads, and keep test files fast (a whole area under
   ~3 minutes; mark anything longer `@pytest.mark.slow`). Never start long background jobs.
5. **Dependencies.** Python 3.13, numpy 2.x, jax/jaxlib 0.11.1 (the Colab TPU version; do not
   upgrade), optax 0.2.9, tiktoken, rustbpe, regex, pyarrow, huggingface_hub. Ask in your report
   before adding anything else; do not `pip install` new packages.
6. **Target hardware.** One TPU v6e (JAX/XLA, static shapes, bf16 activations, fp32 logits and
   normalisation statistics, no scatter on large tables, matmul-centric code). All device code
   must also run on CPU in float32 for tests.
7. **Honesty.** The final report states what was implemented, what was verified and how, what is
   untested, and any deviation from this spec with the reason. Do not claim a test passed unless
   you ran it. Paste the final pytest summary line.
8. **Style.** Plain typed Python, small pure functions, docstrings that state shapes and dtypes,
   no model identifiers or agent names anywhere in the code or docs, MIT attribution comments
   where code is ported (repository + file).
9. Network: huggingface.co, datasets-server.huggingface.co, pypi.org and
   raw.githubusercontent.com are reachable; api.github.com is not. Download only small samples
   into `data_cache/` (git-ignored) unless your brief says otherwise.

## 1. Tokens and chat format

- Byte-level BPE, `VOCAB_SIZE = 8192`. Ids `0..255` are raw bytes, `256..8175` merges, the last
  16 ids are the special tokens of `bes.config.SPECIAL_TOKENS` (`PAD, BOS, SYSTEM, USER,
  ASSISTANT, LIBRARY, END, CALC, EQ, CALC_END`, six reserved).
- `bes/tokenizer/tok.py`: `load() -> Tokenizer` (reads `assets/tokenizer/`), `encode(text) ->
  list[int]` (never emits special ids, even if the text contains their strings),
  `decode(ids) -> str`, `token_bytes(id) -> bytes` (b"" for specials), `vocab_size`.
- A **segment** is one document or one conversation; it always starts with `BOS`.
  - Raw document: `BOS text`.
  - Conversation: `BOS` then turns. Each turn is `ROLE content END` with `ROLE` one of
    `SYSTEM, USER, ASSISTANT, LIBRARY`. A `LIBRARY` turn (retrieved passage, title first) is
    placed immediately before the `USER` turn it serves. Generation is prompted by a trailing
    `ASSISTANT` and stops at `END`.
  - Calculator span inside assistant content: `CALC expr EQ result CALC_END`. The host evaluates
    `expr` exactly and forces `result CALC_END`. For display the span renders as `expr = result`.
- Rows fed to the model pack one or more segments; `seg` gives each position its segment id
  (0, 1, 2, … within the row; padding gets its own id and zero loss weight). A row always starts
  with `BOS` (a document continued from a previous row starts with a fresh `BOS`).

## 2. Exact layer (`bes/exact/`, numpy only)

Everything here is a deterministic function of the token ids, used identically by the training
loader, by generation (incrementally) and by data relabelling. All statistics are **causal** and
**segment-local**: the value at position `t` depends only on tokens `<= t` of the same segment.

### 2.1 Byte-level transducer and registers

One finite-state transducer over bytes defines all formatting state. It is compiled once per
tokenizer into per-token tables so that running it over tokens equals running it over the bytes
(`instr_table.py`; cache under `assets/instr/`). State components (all small): backtick run
(0..2), inside ``` fence, inside `inline code`, inside "double quote", previous byte
alphanumeric, pending sentence terminator, line-start parser state (start / digits seen /
marker seen / inside line). Counters (int32, reset at every role token and at `BOS`): words,
sentences, lines, list items, tokens. Operational definitions (bytes, ASCII classes; all
non-ASCII bytes count as alphanumeric):
- word: starts at an alphanumeric byte whose predecessor is not alphanumeric or an apostrophe;
- sentence: counted when a run of `.`/`!`/`?` that follows a non-space byte is followed by
  whitespace, a closing quote/bracket, or the end of the turn; never inside a code fence;
- line: every `\n`;
- list item: a line that starts (after optional spaces) with `- `, `* `, `+ `, `• ` or
  digits followed by `.` or `)` and a space; never inside a code fence;
- bracket depth: `([{` +1, `)]}` −1, floored at 0, not counted inside code;
- quote parity toggles on `"` outside code; inline parity on single backticks outside a fence;
  fence parity on a run of exactly three or more backticks at line start.
`END` flushes a pending sentence terminator.

`registers.py`
- `compute(tokens: int32[B,T], seg: int32[B,T], targets: optional) -> Registers` with
  `reg: int16[B,T,N_REG]` (fields and cardinalities exactly `bes.config.REG_FIELDS`, values
  already clipped) and `clock: int32[B,T,N_CLOCKS]` = (sentences so far in the segment, lines so
  far in the segment, tokens since the turn started). Raw counters are also returned for tests.
- `Incremental` class with `step(token) -> (reg_row, clock_row)` giving identical values.
- Register values at position `t` describe the state **after** consuming token `t`.
- `rem_sent / rem_item / rem_word` count down inside an assistant turn towards the targets that
  `constraints.compile_from_tokens` extracts from the preceding user turn (0 when there is no
  target; encoding in `REG_FIELDS` comments). `turn_idx` counts role tokens in the segment.

### 2.2 Constraints, verifiers, calculator

`constraints.py`
- `measure(text) -> dict` : exact properties of a response through the same transducer (words,
  sentences, lines, paragraphs, list items, has_code_fence, ends_with_question, all_lower,
  all_upper, json_valid, …; about 25 properties).
- `compile(text) -> ConstraintSpec` : parse explicit, checkable requests from a user message
  (exactly / at most / at least N words, sentences, bullet points / items, paragraphs, lines;
  number words one..twenty; "one-sentence", "a single paragraph"; required keywords in quotes;
  JSON or code-block output). Conservative: when unsure, return no constraint.
- `verify(spec, text) -> list[str]` : violated constraints (empty list = satisfied).
- `render(spec, rng) -> str` : natural instruction text for a spec, from several templates per
  constraint type; `compile(render(spec))` must recover `spec` (property test).
- `compile_from_tokens(tokens) -> targets` used by `registers.compute`.

`calc.py`: `evaluate(expr: str) -> str | None` — whitelisted AST (numbers, `+ - * / // % **`,
parentheses, unary minus; exponent and magnitude limits), exact rational arithmetic, canonical
result string (integers without decimals, otherwise at most 6 significant decimals, no
scientific notation); `None` on anything else. No `eval`.

### 2.3 Precedent candidate tape (`ngram_heads.py`, `tape.py`)

For every position `t` the tape lists up to `CAND_SLOTS = 16` candidate next tokens with
features (`CAND_FEATS`). Slots 0-3 / 4-7 / 8-11 serve n-gram orders 1 / 2 / 3; slots 12-15 the
longest match.
- Order `o`: the context is the last `o` tokens ending at `t`. Among earlier positions `j < t`
  of the same segment whose last `o` tokens equal the context, with successor `x[j+1]`:
  `total` = their number `n`, `types` = number of distinct successors `K`. Candidates are up to
  four distinct successors, most recent first, searched among the 16 most recent such `j`; each
  carries its exact count `n_b` over **all** such `j`.
- Longest match: `L*` = length (capped at 32) of the longest suffix ending at `t` that also ends
  at an earlier `j < t`; if `L* >= 4`, candidates are the distinct successors of the positions
  with match length `L*` (most recent first), `mlen` bucket from `L*`.
- `src` = role of the candidate's most recent source position, or 5 when that position lies in
  the same assistant turn as `t` (self-generated evidence).
- Buckets for `count/total/types/mlen`: 0 → 0, 1 → 1, 2 → 2, 3-4 → 3, 5-8 → 4, 9-16 → 5,
  17-64 → 6, larger → 7.
- Special tokens never appear as candidates; context n-grams never cross a role token or `BOS`.
- Output: `cand_tok: int32[B,T,16]` (−1 = empty), `cand_feat: int8[B,T,16,N_CAND_FEAT]`.
  An incremental class must reproduce the batch function exactly.
- Throughput target: the full exact layer (registers + tape) at >= 60 sequences of 1024 tokens
  per second per core.

## 3. Batch format (loader → device)

A batch is a dict of numpy arrays with a leading micro-batch axis `A` added by the loader:

| key | dtype, shape | meaning |
|---|---|---|
| `tokens` | int32 `[A,B,T+1]` | inputs are `[..., :-1]`, targets `[..., 1:]` |
| `w` | float32 `[A,B,T]` | loss weight of each target (0 for padding, masked spans) |
| `seg` | int32 `[A,B,T]` | segment id of each input position |
| `reg` | int16 `[A,B,T,N_REG]` | registers of the input positions |
| `clock` | int32 `[A,B,T,N_CLOCKS]` | structural clocks |
| `cand_tok` | int32 `[A,B,T,16]` | candidate tape (−1 empty) |
| `cand_feat` | int8 `[A,B,T,16,N_CAND_FEAT]` | candidate features |

Two shapes are compiled: `T=1024, B=32, A=8` and `T=2048, B=16, A=8` (262,144 tokens per step).
