# Proxy blind test: protocol

The proxy blind test is the instrument behind every decision of the 24-hour run (decoding
settings, library on or off, particle count, which checkpoint ships) and behind the final claim
against the two opponents, `tiiuae/Falcon-H1-Tiny-90M-Instruct` and
`facebook/MobileLLM-R1.5-140M`. It runs on the host CPUs through llama.cpp and needs no network
once the files are fetched. Code: `bes/eval/proxy/` and `bes/eval/exact_score.py`; command line:
`python -m bes.eval.proxy`. Plan reference: `docs/PLAN.md` section 5, item 5.

Precision first: with n decisive comparisons (wins plus losses) the 95% interval of a win rate
is about plus or minus 1/sqrt(n). With the 400-prompt set and two opponents a difference
smaller than about 5 points cannot be seen; read every number in this light (section 8).

## 1. Prompt set (`bes/eval/proxy/prompts.jsonl`)

400 original English prompts written for this project (no benchmark items):

| category | prompts | two-turn | content |
|---|---|---|---|
| chat | 100 | 25 | everyday questions, explanations, advice, definitions, simple facts |
| math | 60 | 15 | arithmetic, percentages, unit conversion, one- and two-step word problems |
| code | 60 | 15 | short Python (and a little SQL, JavaScript, Bash), small functions, explain or fix an error |
| writing | 100 | 25 | short stories, emails, summaries of a given paragraph, rewrites, poems |
| instruct | 80 | 20 | explicit, machine-checkable constraints |

Fields: `id`, `category`, `turns` (one or two user messages; a second message depends on the
model's own first answer), `answer` (math: exact number of the last turn, or one entry per
turn), optional `tol`, `constraints`, `notes`. All 60 math answers are recomputed with exact
fractions inside `tests/eval/test_proxy_prompts.py`. 95 prompts carry constraints (all 80
instruct prompts plus 15 writing and chat prompts that state a checkable requirement, such as
"in one sentence"); every count constraint is stated in the prompt text (tested).

**Split.** Not stored: inside each stratum (category x number of turns) prompts are ordered by
SHA-256 of their id and alternately assigned to `dev` and `test`, so each category splits
exactly in half (200 / 200). Use `dev` for every decision during the run and keep `test`
untouched for the final claim. Editing a prompt changes its content hash, so stale outputs
are never paired.

## 2. How every model is run

One interface for all models: `generate(messages, settings, seed) -> Reply` (text, token
counts, latency, finish reason and, when available, the log-probability of the generated
tokens under the model's own unmodified distribution).

- **Conversations.** Each model sees the system message of its arm (if any), the user turns and
  **its own** earlier visible replies. Hidden reasoning is never fed back.
- **Seeds.** Candidate `k` of turn `t` of prompt `p` uses `derive_seed(seed, p, t, k)`;
  identical across models. Each call starts from an empty context, so a reply depends only on
  (model file, messages, settings, seed) and the thread count.
- **Cache.** One line per generation in `data_cache/proxy/cache/<model key>.jsonl`, keyed by a
  hash of (model key, messages, generation settings, seed, harness version), appended with fsync
  by the parent process. Every command resumes: rerunning skips cached generations, and a run
  with K = 4 reuses candidate 0 of a K = 1 run. A run is identified as
  `<model key>/<preset>-<settings hash>-s<seed>`; its conversations are written to
  `data_cache/proxy/outputs/<run id>.jsonl`.
- **Workers.** `--workers N --threads T`: N spawned processes, each with its own copy of the
  model and T llama.cpp threads. `--minutes M` stops starting new generations after M minutes.

### Falcon-H1-Tiny-90M-Instruct

File `Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf` from the publisher's repository
`tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF` (SHA-256 pinned in `opponents.py`), run with the chat
template stored in the file. The publisher gives no sampling parameters
(`generation_config.json` holds token ids only) and documents `llama-cli -cnv`, so the
**recommended** preset is llama.cpp's defaults: temperature 0.8, top-k 40, top-p 0.95,
min-p 0.05, no repetition penalty; at most 1024 new tokens.

### MobileLLM-R1.5-140M

- **File.** The upstream checkpoint is gated. The ungated community file
  `sasa2000/MobileLLM-R1.5-140M-Q8_0-GGUF` stores it as a dense `llama4` model with zero experts,
  which the llama.cpp bundled with llama-cpp-python 0.3.37 refuses to load (patching the expert
  count only reaches a failing assertion). `python -m bes.eval.proxy fetch mobilellm` rewrites
  it, without touching a single quantised value, as the equivalent `qwen3` model: renamed
  hyper-parameters, query and key rows reordered from interleaved to half-split rotary pairs,
  and unit per-head q/k RMS norms (the upstream applies a weightless RMS norm to each head after
  RoPE; that norm commutes with the rotation). Checked against an independent numpy forward pass
  of the original tensors with the upstream semantics: mean KL 0.003 nats per position, 98% of
  argmaxes equal, logit correlation 0.999; a model without the q/k norm scores KL 4.5 (slow test
  in `tests/eval/test_proxy_gguf_convert.py`).
- **Template.** The file has no chat template. `MOBILELLM_TEMPLATE` reproduces the official
  template's rendering for system / user / assistant messages (checked string for string against
  the template of the R1.5 checkpoints). **When no system message is given it injects "Please
  reason step by step, and put your final answer within \boxed{}."**
- **Reasoning.** The model opens every reply with `<think>`. The blind comparison shows only the
  text after `</think>` (a reply still inside the reasoning shows nothing). The publisher's own
  evaluation (`facebookresearch/MobileLLM-R1`, `evaluation/run_eval.sh`) samples with temperature
  0.6, top-p 0.95 and allows 30,000 tokens per call. That is unaffordable on CPU; with a plain
  4096-token cap the model ended inside its reasoning on all three chat prompts of a smoke run
  (4 of 7 replies empty). The presets therefore use **budget forcing**: after 3072 reasoning
  tokens an open `<think>` is closed (`\n</think>\n\n` is appended) and the model answers with the
  remaining 1024 tokens; with it no reply of the same smoke run was empty (3 of 7 still reached
  the cap while answering).
- **Two arms, run fairly.** `recommended`: its own template (with the injected reasoning prompt),
  temperature 0.6, top-p 0.95, budget forcing. `neutral`: the same with the system prompt "You
  are a helpful assistant." in place of the injected one. `noforce`: `recommended` without
  forcing, for reference. Both main arms are compared with ours; MobileLLM is credited with the
  better arm on `dev`, decided before `test` is looked at.

### Ours

`run_ours.py` defines the contract our generator must satisfy (`generator(messages, settings,
seed) -> str | dict`, deterministic, with a cache `key` naming checkpoint and decoding code);
`--model ours --generator module:factory --generator-arg k=v --key K`. Presets: `default`
(our decoder settings from `bes.config.DecodeCfg`: particles, library on), `k1` (one particle),
`nolib` (library off). Stand-ins (`standin-plain`, `standin-verbose`, `standin-loop`,
`standin-empty`) and `gguf_generator` (any GGUF behind the same contract) exist for tests and
rehearsals.

### Common setting

`common`: identical sampling for every model, our decoder's base temperature and top-p
(0.6 / 0.9), no other filter, no repetition penalty, 1024 tokens (MobileLLM: neutral system
prompt, hidden reasoning, 4096 tokens with budget forcing, because its template would otherwise
inject the reasoning prompt and its reasoning does not fit 1024 tokens).

## 3. What any claim must disclose

- Our model draws K samples per reply (particles, K = 4 to 8) and uses a built-in, offline,
  frozen library that counts against its 100M stored numbers. Results are reported with the
  library on and off, and with K = 1.
- **Control arm.** Each opponent is also run with the same K, keeping its candidate with the best
  mean log-probability under its own model (`--samples K --select logprob_mean`), and a
  second variant that first drops failed candidates exactly as our selector does
  (`filtered_logprob_mean`: empty, looping, cut off, unclosed code fence, leaked role markers).
  The llama.cpp backend reads the raw logits of every sampled token, so the log-probability is
  exact for opponents.
- Opponents run as Q8_0 GGUF files on CPU through llama.cpp, with the token caps above; MobileLLM
  hides its reasoning and uses budget forcing.
- Which presets were tried on `dev` and which were chosen, before `test` was opened.

## 4. Blinding (`make_ab.py`)

`python -m bes.eval.proxy pairs --x <our run> --y <opponent run> --name N` pairs every prompt
both runs answered. Who is "A" is a seeded assignment balanced overall and within every
category (difference at most one); the pair order in the file is a second seeded shuffle; pair
ids are salted hashes. Raters and the judge read only `data_cache/proxy/pairs/N/` (no run or
model names). The key (`data_cache/proxy/keys/N.json`: seed, salt, run ids, assignment) is
needed to unblind; its SHA-256 is published in `pairs/N/meta.json`, so a key changed after
rating is detected. Keep `keys/` away from raters. Style can still give a model away (for
example MobileLLM's `\boxed{}` answers); this cannot be hidden and is a reason to trust exact
scores alongside preferences.

## 5. Ratings

**Human** (`rate.py`): `python -m bes.eval.proxy rate --pairs N --rater ID`; keys `a` (A better),
`b` (B better), `t` (tie), `x` (both bad), `s` skip, `u` undo, `q` quit. Every rating is
appended at once; a new session continues with the unrated pairs; each rater gets a different
fixed order. Instruction to raters: judge as a user who asked the question; correctness first,
then whether the request was followed, then usefulness; length is not a merit.

**Automatic judge** (`judge.py`): an instruct model on CPU sees each pair twice, in the blinded
order and swapped. The prompt (`JUDGE_INSTRUCTIONS`) asks for correctness first, then following
every explicit instruction, then helpfulness; it says that length is not a merit beyond what the
request needs, that order and labels must be ignored, that repetition, unfinished text or writing
the user's next message are serious flaws, and to answer C for equal or equally poor replies. The
verdict is read from the next-token distribution over the letters A, B, C (one forward pass, no
generation; the share of probability on the three letters is logged as a format check). **A
side wins only if both orders pick the same reply; otherwise the pair is a tie.** The averaged
probabilities of the two orders are kept as a soft verdict. Replies longer than 4000 characters
are cut for the judge (marked in the prompt and counted in the report). The shared instruction
prefix is reused from the KV cache.

Judge models (files checked through the Hub API, sizes and SHA-256 pinned in `opponents.py`):

| name | file | size | use |
|---|---|---|---|
| `qwen3-4b-2507` | `unsloth/Qwen3-4B-Instruct-2507-GGUF` / `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2.50 GB | default judge for the real run (non-thinking 4B, plain attention, Apache-2.0) |
| `gemma4-e4b` | `unsloth/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_K_M.gguf` | 4.98 GB | second judge from another family (self-preference check) |
| `llama31-8b` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` / `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.92 GB | older, slower alternative |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` / `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 0.49 GB | plumbing tests on a small machine only; far too weak for verdicts |

The three large judges were not downloaded or run in the development container (download cap);
their architectures (`qwen3`, `gemma4`, `llama`) are in the bundled llama.cpp. Cost: on real
outputs a judge prompt is about 860 tokens (median; 90th percentile 1400, of which 280 are the
fixed instructions), so one pair costs two forward passes over roughly 1700 tokens, about 1200
of them new after prefix reuse.

**Known biases of LLM judges and what the protocol does about them**

- *Position*: judges prefer one slot. Both orders plus strict agreement turn pure position
  preference into ties; the report gives the mean preference for the first-shown reply.
- *Length and style*: judges reward long, confident, markdown-heavy answers. The prompt says
  length is not a merit; the report splits outcomes by the length ratio of the two replies
  (x shorter than 2/3, similar, x longer than 3/2) so a verbosity effect is visible; exact
  scores do not care about length.
- *Self-preference*: a judge favours text that resembles its own family's (and its training
  data's) style; our chat data and probably the opponents' were partly generated by large
  open models. Use a second judge from another family on the decisive comparisons and report
  their agreement.
- *Weak verification*: a 4B judge often misses arithmetic or code errors. Math and constraint
  prompts are also scored exactly (section 6); where judge and exact score disagree, trust the
  exact score.
- *Ties*: strict agreement produces many ties when both replies are poor, which lowers power
  (section 8).
- *Clipping*: replies beyond 4000 characters are cut for the judge.

## 6. Exact scores (`bes/eval/exact_score.py`)

- **Math.** The final stated number: the last `\boxed{}`, else the number after the last "answer
  is" / "answer:", else the last bold number (labels such as "**Step 2:**" skipped), else the last
  concluding sentence ("so", "therefore", "in total", ...) or, failing that, the last sentence with
  a number; in those sentences numbers that merely restate the user turns or the model's own
  earlier answers are not taken as the answer ("So, there are 2400 milliliters in 2.4 liters"
  reads 2400). Comparison is exact on fractions, tolerant to thousands separators, currency and
  percent signs, units, LaTeX fractions, number words and rounding to the reference's decimals.
  A model that contradicts itself is judged by its last conclusion. "Reference appears anywhere"
  is reported as an upper bound only.
- **Constraints.** Word, sentence and list-item counts come from the project's exact layer
  (`bes.exact.constraints.measure`, else `bes.exact.transducer.measure_counts`), the definitions
  our decoder works with; `local_counts` implements the same definitions and is used only when
  neither imports (the report names the backend in use; on 5000 random texts the local counts
  agreed with both project functions). Other checks (keywords, case, JSON, endings, paragraphs, bullets,
  numbered steps, ...) are local, listed in `CONSTRAINT_TYPES`. When `bes.exact.constraints`
  offers `compile` and `verify`, the project's own constraint compiler is run on the user message
  as a cross-check.
- **Degenerate output** for every reply: empty, cut at the token limit, looping tail, repeated
  lines (outside code), repeated 4-gram rate, unclosed code fence, leaked role markers, length in
  words.

## 7. Statistics (`score.py`)

Outcomes are from our side: win, loss, tie, both bad (a tie in every rate).

- score = (wins + ties/2) / n, with a Wilson interval over n (conservative for an outcome that
  can be one half);
- decisive win rate = wins / (wins + losses) with its Wilson interval;
- exact two-sided sign test of wins against losses;
- slices by category and by single versus two-turn prompts, the length-ratio buckets and the
  judge diagnostics (orders agreeing, first-position preference, format mass);
- `compare`: paired sign test between two arms of ours against the same opponent on the same
  prompts (per-prompt scores 1, 1/2, 0; ties of the difference dropped);
- `power`: comparisons needed to detect a decisive win rate of 0.5 + d with a two-sided 5% test
  at 80% power, inflated by 1 / (1 - tie rate).

| difference d | decisive comparisons | with 30% ties | with 50% ties |
|---|---|---|---|
| 3 points | 2178 | 3112 | 4356 |
| 5 points | 783 | 1118 | 1566 |
| 10 points | 194 | 277 | 388 |

## 8. Reading the results

1. Look at exact scores first (math correct, constraint prompts satisfied, degenerate rates);
   they need no judge and cannot be swayed by style.
2. Then the score against each opponent with its interval. 200 dev prompts give at most 200
   comparisons per opponent and arm: an interval of about plus or minus 7 points. Differences
   between our own arms (library, particles) should be read through `compare` (paired, same
   prompts), and a change smaller than the interval is "no evidence", not "no effect". The
   plan's rule that the library ships only when it is at least 2 points better cannot be met
   with confidence on this set; require that it is not worse and that exact scores do not drop.
3. Check the length buckets: if we win only where our replies are longer, suspect the judge.
4. Check the judge diagnostics: orders agreeing on few pairs or a large first-position
   preference mean the judge is unreliable for this comparison.
5. The final claim uses `test`, run once, with every preset and arm fixed on `dev`.

## 9. Running it

```
python -m bes.eval.proxy fetch falcon mobilellm qwen3-4b-2507      # download, verify, convert
python -m bes.eval.proxy run --model falcon --preset recommended --split dev --workers 8 --threads 2
python -m bes.eval.proxy run --model mobilellm --preset recommended --split dev --workers 8 --threads 2
python -m bes.eval.proxy run --model mobilellm --preset neutral --split dev --workers 8 --threads 2
python -m bes.eval.proxy run --model ours --generator <module:factory> --generator-arg ckpt=<dir> --key <ckpt+code> --split dev
python -m bes.eval.proxy exact <run id> --split dev
python -m bes.eval.proxy pairs --x <our run> --y <opponent run> --name ours-vs-falcon-dev --split dev
python -m bes.eval.proxy judge --pairs ours-vs-falcon-dev --judge qwen3-4b-2507 --threads 16
python -m bes.eval.proxy score --pairs ours-vs-falcon-dev
python -m bes.eval.proxy all --ours <module:factory> --ours-arg ckpt=<dir> --ours-key <key> --judge qwen3-4b-2507 --split dev
```

Every command can be interrupted and rerun. Reports go to `data_cache/proxy/reports/`.

**Cost and scheduling.** Opponent outputs do not depend on our model: generate them for both
splits before the 24-hour window. Measured here on a shared 4-core machine: Falcon averaged
54.6 tokens per second per single-thread process and 344 tokens per reply (the 200 dev prompts,
248 replies, took 13 minutes with two processes). MobileLLM spends most of its 4096-token budget
reasoning: from 30 seconds to 4 minutes per reply (mostly 2 to 4 minutes) on one loaded thread;
about 500 replies for the full set, so roughly 30 minutes on 40 host cores.

## 10. Results so far: Falcon-H1-Tiny-90M-Instruct, `dev`, recommended preset

200 prompts, 248 replies (run `falcon-e60034381f4a/recommended-5ccec0259b-s0`):

| measure | value |
|---|---|
| math, last answer correct (30 prompts) | 66.7% |
| math, all answers incl. first turns (37) | 70.3% |
| constraint prompts fully satisfied (47) | 78.7% (single constraints 39 of 50) |
| replies cut at 1024 tokens | 1.6% (4 of 248) |
| empty / looping / leaked role markers | 0% / 0% / 0% |
| mean / median / 90th percentile length | 197 / 174 / 384 words (chat 273, writing 240, code 224, instruct 113, math 84) |

Where it is weak: wrong arithmetic in multi-step problems (several confident wrong answers
after a correct intermediate result), avoiding forbidden words (1 of 3 prompts), "no commas",
splitting into paragraphs, answering with one allowed word only, exact sentence counts (4 of 6),
and long replies to simple chat questions. Every math verdict and every failed constraint of this run
was checked by hand against the reply.

## Appendix: the judge prompt

System message:

```text
You are a careful, impartial judge of replies written by chat assistants.
```

User message (single turn; for two-turn prompts the two blocks are `[Conversation with
assistant A]` ... `[End of conversation with assistant A]` with `User:` / `Assistant A:` lines,
preceded by a sentence asking to judge the replies over the whole conversation):

```text
Compare two assistant replies to the same user request and decide which one serves the user better.

Judge, in this order of importance:
1. Correctness: facts, arithmetic, code and reasoning must be right. A wrong final answer is a serious flaw, even if it is well written.
2. Following the request: every explicit instruction must be obeyed (length, number of sentences, items or words, format, required or forbidden words, tone).
3. Helpfulness: the reply addresses what was asked, clearly and completely, without padding.

Rules:
- Length is not a merit. Extra length only counts when the request needs it; prefer the shorter reply when both are equally correct and complete.
- Ignore the order in which the replies are shown and the labels A and B.
- Repetition, unfinished text, gibberish, or a reply that writes the user's next message are serious flaws.
- Answer C if the replies are about equally good or both are poor.

[User request]
<user message>
[End of user request]

[Reply A]
<reply A, or (empty reply)>
[End of reply A]

[Reply B]
<reply B>
[End of reply B]

Which reply is better? Answer with a single letter: A, B or C.
```

The verdict is the most probable of the tokens `A`, `B`, `C` right after the judge model's
generation prompt; the same pair is then rendered with A and B swapped.
