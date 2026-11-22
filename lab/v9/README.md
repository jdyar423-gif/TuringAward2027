# Lab behind docs/V9.md

Scripts and result logs of the small CPU experiments that `docs/V9.md` section 2 summarises.
They are not part of the `bes` package and are not run by the test-suite.

* `lab12.py`, `lab8.py`: one training run of the lab model (d = 48, 2 shared blocks x 4
  loops, FFN 192, 1,040 token types, 8,192 tokens per step; AdamW, one core). An arm is a set
  of flags; `lab12.py --help` lists them. `lab8.py` is the earlier version that ran the
  class-specific attention arms.
* `queue3.py`: runs a list of arms, one per core. `cmp.py <arm> ...`: loss differences against
  the base arm `k_base` at the four checkpoints.
* `atomlaw.py`: the single law for all feed-forward arms (docs/V9.md section 2, item 3).
  `waterfill.py`: the allocation model of section 3 fitted on two arms and tested on a third.
* `runs/<arm>.log`: the configuration line, the stored / read counts and the evaluation
  records of every arm cited in the document (and of the arms that were dropped).
* `micro/`: the micro ablations of docs/V9.md section 8, which are not lab runs: they train
  the repository's model at the micro width with `scripts/micro_ablate.py`. `arms.json` holds
  the held-out losses and per-row sums of every arm at the three checkpoints, `table.py`
  prints the table. Two arms (`attn_gate_unscaled`, `v9_pages_gate_unscaled`) ran with the
  first version of the attention gate, whose logit had no `1/d` scale; that code is not in the
  repository.

Data. The runs read `web1k.u16`, `chat1k.u16`, `chat1k_role.u8`, `chat1k_turn.i32`,
`tree_leaf_1k.npy` and `unigram_1k.npy` from the script directory (or `LAB_DATA` for the two
analysis scripts): web and chat text of the data cache tokenised with the 256 bytes, the
first 768 merges of the repository tokenizer and the 16 special ids (1,040 types), the role
and turn index of every chat token, and a frequency-balanced token taxonomy. These files are
not in the repository (about 0.5 GB) and the script that wrote them was not kept; the logs
are the record.

Arms and flags (all with `--tokens 12e6 --lr 5e-3 --lr-emb 5e-2` unless the log says
otherwise):

| arm | script and flags |
|---|---|
| `k_base` | `lab8.py` (no flags) |
| `k_f384`, `k_f768` | `--f 384`, `--f 768` |
| `k_exp8_bal`, `k_exp8_tree`, `q_ffn32` | `--experts 8 --f 0 --classes bal`, `... --classes tree`, `--experts 32 --f 0 --classes tree` |
| `q_attn8_qk`, `q_attn8_q`, `q_attn8_k`, `q_attn8_v`, `q_attn8_o`, `q_attn8_vo`, `q_attn8` | `--attn-experts 8 --attn-parts qk` / `q` / `k` / `v` / `o` / `vo` / `qkvo` (`q` and `k` to 6.6M tokens) |
| `q_ffn8_vo12` | `--attn-experts 8 --attn-parts vo --experts 8 --f 0 --route tok --classes tree` |
| `g_ffn`, `g_vo`, `g_o`, `g_all` | `lab12.py --tokgain`, `--vgain --ogain`, `--ogain`, `--tokgain --vgain --ogain` |
| `g_ve` | `lab12.py --ve` |
| `x_lex2_lr`, `x_lex2n_lr` | `lab12.py --lex 2 --lex-lr 5e-3`, `... --lex-norm` |
| `x_lexg2_lr`, `x_lexg2c8_lr`, `x_lexg2c4_lr` | `lab12.py --lex 2 --lex-gamma 1.0 --lex-max 32 --lex-lr 5e-3`, `... --lex-max 8`, `... --lex-max 4` |
| `x_lexgp8_lr` | `lab12.py --lex 8 --lex-proj 12 --lex-gamma 1.0 --lex-max 64 --lex-lr 5e-3` |
| `x_lexp8n_lr` | `lab12.py --lex 8 --lex-proj 12 --lex-norm --lex-lr 5e-3` |
| `x_v9`, `x_v9_ve`, `x_v9_v` | the previous arm with `--ogain`, with `--ogain --ve`, with `--ogain --vgain` |
| `x_v9_lr3`, `x_v9_lr03` | `x_v9` with `--lex-lr 1.5e-2`, with `--lex-lr 1.5e-3` |

Private neurons need their own learning rate (`--lex-lr 5e-3`): at the embedding rate
(`--lr-emb`, ten times higher) the uniform arm was above the base after 3.3M tokens (+0.043);
that run was stopped and its log was not kept. The `k_*` arms ran with earlier versions of the
script (the same model code with the flags listed); `runs/<arm>.json` holds the full argument
list of every arm.
