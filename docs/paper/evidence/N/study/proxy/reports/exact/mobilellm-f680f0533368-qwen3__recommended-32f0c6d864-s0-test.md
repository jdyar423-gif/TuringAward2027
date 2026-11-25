# Exact scores of `mobilellm-f680f0533368-qwen3/recommended-32f0c6d864-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.4 (all 38 answers incl. first turns: 0.5; lenient, reference appears anywhere: 0.5), extraction {'boxed': 21, 'answer_key': 2, 'bold': 5, 'last': 2}
- constraints: 48 prompts, all satisfied 0.1875, single constraints satisfied 0.2143 of 56
- degenerate replies 0.254: empty 0.004, cut at the token limit 0.1984, looping tail 0.0357, repeated lines 0.0595, unclosed code fence 0.0198, role markers 0.0; mean repeated-4-gram rate 0.1676
- length in words: mean 238.6, median 108, p90 735

| constraint type | n | satisfied |
|---|---|---|
| bullets | 6 | 0.0 |
| case | 3 | 0.0 |
| ends_with | 3 | 0.0 |
| ends_with_question | 2 | 0.0 |
| frequency | 1 | 1.0 |
| include | 3 | 1.0 |
| json | 2 | 0.0 |
| json_keys | 1 | 0.0 |
| lines | 1 | 0.0 |
| no_commas | 2 | 1.0 |
| numbered | 2 | 0.5 |
| one_of | 1 | 0.0 |
| paragraphs | 2 | 0.5 |
| placeholders | 1 | 1.0 |
| postscript | 1 | 0.0 |
| quoted | 1 | 0.0 |
| sections | 1 | 0.0 |
| sentences | 10 | 0.0 |
| separated_parts | 1 | 0.0 |
| starts_with | 2 | 0.0 |
| title | 1 | 1.0 |
| words | 9 | 0.2222 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 0.0 | 0.1746 | 229.4 |
| code | 30 | None | None | 0.5263 | 356.6 |
| instruct | 40 | None | 0.225 | 0.18 | 175.5 |
| math | 30 | 0.4 | None | 0.2632 | 255.4 |
| writing | 50 | None | 0.0 | 0.2222 | 216.7 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.2708}
