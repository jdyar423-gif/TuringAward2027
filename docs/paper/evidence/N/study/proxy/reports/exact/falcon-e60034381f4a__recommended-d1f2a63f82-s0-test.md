# Exact scores of `falcon-e60034381f4a/recommended-d1f2a63f82-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.6 (all 38 answers incl. first turns: 0.6842; lenient, reference appears anywhere: 0.6), extraction {'conclusion': 9, 'boxed': 18, 'bold': 2, 'last': 1}
- constraints: 48 prompts, all satisfied 0.875, single constraints satisfied 0.8929 of 56
- degenerate replies 0.0: empty 0.0, cut at the token limit 0.0, looping tail 0.0, repeated lines 0.0, unclosed code fence 0.0, role markers 0.0; mean repeated-4-gram rate 0.0205
- length in words: mean 168.4, median 145, p90 353

| constraint type | n | satisfied |
|---|---|---|
| bullets | 6 | 1.0 |
| case | 3 | 1.0 |
| ends_with | 3 | 1.0 |
| ends_with_question | 2 | 1.0 |
| frequency | 1 | 1.0 |
| include | 3 | 1.0 |
| json | 2 | 0.5 |
| json_keys | 1 | 1.0 |
| lines | 1 | 1.0 |
| no_commas | 2 | 1.0 |
| numbered | 2 | 0.5 |
| one_of | 1 | 1.0 |
| paragraphs | 2 | 0.5 |
| placeholders | 1 | 1.0 |
| postscript | 1 | 1.0 |
| quoted | 1 | 1.0 |
| sections | 1 | 1.0 |
| sentences | 10 | 0.9 |
| separated_parts | 1 | 1.0 |
| starts_with | 2 | 1.0 |
| title | 1 | 1.0 |
| words | 9 | 0.7778 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 1.0 | 0.0 | 258.9 |
| code | 30 | None | None | 0.0 | 206.2 |
| instruct | 40 | None | 0.85 | 0.0 | 94.3 |
| math | 30 | 0.6 | None | 0.0 | 87.8 |
| writing | 50 | None | 1.0 | 0.0 | 162.5 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.9167}
