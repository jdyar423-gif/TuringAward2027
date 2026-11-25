# Exact scores of `falcon-e60034381f4a/recommended-2dcb1c92a6-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.5667 (all 38 answers incl. first turns: 0.6579; lenient, reference appears anywhere: 0.6333), extraction {'bold': 5, 'boxed': 14, 'conclusion': 11}
- constraints: 48 prompts, all satisfied 0.7292, single constraints satisfied 0.75 of 56
- degenerate replies 0.004: empty 0.0, cut at the token limit 0.0, looping tail 0.0, repeated lines 0.0, unclosed code fence 0.004, role markers 0.0; mean repeated-4-gram rate 0.0262
- length in words: mean 174.3, median 147, p90 363

| constraint type | n | satisfied |
|---|---|---|
| bullets | 6 | 0.8333 |
| case | 3 | 0.3333 |
| ends_with | 3 | 1.0 |
| ends_with_question | 2 | 1.0 |
| frequency | 1 | 1.0 |
| include | 3 | 1.0 |
| json | 2 | 0.5 |
| json_keys | 1 | 1.0 |
| lines | 1 | 1.0 |
| no_commas | 2 | 1.0 |
| numbered | 2 | 0.5 |
| one_of | 1 | 0.0 |
| paragraphs | 2 | 0.5 |
| placeholders | 1 | 1.0 |
| postscript | 1 | 1.0 |
| quoted | 1 | 1.0 |
| sections | 1 | 1.0 |
| sentences | 10 | 0.7 |
| separated_parts | 1 | 1.0 |
| starts_with | 2 | 1.0 |
| title | 1 | 1.0 |
| words | 9 | 0.5556 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 1.0 | 0.0 | 271.0 |
| code | 30 | None | None | 0.0263 | 210.5 |
| instruct | 40 | None | 0.725 | 0.0 | 103.2 |
| math | 30 | 0.5667 | None | 0.0 | 88.7 |
| writing | 50 | None | 0.7143 | 0.0 | 163.7 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.75}
