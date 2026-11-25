# Exact scores of `falcon-e60034381f4a/recommended-5ccec0259b-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.5 (all 38 answers incl. first turns: 0.6053; lenient, reference appears anywhere: 0.6), extraction {'conclusion': 10, 'boxed': 14, 'bold': 5, 'answer_key': 1}
- constraints: 48 prompts, all satisfied 0.6875, single constraints satisfied 0.7321 of 56
- degenerate replies 0.0079: empty 0.0, cut at the token limit 0.004, looping tail 0.0, repeated lines 0.0, unclosed code fence 0.004, role markers 0.0; mean repeated-4-gram rate 0.0201
- length in words: mean 173.4, median 155, p90 350

| constraint type | n | satisfied |
|---|---|---|
| bullets | 6 | 0.8333 |
| case | 3 | 1.0 |
| ends_with | 3 | 1.0 |
| ends_with_question | 2 | 1.0 |
| frequency | 1 | 1.0 |
| include | 3 | 1.0 |
| json | 2 | 0.0 |
| json_keys | 1 | 1.0 |
| lines | 1 | 0.0 |
| no_commas | 2 | 1.0 |
| numbered | 2 | 0.5 |
| one_of | 1 | 1.0 |
| paragraphs | 2 | 0.5 |
| placeholders | 1 | 1.0 |
| postscript | 1 | 1.0 |
| quoted | 1 | 0.0 |
| sections | 1 | 0.0 |
| sentences | 10 | 0.6 |
| separated_parts | 1 | 1.0 |
| starts_with | 2 | 0.5 |
| title | 1 | 1.0 |
| words | 9 | 0.7778 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 1.0 | 0.0 | 263.5 |
| code | 30 | None | None | 0.0 | 214.8 |
| instruct | 40 | None | 0.725 | 0.0 | 107.2 |
| math | 30 | 0.5 | None | 0.0263 | 82.3 |
| writing | 50 | None | 0.4286 | 0.0159 | 165.6 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.7083}
