# Exact scores of `ours-export-step51812-ef5599caa3974edc/final-trained-9a89782942-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.2 (all 38 answers incl. first turns: 0.2895; lenient, reference appears anywhere: 0.2), extraction {'conclusion': 22, 'boxed': 1, 'last': 5, 'answer_key': 2}
- constraints: 48 prompts, all satisfied 0.8542, single constraints satisfied 0.8571 of 56
- degenerate replies 0.0119: empty 0.0, cut at the token limit 0.0119, looping tail 0.0, repeated lines 0.0, unclosed code fence 0.0, role markers 0.0; mean repeated-4-gram rate 0.0384
- length in words: mean 119.2, median 99, p90 252

| constraint type | n | satisfied |
|---|---|---|
| bullets | 6 | 1.0 |
| case | 3 | 1.0 |
| ends_with | 3 | 1.0 |
| ends_with_question | 2 | 0.5 |
| frequency | 1 | 1.0 |
| include | 3 | 0.3333 |
| json | 2 | 0.0 |
| json_keys | 1 | 1.0 |
| lines | 1 | 1.0 |
| no_commas | 2 | 1.0 |
| numbered | 2 | 1.0 |
| one_of | 1 | 1.0 |
| paragraphs | 2 | 1.0 |
| placeholders | 1 | 1.0 |
| postscript | 1 | 1.0 |
| quoted | 1 | 1.0 |
| sections | 1 | 0.0 |
| sentences | 10 | 1.0 |
| separated_parts | 1 | 1.0 |
| starts_with | 2 | 1.0 |
| title | 1 | 1.0 |
| words | 9 | 0.7778 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 1.0 | 0.0 | 137.3 |
| code | 30 | None | None | 0.0 | 131.9 |
| instruct | 40 | None | 0.825 | 0.06 | 96.8 |
| math | 30 | 0.2 | None | 0.0 | 59.0 |
| writing | 50 | None | 1.0 | 0.0 | 147.4 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.9375}
