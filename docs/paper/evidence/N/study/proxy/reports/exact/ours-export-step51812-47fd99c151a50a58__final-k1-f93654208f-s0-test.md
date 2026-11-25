# Exact scores of `ours-export-step51812-47fd99c151a50a58/final-k1-f93654208f-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.1667 (all 38 answers incl. first turns: 0.2632; lenient, reference appears anywhere: 0.1667), extraction {'conclusion': 23, 'answer_key': 3, 'last': 4}
- constraints: 48 prompts, all satisfied 0.7708, single constraints satisfied 0.7857 of 56
- degenerate replies 0.0675: empty 0.0, cut at the token limit 0.0675, looping tail 0.0, repeated lines 0.0, unclosed code fence 0.0, role markers 0.0; mean repeated-4-gram rate 0.0537
- length in words: mean 149.7, median 105, p90 363

| constraint type | n | satisfied |
|---|---|---|
| bullets | 6 | 1.0 |
| case | 3 | 1.0 |
| ends_with | 3 | 0.6667 |
| ends_with_question | 2 | 0.0 |
| frequency | 1 | 0.0 |
| include | 3 | 0.3333 |
| json | 2 | 0.0 |
| json_keys | 1 | 1.0 |
| lines | 1 | 1.0 |
| no_commas | 2 | 1.0 |
| numbered | 2 | 1.0 |
| one_of | 1 | 1.0 |
| paragraphs | 2 | 0.5 |
| placeholders | 1 | 1.0 |
| postscript | 1 | 1.0 |
| quoted | 1 | 1.0 |
| sections | 1 | 1.0 |
| sentences | 10 | 1.0 |
| separated_parts | 1 | 1.0 |
| starts_with | 2 | 0.5 |
| title | 1 | 0.0 |
| words | 9 | 0.8889 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 1.0 | 0.0 | 168.7 |
| code | 30 | None | None | 0.0789 | 154.7 |
| instruct | 40 | None | 0.725 | 0.14 | 116.4 |
| math | 30 | 0.1667 | None | 0.0 | 64.9 |
| writing | 50 | None | 1.0 | 0.1111 | 205.3 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.875}
