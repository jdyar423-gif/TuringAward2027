# Exact scores of `mobilellm-f680f0533368-qwen3/recommended-aca173fa2c-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.3667 (all 38 answers incl. first turns: 0.4474; lenient, reference appears anywhere: 0.4), extraction {'boxed': 24, 'bold': 3, 'answer_key': 1, 'last': 1, 'conclusion': 1}
- constraints: 48 prompts, all satisfied 0.1667, single constraints satisfied 0.2143 of 56
- degenerate replies 0.2024: empty 0.0, cut at the token limit 0.1865, looping tail 0.0159, repeated lines 0.0278, unclosed code fence 0.0079, role markers 0.0; mean repeated-4-gram rate 0.1355
- length in words: mean 217.6, median 105, p90 722

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
| sentences | 10 | 0.1 |
| separated_parts | 1 | 0.0 |
| starts_with | 2 | 0.0 |
| title | 1 | 1.0 |
| words | 9 | 0.1111 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 0.0 | 0.1746 | 219.5 |
| code | 30 | None | None | 0.4474 | 347.8 |
| instruct | 40 | None | 0.2 | 0.14 | 173.9 |
| math | 30 | 0.3667 | None | 0.2368 | 224.7 |
| writing | 50 | None | 0.0 | 0.1111 | 167.5 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.2708}
