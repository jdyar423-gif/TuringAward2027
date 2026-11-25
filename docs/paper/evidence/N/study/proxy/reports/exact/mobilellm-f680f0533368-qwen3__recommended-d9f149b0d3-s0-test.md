# Exact scores of `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0 (test)`

- prompts: 200 (252 replies); counting backend: `constraints.measure`
- math: 30 prompts, last answer correct 0.4 (all 38 answers incl. first turns: 0.4737; lenient, reference appears anywhere: 0.4), extraction {'boxed': 19, 'conclusion': 3, 'answer_key': 5, 'last': 3}
- constraints: 48 prompts, all satisfied 0.1667, single constraints satisfied 0.1786 of 56
- degenerate replies 0.627: empty 0.0079, cut at the token limit 0.5952, looping tail 0.0556, repeated lines 0.2143, unclosed code fence 0.0278, role markers 0.0; mean repeated-4-gram rate 0.2576
- length in words: mean 474.1, median 395, p90 836

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
| no_commas | 2 | 0.0 |
| numbered | 2 | 0.0 |
| one_of | 1 | 0.0 |
| paragraphs | 2 | 0.0 |
| placeholders | 1 | 1.0 |
| postscript | 1 | 0.0 |
| quoted | 1 | 0.0 |
| sections | 1 | 0.0 |
| sentences | 10 | 0.3 |
| separated_parts | 1 | 0.0 |
| starts_with | 2 | 0.0 |
| title | 1 | 1.0 |
| words | 9 | 0.1111 |

| category | n | math correct | constraints ok | degenerate | mean words |
|---|---|---|---|---|---|
| chat | 50 | None | 1.0 | 0.6032 | 488.6 |
| code | 30 | None | None | 0.7368 | 506.6 |
| instruct | 40 | None | 0.15 | 0.62 | 490.1 |
| math | 30 | 0.4 | None | 0.5263 | 412.8 |
| writing | 50 | None | 0.1429 | 0.6508 | 464.1 |

Project constraint compiler cross-check: {'n': 48, 'no_violation': 0.25}
