# Blind comparison `s2-default-vs-mobilellm` rated by `judge-opus`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 20/7/33 | 0.608 | [0.482, 0.722] | 0.741 | [0.553, 0.868] | 0.0192 |
| chat | 15 | 5/0/10 | 0.667 | [0.417, 0.848] | 1.000 | [0.566, 1.000] | 0.0625 |
| code | 9 | 1/2/6 | 0.444 | [0.189, 0.733] | 0.333 | [0.061, 0.792] | 1.0000 |
| instruct | 12 | 9/0/3 | 0.875 | [0.598, 0.971] | 1.000 | [0.701, 1.000] | 0.0039 |
| math | 9 | 0/5/4 | 0.222 | [0.063, 0.547] | 0.000 | [0.000, 0.434] | 0.0625 |
| writing | 15 | 5/0/10 | 0.667 | [0.417, 0.848] | 1.000 | [0.566, 1.000] | 0.0625 |
| single_turn | 45 | 13/7/25 | 0.567 | [0.422, 0.701] | 0.650 | [0.433, 0.819] | 0.2632 |
| multi_turn | 15 | 7/0/8 | 0.733 | [0.480, 0.891] | 1.000 | [0.646, 1.000] | 0.0156 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 44 | 17/4/23 | 0.648 | [0.500, 0.772] | 0.809 | [0.600, 0.923] | 0.0072 |
| similar | 7 | 1/3/3 | 0.357 | [0.118, 0.697] | 0.250 | [0.046, 0.699] | 0.6250 |
| x_longer | 9 | 2/0/7 | 0.611 | [0.309, 0.847] | 1.000 | [0.342, 1.000] | 0.5000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 4840, +0.05: 1739, +0.10: 431

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 55%, detecting a difference of 10 points needs 431 comparisons and 5 points 1739 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 98.3% of pairs; mean preference for the first-shown reply +0.008; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
