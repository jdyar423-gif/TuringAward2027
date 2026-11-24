# Blind comparison `s2-default-vs-mobilellm` rated by `judge-gemma4-e4b-85a896a0`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 2/1/57 | 0.508 | [0.385, 0.630] | 0.667 | [0.208, 0.939] | 1.0000 |
| chat | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |
| code | 9 | 0/1/8 | 0.444 | [0.189, 0.733] | 0.000 | [0.000, 0.793] | 1.0000 |
| instruct | 12 | 2/0/10 | 0.583 | [0.320, 0.807] | 1.000 | [0.342, 1.000] | 0.5000 |
| math | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| writing | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |
| single_turn | 45 | 2/1/42 | 0.511 | [0.370, 0.650] | 0.667 | [0.208, 0.939] | 1.0000 |
| multi_turn | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 44 | 2/0/42 | 0.523 | [0.379, 0.662] | 1.000 | [0.342, 1.000] | 0.5000 |
| similar | 7 | 0/1/6 | 0.429 | [0.158, 0.750] | 0.000 | [0.000, 0.793] | 1.0000 |
| x_longer | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 43558, +0.05: 15651, +0.10: 3877

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 95%, detecting a difference of 10 points needs 3877 comparisons and 5 points 15651 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 93.3% of pairs; mean preference for the first-shown reply +0.008; mean probability mass on A/B/C 0.999; replies clipped in 30 pairs.
