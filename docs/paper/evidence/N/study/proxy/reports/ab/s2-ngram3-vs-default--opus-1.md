# Blind comparison `s2-ngram3-vs-default` rated by `opus-1`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-ngram3-908483c4c4-s0`
- y (opponent): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 4/8/48 | 0.467 | [0.346, 0.591] | 0.333 | [0.138, 0.609] | 0.3877 |
| chat | 15 | 2/2/11 | 0.500 | [0.274, 0.726] | 0.500 | [0.150, 0.850] | 1.0000 |
| code | 9 | 1/0/8 | 0.556 | [0.267, 0.811] | 1.000 | [0.206, 1.000] | 1.0000 |
| instruct | 12 | 0/3/9 | 0.375 | [0.165, 0.646] | 0.000 | [0.000, 0.561] | 0.2500 |
| math | 9 | 0/2/7 | 0.389 | [0.153, 0.691] | 0.000 | [0.000, 0.658] | 0.5000 |
| writing | 15 | 1/1/13 | 0.500 | [0.274, 0.726] | 0.500 | [0.095, 0.905] | 1.0000 |
| single_turn | 45 | 2/5/38 | 0.467 | [0.329, 0.609] | 0.286 | [0.082, 0.641] | 0.4531 |
| multi_turn | 15 | 2/3/10 | 0.467 | [0.248, 0.699] | 0.400 | [0.118, 0.769] | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 3 | 0/0/3 | 0.500 | [0.125, 0.875] |  |  | 1.0000 |
| similar | 50 | 4/6/40 | 0.480 | [0.348, 0.615] | 0.400 | [0.168, 0.687] | 0.7539 |
| x_longer | 7 | 0/2/5 | 0.357 | [0.118, 0.697] | 0.000 | [0.000, 0.658] | 0.5000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 10890, +0.05: 3913, +0.10: 970

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 80%, detecting a difference of 10 points needs 970 comparisons and 5 points 3913 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
