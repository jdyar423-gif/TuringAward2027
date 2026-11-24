# Blind comparison `s1-short-vs-default` rated by `opus-1`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-short-37e897ff55-s0`
- y (opponent): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 1/1/58 | 0.500 | [0.377, 0.623] | 0.500 | [0.095, 0.905] | 1.0000 |
| chat | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |
| code | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| instruct | 12 | 1/1/10 | 0.500 | [0.254, 0.746] | 0.500 | [0.095, 0.905] | 1.0000 |
| math | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| writing | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |
| single_turn | 45 | 1/0/44 | 0.511 | [0.370, 0.650] | 1.000 | [0.206, 1.000] | 1.0000 |
| multi_turn | 15 | 0/1/14 | 0.467 | [0.248, 0.699] | 0.000 | [0.000, 0.793] | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 7 | 1/1/5 | 0.500 | [0.202, 0.798] | 0.500 | [0.095, 0.905] | 1.0000 |
| similar | 53 | 0/0/53 | 0.500 | [0.370, 0.630] |  |  | 1.0000 |
| x_longer | 0 | | | | | | |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 43558, +0.05: 15651, +0.10: 3877

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 95%, detecting a difference of 10 points needs 3877 comparisons and 5 points 15651 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
