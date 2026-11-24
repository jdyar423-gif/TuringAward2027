# Blind comparison `s1-k1-vs-default` rated by `opus-1`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-k1-f93654208f-s0`
- y (opponent): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 1/3/56 | 0.483 | [0.362, 0.607] | 0.250 | [0.046, 0.699] | 0.6250 |
| chat | 15 | 1/0/14 | 0.533 | [0.301, 0.752] | 1.000 | [0.206, 1.000] | 1.0000 |
| code | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| instruct | 12 | 0/1/11 | 0.458 | [0.223, 0.714] | 0.000 | [0.000, 0.793] | 1.0000 |
| math | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| writing | 15 | 0/2/13 | 0.433 | [0.223, 0.671] | 0.000 | [0.000, 0.658] | 0.5000 |
| single_turn | 45 | 0/2/43 | 0.478 | [0.339, 0.620] | 0.000 | [0.000, 0.658] | 0.5000 |
| multi_turn | 15 | 1/1/13 | 0.500 | [0.274, 0.726] | 0.500 | [0.095, 0.905] | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 11 | 0/2/9 | 0.409 | [0.181, 0.684] | 0.000 | [0.000, 0.658] | 0.5000 |
| similar | 27 | 1/0/26 | 0.518 | [0.340, 0.693] | 1.000 | [0.206, 1.000] | 1.0000 |
| x_longer | 22 | 0/1/21 | 0.477 | [0.288, 0.673] | 0.000 | [0.000, 0.793] | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 32669, +0.05: 11738, +0.10: 2908

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 93%, detecting a difference of 10 points needs 2908 comparisons and 5 points 11738 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
