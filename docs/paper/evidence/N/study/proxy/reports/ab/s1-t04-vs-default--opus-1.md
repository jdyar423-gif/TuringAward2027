# Blind comparison `s1-t04-vs-default` rated by `opus-1`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-t04-eb17090d64-s0`
- y (opponent): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 5/6/49 | 0.492 | [0.369, 0.615] | 0.455 | [0.213, 0.720] | 1.0000 |
| chat | 15 | 2/1/12 | 0.533 | [0.301, 0.752] | 0.667 | [0.208, 0.939] | 1.0000 |
| code | 9 | 1/1/7 | 0.500 | [0.227, 0.773] | 0.500 | [0.095, 0.905] | 1.0000 |
| instruct | 12 | 0/3/9 | 0.375 | [0.165, 0.646] | 0.000 | [0.000, 0.561] | 0.2500 |
| math | 9 | 1/0/8 | 0.556 | [0.267, 0.811] | 1.000 | [0.206, 1.000] | 1.0000 |
| writing | 15 | 1/1/13 | 0.500 | [0.274, 0.726] | 0.500 | [0.095, 0.905] | 1.0000 |
| single_turn | 45 | 3/3/39 | 0.500 | [0.360, 0.640] | 0.500 | [0.188, 0.812] | 1.0000 |
| multi_turn | 15 | 2/3/10 | 0.467 | [0.248, 0.699] | 0.400 | [0.118, 0.769] | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 10 | 0/1/9 | 0.450 | [0.201, 0.726] | 0.000 | [0.000, 0.793] | 1.0000 |
| similar | 40 | 4/2/34 | 0.525 | [0.375, 0.671] | 0.667 | [0.300, 0.903] | 0.6875 |
| x_longer | 10 | 1/3/6 | 0.400 | [0.168, 0.687] | 0.250 | [0.046, 0.699] | 0.6250 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 11880, +0.05: 4269, +0.10: 1058

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 82%, detecting a difference of 10 points needs 1058 comparisons and 5 points 4269 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
