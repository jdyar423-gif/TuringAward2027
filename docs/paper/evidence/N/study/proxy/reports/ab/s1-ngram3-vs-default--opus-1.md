# Blind comparison `s1-ngram3-vs-default` rated by `opus-1`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-ngram3-908483c4c4-s0`
- y (opponent): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 6/3/51 | 0.525 | [0.401, 0.646] | 0.667 | [0.354, 0.879] | 0.5078 |
| chat | 15 | 3/0/12 | 0.600 | [0.357, 0.802] | 1.000 | [0.439, 1.000] | 0.2500 |
| code | 9 | 0/1/8 | 0.444 | [0.189, 0.733] | 0.000 | [0.000, 0.793] | 1.0000 |
| instruct | 12 | 3/1/8 | 0.583 | [0.320, 0.807] | 0.750 | [0.301, 0.954] | 0.6250 |
| math | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| writing | 15 | 0/1/14 | 0.467 | [0.248, 0.699] | 0.000 | [0.000, 0.793] | 1.0000 |
| single_turn | 45 | 4/3/38 | 0.511 | [0.370, 0.650] | 0.571 | [0.251, 0.842] | 1.0000 |
| multi_turn | 15 | 2/0/13 | 0.567 | [0.329, 0.777] | 1.000 | [0.342, 1.000] | 0.5000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 9 | 3/0/6 | 0.667 | [0.354, 0.879] | 1.000 | [0.439, 1.000] | 0.2500 |
| similar | 43 | 3/2/38 | 0.512 | [0.367, 0.654] | 0.600 | [0.231, 0.882] | 1.0000 |
| x_longer | 8 | 0/1/7 | 0.438 | [0.174, 0.741] | 0.000 | [0.000, 0.793] | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 14520, +0.05: 5217, +0.10: 1293

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 85%, detecting a difference of 10 points needs 1293 comparisons and 5 points 5217 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
