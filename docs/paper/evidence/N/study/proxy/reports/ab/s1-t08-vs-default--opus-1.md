# Blind comparison `s1-t08-vs-default` rated by `opus-1`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-t08-abbdeb7de0-s0`
- y (opponent): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 2/7/51 | 0.458 | [0.339, 0.583] | 0.222 | [0.063, 0.547] | 0.1797 |
| chat | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |
| code | 9 | 0/2/7 | 0.389 | [0.153, 0.691] | 0.000 | [0.000, 0.658] | 0.5000 |
| instruct | 12 | 1/2/9 | 0.458 | [0.223, 0.714] | 0.333 | [0.061, 0.792] | 1.0000 |
| math | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| writing | 15 | 1/3/11 | 0.433 | [0.223, 0.671] | 0.250 | [0.046, 0.699] | 0.6250 |
| single_turn | 45 | 0/6/39 | 0.433 | [0.299, 0.578] | 0.000 | [0.000, 0.390] | 0.0312 |
| multi_turn | 15 | 2/1/12 | 0.533 | [0.301, 0.752] | 0.667 | [0.208, 0.939] | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 11 | 2/2/7 | 0.500 | [0.246, 0.754] | 0.500 | [0.150, 0.850] | 1.0000 |
| similar | 38 | 0/5/33 | 0.434 | [0.290, 0.591] | 0.000 | [0.000, 0.434] | 0.0625 |
| x_longer | 11 | 0/0/11 | 0.500 | [0.246, 0.754] |  |  | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 14520, +0.05: 5217, +0.10: 1293

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 85%, detecting a difference of 10 points needs 1293 comparisons and 5 points 5217 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
