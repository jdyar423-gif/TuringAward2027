# Blind comparison `s1-default-vs-mobilellm` rated by `opus-1`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 15/8/37 | 0.558 | [0.433, 0.677] | 0.652 | [0.449, 0.812] | 0.2100 |
| chat | 15 | 3/0/12 | 0.600 | [0.357, 0.802] | 1.000 | [0.439, 1.000] | 0.2500 |
| code | 9 | 1/1/7 | 0.500 | [0.227, 0.773] | 0.500 | [0.095, 0.905] | 1.0000 |
| instruct | 12 | 7/0/5 | 0.792 | [0.509, 0.933] | 1.000 | [0.646, 1.000] | 0.0156 |
| math | 9 | 0/7/2 | 0.111 | [0.020, 0.435] | 0.000 | [0.000, 0.354] | 0.0156 |
| writing | 15 | 4/0/11 | 0.633 | [0.387, 0.825] | 1.000 | [0.510, 1.000] | 0.1250 |
| single_turn | 45 | 10/6/29 | 0.544 | [0.401, 0.681] | 0.625 | [0.386, 0.815] | 0.4545 |
| multi_turn | 15 | 5/2/8 | 0.600 | [0.357, 0.802] | 0.714 | [0.359, 0.918] | 0.4531 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 45 | 13/4/28 | 0.600 | [0.455, 0.730] | 0.765 | [0.527, 0.904] | 0.0490 |
| similar | 9 | 1/4/4 | 0.333 | [0.121, 0.646] | 0.200 | [0.036, 0.625] | 0.3750 |
| x_longer | 6 | 1/0/5 | 0.583 | [0.241, 0.861] | 1.000 | [0.206, 1.000] | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 5682, +0.05: 2042, +0.10: 506

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 62%, detecting a difference of 10 points needs 506 comparisons and 5 points 2042 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
