# Blind comparison `s2-default-vs-mobilellm` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 7/2/51 | 0.542 | [0.417, 0.661] | 0.778 | [0.453, 0.937] | 0.1797 |
| chat | 15 | 1/0/14 | 0.533 | [0.301, 0.752] | 1.000 | [0.206, 1.000] | 1.0000 |
| code | 9 | 0/1/8 | 0.444 | [0.189, 0.733] | 0.000 | [0.000, 0.793] | 1.0000 |
| instruct | 12 | 6/0/6 | 0.750 | [0.468, 0.911] | 1.000 | [0.610, 1.000] | 0.0312 |
| math | 9 | 0/1/8 | 0.444 | [0.189, 0.733] | 0.000 | [0.000, 0.793] | 1.0000 |
| writing | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |
| single_turn | 45 | 6/2/37 | 0.544 | [0.401, 0.681] | 0.750 | [0.409, 0.928] | 0.2891 |
| multi_turn | 15 | 1/0/14 | 0.533 | [0.301, 0.752] | 1.000 | [0.206, 1.000] | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 44 | 7/1/36 | 0.568 | [0.422, 0.703] | 0.875 | [0.529, 0.978] | 0.0703 |
| similar | 7 | 0/1/6 | 0.429 | [0.158, 0.750] | 0.000 | [0.000, 0.793] | 1.0000 |
| x_longer | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 14520, +0.05: 5217, +0.10: 1293

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 85%, detecting a difference of 10 points needs 1293 comparisons and 5 points 5217 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 80.0% of pairs; mean preference for the first-shown reply -0.000; mean probability mass on A/B/C 0.988; replies clipped in 30 pairs.
