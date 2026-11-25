# Blind comparison `final-prefopt-vs-falcon-k6ds` rated by `judge-opus`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-d1f2a63f82-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 29/116/55 | 0.282 | [0.225, 0.348] | 0.200 | [0.143, 0.273] | 0.0000 |
| chat | 50 | 7/25/18 | 0.320 | [0.208, 0.458] | 0.219 | [0.110, 0.388] | 0.0021 |
| code | 30 | 11/12/7 | 0.483 | [0.317, 0.654] | 0.478 | [0.292, 0.670] | 1.0000 |
| instruct | 40 | 7/22/11 | 0.312 | [0.191, 0.467] | 0.241 | [0.122, 0.421] | 0.0081 |
| math | 30 | 3/20/7 | 0.217 | [0.106, 0.391] | 0.130 | [0.045, 0.321] | 0.0005 |
| writing | 50 | 1/37/12 | 0.140 | [0.070, 0.262] | 0.026 | [0.005, 0.135] | 0.0000 |
| single_turn | 148 | 21/84/43 | 0.287 | [0.220, 0.365] | 0.200 | [0.135, 0.286] | 0.0000 |
| multi_turn | 52 | 8/32/12 | 0.269 | [0.168, 0.403] | 0.200 | [0.105, 0.348] | 0.0002 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 89 | 19/37/33 | 0.399 | [0.303, 0.503] | 0.339 | [0.229, 0.470] | 0.0222 |
| similar | 82 | 10/55/17 | 0.226 | [0.149, 0.327] | 0.154 | [0.086, 0.261] | 0.0000 |
| x_longer | 29 | 0/24/5 | 0.086 | [0.027, 0.242] | 0.000 | [0.000, 0.138] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 3004, +0.05: 1080, +0.10: 268

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 28%, detecting a difference of 10 points needs 268 comparisons and 5 points 1080 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 92.5% of pairs; mean preference for the first-shown reply +0.018; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
