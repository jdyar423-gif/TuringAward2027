# Blind comparison `s2-default-vs-falcon` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 1/16/43 | 0.375 | [0.264, 0.501] | 0.059 | [0.011, 0.270] | 0.0003 |
| chat | 15 | 1/3/11 | 0.433 | [0.223, 0.671] | 0.250 | [0.046, 0.699] | 0.6250 |
| code | 9 | 0/3/6 | 0.333 | [0.121, 0.646] | 0.000 | [0.000, 0.561] | 0.2500 |
| instruct | 12 | 0/4/8 | 0.333 | [0.138, 0.609] | 0.000 | [0.000, 0.490] | 0.1250 |
| math | 9 | 0/3/6 | 0.333 | [0.121, 0.646] | 0.000 | [0.000, 0.561] | 0.2500 |
| writing | 15 | 0/3/12 | 0.400 | [0.198, 0.642] | 0.000 | [0.000, 0.561] | 0.2500 |
| single_turn | 45 | 1/11/33 | 0.389 | [0.261, 0.535] | 0.083 | [0.015, 0.354] | 0.0063 |
| multi_turn | 15 | 0/5/10 | 0.333 | [0.152, 0.583] | 0.000 | [0.000, 0.434] | 0.0625 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 38 | 1/8/29 | 0.408 | [0.267, 0.566] | 0.111 | [0.020, 0.435] | 0.0391 |
| similar | 17 | 0/7/10 | 0.294 | [0.133, 0.531] | 0.000 | [0.000, 0.354] | 0.0156 |
| x_longer | 5 | 0/1/4 | 0.400 | [0.118, 0.769] | 0.000 | [0.000, 0.793] | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 7687, +0.05: 2762, +0.10: 685

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 72%, detecting a difference of 10 points needs 685 comparisons and 5 points 2762 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 60.0% of pairs; mean preference for the first-shown reply -0.114; mean probability mass on A/B/C 0.996; replies clipped in 0 pairs.
