# Blind comparison `s2-default-vs-falcon` rated by `judge-gemma4-e4b-85a896a0`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 0/3/57 | 0.475 | [0.354, 0.599] | 0.000 | [0.000, 0.561] | 0.2500 |
| chat | 15 | 0/1/14 | 0.467 | [0.248, 0.699] | 0.000 | [0.000, 0.793] | 1.0000 |
| code | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| instruct | 12 | 0/1/11 | 0.458 | [0.223, 0.714] | 0.000 | [0.000, 0.793] | 1.0000 |
| math | 9 | 0/0/9 | 0.500 | [0.227, 0.773] |  |  | 1.0000 |
| writing | 15 | 0/1/14 | 0.467 | [0.248, 0.699] | 0.000 | [0.000, 0.793] | 1.0000 |
| single_turn | 45 | 0/3/42 | 0.467 | [0.329, 0.609] | 0.000 | [0.000, 0.561] | 0.2500 |
| multi_turn | 15 | 0/0/15 | 0.500 | [0.274, 0.726] |  |  | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 38 | 0/1/37 | 0.487 | [0.337, 0.639] | 0.000 | [0.000, 0.793] | 1.0000 |
| similar | 17 | 0/1/16 | 0.471 | [0.262, 0.690] | 0.000 | [0.000, 0.793] | 1.0000 |
| x_longer | 5 | 0/1/4 | 0.400 | [0.118, 0.769] | 0.000 | [0.000, 0.793] | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 43558, +0.05: 15651, +0.10: 3877

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 95%, detecting a difference of 10 points needs 3877 comparisons and 5 points 15651 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 83.3% of pairs; mean preference for the first-shown reply -0.065; mean probability mass on A/B/C 0.999; replies clipped in 0 pairs.
