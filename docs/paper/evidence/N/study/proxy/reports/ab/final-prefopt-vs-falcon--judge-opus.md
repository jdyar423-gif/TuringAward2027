# Blind comparison `final-prefopt-vs-falcon` rated by `judge-opus`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 35/108/57 | 0.318 | [0.257, 0.385] | 0.245 | [0.182, 0.321] | 0.0000 |
| chat | 50 | 10/24/16 | 0.360 | [0.241, 0.499] | 0.294 | [0.168, 0.462] | 0.0243 |
| code | 30 | 10/13/7 | 0.450 | [0.288, 0.623] | 0.435 | [0.256, 0.632] | 0.6776 |
| instruct | 40 | 7/21/12 | 0.325 | [0.201, 0.480] | 0.250 | [0.127, 0.434] | 0.0125 |
| math | 30 | 4/18/8 | 0.267 | [0.142, 0.445] | 0.182 | [0.073, 0.385] | 0.0043 |
| writing | 50 | 4/32/14 | 0.220 | [0.128, 0.352] | 0.111 | [0.044, 0.253] | 0.0000 |
| single_turn | 148 | 28/78/42 | 0.331 | [0.260, 0.410] | 0.264 | [0.190, 0.355] | 0.0000 |
| multi_turn | 52 | 7/30/15 | 0.279 | [0.175, 0.413] | 0.189 | [0.095, 0.342] | 0.0002 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 93 | 26/41/26 | 0.419 | [0.324, 0.521] | 0.388 | [0.281, 0.508] | 0.0864 |
| similar | 72 | 7/40/25 | 0.271 | [0.182, 0.383] | 0.149 | [0.074, 0.277] | 0.0000 |
| x_longer | 35 | 2/27/6 | 0.143 | [0.063, 0.294] | 0.069 | [0.019, 0.220] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 3046, +0.05: 1095, +0.10: 272

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 28%, detecting a difference of 10 points needs 272 comparisons and 5 points 1095 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 90.5% of pairs; mean preference for the first-shown reply +0.005; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
