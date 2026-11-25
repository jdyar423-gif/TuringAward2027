# Blind comparison `final-prefopt-vs-falcon` rated by `judge-opus-neutral`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 33/109/58 | 0.310 | [0.250, 0.377] | 0.232 | [0.171, 0.308] | 0.0000 |
| chat | 50 | 9/24/17 | 0.350 | [0.233, 0.489] | 0.273 | [0.151, 0.442] | 0.0135 |
| code | 30 | 9/13/8 | 0.433 | [0.274, 0.608] | 0.409 | [0.233, 0.613] | 0.5235 |
| instruct | 40 | 8/21/11 | 0.338 | [0.211, 0.492] | 0.276 | [0.147, 0.457] | 0.0241 |
| math | 30 | 4/18/8 | 0.267 | [0.142, 0.445] | 0.182 | [0.073, 0.385] | 0.0043 |
| writing | 50 | 3/33/14 | 0.200 | [0.112, 0.330] | 0.083 | [0.029, 0.218] | 0.0000 |
| single_turn | 148 | 25/80/43 | 0.314 | [0.245, 0.393] | 0.238 | [0.167, 0.328] | 0.0000 |
| multi_turn | 52 | 8/29/15 | 0.298 | [0.191, 0.433] | 0.216 | [0.114, 0.372] | 0.0008 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 93 | 24/40/29 | 0.414 | [0.319, 0.515] | 0.375 | [0.267, 0.497] | 0.0599 |
| similar | 72 | 7/42/23 | 0.257 | [0.170, 0.368] | 0.143 | [0.071, 0.267] | 0.0000 |
| x_longer | 35 | 2/27/6 | 0.143 | [0.063, 0.294] | 0.069 | [0.019, 0.220] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 3068, +0.05: 1103, +0.10: 274

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 29%, detecting a difference of 10 points needs 274 comparisons and 5 points 1103 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 92.5% of pairs; mean preference for the first-shown reply +0.010; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
