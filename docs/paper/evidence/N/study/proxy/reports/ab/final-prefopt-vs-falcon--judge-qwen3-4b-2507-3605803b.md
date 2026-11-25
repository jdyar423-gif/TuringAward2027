# Blind comparison `final-prefopt-vs-falcon` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 12/61/127 | 0.378 | [0.313, 0.446] | 0.164 | [0.097, 0.266] | 0.0000 |
| chat | 50 | 3/10/37 | 0.430 | [0.303, 0.567] | 0.231 | [0.082, 0.503] | 0.0923 |
| code | 30 | 3/4/23 | 0.483 | [0.317, 0.654] | 0.429 | [0.158, 0.750] | 1.0000 |
| instruct | 40 | 4/17/19 | 0.338 | [0.211, 0.492] | 0.191 | [0.077, 0.400] | 0.0072 |
| math | 30 | 0/12/18 | 0.300 | [0.167, 0.479] | 0.000 | [0.000, 0.242] | 0.0005 |
| writing | 50 | 2/18/30 | 0.340 | [0.224, 0.478] | 0.100 | [0.028, 0.301] | 0.0004 |
| single_turn | 148 | 10/45/93 | 0.382 | [0.307, 0.462] | 0.182 | [0.102, 0.303] | 0.0000 |
| multi_turn | 52 | 2/16/34 | 0.365 | [0.248, 0.501] | 0.111 | [0.031, 0.328] | 0.0013 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 93 | 11/14/68 | 0.484 | [0.385, 0.584] | 0.440 | [0.267, 0.629] | 0.6900 |
| similar | 72 | 1/25/46 | 0.333 | [0.235, 0.448] | 0.038 | [0.007, 0.189] | 0.0000 |
| x_longer | 35 | 0/22/13 | 0.186 | [0.091, 0.343] | 0.000 | [0.000, 0.149] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 5967, +0.05: 2144, +0.10: 532

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 64%, detecting a difference of 10 points needs 532 comparisons and 5 points 2144 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 60.5% of pairs; mean preference for the first-shown reply -0.134; mean probability mass on A/B/C 0.983; replies clipped in 1 pairs.
