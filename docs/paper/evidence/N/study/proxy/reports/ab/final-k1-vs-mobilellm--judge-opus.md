# Blind comparison `final-k1-vs-mobilellm` rated by `judge-opus`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-k1-f93654208f-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 99/16/85 | 0.708 | [0.641, 0.766] | 0.861 | [0.786, 0.912] | 0.0000 |
| chat | 50 | 28/2/20 | 0.760 | [0.626, 0.857] | 0.933 | [0.787, 0.982] | 0.0000 |
| code | 30 | 13/4/13 | 0.650 | [0.471, 0.795] | 0.765 | [0.527, 0.904] | 0.0490 |
| instruct | 40 | 24/0/16 | 0.800 | [0.652, 0.895] | 1.000 | [0.862, 1.000] | 0.0000 |
| math | 30 | 6/10/14 | 0.433 | [0.274, 0.608] | 0.375 | [0.185, 0.614] | 0.4545 |
| writing | 50 | 28/0/22 | 0.780 | [0.648, 0.873] | 1.000 | [0.879, 1.000] | 0.0000 |
| single_turn | 148 | 67/13/68 | 0.682 | [0.604, 0.752] | 0.838 | [0.742, 0.902] | 0.0000 |
| multi_turn | 52 | 32/3/17 | 0.779 | [0.649, 0.870] | 0.914 | [0.776, 0.970] | 0.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 140 | 71/9/60 | 0.721 | [0.642, 0.789] | 0.887 | [0.800, 0.940] | 0.0000 |
| similar | 23 | 6/5/12 | 0.522 | [0.330, 0.708] | 0.545 | [0.280, 0.787] | 1.0000 |
| x_longer | 37 | 22/2/13 | 0.770 | [0.613, 0.876] | 0.917 | [0.742, 0.977] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 3788, +0.05: 1361, +0.10: 338

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 42%, detecting a difference of 10 points needs 338 comparisons and 5 points 1361 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 91.5% of pairs; mean preference for the first-shown reply +0.007; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
