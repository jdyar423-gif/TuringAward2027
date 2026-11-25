# Blind comparison `final-prefopt-vs-trained` rated by `judge-opus`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `ours-export-step51812-ef5599caa3974edc/final-trained-9a89782942-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 59/34/107 | 0.562 | [0.493, 0.629] | 0.634 | [0.533, 0.725] | 0.0124 |
| chat | 50 | 16/8/26 | 0.580 | [0.442, 0.706] | 0.667 | [0.467, 0.820] | 0.1516 |
| code | 30 | 10/1/19 | 0.650 | [0.471, 0.795] | 0.909 | [0.623, 0.984] | 0.0117 |
| instruct | 40 | 14/8/18 | 0.575 | [0.422, 0.715] | 0.636 | [0.429, 0.803] | 0.2863 |
| math | 30 | 5/5/20 | 0.500 | [0.332, 0.668] | 0.500 | [0.237, 0.763] | 1.0000 |
| writing | 50 | 14/12/24 | 0.520 | [0.385, 0.652] | 0.538 | [0.355, 0.712] | 0.8450 |
| single_turn | 148 | 45/25/78 | 0.568 | [0.487, 0.645] | 0.643 | [0.526, 0.745] | 0.0225 |
| multi_turn | 52 | 14/9/29 | 0.548 | [0.414, 0.675] | 0.609 | [0.408, 0.778] | 0.4049 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 19 | 8/1/10 | 0.684 | [0.460, 0.846] | 0.889 | [0.565, 0.980] | 0.0391 |
| similar | 144 | 42/19/83 | 0.580 | [0.498, 0.657] | 0.689 | [0.564, 0.791] | 0.0044 |
| x_longer | 37 | 9/14/14 | 0.432 | [0.287, 0.591] | 0.391 | [0.222, 0.592] | 0.4049 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 4684, +0.05: 1683, +0.10: 417

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 54%, detecting a difference of 10 points needs 417 comparisons and 5 points 1683 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 91.5% of pairs; mean preference for the first-shown reply +0.010; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
