# Blind comparison `final-prefopt-vs-mobilellm-k6ds` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-aca173fa2c-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 31/8/161 | 0.557 | [0.488, 0.625] | 0.795 | [0.645, 0.892] | 0.0003 |
| chat | 50 | 6/2/42 | 0.540 | [0.404, 0.670] | 0.750 | [0.409, 0.928] | 0.2891 |
| code | 30 | 9/2/19 | 0.617 | [0.439, 0.768] | 0.818 | [0.523, 0.949] | 0.0654 |
| instruct | 40 | 6/0/34 | 0.575 | [0.422, 0.715] | 1.000 | [0.610, 1.000] | 0.0312 |
| math | 30 | 4/4/22 | 0.500 | [0.332, 0.668] | 0.500 | [0.215, 0.785] | 1.0000 |
| writing | 50 | 6/0/44 | 0.560 | [0.423, 0.688] | 1.000 | [0.610, 1.000] | 0.0312 |
| single_turn | 148 | 28/8/112 | 0.568 | [0.487, 0.645] | 0.778 | [0.619, 0.883] | 0.0012 |
| multi_turn | 52 | 3/0/49 | 0.529 | [0.396, 0.658] | 1.000 | [0.439, 1.000] | 0.2500 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 94 | 19/3/72 | 0.585 | [0.484, 0.679] | 0.864 | [0.667, 0.953] | 0.0009 |
| similar | 49 | 6/1/42 | 0.551 | [0.413, 0.681] | 0.857 | [0.487, 0.974] | 0.1250 |
| x_longer | 57 | 6/4/47 | 0.517 | [0.391, 0.642] | 0.600 | [0.313, 0.832] | 0.7539 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 11169, +0.05: 4013, +0.10: 995

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 80%, detecting a difference of 10 points needs 995 comparisons and 5 points 4013 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 68.5% of pairs; mean preference for the first-shown reply -0.054; mean probability mass on A/B/C 0.993; replies clipped in 28 pairs.
