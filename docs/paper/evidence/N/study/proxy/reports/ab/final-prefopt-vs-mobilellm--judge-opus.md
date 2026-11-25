# Blind comparison `final-prefopt-vs-mobilellm` rated by `judge-opus`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 116/15/69 | 0.752 | [0.688, 0.807] | 0.885 | [0.820, 0.929] | 0.0000 |
| chat | 50 | 28/1/21 | 0.770 | [0.637, 0.865] | 0.966 | [0.828, 0.994] | 0.0000 |
| code | 30 | 16/2/12 | 0.733 | [0.555, 0.858] | 0.889 | [0.672, 0.969] | 0.0013 |
| instruct | 40 | 29/0/11 | 0.863 | [0.724, 0.938] | 1.000 | [0.883, 1.000] | 0.0000 |
| math | 30 | 7/12/11 | 0.417 | [0.260, 0.593] | 0.368 | [0.192, 0.590] | 0.3593 |
| writing | 50 | 36/0/14 | 0.860 | [0.738, 0.930] | 1.000 | [0.904, 1.000] | 0.0000 |
| single_turn | 148 | 83/11/54 | 0.743 | [0.667, 0.807] | 0.883 | [0.802, 0.933] | 0.0000 |
| multi_turn | 52 | 33/4/15 | 0.779 | [0.649, 0.870] | 0.892 | [0.753, 0.957] | 0.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 140 | 77/9/54 | 0.743 | [0.665, 0.808] | 0.895 | [0.813, 0.944] | 0.0000 |
| similar | 28 | 17/5/6 | 0.714 | [0.529, 0.848] | 0.773 | [0.566, 0.899] | 0.0169 |
| x_longer | 32 | 22/1/9 | 0.828 | [0.664, 0.921] | 0.957 | [0.790, 0.992] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 3326, +0.05: 1195, +0.10: 296

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 34%, detecting a difference of 10 points needs 296 comparisons and 5 points 1195 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 96.0% of pairs; mean preference for the first-shown reply -0.005; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
