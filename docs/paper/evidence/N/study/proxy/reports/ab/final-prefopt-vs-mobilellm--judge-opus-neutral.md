# Blind comparison `final-prefopt-vs-mobilellm` rated by `judge-opus-neutral`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 112/15/73 | 0.743 | [0.678, 0.798] | 0.882 | [0.814, 0.927] | 0.0000 |
| chat | 50 | 28/1/21 | 0.770 | [0.637, 0.865] | 0.966 | [0.828, 0.994] | 0.0000 |
| code | 30 | 16/2/12 | 0.733 | [0.555, 0.858] | 0.889 | [0.672, 0.969] | 0.0013 |
| instruct | 40 | 28/0/12 | 0.850 | [0.709, 0.929] | 1.000 | [0.879, 1.000] | 0.0000 |
| math | 30 | 6/12/12 | 0.400 | [0.246, 0.577] | 0.333 | [0.163, 0.562] | 0.2379 |
| writing | 50 | 34/0/16 | 0.840 | [0.715, 0.917] | 1.000 | [0.898, 1.000] | 0.0000 |
| single_turn | 148 | 80/11/57 | 0.733 | [0.657, 0.798] | 0.879 | [0.796, 0.931] | 0.0000 |
| multi_turn | 52 | 32/4/16 | 0.769 | [0.639, 0.863] | 0.889 | [0.747, 0.956] | 0.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 140 | 76/9/55 | 0.739 | [0.661, 0.805] | 0.894 | [0.811, 0.943] | 0.0000 |
| similar | 28 | 15/5/8 | 0.679 | [0.493, 0.821] | 0.750 | [0.531, 0.888] | 0.0414 |
| x_longer | 32 | 21/1/10 | 0.812 | [0.647, 0.911] | 0.955 | [0.782, 0.992] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 3430, +0.05: 1233, +0.10: 306

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 36%, detecting a difference of 10 points needs 306 comparisons and 5 points 1233 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 96.0% of pairs; mean preference for the first-shown reply -0.010; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
