# Blind comparison `final-prefopt-vs-mobilellm` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 59/5/136 | 0.635 | [0.566, 0.699] | 0.922 | [0.830, 0.966] | 0.0000 |
| chat | 50 | 12/0/38 | 0.620 | [0.481, 0.741] | 1.000 | [0.757, 1.000] | 0.0005 |
| code | 30 | 13/0/17 | 0.717 | [0.538, 0.846] | 1.000 | [0.772, 1.000] | 0.0002 |
| instruct | 40 | 17/0/23 | 0.713 | [0.559, 0.829] | 1.000 | [0.816, 1.000] | 0.0000 |
| math | 30 | 5/4/21 | 0.517 | [0.346, 0.683] | 0.556 | [0.267, 0.811] | 1.0000 |
| writing | 50 | 12/1/37 | 0.610 | [0.472, 0.733] | 0.923 | [0.667, 0.986] | 0.0034 |
| single_turn | 148 | 45/5/98 | 0.635 | [0.555, 0.708] | 0.900 | [0.786, 0.957] | 0.0000 |
| multi_turn | 52 | 14/0/38 | 0.635 | [0.499, 0.752] | 1.000 | [0.785, 1.000] | 0.0001 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 140 | 49/1/90 | 0.671 | [0.590, 0.744] | 0.980 | [0.895, 0.997] | 0.0000 |
| similar | 28 | 4/3/21 | 0.518 | [0.342, 0.689] | 0.571 | [0.251, 0.842] | 1.0000 |
| x_longer | 32 | 6/1/25 | 0.578 | [0.408, 0.732] | 0.857 | [0.487, 0.974] | 0.1250 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 6806, +0.05: 2446, +0.10: 606

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 68%, detecting a difference of 10 points needs 606 comparisons and 5 points 2446 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 71.5% of pairs; mean preference for the first-shown reply -0.035; mean probability mass on A/B/C 0.995; replies clipped in 88 pairs.
