# Blind comparison `final-prefopt-vs-mobilellm-k6lp` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-32f0c6d864-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 39/8/153 | 0.578 | [0.508, 0.644] | 0.830 | [0.699, 0.911] | 0.0000 |
| chat | 50 | 10/2/38 | 0.580 | [0.442, 0.706] | 0.833 | [0.552, 0.953] | 0.0386 |
| code | 30 | 8/3/19 | 0.583 | [0.407, 0.740] | 0.727 | [0.434, 0.902] | 0.2266 |
| instruct | 40 | 10/0/30 | 0.625 | [0.470, 0.758] | 1.000 | [0.723, 1.000] | 0.0020 |
| math | 30 | 6/3/21 | 0.550 | [0.377, 0.712] | 0.667 | [0.354, 0.879] | 0.5078 |
| writing | 50 | 5/0/45 | 0.550 | [0.413, 0.679] | 1.000 | [0.566, 1.000] | 0.0625 |
| single_turn | 148 | 33/7/108 | 0.588 | [0.507, 0.664] | 0.825 | [0.680, 0.912] | 0.0000 |
| multi_turn | 52 | 6/1/45 | 0.548 | [0.414, 0.675] | 0.857 | [0.487, 0.974] | 0.1250 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 99 | 25/1/73 | 0.621 | [0.523, 0.711] | 0.962 | [0.811, 0.993] | 0.0000 |
| similar | 36 | 5/2/29 | 0.542 | [0.383, 0.692] | 0.714 | [0.359, 0.918] | 0.4531 |
| x_longer | 65 | 9/5/51 | 0.531 | [0.411, 0.647] | 0.643 | [0.388, 0.837] | 0.4239 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 9268, +0.05: 3330, +0.10: 825

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 76%, detecting a difference of 10 points needs 825 comparisons and 5 points 3330 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 71.0% of pairs; mean preference for the first-shown reply -0.054; mean probability mass on A/B/C 0.986; replies clipped in 29 pairs.
