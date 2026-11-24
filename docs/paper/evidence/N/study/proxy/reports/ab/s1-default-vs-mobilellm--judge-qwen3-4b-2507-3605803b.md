# Blind comparison `s1-default-vs-mobilellm` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 7/4/49 | 0.525 | [0.401, 0.646] | 0.636 | [0.354, 0.848] | 0.5488 |
| chat | 15 | 0/1/14 | 0.467 | [0.248, 0.699] | 0.000 | [0.000, 0.793] | 1.0000 |
| code | 9 | 0/1/8 | 0.444 | [0.189, 0.733] | 0.000 | [0.000, 0.793] | 1.0000 |
| instruct | 12 | 5/0/7 | 0.708 | [0.428, 0.887] | 1.000 | [0.566, 1.000] | 0.0625 |
| math | 9 | 1/2/6 | 0.444 | [0.189, 0.733] | 0.333 | [0.061, 0.792] | 1.0000 |
| writing | 15 | 1/0/14 | 0.533 | [0.301, 0.752] | 1.000 | [0.206, 1.000] | 1.0000 |
| single_turn | 45 | 6/3/36 | 0.533 | [0.391, 0.671] | 0.667 | [0.354, 0.879] | 0.5078 |
| multi_turn | 15 | 1/1/13 | 0.500 | [0.274, 0.726] | 0.500 | [0.095, 0.905] | 1.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 45 | 7/0/38 | 0.578 | [0.433, 0.710] | 1.000 | [0.646, 1.000] | 0.0156 |
| similar | 9 | 0/3/6 | 0.333 | [0.121, 0.646] | 0.000 | [0.000, 0.561] | 0.2500 |
| x_longer | 6 | 0/1/5 | 0.417 | [0.140, 0.759] | 0.000 | [0.000, 0.793] | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 11880, +0.05: 4269, +0.10: 1058

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 82%, detecting a difference of 10 points needs 1058 comparisons and 5 points 4269 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 78.3% of pairs; mean preference for the first-shown reply -0.000; mean probability mass on A/B/C 0.977; replies clipped in 30 pairs.
