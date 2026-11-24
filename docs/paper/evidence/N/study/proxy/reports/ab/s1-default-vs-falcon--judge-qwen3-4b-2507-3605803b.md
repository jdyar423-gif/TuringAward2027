# Blind comparison `s1-default-vs-falcon` rated by `judge-qwen3-4b-2507-3605803b`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 2/16/42 | 0.383 | [0.271, 0.510] | 0.111 | [0.031, 0.328] | 0.0013 |
| chat | 15 | 0/4/11 | 0.367 | [0.175, 0.613] | 0.000 | [0.000, 0.490] | 0.1250 |
| code | 9 | 0/3/6 | 0.333 | [0.121, 0.646] | 0.000 | [0.000, 0.561] | 0.2500 |
| instruct | 12 | 0/4/8 | 0.333 | [0.138, 0.609] | 0.000 | [0.000, 0.490] | 0.1250 |
| math | 9 | 1/3/5 | 0.389 | [0.153, 0.691] | 0.250 | [0.046, 0.699] | 0.6250 |
| writing | 15 | 1/2/12 | 0.467 | [0.248, 0.699] | 0.333 | [0.061, 0.792] | 1.0000 |
| single_turn | 45 | 2/12/31 | 0.389 | [0.261, 0.535] | 0.143 | [0.040, 0.399] | 0.0129 |
| multi_turn | 15 | 0/4/11 | 0.367 | [0.175, 0.613] | 0.000 | [0.000, 0.490] | 0.1250 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 40 | 2/8/30 | 0.425 | [0.285, 0.578] | 0.200 | [0.057, 0.510] | 0.1094 |
| similar | 15 | 0/7/8 | 0.267 | [0.109, 0.519] | 0.000 | [0.000, 0.354] | 0.0156 |
| x_longer | 5 | 0/1/4 | 0.400 | [0.118, 0.769] | 0.000 | [0.000, 0.793] | 1.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 7260, +0.05: 2609, +0.10: 647

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 70%, detecting a difference of 10 points needs 647 comparisons and 5 points 2609 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 63.3% of pairs; mean preference for the first-shown reply -0.096; mean probability mass on A/B/C 0.995; replies clipped in 0 pairs.
