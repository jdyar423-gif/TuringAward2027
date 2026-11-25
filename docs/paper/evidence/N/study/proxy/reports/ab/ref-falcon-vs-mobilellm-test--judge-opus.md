# Blind comparison `ref-falcon-vs-mobilellm-test` rated by `judge-opus`

- x (ours): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- y (opponent): `mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0`
- rated pairs: 200 of 200

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 200 | 144/6/50 | 0.845 | [0.788, 0.889] | 0.960 | [0.915, 0.982] | 0.0000 |
| chat | 50 | 37/1/12 | 0.860 | [0.738, 0.930] | 0.974 | [0.865, 0.995] | 0.0000 |
| code | 30 | 17/2/11 | 0.750 | [0.573, 0.870] | 0.895 | [0.686, 0.971] | 0.0007 |
| instruct | 40 | 31/0/9 | 0.887 | [0.754, 0.953] | 1.000 | [0.890, 1.000] | 0.0000 |
| math | 30 | 15/3/12 | 0.700 | [0.521, 0.833] | 0.833 | [0.608, 0.942] | 0.0075 |
| writing | 50 | 44/0/6 | 0.940 | [0.838, 0.979] | 1.000 | [0.920, 1.000] | 0.0000 |
| single_turn | 148 | 105/5/38 | 0.838 | [0.770, 0.888] | 0.955 | [0.898, 0.980] | 0.0000 |
| multi_turn | 52 | 39/1/12 | 0.865 | [0.747, 0.933] | 0.975 | [0.871, 0.996] | 0.0000 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 137 | 101/3/33 | 0.858 | [0.789, 0.906] | 0.971 | [0.919, 0.990] | 0.0000 |
| similar | 25 | 16/2/7 | 0.780 | [0.587, 0.898] | 0.889 | [0.672, 0.969] | 0.0013 |
| x_longer | 38 | 27/1/10 | 0.842 | [0.696, 0.926] | 0.964 | [0.823, 0.994] | 0.0000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 2904, +0.05: 1044, +0.10: 259

With n = 200 comparisons the 95% interval of a score is about ±7 points; at a tie rate of 25%, detecting a difference of 10 points needs 259 comparisons and 5 points 1044 (two-sided 5% test, 80% power), so a smaller lead on n = 200 is no evidence either way.

Judge diagnostics: both orders agree on 93.5% of pairs; mean preference for the first-shown reply +0.013; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
