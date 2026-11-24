# Blind comparison `s1-default-vs-falcon` rated by `opus-1`

- x (ours): `ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 2/43/15 | 0.158 | [0.087, 0.271] | 0.044 | [0.012, 0.148] | 0.0000 |
| chat | 15 | 0/10/5 | 0.167 | [0.053, 0.416] | 0.000 | [0.000, 0.278] | 0.0020 |
| code | 9 | 2/6/1 | 0.278 | [0.090, 0.598] | 0.250 | [0.071, 0.591] | 0.2891 |
| instruct | 12 | 0/9/3 | 0.125 | [0.029, 0.403] | 0.000 | [0.000, 0.299] | 0.0039 |
| math | 9 | 0/6/3 | 0.167 | [0.040, 0.493] | 0.000 | [0.000, 0.390] | 0.0312 |
| writing | 15 | 0/12/3 | 0.100 | [0.023, 0.340] | 0.000 | [0.000, 0.242] | 0.0005 |
| single_turn | 45 | 2/28/15 | 0.211 | [0.117, 0.350] | 0.067 | [0.018, 0.213] | 0.0000 |
| multi_turn | 15 | 0/15/0 | 0.000 | [0.000, 0.204] | 0.000 | [0.000, 0.204] | 0.0001 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 40 | 2/27/11 | 0.188 | [0.096, 0.334] | 0.069 | [0.019, 0.220] | 0.0000 |
| similar | 15 | 0/13/2 | 0.067 | [0.012, 0.298] | 0.000 | [0.000, 0.228] | 0.0002 |
| x_longer | 5 | 0/3/2 | 0.200 | [0.036, 0.625] | 0.000 | [0.000, 0.561] | 0.2500 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 2904, +0.05: 1044, +0.10: 259

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 25%, detecting a difference of 10 points needs 259 comparisons and 5 points 1044 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
