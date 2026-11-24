# Blind comparison `s2-default-vs-falcon` rated by `judge-opus`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 3/39/18 | 0.200 | [0.118, 0.318] | 0.071 | [0.025, 0.190] | 0.0000 |
| chat | 15 | 0/10/5 | 0.167 | [0.053, 0.416] | 0.000 | [0.000, 0.278] | 0.0020 |
| code | 9 | 1/6/2 | 0.222 | [0.063, 0.547] | 0.143 | [0.026, 0.513] | 0.1250 |
| instruct | 12 | 2/7/3 | 0.292 | [0.113, 0.572] | 0.222 | [0.063, 0.547] | 0.1797 |
| math | 9 | 0/5/4 | 0.222 | [0.063, 0.547] | 0.000 | [0.000, 0.434] | 0.0625 |
| writing | 15 | 0/11/4 | 0.133 | [0.037, 0.379] | 0.000 | [0.000, 0.259] | 0.0010 |
| single_turn | 45 | 3/25/17 | 0.256 | [0.151, 0.399] | 0.107 | [0.037, 0.272] | 0.0000 |
| multi_turn | 15 | 0/14/1 | 0.033 | [0.004, 0.254] | 0.000 | [0.000, 0.215] | 0.0001 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 38 | 2/26/10 | 0.184 | [0.092, 0.334] | 0.071 | [0.020, 0.227] | 0.0000 |
| similar | 17 | 1/11/5 | 0.206 | [0.078, 0.442] | 0.083 | [0.015, 0.354] | 0.0063 |
| x_longer | 5 | 0/2/3 | 0.300 | [0.073, 0.701] | 0.000 | [0.000, 0.658] | 0.5000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 3112, +0.05: 1118, +0.10: 277

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 30%, detecting a difference of 10 points needs 277 comparisons and 5 points 1118 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.

Judge diagnostics: both orders agree on 93.3% of pairs; mean preference for the first-shown reply +0.000; mean probability mass on A/B/C 1.000; replies clipped in 0 pairs.
