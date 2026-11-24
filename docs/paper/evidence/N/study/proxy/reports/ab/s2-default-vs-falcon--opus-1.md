# Blind comparison `s2-default-vs-falcon` rated by `opus-1`

- x (ours): `ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0`
- y (opponent): `falcon-e60034381f4a/recommended-5ccec0259b-s0`
- rated pairs: 60 of 60

Score = (wins + ties/2) / n; ties include 'both bad'. Decisive = wins / (wins + losses). Intervals are 95% Wilson; p is the exact two-sided sign test of wins against losses.

| slice | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| overall | 60 | 3/41/16 | 0.183 | [0.106, 0.299] | 0.068 | [0.024, 0.182] | 0.0000 |
| chat | 15 | 0/10/5 | 0.167 | [0.053, 0.416] | 0.000 | [0.000, 0.278] | 0.0020 |
| code | 9 | 1/7/1 | 0.167 | [0.040, 0.493] | 0.125 | [0.022, 0.471] | 0.0703 |
| instruct | 12 | 2/7/3 | 0.292 | [0.113, 0.572] | 0.222 | [0.063, 0.547] | 0.1797 |
| math | 9 | 0/5/4 | 0.222 | [0.063, 0.547] | 0.000 | [0.000, 0.434] | 0.0625 |
| writing | 15 | 0/12/3 | 0.100 | [0.023, 0.340] | 0.000 | [0.000, 0.242] | 0.0005 |
| single_turn | 45 | 3/27/15 | 0.233 | [0.134, 0.375] | 0.100 | [0.035, 0.256] | 0.0000 |
| multi_turn | 15 | 0/14/1 | 0.033 | [0.004, 0.254] | 0.000 | [0.000, 0.215] | 0.0001 |

Length control (ratio of x's words to y's words):

| bucket | n | W/L/T | score | score 95% | decisive | decisive 95% | p |
|---|---|---|---|---|---|---|---|
| x_shorter | 38 | 2/27/9 | 0.171 | [0.083, 0.319] | 0.069 | [0.019, 0.220] | 0.0000 |
| similar | 17 | 1/12/4 | 0.176 | [0.062, 0.410] | 0.077 | [0.014, 0.333] | 0.0034 |
| x_longer | 5 | 0/2/3 | 0.300 | [0.073, 0.701] | 0.000 | [0.000, 0.658] | 0.5000 |

Comparisons needed (power 0.8, alpha 0.05, at the observed tie rate): +0.03: 2970, +0.05: 1068, +0.10: 265

With n = 60 comparisons the 95% interval of a score is about ±12 points; at a tie rate of 27%, detecting a difference of 10 points needs 265 comparisons and 5 points 1068 (two-sided 5% test, 80% power), so a smaller lead on n = 60 is no evidence either way.
