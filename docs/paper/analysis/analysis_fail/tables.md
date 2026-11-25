### loss_falcon_ours

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 6 | 2 | 9 | 0 | 12 | 29 |
| LOOP | 2 | 1 | 2 | 0 | 9 | 14 |
| FACT | 14 | 0 | 3 | 0 | 2 | 19 |
| MATH | 0 | 0 | 0 | 17 | 0 | 17 |
| CODE | 0 | 10 | 0 | 0 | 0 | 10 |
| INSTR | 1 | 0 | 6 | 0 | 7 | 14 |
| THIN | 1 | 0 | 1 | 1 | 2 | 5 |
| total | 24 | 13 | 21 | 18 | 32 | 108 |

### loss_falcon_theirs

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 1 | 0 | 0 | 0 | 0 | 1 |
| FACT | 9 | 0 | 1 | 0 | 4 | 14 |
| MATH | 0 | 0 | 0 | 3 | 0 | 3 |
| CODE | 0 | 7 | 0 | 0 | 0 | 7 |
| INSTR | 0 | 0 | 1 | 0 | 5 | 6 |
| THIN | 2 | 0 | 0 | 0 | 0 | 2 |
| OK | 12 | 6 | 19 | 15 | 23 | 75 |
| total | 24 | 13 | 21 | 18 | 32 | 108 |

### loss_falcon_ours_neutral

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 6 | 2 | 9 | 0 | 13 | 30 |
| LOOP | 2 | 1 | 2 | 0 | 9 | 14 |
| FACT | 13 | 0 | 3 | 0 | 2 | 18 |
| MATH | 0 | 0 | 0 | 17 | 0 | 17 |
| CODE | 0 | 10 | 0 | 0 | 0 | 10 |
| INSTR | 1 | 0 | 6 | 0 | 7 | 14 |
| THIN | 2 | 0 | 1 | 1 | 2 | 6 |
| total | 24 | 13 | 21 | 18 | 33 | 109 |

### bb_falcon_ours

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 2 | 0 | 3 | 0 | 1 | 6 |
| LOOP | 2 | 1 | 2 | 0 | 3 | 8 |
| FACT | 3 | 0 | 2 | 0 | 1 | 6 |
| MATH | 0 | 0 | 0 | 6 | 0 | 6 |
| CODE | 0 | 6 | 0 | 0 | 0 | 6 |
| INSTR | 0 | 0 | 0 | 0 | 3 | 3 |
| TRUNC | 0 | 0 | 1 | 0 | 0 | 1 |
| total | 7 | 7 | 8 | 6 | 8 | 36 |

### bb_falcon_theirs

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 3 | 0 | 2 | 0 | 5 | 10 |
| FACT | 4 | 0 | 4 | 0 | 1 | 9 |
| MATH | 0 | 0 | 0 | 6 | 0 | 6 |
| CODE | 0 | 7 | 0 | 0 | 0 | 7 |
| INSTR | 0 | 0 | 2 | 0 | 2 | 4 |
| total | 7 | 7 | 8 | 6 | 8 | 36 |

### bb_mobilellm_ours

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 1 | 2 | 4 | 0 | 4 | 11 |
| LOOP | 0 | 1 | 0 | 0 | 0 | 1 |
| FACT | 17 | 0 | 5 | 0 | 1 | 23 |
| MATH | 0 | 0 | 0 | 9 | 0 | 9 |
| CODE | 0 | 7 | 0 | 0 | 0 | 7 |
| INSTR | 0 | 0 | 0 | 0 | 7 | 7 |
| THIN | 0 | 0 | 1 | 1 | 1 | 3 |
| total | 18 | 10 | 10 | 10 | 13 | 61 |

### bb_mobilellm_theirs

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 4 | 2 | 5 | 0 | 7 | 18 |
| LOOP | 4 | 6 | 5 | 2 | 5 | 22 |
| FACT | 7 | 0 | 0 | 0 | 0 | 7 |
| MATH | 0 | 0 | 0 | 4 | 0 | 4 |
| CODE | 0 | 2 | 0 | 0 | 0 | 2 |
| TRUNC | 2 | 0 | 0 | 4 | 0 | 6 |
| THIN | 1 | 0 | 0 | 0 | 1 | 2 |
| total | 18 | 10 | 10 | 10 | 13 | 61 |

### win_mobilellm_theirs

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 10 | 3 | 18 | 2 | 23 | 56 |
| LOOP | 13 | 3 | 8 | 1 | 12 | 37 |
| FACT | 2 | 0 | 0 | 0 | 0 | 2 |
| MATH | 0 | 0 | 0 | 4 | 0 | 4 |
| CODE | 0 | 9 | 0 | 0 | 0 | 9 |
| INSTR | 0 | 0 | 1 | 0 | 1 | 2 |
| TRUNC | 0 | 0 | 1 | 0 | 0 | 1 |
| THIN | 3 | 1 | 1 | 0 | 0 | 5 |
| total | 28 | 16 | 29 | 7 | 36 | 116 |

### win_mobilellm_ours

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 0 | 0 | 1 | 0 | 2 | 3 |
| LOOP | 2 | 0 | 2 | 0 | 7 | 11 |
| FACT | 6 | 0 | 3 | 0 | 1 | 10 |
| MATH | 0 | 0 | 0 | 3 | 0 | 3 |
| CODE | 0 | 8 | 0 | 0 | 0 | 8 |
| INSTR | 2 | 0 | 11 | 0 | 8 | 21 |
| THIN | 2 | 0 | 1 | 0 | 1 | 4 |
| OK | 16 | 8 | 11 | 4 | 17 | 56 |
| total | 28 | 16 | 29 | 7 | 36 | 116 |

### win_falcon_theirs

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 2 | 0 | 3 | 0 | 1 | 6 |
| FACT | 8 | 0 | 1 | 0 | 1 | 10 |
| MATH | 0 | 0 | 0 | 3 | 0 | 3 |
| CODE | 0 | 10 | 0 | 0 | 0 | 10 |
| INSTR | 0 | 0 | 3 | 0 | 2 | 5 |
| WEAK | 0 | 0 | 0 | 1 | 0 | 1 |
| total | 10 | 10 | 7 | 4 | 4 | 35 |

### win_falcon_ours

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 1 | 0 | 0 | 0 | 0 | 1 |
| FACT | 1 | 0 | 1 | 0 | 1 | 3 |
| CODE | 0 | 2 | 0 | 0 | 0 | 2 |
| INSTR | 0 | 0 | 2 | 0 | 0 | 2 |
| THIN | 2 | 0 | 1 | 0 | 0 | 3 |
| OK | 6 | 8 | 3 | 4 | 3 | 24 |
| total | 10 | 10 | 7 | 4 | 4 | 35 |

### loss_mobilellm_ours

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| FACT | 1 | 0 | 0 | 0 | 0 | 1 |
| MATH | 0 | 0 | 0 | 12 | 0 | 12 |
| CODE | 0 | 2 | 0 | 0 | 0 | 2 |
| total | 1 | 2 | 0 | 12 | 0 | 15 |

### loss_mobilellm_theirs

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 0 | 0 | 0 | 5 | 0 | 5 |
| CODE | 0 | 2 | 0 | 0 | 0 | 2 |
| OK | 1 | 0 | 0 | 7 | 0 | 8 |
| total | 1 | 2 | 0 | 12 | 0 | 15 |

### loss_falcon_ours_other_tiebreak

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH | 4 | 2 | 6 | 0 | 9 | 21 |
| LOOP | 2 | 0 | 1 | 0 | 8 | 11 |
| FACT | 14 | 0 | 2 | 0 | 1 | 17 |
| MATH | 0 | 0 | 0 | 17 | 0 | 17 |
| CODE | 0 | 11 | 0 | 0 | 0 | 11 |
| INSTR | 1 | 0 | 10 | 0 | 9 | 20 |
| TRUNC | 0 | 0 | 0 | 0 | 1 | 1 |
| THIN | 3 | 0 | 2 | 1 | 4 | 10 |
| total | 24 | 13 | 21 | 18 | 32 | 108 |
