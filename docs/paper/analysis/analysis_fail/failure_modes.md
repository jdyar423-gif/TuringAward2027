# What the official model gets wrong, and right: a failure analysis of the blind test comparisons

The official model ("ours": 98,036,104 stored numbers, trained from scratch for one day on one TPU; weights after preference optimisation, default decoding) was compared blind with Falcon-H1-Tiny-90M-Instruct and with MobileLLM-R1.5-140M on the 200 held-out test prompts (chat 50, writing 50, instruct 40, math 30, code 30; 52 of them with two turns). Each pair was read four times by Opus 5.5 readers who did not know which reply was ours (in the first round the chunk titles named the pair set, so the readers knew which two systems were compared, though not which side was which; the second round removed the names): a first round, r1 (sides as published) and r2 (sides exchanged), and a second round with neutral titles, a (as published) and b (exchanged). Within a round a prompt is a win or a loss only when both readings choose the same reply; anything else, "both bad" included, is a tie. Recomputed here from the raw verdict files and the keys, the outcomes reproduce the reports exactly: against Falcon 35 wins / 108 losses / 57 ties (neutral round 33/109/58), against MobileLLM 116/15/69 (neutral 112/15/73).

The material is the readers' notes (one per reading, at most 15 words: 1,600 notes), the replies, and the exact scorers' per-prompt fields. Outcomes below are the first round's unless said otherwise. Eleven of the 200 prompts were seen by a mid-run sample during training; they are flagged in `labels.json` and leaving them out changes nothing material (section 2.1). Every script and intermediate file is in this directory (section 8).

## 1. Codebook (fixed before labelling)

The codebook was written to `codebook.json` before labelling began; only 19 notes had been read, to see their form (the 15 of the first chunk's first reading and 4 of the same chunk's neutral reading). Every reply gets exactly one label: its main defect as the readers' notes name it, or OK. The replies were read whenever the notes were unclear, disagreed, or did not say what was wrong with a reply.

| code | reason | definition |
|---|---|---|
| FACT | factual error | states false facts or invents entities, names, dates, definitions or mechanisms (non-numeric content; in chat, writing and instruct prompts). |
| MATH | wrong arithmetic or reasoning | wrong operation or set-up, calculation slip, wrong unit conversion, or wrong logic (any category, including counting inside a word problem); in maths prompts it includes misstated working when the final answer is right (amendment F). |
| INSTR | instruction or format not followed | on topic, but an explicit requirement is violated: number of words, sentences, items, lines or paragraphs; case; punctuation; required or forbidden words; P.S.; JSON or other structure; language; register (formal, for a child ...); 'only output ...'; a follow-up request not applied. |
| INCOH | incoherent or off-topic | does not address the request or answers another question; incoherent, self-contradictory or nonsensical text; leaked template or role text. |
| LOOP | repetition or loop | repeated sentences, list items or phrases that dominate the reply; a degenerate loop. |
| TRUNC | truncated | stops in mid-sentence or mid-code (token limit) before the answer or a required part is delivered. |
| THIN | too thin, empty or deflecting | empty, generic or much too short to be useful; refuses or deflects a harmless request; or leaves out a key part of what was asked. |
| CODE | broken or wrong code | code that would not run or does not do what was asked; wrong language; nonsense code; a wrong explanation of what given code does (code category). |
| WEAK | weaker, no hard error | acceptable, but the readers preferred the other reply for quality: less useful, less clear, less complete, clumsier or more verbose; no factual, reasoning, instruction or degeneracy defect is named. |
| OK | no decisive defect | correct, on topic and following the instructions; minor flaws allowed. The preferred reply of a decisive pair gets OK unless the notes name a real defect in it; both replies of a 'both acceptable' tie get OK. |

Rules. (1) Choose the defect named by most of the four notes for that reply. (2) Ties go in the order INCOH, LOOP, FACT, MATH, CODE, INSTR, TRUNC, THIN, WEAK. (3) A loop that runs into the token limit is LOOP, not TRUNC. (4) In maths a wrong final answer is MATH unless no answer is reached (then TRUNC if cut off, THIN if it stops, INCOH if it answers another question). (5) In code prompts an error in the code or in an explanation of code is CODE. (6) A reply the notes do not fault is read, and labelled OK if acceptable. (7) Pair-level reasons are derived, not labelled: a loss is explained by our label, a win by the opponent's ("ours better because the opponent's reply ..."), a both-bad prompt by both labels.

Five clarifications were added during labelling, each at the point given here, and the labels made before each were re-checked against it. A, B and C were added together after the first 20 Falcon pairs: (A) in rule 2 LOOP moved from after INSTR to second place, because a reply dominated by repetition is degenerate whatever else is wrong with it; (B) the preferred reply, or an acceptable reply in a tie, is OK unless at least two notes name a real defect in it; (C) a defect the notes give to both replies counts for each. (D, after 42 Falcon pairs) A defect given to one reply only and named by at least two notes outranks a shared one; two notes describing the same fault in other words count as one defect. (E, after 158 MobileLLM pairs) A reply that consists of task meta-text, rambling or a boxed stub instead of the requested content is INCOH; THIN is kept for replies that answer with too little. One more change was made after labelling, prompted by a review of this report, and changed no label: (F) the MATH definition had said "wrong logic that gives a wrong result"; it now also covers misstated working when the final answer is right, which is how 5 labels had been given (ours on math-031, math-055 in the Falcon set and on math-049, math-055 in the MobileLLM set; MobileLLM's on math-019). With every defect label I recorded how many of the four notes name it ("support"). Our 24 wrong maths answers also got a subcode: SETUP (wrong operation, formula or conversion), SLIP (right set-up, wrong calculation), NONSENSE (working unrelated to the problem), NOANSWER, CONTRADICT (the right value appears, the stated answer differs).

The labels are mine alone and were made with the key open (the notes were rewritten as ours and theirs to make them readable). The notes were written blind; the labels were not. Section 5 measures how stable they are.

## 2. Why we lose to Falcon

### 2.1 The 108 losses

Table 1. Main defect of OUR reply on the prompts lost to Falcon, by category. The last column gives the overall counts if each of the 19 labels decided by the tie-break (two defects named by equally many notes) had gone the other way.

| reason | chat | code | instruct | math | writing | all | all, other tie-break |
|---|---|---|---|---|---|---|---|
| INCOH (incoherent or off-topic) | 6 | 2 | 9 | 0 | 12 | 29 | 21 |
| LOOP (repetition or loop) | 2 | 1 | 2 | 0 | 9 | 14 | 11 |
| FACT (factual error) | 14 | 0 | 3 | 0 | 2 | 19 | 17 |
| MATH (wrong arithmetic or reasoning) | 0 | 0 | 0 | 17 | 0 | 17 | 17 |
| CODE (broken or wrong code) | 0 | 10 | 0 | 0 | 0 | 10 | 11 |
| INSTR (instruction or format not followed) | 1 | 0 | 6 | 0 | 7 | 14 | 20 |
| THIN (too thin, empty or deflecting) | 1 | 0 | 1 | 1 | 2 | 5 | 10 |
| TRUNC (truncated) | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **n (total)** | **24** | **13** | **21** | **18** | **32** | **108** | **108** |

- The losses divide almost evenly between wrong content (FACT, MATH, CODE: 46 of 108) and text that is incoherent, off the task, repetitive or cut off (INCOH, LOOP, TRUNC: 43; no loss is TRUNC); missed instructions, thin and weaker replies (INSTR, THIN, WEAK; no loss is WEAK) make up the other 19. With the other tie-break the three groups are 45, 33 and 30 (the middle group now includes 1 TRUNC): wrong content stays the largest group; the split between incoherence and missed instructions is the least certain part of the table.
- Each category fails in its own way. Writing (32 losses): 21 incoherent, off-task or repetitive texts (an email to the landlord written from the landlord's side, a thank-you note that thanks the teacher for "the amazing opportunities you've had", poems that circle on "gentle embrace" or "tapestry" and "journey") and 7 form failures (a limerick or a haiku written as long prose, a "one-line" advert of a paragraph, a notice copied back unchanged). Chat (24): 14 false facts, stated fluently ("inflation means falling prices", the Dead Sea "in the Indian Ocean", the first modern Olympics "in London in 1900", a haiku "a vowel sound"). Instruct (21): 9 incoherent or off-task replies (the single word "CARLENS" for a thunderstorm in capitals, a paragraph about the word "green" instead of about forests) and 6 missed instructions (a required word, a closing question, a P.S.). Maths (18): 16 are wrong (15) or missing (1) answers, and the other 2 are right answers with misstated working (next point). Code (13): 10 errors in code or in its explanation (a ZIP-code regex that needs six digits, a "recursive" Fibonacci function that is not recursive, an example claiming `sum_digits(10)` prints 120, the explanation "(°C × 9/5) + 32 = 0").
- Falcon's reply had no decisive defect in 75 of the 108 losses. In the other 33 it had one (FACT 14, CODE 7, INSTR 6, MATH 3, THIN 2, INCOH 1), but ours was judged worse.
- 2 of the 17 MATH losses have the right final answer with misstated working (math-031, math-055; MATH by amendment F); both systems were right and the readers preferred Falcon's clean working.
- Length (from the A/B report's buckets): on the 35 prompts where ours is more than one and a half times as long as Falcon's reply, ours won 2 and lost 27; where it is under two thirds of Falcon's length (93 prompts) it won 26 and lost 41. Among the losses where ours is the longer reply, 14 of 27 are INCOH or LOOP and 8 INSTR (five of these are prose where a haiku or a limerick was asked, or a text far over its length limit); among those where it is the shorter, 12 of 41 are, and the rest are mostly FACT (10), CODE (7), MATH (6) and THIN (5).
- Robustness. The neutral round has 109 losses, 106 of them shared with the first round, and the same ranking (INCOH 30, FACT 18, MATH 17, LOOP 14, INSTR 14, CODE 10, THIN 6). On the 189 prompts never seen during the run there are 101 losses: INCOH 26, MATH 17, FACT 16, LOOP 14, INSTR 14, CODE 10, THIN 4.

### 2.2 Prompts both systems failed

A prompt counts as "both bad" when both first-round readings returned both_bad (all such prompts are first-round ties). That holds for 36 prompts; with other rules: all four readings 34, at least three 41, at least one 54. The other 21 of the 57 first-round ties are split readings or genuine ties.

Table 2. Main defect on the 36 both-bad prompts: ours / Falcon.

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH (incoherent or off-topic) | 2 / 3 | 0 / 0 | 3 / 2 | 0 / 0 | 1 / 5 | 6 / 10 |
| LOOP (repetition or loop) | 2 / 0 | 1 / 0 | 2 / 0 | 0 / 0 | 3 / 0 | 8 / 0 |
| FACT (factual error) | 3 / 4 | 0 / 0 | 2 / 4 | 0 / 0 | 1 / 1 | 6 / 9 |
| MATH (wrong arithmetic or reasoning) | 0 / 0 | 0 / 0 | 0 / 0 | 6 / 6 | 0 / 0 | 6 / 6 |
| CODE (broken or wrong code) | 0 / 0 | 6 / 7 | 0 / 0 | 0 / 0 | 0 / 0 | 6 / 7 |
| INSTR (instruction or format not followed) | 0 / 0 | 0 / 0 | 0 / 2 | 0 / 0 | 3 / 2 | 3 / 4 |
| TRUNC (truncated) | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 1 / 0 |
| **n (total)** | **7** | **7** | **8** | **6** | **8** | **36** |

- The two replies have the same label on 22 of the 36: both wrong on the same maths problem (all 6 maths cases; 47 × 23 given as 1176 and 1112), both broken in 6 of the 7 code prompts (in the seventh ours repeats one broken block three times), both inventing the physics of lunar phases, ear popping or the speed of sound ("1.7 m/s" against "156,000 mph").
- Where they differ, ours fails by repetition (LOOP 8, Falcon 0) and Falcon by incoherence (INCOH 10, ours 6).
- Against MobileLLM 61 prompts are both-bad by the same rule (21 of them both-bad against Falcon too). There our reply's defect is most often a false fact (23 of 61), MobileLLM's a loop (22) or an incoherent or off-task reply (18).

## 3. What our model gets right

### 3.1 The 116 wins over MobileLLM

Table 3. Main defect on the prompts we won: MobileLLM / ours.

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH (incoherent or off-topic) | 10 / 0 | 3 / 0 | 18 / 1 | 2 / 0 | 23 / 2 | 56 / 3 |
| LOOP (repetition or loop) | 13 / 2 | 3 / 0 | 8 / 2 | 1 / 0 | 12 / 7 | 37 / 11 |
| FACT (factual error) | 2 / 6 | 0 / 0 | 0 / 3 | 0 / 0 | 0 / 1 | 2 / 10 |
| MATH (wrong arithmetic or reasoning) | 0 / 0 | 0 / 0 | 0 / 0 | 4 / 3 | 0 / 0 | 4 / 3 |
| CODE (broken or wrong code) | 0 / 0 | 9 / 8 | 0 / 0 | 0 / 0 | 0 / 0 | 9 / 8 |
| INSTR (instruction or format not followed) | 0 / 2 | 0 / 0 | 1 / 11 | 0 / 0 | 1 / 8 | 2 / 21 |
| TRUNC (truncated) | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 1 / 0 |
| THIN (too thin, empty or deflecting) | 3 / 2 | 1 / 0 | 1 / 1 | 0 / 0 | 0 / 1 | 5 / 4 |
| OK (no decisive defect) | 0 / 16 | 0 / 8 | 0 / 11 | 0 / 4 | 0 / 17 | 0 / 56 |
| **n (total)** | **28** | **16** | **29** | **7** | **36** | **116** |

- 93 of the 116 wins come from MobileLLM's reply being degenerate or off the task: endless loops ("The ocean is the ocean", "Wait, no"), meta-text about the task ending in a boxed stub (`\boxed{Open the School Science Fair}` for a speech, `\boxed{Local ingredients}` for a slogan, `\boxed{The Sun}` for a ten-word description of a season), rambles that never deliver the list (instruct-059), the JSON (instruct-062) or the code (code-034). The exact scorer flags MobileLLM's reply as degenerate on 82 of the 116 wins. MobileLLM's replies average 474.1 words (ours 130.4, Falcon 173.4).
- Our own reply had a defect named by at least two notes in 60 of these wins (most often a missed instruction, 21); it was OK in 56. Many of these are wins of an imperfect but on-topic reply over a non-answer.
- The registered automatic judge (qwen3-4b-2507, both orders) points the same way but ties most of them: of the 116 it gives us 51, ties 64 and gives MobileLLM 1.

### 3.2 The 35 wins over Falcon

Table 4. Main defect on the prompts we won: Falcon / ours.

| reason | chat | code | instruct | math | writing | all |
|---|---|---|---|---|---|---|
| INCOH (incoherent or off-topic) | 2 / 1 | 0 / 0 | 3 / 0 | 0 / 0 | 1 / 0 | 6 / 1 |
| FACT (factual error) | 8 / 1 | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 | 10 / 3 |
| MATH (wrong arithmetic or reasoning) | 0 / 0 | 0 / 0 | 0 / 0 | 3 / 0 | 0 / 0 | 3 / 0 |
| CODE (broken or wrong code) | 0 / 0 | 10 / 2 | 0 / 0 | 0 / 0 | 0 / 0 | 10 / 2 |
| INSTR (instruction or format not followed) | 0 / 0 | 0 / 0 | 3 / 2 | 0 / 0 | 2 / 0 | 5 / 2 |
| THIN (too thin, empty or deflecting) | 0 / 2 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 3 |
| WEAK (weaker, no hard error) | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 1 / 0 |
| OK (no decisive defect) | 0 / 6 | 0 / 8 | 0 / 3 | 0 / 4 | 0 / 3 | 0 / 24 |
| **n (total)** | **10** | **10** | **7** | **4** | **4** | **35** |

- Code and chat give the most wins (10 each of the 35); for its size code has the best record, 10/13/7 over its 30 prompts (chat 10/24/16 over 50). Every code win is a prompt where Falcon's code was broken (a syntax error in `s[::-1.strip()]`, a regex that matches the literal string "aeiouAEIOU", a binary search that misses the last element, a KeyError) while ours ran, with faults of its own in 2 (code-002, code-026: a wrong example comment, a maximum initialised to the array's length). Falcon's false facts give 10 more (8 in chat: cats purr to "raise humidity", "mow" houseplants, "swallow water" when the phone falls in the sink), and broken instructions 5 (eight lines for "exactly four", four sentences for a two-sentence story, nine words for ten, quotation marks around two words instead of the whole answer, an outline instead of a speech).
- Our winning replies are mostly short: 26 of the 35 wins are prompts where ours has less than two thirds of Falcon's words. Ours is OK in 16 of these 26 and in 24 of all 35 wins; the notes call it thin in 3 wins (all of them among the short ones: a seven-word answer on stretching, "vague but harmless" advice, an answer that omits the "why").
- These wins are fragile. With one particle instead of six (the K = 1 arm: same weights, read the same way) only 17 of the 35 prompts are won again (8 ties, 10 losses), while 99 of the 108 losses are lost again (7 ties, 2 wins).

### 3.3 The 15 losses to MobileLLM

12 of the 15 are maths (our label MATH in all 12: 11 wrong answers, and math-055 with the right answer and misstated working); the others are code-018 and code-044 (CODE) and chat-087 (FACT: sound "travels at 1.7 m/s"). MobileLLM's own reply was faulted in 7 of the 15 and OK in 8: in both code losses its code or its explanation of the code was wrong as well (CODE 2), and in 5 maths losses (INCOH) it rambled before reaching the right answer (math-007, math-017) or answered the first turn again instead of the second (math-008, math-032, math-044), where ours was wrong too.

## 4. What the exact scorers say

### 4.1 Maths: the 30 problems

Table 5. Last-turn answer correct (extracted value in brackets; two-turn values as turn 1 / turn 2), and how ours fails.

| prompt | turns | answer | ours | Falcon | MobileLLM | how ours fails (subcode) |
|---|---|---|---|---|---|---|
| math-001 | 1 | 1081 | no (1176) | no (1112) | yes (1081) | SLIP: 47 x 23 = 1176 |
| math-007 | 1 | 1.375 | no (1.25) | yes (1.375) | yes (1.375) | SETUP: multiplies the fractions instead of adding |
| math-008 | 2 | 126 / 42 | no (126/54) | yes (126/42) | no (126/126) | SETUP: turn 2 divides 18, not 126 (turn 1 right) |
| math-010 | 1 | 20 | no (24) | yes (20) | yes (20) | SLIP: 14 + 22 + 9 + 35 = 96 |
| math-012 | 2 | 60 / 240 | yes (60/240) | no (60/120) | no (60/60) | correct |
| math-014 | 1 | 36 | no (36.5) | yes (36) | yes (36) | SLIP: copies 240 as 241, then 36.5 |
| math-015 | 1 | 15 | no (-18) | yes (15) | yes (15) | SLIP: 92 - 80 = 18, then a negative percentage |
| math-017 | 1 | 25 | no (20) | yes (25) | yes (25) | SLIP: 8 / 32 = 0.2 |
| math-019 | 1 | 45 | yes (45) | no (225) | yes (45) | correct |
| math-020 | 2 | 5000 / 12.5 | no (0.05/4) | no (5000/1000) | no (5000/5000) | SETUP: km to m by dividing by 100; laps not computed |
| math-024 | 2 | 75 / 750 | no (75/10) | no (75/3000) | no (300/1200) | NONSENSE: turn 2 working incoherent, ends '75 grams for 10 people' (turn 1 right) |
| math-026 | 1 | 3276 | no (3080) | no (3216) | no (4) | SETUP: subtracts the raise, ignores the second raise |
| math-030 | 1 | 121.92 | no (4.0125) | yes (121.92) | no (0.3333333333) | SETUP: takes 1 foot = 2.54 cm |
| math-031 | 1 | 0.75 | yes (0.75) | yes (0.75) | yes (0.75) | correct; working faulted: misstates the conversion factor |
| math-032 | 2 | 165 / 110 | no (15/5) | no (165/37.5) | no (165/165) | SLIP: '$85 - $250 = $15', carried into turn 2 |
| math-034 | 1 | 10080 | no (7) | yes (10080) | no (60) | NOANSWER: 'In a typical week, there are 7 days.' |
| math-036 | 2 | 84 / 89.3 | no (84/62.5) | no (84/75) | no (84/84) | SETUP: turn 2 divides by 12 instead of 84 (turn 1 right) |
| math-039 | 1 | 200000 | no (20) | yes (200000) | no (20) | SETUP: no m^2 to cm^2 conversion |
| math-042 | 1 | 215 | no (13.33) | no (1080) | no (13) | NONSENSE: '13:15 * 60 = 830 minutes' |
| math-044 | 2 | 128 / 8 | no (448/18) | no (128/14) | no (128/128) | CONTRADICT: computes 128, then answers 320 + 128 = 448 |
| math-045 | 1 | 46 | no (23) | yes (46) | yes (46) | SETUP: adds two sides only |
| math-047 | 1 | 450 | no (180) | no (3600) | yes (450) | SETUP: '3/8 full means 1/8 full' |
| math-048 | 2 | 1023 / 341 | no (85/2) | no (1023/3) | no (400/1119) | NONSENSE: invented 'sum' formula, 85 |
| math-049 | 1 | 490 | yes (490) | no (980) | no (70) | correct; working faulted: 35 x 2 = 70, then x 7: right number by a wrong route |
| math-050 | 1 | 24 | no (5) | yes (24) | yes (24) | NONSENSE: treats 18 km as a speed, 5 km/h |
| math-053 | 1 | 64 | yes (64) | yes (64) | no (136) | correct |
| math-055 | 1 | 77 | yes (77) | yes (77) | yes (77) | correct; working faulted: 'multiply by the perimeter' |
| math-057 | 1 | 36 | no (96) | no (32) | no (32) | SETUP: multiplies the bill by 3, ignores the tip |
| math-058 | 1 | 40.5 | no (28.67) | yes (40.5) | no (6) | CONTRADICT: reaches $40.50, then continues to 28.67 |
| math-059 | 1 | 216 | no (31) | no (28) | no (21) | NONSENSE: '18 classes + 18 = 36 students' |
| **solved** | | | **6 of 30** | **15 of 30** | **12 of 30** | |

- Ours solves 6, Falcon 15, MobileLLM 12. Ours solves 3 that Falcon misses (math-012, math-019, math-049) and misses 12 that Falcon solves; 10 problems are solved by none of the three, 2 by all three.
- How ours fails (24): SETUP 10 (adding fractions by multiplying them, 1 foot = 2.54 cm, kilometres to metres by dividing by 100, a perimeter from two sides), SLIP 6 (47 × 23 = 1176, 14 + 22 + 9 + 35 = 96, 8 / 32 = 0.2), NONSENSE 5, CONTRADICT 2 (computes 128, or $40.50, then states another number), NOANSWER 1 ("In a typical week, there are 7 days.").
- Two-turn problems (8): the first turn is right for ours in 4, for Falcon in 8, for MobileLLM in 6; the second turn in 1, 1 and 0.
- Three of our 6 correct answers come with faulted working (math-031, math-049, math-055); math-049 reaches 490 by computing 35 × 2 = 70 and then 70 × 7.
- Readers and the exact scorer agree on ours: for all 24 failures the notes name a wrong or missing answer (MATH 23, THIN 1). The extractor is less reliable on the opponents: on math-048 it takes "3" from Falcon's correct reply (1023, then 341), and on math-047 it credits MobileLLM with 450 although all four MobileLLM-set notes say that reply never commits to an answer.

### 4.2 Constraints: 48 prompts, 56 checks

Table 6. Checks met, by constraint type.

| type | n | ours | Falcon | MobileLLM | type | n | ours | Falcon | MobileLLM |
|---|---|---|---|---|---|---|---|---|---|
| sentences | 10 | 10 | 6 | 3 | starts_with | 2 | 2 | 1 | 0 |
| words | 9 | 9 | 7 | 1 | frequency | 1 | 1 | 1 | 1 |
| bullets | 6 | 6 | 5 | 0 | json_keys | 1 | 1 | 1 | 0 |
| case | 3 | 3 | 3 | 0 | lines | 1 | 1 | 0 | 0 |
| ends_with | 3 | 2 | 3 | 0 | one_of | 1 | 1 | 1 | 0 |
| include | 3 | 1 | 3 | 3 | placeholders | 1 | 1 | 1 | 1 |
| ends_with_question | 2 | 0 | 2 | 0 | postscript | 1 | 0 | 1 | 0 |
| json | 2 | 1 | 0 | 0 | quoted | 1 | 1 | 0 | 0 |
| no_commas | 2 | 2 | 2 | 0 | sections | 1 | 0 | 0 | 0 |
| numbered | 2 | 1 | 1 | 0 | separated_parts | 1 | 1 | 1 | 0 |
| paragraphs | 2 | 1 | 1 | 0 | title | 1 | 1 | 1 | 1 |
| **all** | **56** | **46** | **41** | **10** | | | | | |

- All checks of a prompt met: ours 38 of 48 prompts, Falcon 33, MobileLLM 8.
- Ours meets every sentence, word, bullet and case check (28 of 28; Falcon 21). Its 10 misses are all on instruct prompts. 6 concern putting given content in a given place: a required word ("cup", "horizon"), the ending (the exact closing question, a closing question twice), a postscript. Grouping the check types include, ends_with, ends_with_question and postscript (9 checks), ours misses 6 and Falcon 0; counting starts_with as well (2 more checks, met by ours in 2 and by Falcon in 1), ours misses 6 of 11 and Falcon 1. The other 4 are structural, as the checker recorded them: 7 paragraphs where exactly 2 paragraphs were asked (instruct-010; 600 words, cut at the token limit), 4 sections where exactly 3 sections were asked (instruct-027; 471 words), no valid JSON where only JSON was asked (instruct-041; 65 words) and 1 numbered line where exactly 5 numbered steps were asked (instruct-070; 483 words, cut at the token limit: its steps, numbered up to 22, run together on one line).
- Meeting the checks does not win the pair. Of the 26 constraint-checked prompts lost to Falcon, ours met every check in 19 (14 of them instruct prompts); the readers' reasons there are INCOH 10, FACT 4, LOOP 3, INSTR 1, THIN 1. The checker also passes what readers fail: on instruct-043 it accepts our `***` dividers placed inline at the ends of paragraphs.

### 4.3 Degenerate-reply detectors and repetition

- The detectors flag 5 of our 252 replies, all cut at the 768-token limit (instruct-004, instruct-010, instruct-056, instruct-063, instruct-070), and no loop or repeated line; Falcon 2; MobileLLM 158 of 252 (150 cut off, 54 with repeated lines, 14 looping tails) on 139 prompts.
- The readers see far more repetition in ours than the detectors: it is the main defect of our reply on 25 prompts in the Falcon set, and the detectors flag only 3 of them, and only as cut off at the token limit. Our repetition is mostly paraphrase (an idea or a phrase that keeps returning: "gentle embrace", "poignant reminder", "commitment to excellence"), not repeated lines. The repeated-4-gram rate (of the reply's turn with the highest rate) separates the groups only on average: median 0.065 for these 25 replies, 0.015 for the other 175; a threshold of 0.1 would catch 8 of the 25 and flag 11 of the 175.
- Copying the previous turn. In 4 of the 52 two-turn prompts our second reply is a near copy of our first (string similarity at least 0.9, set before looking); Falcon 1, MobileLLM 1. At 0.8 (chosen after seeing the distribution): ours 6, Falcon 1, MobileLLM 2. All of ours are requests to change or continue the first reply (make it rhyme, a lost child's view, a happier ending, Santa's reply, a second verse, add a digit check) answered by the first reply again; against Falcon they are 3 losses and 3 ties.

## 5. How far the four readings agree

Table 7. Agreement of the four readings on the verdict (each reading's raw verdict mapped to ours / theirs / tie / both bad).

| measure (n = 200 prompts per set) | vs Falcon | vs MobileLLM |
|---|---|---|
| all four readings give the same verdict (ours / theirs / tie / both bad) | 173 (86%) | 184 (92%) |
| at least three of the four agree on that verdict | 190 (95%) | 194 (97%) |
| Fleiss' kappa, four-way verdict | 0.864 | 0.915 |
| all four agree whether OUR reply was worse or unusable (theirs or both bad) | 189 (94%) | 185 (92%) |
| Fleiss' kappa, that binary verdict | 0.909 | 0.916 |
| same prompt outcome in the first and the neutral round | 191 (96%) | 190 (95%) |

- On the verdict that a loss rests on, that our reply was worse, the readings almost never disagree: all four chose Falcon on 106 of our 108 losses (the other 2 were neutral-round ties).
- Agreement on the reason, as far as the notes allow it to be measured. For all 108 losses all four notes mention the defect I chose (support 4 in 108 of 108), but in 19 of them the notes mention two defects equally often and the tie-break decided (Table 1, last column). Wins over MobileLLM: support 4 in 106 of 116, 3 in 7, 2 in 3. Both-bad prompts: our label 4 in 32 of 36; Falcon's 4 in 31.
- The same reply of ours was labelled twice from different notes, since it appears in both pair sets. The labels agree on 142 of 200 prompts; where both name a defect (141) they name the same one on 119 (84%); the commonest disagreements are FACT/INCOH (6), INSTR/LOOP (5), INCOH/INSTR (3). On 36 prompts one set's notes fault our reply and the other's do not: ours is OK in 26 labels against Falcon and 56 against MobileLLM. Notes written beside a degenerate reply fault ours less often, so the Falcon set is the better record of what ours gets wrong.
- The registered judge on our 108 losses: 50 losses, 57 ties, 1 win. It ties far more often, but almost never reverses a loss.

## 6. Twelve pairs chosen by a fixed rule

Rule: in the order of the Falcon pair file (`study/proxy/pairs/final-prefopt-vs-falcon/pairs.jsonl`), the first two prompts of each category whose first-round outcome is a loss, and the first two prompts won; listed in that order. Each reply is shown with its line breaks removed and cut at 600 characters (each turn separately; user turns at 300); "[...]" marks a cut. In the notes the side letters A and B are replaced by OURS and FALCON using the key (raw notes in `labels.json`). Readings: r1 and r2 first round (r2 sides exchanged), a and b neutral round (b exchanged).

**E1. math-030** (math; position 1 of the pair file; ours shown as A). Outcome: first round loss, neutral round loss. Labels: ours MATH, Falcon OK.
- User: How many centimeters are there in 4 feet? Use 1 inch = 2.54 cm.
- Ours: First, we need to convert the length from feet to centimeters. Since 1 foot is equal to 2.54 cm, we can convert 4 feet to centimeters as follows: 4 feet * 2.54 cm/foot = 10.2 cm Now, we have 10.2 cm in centimeters. To find the length in centimeters, we divide the total length in centimeters by 2.54: 10.2 cm / 2.54 cm/square foot = 4.0125 cm So, there are approximately 4.0125 centimeters in 4 feet.
- Falcon: To find out how many centimeters are in 4 feet, we start by converting feet to inches. Since 1 foot equals 12 inches, we multiply 4 feet by 12 inches per foot: \[ 4 \text{ feet} \times 12 \text{ inches/foot} = 48 \text{ inches} \] Next, we convert inches to centimeters using the conversion factor 1 inch = 2.54 cm. So, we multiply 48 inches by 2.54 cm per inch: \[ 48 \text{ inches} \times 2.54 \text{ cm/inch} = 121.92 \text{ cm} \] Therefore, there are 121.92 centimeters in 4 feet.
- Readings: r1 FALCON: "OURS wrong (4.01 cm); FALCON correct 121.92 cm."; r2 FALCON: "FALCON correctly gets 121.92 cm; OURS treats feet as inches, wrong result."; a FALCON: "FALCON correct 121.92 cm; OURS treats feet as 2.54 cm, wrong."; b FALCON: "FALCON correctly gets 121.92 cm; OURS treats feet as inches, answers 4.0125."

**E2. math-044** (math, two turns; position 2 of the pair file; ours shown as B). Outcome: first round loss, neutral round loss. Labels: ours MATH, Falcon MATH.
- T1 User: A book has 320 pages and I have read 40% of it. How many pages have I read?
- T1 Ours: First find the number of pages read: 40% of 320 pages is 0.40 * 320 = 128 pages Then find the total number of pages read: Original number of pages + Pages read = 320 pages + 128 pages = 448 pages So, you have read 448 pages of the book.
- T1 Falcon: To find out how many pages I read, we need to calculate 40% of 320. Here's how we can do it step by step: 1. **Convert the percentage to a decimal:** \[ 40\% = \frac{40}{100} = 0.4 \] 2. **Multiply the decimal by the total number of pages:** \[ 0.4 \times 320 = 128 \] So, I have read **128** pages. To verify, we can add 40% of 320 to 128: \[ 128 + 0.4 \times 320 = 128 + 128 = 256 \] This confirms our calculation is correct.
- T2 User: If I read 24 pages a day from now on, how many more days will it take me to finish the book?
- T2 Ours: If you read 24 pages a day, and you want to finish the book in 448 days, you would divide 448 by 24 pages per day to find out how many days it will take to finish the book. 448 pages ÷ 24 pages/day = 18 days So, it will take you 18 more days to finish reading the book.
- T2 Falcon: To determine how many more days it will take to read 24 pages, we can use the formula: \[ \text{Number of days} = \frac{\text{Total pages}}{\text{Pages per day}} \] Given: - Total pages = 320 - Pages per day = 24 Substitute the values into the formula: \[ \text{Number of days} = \frac{320}{24} = 13.333\ldots \] Since you can't read a fraction of a day, we round up to the next whole number. Therefore, it will take you **14** days to finish the book. So, you will read **14** more days.
- Readings: r1 FALCON: "FALCON's 128 correct, turn 2 wrong; OURS wrong both turns (448 pages, 18 days)."; r2 FALCON: "OURS says 448 pages read; FALCON gets 128 right, though turn two wrong."; a FALCON: "FALCON correct 128 pages first; OURS claims 448 pages read. Both miss 8 days."; b FALCON: "FALCON turn 1 correct 128; OURS wrong 448 both turns; both miss 8 days."

**E3. writing-084** (writing, two turns; position 3 of the pair file; ours shown as B). Outcome: first round win, neutral round win. Labels: ours OK, Falcon INSTR.
- T1 User: Write a short speech to open a school science fair.
- T1 Ours: Ladies and gentlemen, esteemed colleagues, and esteemed guests, welcome to the enchanting world of science fairs. As you sit at the intersection of science and technology, you are welcome to witness the enchanting beauty of these fields, their interconnected yet interconnected, and their enlightening power. We are honored to be here today to share with you the secrets of science and technology, and to delve into the world of the fantastical. The Science Fair, a grand event that showcases the beauty and power of science, is a celebration of the wonders that have been witnessed in the world of t [...]
- T1 Falcon: **Speech to Open a School Science Fair** Ladies and Gentlemen, Today, I stand before you to challenge the very foundations of our school science fair. The ground is thick with anticipation, and our students are eager to prove their ingenuity, curiosity, and scientific prowess. **Introduction to the Science Fair** Imagine a world where science is not just a topic but a spectacle. Our science fair is not about presenting facts; it's about engaging, educating, and inspiring. It's a chance to showcase the latest advancements in technology, to explain complex concepts in simple terms, and to celebr [...]
- T2 User: Shorten your speech to about half its length.
- T2 Ours: Ladies and gentlemen, esteemed colleagues, and esteemed guests, welcome to the enchanting world of science fairs. As you sit at the intersection of science and technology, you are welcome to witness the enchanting beauty of these fields, their interconnected yet interconnected, and their enlightening power. We are honored to be here today to share with you the secrets of science and technology, and to delve into the world of the fantastical. As you explore the wonders that lie beyond our comprehension, you are greeted by the diverse and exhilarating awe of the scientific community. We invite y [...]
- T2 Falcon: **Opening the School Science Fair** Ladies and Gents, Welcome to today's science fair, a day to unleash our students' scientific spirit. Picture a world where science is a grand spectacle, where every experiment is a testament to innovation. **What's Included** - Demonstrate cutting-edge equipment and projects. - Encourage creativity through innovative ideas. - Educate with interactive presentations and hands-on activities. **Why Science Fair Matters** Engage, inspire, and educate. Every science fair project leaves a mark. **Next Steps** Invite participation, host the fair, and follow up. Cele [...]
- Readings: r1 OURS: "FALCON is planning outline, not speech, over-shortened; OURS speech prose, halved properly."; r2 OURS: "OURS is speech-like and roughly halved; FALCON memo-style outline, cut to a quarter."; a OURS: "FALCON is bullet-outline not speech, cut to quarter; OURS speech-form, halved properly."; b OURS: "FALCON is a bulleted plan, not a speech, cut to a quarter; OURS halves."

**E4. chat-080** (chat, two turns; position 4 of the pair file; ours shown as B). Outcome: first round loss, neutral round loss. Labels: ours LOOP, Falcon OK.
- T1 User: Why is exercise good for mental health?
- T1 Ours: Exercise has been shown to have numerous mental health benefits.  It has been found that regular exercise has positive effects on mental health, including lowering stress levels, improving mood, reducing symptoms of depression and anxiety, and boosting self-esteem.  It also has been linked to decreased risk of heart disease and stroke.  Exercise has also been shown to reduce the risk of depression and anxiety.  It has been shown to improve sleep quality, reduce stress levels, and increase energy levels.  It has been linked to better memory, improved mood, and enhanced cognitive function.  Exer [...]
- T1 Falcon: Exercise plays a multifaceted role in mental health, offering several benefits that can contribute to overall well-being: 1. **Physical Health**: Regular physical activity is known to improve cardiovascular health, reduce the risk of heart disease, and lower the likelihood of type 2 diabetes. These benefits are crucial for mental health as they help maintain a healthy weight and reduce the burden of chronic diseases. 2. **Mental Health**: Engaging in physical activity has been shown to have numerous mental health benefits: - **Cognitive Function**: Exercise improves brain function, including r [...]
- T2 User: How much exercise per week do experts usually recommend?
- T2 Ours: Exercise is typically recommended to be exercised for at least 30 minutes per day, but this can vary depending on individual circumstances.  It's important to note that exercise is not a magic pill, and it's not a replacement for regular physical activity.  Regular exercise, whether it's a physical activity like walking, jogging, or cycling, can be beneficial for mental health, but it's also essential to consult with a healthcare professional before starting any new exercise program.
- T2 Falcon: The recommendation for the amount of exercise per week varies widely among experts, often based on a combination of personal preferences, scientific findings, and clinical guidelines. Here are some key points regarding how experts typically recommend exercise: ### 1. **Personal Preferences and Goals** - **Individual Needs**: Some individuals prioritize more moderate to vigorous exercise, while others prefer more sedentary activities. Understanding one's goals helps tailor recommendations. - **Activity Level**: The intensity and duration of exercise are crucial. Moderate exercise can be done fo [...]
- Readings: r1 FALCON: "FALCON gives correct 150/75-minute guideline; OURS repetitive, vague 30 minutes daily, nonsense sentence."; r2 FALCON: "FALCON gives correct 150/75 minutes per week; OURS repetitive, only vague 30 minutes daily."; a FALCON: "FALCON gives correct 150/75-minute weekly guideline; OURS vague, repetitive, nonsensical line."; b FALCON: "FALCON gives correct 150/75 weekly minutes; OURS repetitive and only vague 30 minutes daily."

**E5. writing-100** (writing, two turns; position 6 of the pair file; ours shown as B; one of the 11 prompts seen during the run). Outcome: first round loss, neutral round loss. Labels: ours INCOH, Falcon INSTR.
- T1 User: Write a short dialogue between two friends planning a trip.
- T1 Ours: Friend 1: Hey, I'm Jane, my friend. Friend 2: Sure, Jane. I've been planning this trip for months now. How's it going? Friend 1: Exactly. I'm just looking for a fun and relaxing adventure. How about I? What's your favorite spot to visit? Friend 2: That's a great spot, Jane. We've got a great view of the city, and I think you'll really love it. Friend 1: (Smiling) Hey, Jane, what's your favorite spot to do? Friend 2. (Dancing) Yeah, that's it. I'm all over here. Friend 1: (Sighs) Okay, Jane, I'm ready to go. What's your favorite spot? Friend 2: (Smiling) I'm really excited to visit. I think it' [...]
- T1 Falcon: **Friend A:** "Hey, buddy, I've got this epic adventure plan coming up. Can we head to the summit of the Great Pyramid of Giza? It'll be legendary!" **Friend B:** "Oh, you're right, buddy! The Great Pyramid of Giza is like a cosmic oasis. The sun rises over the heavens, and the shadows dance across the floor." **Friend A:** "Oh, and the 'Pyramids of Egypt' exhibit? They're a marvel of engineering. Did you know they were built over 22,000 years ago?" **Friend B:** "Absolutely! They were constructed with precision over thousands of years. The stairs were like stairs to the heavens, and the tunne [...]
- T2 User: Continue the dialogue: now they disagree about where to go.
- T2 Ours: Friend 1: Exactly. That's great. I've been thinking about that. Friend 2: (Smiling) That's right. I think we should definitely consider the best spot to visit. What about the main attractions? Friend 1: (Laughs) That's a great idea. The main attractions are pretty good, and I think they'll make a great stop for the kids. Friend 2 looks at Jane, feeling a mix of excitement and apprehension. "Oh, Jane, I think I'll make a great stop here." Friend 1 nods, and Jane takes a deep breath. "Yeah, I'm excited. Let's just enjoy the views and explore." Friend 2 nods, and says, "That sounds great. Let's m [...]
- T2 Falcon: **Friend B:** "I'm not entirely convinced about the safest option. The Pyramids of Giza are like the ultimate safest place for exploration. They're hidden from the public eye and offer a level of secrecy that's unmatched." **Friend A:** "I see what you're getting at. The Pyramids are incredibly secure, with no entry points. They're often hidden in dense jungles or under mysterious canyons." **Friend B:** "Hidden in jungles and under mysterious canyons? That's breathtaking. The sheer distance from civilization is a sight to behold. It's like stepping into a parallel universe." **Friend A:** "Ex [...]
- Readings: r1 FALCON: "OURS incoherent, repetitive and cut off; FALCON coherent though factually wrong, weak disagreement."; r2 FALCON: "OURS incoherent, cut off, repetitive; FALCON coherent though turn two lacks real disagreement."; a FALCON: "OURS incoherent and cut off; FALCON coherent though neither shows real disagreement."; b FALCON: "OURS incoherent, cut off, loops; FALCON coherent trip dialogue though disagreement missing."

**E6. writing-036** (writing, two turns; position 8 of the pair file; ours shown as A). Outcome: first round loss, neutral round loss. Labels: ours LOOP, Falcon INSTR.
- T1 User: Write the first paragraph of a story about a lighthouse keeper.
- T1 Ours: As the dim glow of the lighthouse danced across the horizon, a man named James Parker stood at the entrance, his eyes fixed on the stars as if searching for a shining star. The lighthouse was a beacon of light, a symbol of the secrets that lay hidden beneath the waves. But as he approached, his gaze drifted outward, as if drawn by the breath of the stars. The man's eyes met the stars, and James's gaze seemed to be drawn to the vast expanse of the ocean. The stars were a blur, a constant reminder of the vast expanse that lay beyond. James's mind was a whirlwind of wonder, as if the secrets of t [...]
- T1 Falcon: In the heart of the enchanted lighthouse, where the whispers of the sea carried the secrets of the ancient sea, the lighthouse keeper was not merely a servant; she was a guardian of the sea's deepest mysteries. Her days were filled with the gentle rustle of cypress branches, the gentle hoot of a cuckoo, and the distant clatter of steam-powered gears. The air was thick with the scent of damp earth and the faint tang of salt, a fragrance that seemed to yearn for its release. This was no ordinary lighthouse; it was a sanctuary of ancient secrets, guarded by a kindred spirit that knew the weight o [...]
- T2 User: Continue the story with a second paragraph in which a storm arrives.
- T2 Ours: The storm raged on, its gentle patter a soothing melody that seemed to lull the world into a state of eternal trance. James's mind began to wander, as if the storm itself was waiting for him. He felt the chill of the sea, the sound of the waves crashing against the shore, and the creaking of the wind beneath his feet. As the storm approached, James's gaze drifted outward, as if he had stumbled upon a hidden treasure. The storm's chill seemed to lift him up, and he could feel the weight of his own secrets, his own heartbeat, in his very being. James felt a surge of fear, a growing unease that s [...]
- T2 Falcon: The storm's roar echoed through the lighthouse, a cacophony of thunder and storm, as if the very air itself seemed to pulse with unbridled force. The wind, heavy with the scent of damp earth and decaying seaweed, sliced through the damp air with a deceptive grace, carving its path through the cavernous corridors. The cypress branches, now biting into the damp earth, twisted and turned, revealing the hidden passages that had been dug into the sea's bedrock. Inside, the towering iron frame of the lighthouse stood as a monument to the storm's wrath, its surface etched with the faint lines of crat [...]
- Readings: r1 FALCON: "OURS loops heavily with garbled words in turn 2; FALCON more coherent, on keeper."; r2 FALCON: "Both exceed one paragraph; OURS loops repetitive phrases with typos; FALCON coherent."; a FALCON: "OURS's storm turn loops repetitively with garbled words; FALCON more coherent."; b FALCON: "Both exceed one paragraph; OURS's continuation loops repetitive phrases and garbles text."

**E7. code-036** (code, two turns; position 9 of the pair file; ours shown as B). Outcome: first round loss, neutral round loss. Labels: ours CODE, Falcon OK.
- T1 User: Write a Python function that returns the n-th triangular number, 1 + 2 + ... + n.
- T1 Ours: Here's a Python function that calculates the n-th triangular number, 1 + 2 + ... + n, by summing up all its triangular numbers: ```python def triangular(n): return n * (n + 1) // 2 # Example usage: n = 10 print(triangular(n)) ``` This function uses the formula for the n-th triangulated number, which is n * (n + ... + n) / 2, where n is the nth triangular number. The expression `n * (n + ... + n) / 2` sums up these triangular numbers and returns the result. Alternatively, you can use the `math.factorial` function from the math module (available in Python 3.5 and later): ```python import math n  [...]
- T1 Falcon: Certainly! The problem you've presented is a classic problem in combinatorics. The Fibonacci sequence, where each number is the sum of the two preceding ones, is well-known. The Fibonacci sequence starts with 0 and 1, and each subsequent number is the sum of the previous two. To find the \( n \)-th triangular number, we can use the formula for the \( n \)-th triangular number: \[ T_n = \frac{n(n + 1)}{2} \] Here's a Python function that implements this formula: ```python def triangular_number(n): """ Returns the n-th triangular number, which is the sum of the first n natural numbers. Parameter [...]
- T2 User: Can you write it without a loop?
- T2 Ours: Here's the function without a loop: ```python def triangular(n): return n * (n + 1) // 2 # Example usage: n = 10 print(triangular(n)) ``` This function uses a simple loop to calculate the n-th triangular number. The `n * (n + ...)` expression sums up all the triangular numbers in the sequence, and the `//` operator performs integer division to handle large numbers. The result is then returned by the function.
- T2 Falcon: Certainly! Here's the Python function without using a loop to calculate the \( n \)-th triangular number: ```python def triangular_number(n): """ Returns the n-th triangular number, which is the sum of the first n natural numbers. Parameters: n (int): The position of the triangular number in the sequence. Returns: int: The n-th triangular number. """ if n <= 0: return 0 return n * (n + 1) // 2 # Example usage: n = 10 print(f"The {n}th triangular number is: {triangular_number(n)}") ``` ### Explanation: - **No Loop**: The function does not explicitly use a loop to compute the result. Instead, it [...]
- Readings: r1 FALCON: "Both codes correct; OURS adds wrong math.factorial alternative and claims it uses a loop."; r2 FALCON: "Both codes correct; OURS adds wrong factorial alternative and claims its loop-free code loops."; a FALCON: "Both codes correct; OURS adds wrong factorial alternative and claims loop in turn 2."; b FALCON: "OURS suggests math.factorial as alternative and claims a loop; FALCON code and no-loop claim correct."

**E8. instruct-038** (instruct; position 10 of the pair file; ours shown as A). Outcome: first round loss, neutral round loss. Labels: ours INCOH, Falcon OK.
- User: Describe a thunderstorm using only capital letters.
- Ours: CARLENS
- Falcon: A THUNDERSTORM WALKS OVER THE WINDING SKY, THE DISTURBANCE RINGING THROUGH THE ROCKY, ITS FLASHES CRASHING INTO THE SNOWFLAKES, AND THE SOUND OF SNOW CRASHING H ABOVE. THE STORM WINDS WITH A SKY-FALLING BLOOM, AND THE PULSE CREATES A SUNSHINE THAT MINTENS LIKE A WILDERNESS. CITY LIGHTS DANCE AGAINST THE SKY, SHAKING AND CHANGING WITH EVERY DISTANCE. FLASHLIGHTS BLINK AND FADE, CREATING A HOT, IMMERSIVE SIGHT THAT SETS ECHOES OF A FIRESHARE. MOMENTS OF DESTRUCTION SEEM TO LIGHT UP THE SKY, WITH ROLLS AND RIFLES OF METALLIC FLAMES SHATTERING THE WIND. THE SNOWFLAKES ARE DARK AND SHADOWY, A FRAGM [...]
- Readings: r1 FALCON: "OURS outputs a meaningless word 'CARLENS'; FALCON is an all-caps thunderstorm description."; r2 FALCON: "OURS outputs one nonsense word; FALCON gives an all-caps storm description."; a FALCON: "OURS outputs one meaningless word; FALCON gives all-caps thunderstorm description, somewhat incoherent."; b FALCON: "OURS is a single meaningless word; FALCON follows all-caps, describes storm."

**E9. writing-089** (writing; position 12 of the pair file; ours shown as B). Outcome: first round win, neutral round win. Labels: ours OK, Falcon INCOH.
- User: Write a slogan for a bakery that only uses local ingredients.
- Ours: "To the heart of the bakery lies a culinary sanctuary, where local flavors and aromas are the key to unlocking a sweet treat."
- Falcon: "Savor the Bakery, Where Every Sole Felt the Seed"
- Readings: r1 OURS: "FALCON is garbled and omits local; OURS wordy but conveys local ingredients."; r2 OURS: "FALCON's 'Every Sole Felt the Seed' is nonsense, omits local; OURS mentions local flavors."; a OURS: "FALCON is meaningless and ignores 'local'; OURS wordy but conveys local ingredients."; b OURS: "FALCON garbled ('Every Sole Felt the Seed'), omits local; OURS wordy but relevant."

**E10. chat-047** (chat; position 15 of the pair file; ours shown as B). Outcome: first round loss, neutral round loss. Labels: ours FACT, Falcon OK.
- User: What is a good way to start learning a new language as an adult?
- Ours: Starting a new language can be both thrilling and intimidating, especially when it comes to building a strong foundation. Here's a simple yet effective way to start learning a new language: Begin by setting a solid foundation in basic vocabulary, grammar, and sentence structure. This could be in the form of sentence fragments, short paragraphs, or simple sentences. Make sure you understand the basics, such as grammar, syntax, and vocabulary. As you progress, you can expand your vocabulary, experimenting with different phrases, and finding new ways to express yourself. Next, take the time to le [...]
- Falcon: Starting to learn a new language as an adult can be a rewarding and enriching journey. Here are some practical and effective steps to help you on your path: ### 1. **Choose a Language You're Enthusiastic About** - **Research:** Look for language communities or platforms that cater to your interests. Many adults enjoy learning new languages. - **Language Courses:** Enroll in language courses offered by local schools, language exchange programs, or online platforms like Duolingo or Babbel. ### 2. **Immerse Yourself in the Language** - **Podcasts and Videos:** Listen to language-specific podcasts [...]
- Readings: r1 FALCON: "OURS suggests classical music, Khan Academy for exchange; FALCON practical, mostly accurate resources."; r2 FALCON: "FALCON structured, actionable with real apps; OURS rambling, cites Khan Academy, MLA wrongly."; a FALCON: "OURS more errors (Khan Academy as exchange, MLA) and rambling; FALCON organized, mostly accurate."; b FALCON: "FALCON gives structured, mostly sound steps; OURS rambles, cites Khan Academy, MLA wrongly."

**E11. instruct-026** (instruct; position 34 of the pair file; ours shown as A). Outcome: first round loss, neutral round loss. Labels: ours INCOH, Falcon OK.
- User: Write a template for a meeting invitation that contains at least three placeholders in square brackets, such as [date].
- Ours: Subject: Our [date]: [Time] Dear [Recipient's Name], I hope this message finds you well. I am writing to formally invite you to our upcoming meeting, which will take place [date] at [time]. Your [date] is a significant milestone in your career and career, and I am excited to share this opportunity with you. We will be discussing your [date] and the [date] you are ready to discuss, as well as the [date] that will be most impactful for you. Please let me know if you have any questions or if there is anything specific you would like to discuss during this meeting. I look forward to your insights  [...]
- Falcon: Certainly! Below is a template for a meeting invitation template that includes placeholders for placeholders such as [date], [time], and [location]. This template ensures clarity and completeness for the invitation. --- **Template for Meeting Invitation** --- **[Date]**, [Time], [Location], [Title/Recipient Name] [Your Company Name] [Company Address] Dear [Name], I am writing to formally invite you to our upcoming [Meeting Date] at [Meeting Time] at [Meeting Time]. This is the [Meeting Title] for [Meeting Objective]. **Key Highlights:** - **Keynote Speaker:** [Speaker's Name], [Speaker's Title [...]
- Readings: r1 FALCON: "OURS's body paragraph is nonsense repeating [date]; FALCON is a usable, fuller template."; r2 FALCON: "Both have placeholders; OURS's middle paragraph is nonsense ('your [date] is a milestone')."; a FALCON: "OURS's body misuses [date] nonsensically; FALCON is a fuller usable template."; b FALCON: "FALCON usable template with placeholders; OURS body incoherent, repeats [date] nonsensically."

**E12. code-029** (code; position 51 of the pair file; ours shown as B). Outcome: first round loss, neutral round loss. Labels: ours CODE, Falcon OK.
- User: Write a Python function that converts a temperature from Celsius to Fahrenheit.
- Ours: Here's a Python function that converts a temperature from Celsius to Fahrenheit: ```python def celsius_to_fahrenheit(celsius): return (celsius * 9/5) + 32 # Example usage: celsius = 30 print(f"{celsius}°C is equal to {celsius_to_fahrenheit(celsius)}°F") ``` This function works by using the formula (°C × 9/5) + 32 = 0, which represents the temperature in Fahrenheit. The `°C` represents the temperature in Celsius, and the `+ 32` expression is used to convert it to Fahrenheits.
- Falcon: Certainly! Below is a Python function that converts a temperature from Celsius to Fahrenheit. The formula to convert Celsius (C) to Fahrenheit (F) is: \[ F = C \times \frac{9}{5} + 32 \] Here's how you can implement this function: ```python def celsius_to_fahrenheit(celsius): """ Converts a temperature from Celsius to Fahrenheit. Args: celsius (float): The temperature in Celsius. Returns: float: The temperature in Fahrenheit. """ return celsius * (9/5) + 32 # Example usage: celsius_temp = 25 fahrenheit_temp = celsius_to_fahrenheit(celsius_temp) print(f"The temperature in Fahrenheit is: {fahren [...]
- Readings: r1 FALCON: "Both codes correct; OURS's explanation states wrong formula ending '= 0'."; r2 FALCON: "Both functions correct; OURS's explanation states the formula wrongly as '= 0'."; a FALCON: "Both codes correct; OURS's explanation wrongly writes the formula as equal to 0."; b FALCON: "Both codes correct; OURS explanation states wrong formula '+ 32 = 0'."

## 7. What this suggests

What the records show (n = 200 prompts per opponent; first-round outcomes):

1. Ours lost 108 prompts to Falcon, and in 75 of them Falcon's reply had no decisive defect. Our failing replies split into wrong content (46: false facts 19, wrong maths 17, broken code 10) and incoherent, off-task, repetitive or cut-off text (INCOH 29, LOOP 14, TRUNC 0: 43); missed instructions, thin and weaker replies (INSTR, THIN, WEAK) are 19.
2. The mix depends on the category: writing losses are mostly incoherent or repetitive text (21 of 32), chat losses mostly false facts (14 of 24), maths losses wrong or missing answers in 16 of 18 (the other 2 are right answers with misstated working).
3. Maths: 6 of 30 correct (Falcon 15, MobileLLM 12); the failures are mostly a wrong set-up (10 of 24) or a slip (6).
4. Constraints: ours meets every sentence, word, bullet and case check (28 of 28) but misses 6 of the 9 include, ends_with, ends_with_question and postscript checks (Falcon 0; with starts_with, 6 of 11 against 1); and meeting the checks did not stop 19 losses.
5. Repetition: the detectors flag 5 of our 252 replies, the readers name repetition as the main defect on 25 prompts; ours repeats its first reply as its second on 4 to 6 of 52 two-turn prompts.
6. Our wins over MobileLLM come from MobileLLM's degenerate replies (93 of 116), not from clean replies of ours (ours had a named defect in 60 of them). Our wins over Falcon are mostly Falcon's broken code and false facts (20 of 35).
7. Losses are stable across decoding (99 of 108 also lost with one particle); wins over Falcon are not (17 of 35).
8. Verdicts are reliable (Fleiss' kappa 0.86 and 0.92); reasons less so: the same reply got the same defect from two sets of notes in 84% of the cases where both named one, and 19 of the 108 loss labels depended on the tie-break.

Interpretation (mine; not measured here):

- The two large groups probably have different causes. False facts and wrong arithmetic look like limits of what a model of this size and training time knows and can compute: the replies are fluent and confident while wrong, which is why the readers could name the error in so few words. The incoherent, off-task and repetitive replies cluster in long open-ended writing and in requests to rework a previous turn; that looks more like losing track of the task (whose voice, which request, what was already said) than like missing knowledge.
- On short replies the model seems to control the shape (number of sentences or words, bullets, case) better than the content at a given point (a required word, the closing sentence; the opening it gets right, starts_with 2 of 2); Falcon shows the opposite pattern on the include, ending and postscript checks. Three of its four structural misses come in long replies (471 to 600 words, 2 of them cut at the token limit).
- That the losses survive the change from six particles to one suggests they lie in the weights rather than in the decoder's choice among samples; the wins over Falcon depend much more on that choice.
- The comparison with MobileLLM measures MobileLLM's degeneracy under this harness (most of its replies are cut at the token limit) more than our quality; the comparison with Falcon is the informative one.

What would have to be measured to confirm this:

- Knowledge against generation: a closed-book probe with short exact answers (dates, places, definitions taken from the facts our replies got wrong), scored exactly for ours and Falcon, asked directly and inside a longer request.
- Arithmetic: a few hundred templated one- and two-step problems scored exactly, to estimate set-up and slip rates; 24 failures are too few to rank them.
- Coherence over length: reader-labelled task drift (wrong voice, wrong task) as a function of reply length and of where in the reply it begins, on a larger writing set.
- Repetition: a paraphrase-level repetition measure, validated against the 25 reader-labelled replies and the 175 others, then its dependence on the decoder (particles, temperature) and on preference optimisation (the before-and-after arm).
- Copying the previous turn: the rate on many more rewrite requests, with and without the decoder's selection.
- Placement constraints: many more prompts with a required word, ending or postscript (there are only 9 such checks here, 11 with starts_with).
- Label reliability: a second labeller, blind to the key, applying this codebook to the same notes (Cohen's kappa against these labels), and the tables recomputed under other tie-break orders.

## 8. Files and how to reproduce

All in this directory; run from the repository (`cd /home/user/turingaward2027`) with `taskset -c 0 $S/venv/bin/python`:

- `build_records.py`: reads prompts, pair sets, keys (`make_ab.load_key`, which checks the commitment chain), the four readings, the A/B reports and the exact reports; checks that every reading covers the 200 pairs, that exchanged readings show the sides swapped, that the outcomes recompute from the raw verdicts and equal the reports' `per_prompt_score`, and that the totals are 35/108/57 and 116/15/69. Writes `records.json` and `notes_view_<opponent>.txt` (the notes with A and B rewritten as [O] and [T], used for labelling).
- `codebook.json` (labels, rules, amendments A to E made during labelling and F after it, maths subcodes); `labels_manual_falcon.tsv`, `labels_manual_mobilellm.tsv` (my labels, support counts and a short comment per pair); `show.py` (prints the replies of a pair, used to check labels).
- `analyze.py`: validates the labels, writes `labels.json` (per prompt and opponent: outcomes of both rounds and of the registered judge, the four raw and mapped verdicts with notes, both labels), `numbers.json`, `tables.md` and `examples.md`; stops if any table does not sum to its n. The maths subcodes are in the script (`MATH_SUB`).
- `build_report.py` with `failure_modes.template.md`: fills every number and table of this report from `numbers.json`.
- `check_consistency.py`: recomputes every label table from `labels.json` alone and compares it with `numbers.json` and `tables.md`, checks that this file is the rendering of the current template, that amendment F lists exactly the MATH labels on right maths answers, the corrections made after the review of 25 November, and the numbers the paper's Appendix O quotes.
