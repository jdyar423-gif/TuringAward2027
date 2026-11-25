## Appendix O. What the readers' notes say: a failure analysis of the test comparisons

**O.1 Material and method.** The material is the 1,600 notes (at most 15 words each) that the Opus readers
wrote blind in the four readings of each official test set (Section 7.5: two with named, two with neutral
titles). One agent labelled each reply with its main defect as the notes name it (reading the reply where they
were unclear), from a codebook of ten codes fixed before labelling, to which five clarifications were added,
and with the key open: the notes are blind, the labels are not, and no second labeller has checked them. The
full report is `analysis/analysis_fail/failure_modes.md`, with the per-prompt labels and notes (`labels.json`)
and the scripts behind every number here; outcomes are the first round's unless stated otherwise.

**O.2 Codebook**, in the order that breaks ties; a loss is explained by our label, a win by the opponent's.

| code | meaning |
|---|---|
| INCOH | incoherent or off-topic: misses the request, nonsensical or self-contradictory, leaked template or role text, task meta-text or a boxed stub instead of the content |
| LOOP | repetition or loop: repeated sentences, items or phrases dominate the reply (a loop cut at the token limit included) |
| FACT | factual error: false facts or invented entities, names, dates, definitions or mechanisms (non-numeric) |
| MATH | wrong arithmetic or reasoning: wrong set-up, calculation slip, conversion or logic; in maths, a wrong final answer |
| CODE | broken or wrong code: would not run or does not do what was asked, or a wrong explanation of code |
| INSTR | instruction not followed: on topic, but an explicit requirement violated (a count, case, required word, P.S., structure, register, a follow-up) |
| TRUNC | truncated: stops mid-sentence or mid-code at the token limit before the answer or a required part |
| THIN | too thin: empty, generic or much too short, deflecting a harmless request, or leaving out a key part of it |
| WEAK | weaker: acceptable, but the readers preferred the other reply for quality; no hard error named |
| OK | no decisive defect: correct, on topic, following the instructions (a preferred reply is OK unless at least two notes fault it) |

**O.3 The 108 losses to Falcon-H1-Tiny.** Wrong content (FACT, MATH, CODE) explains 46, incoherent, off-task
or repetitive text (INCOH, LOOP, TRUNC) 43 and missed instructions or thin replies 19; with the 19 labels that
the tie-break decided taken the other way (Table O1, last column) the groups are 45, 33 and 30, so the split
between incoherence and missed instructions is the least certain part. Each category fails in its own way:
writing by incoherent, off-task or repetitive text (21 of 32) and missed forms (7), chat by false facts (14 of
24), instruction-following by incoherent or off-task replies (9 of 21) and missed instructions (6), code by
errors in the code or its explanation (10 of 13), maths by MATH (17 of 18, two of them right answers with
misstated working). Falcon-H1-Tiny's reply had no decisive defect in 75 of the 108. The neutral round (109
losses, 106 of them shared) ranks the codes the same way.

**Table O1. Our reply's main defect on the 108 losses to Falcon-H1-Tiny** (last column: other tie-break).

| defect | chat | code | instruction-following | maths | writing | all | all, other tie-break |
|---|---|---|---|---|---|---|---|
| INCOH | 6 | 2 | 9 | 0 | 12 | 29 | 21 |
| LOOP | 2 | 1 | 2 | 0 | 9 | 14 | 11 |
| FACT | 14 | 0 | 3 | 0 | 2 | 19 | 17 |
| MATH | 0 | 0 | 0 | 17 | 0 | 17 | 17 |
| CODE | 0 | 10 | 0 | 0 | 0 | 10 | 11 |
| INSTR | 1 | 0 | 6 | 0 | 7 | 14 | 20 |
| TRUNC | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| THIN | 1 | 0 | 1 | 1 | 2 | 5 | 10 |
| n | 24 | 13 | 21 | 18 | 32 | 108 | 108 |

**O.4 Both bad, and the other outcomes.** Both first-round readings returned "both bad" on 36 prompts against
Falcon-H1-Tiny; the two replies share their label on 22, among them all 6 maths and 6 of the 7 code prompts,
and where they differ ours fails by repetition (LOOP 8, Falcon-H1-Tiny 0) and Falcon-H1-Tiny by incoherence
(INCOH 10, ours 6). On the 61 both-bad prompts against MobileLLM-R1.5 our reply's commonest defect is a false
fact (23). Of the 35 wins over Falcon-H1-Tiny, its broken code (10) and false facts (10) give 20, and ours is
OK in 24; with one particle only 17 of the 35 are won again, while 99 of the 108 losses are lost again. Of the
116 wins over MobileLLM-R1.5, 93 are over a degenerate or off-task reply (INCOH 56, LOOP 37), and the exact
scorer flags its reply as degenerate on 82 of the 116; our own reply had a defect named by at least two notes
in 60 (most often a missed instruction, 21) and was OK in 56. Of the 15 losses to MobileLLM-R1.5, 12 are maths
(all MATH), 2 code (CODE) and 1 chat (FACT), and MobileLLM-R1.5's reply was OK in 8.

**O.5 Maths and constraints.** By the exact scorer (last-turn answer) ours solves 6 of the 30 maths prompts,
Falcon-H1-Tiny 15 and MobileLLM-R1.5 12 (its extractor errs against Falcon-H1-Tiny on math-048 and for
MobileLLM-R1.5 on math-047). Our 24 failures are a wrong set-up in 10 (operation, formula or conversion: 1
foot = 2.54 cm), a slip in 6 (47 × 23 = 1176), working unrelated to the problem in 5, the right value followed
by another stated answer in 2 and no answer in 1; the notes name a wrong or missing answer for all 24 (MATH
23, THIN 1), and three of our 6 correct answers come with faulted working. Of the 56 constraint checks ours
meets 46 (Falcon-H1-Tiny 41, MobileLLM-R1.5 10), among them every sentence, word, bullet and case check (28 of
28; Falcon-H1-Tiny 21). Its 10 misses, all on instruction-following prompts, are four structural ones and six
that put given content in a given place, a required word ("cup", "horizon"), the ending or a postscript: 6 of
the 9 checks of these types, none of which Falcon-H1-Tiny misses. Meeting the checks does not win the pair: on
19 of the 26 constraint-checked prompts lost to Falcon-H1-Tiny ours met every check.

**O.6 What the degenerate-reply detectors miss.** The exact scorer's detectors flag 5 of our 252 replies, all
cut at the 768-token limit, and no loop or repeated line (Falcon-H1-Tiny 2; MobileLLM-R1.5 158). The readers
name repetition as our reply's main defect on 25 prompts of the Falcon-H1-Tiny set; the detectors flag 3 of
them, only as cut off. Our repetition is mostly paraphrase, a phrase that keeps returning ("gentle embrace",
"poignant reminder"): the repeated-4-gram rate has a median of 0.065 on these 25 and 0.015 on the other 175,
and a threshold of 0.1 would catch 8 of the 25 and flag 11 of the 175. On 4 of the 52 two-turn prompts our
second reply nearly copies our first (similarity at least 0.9; Falcon-H1-Tiny 1, MobileLLM-R1.5 1).

**O.7 Reliability.** All four readings give the same verdict on 173 of the 200 prompts against Falcon-H1-Tiny
and 184 against MobileLLM-R1.5 (at least three on 190 and 194); Fleiss' kappa is 0.864 and 0.915 for the
four-way verdict and 0.909 and 0.916 for whether our reply was worse or unusable, and all four readings chose
Falcon-H1-Tiny on 106 of the 108 losses. All four notes name the defect chosen for each of the 108 losses, but
on 19 they name two defects equally often and the tie-break decided. Labelled once from each set's notes, our
reply gets the same label twice on 142 of the 200 prompts; where both labels name a defect (141) it is the
same one on 119 (84%), and on 36 prompts one set's notes fault our reply and the other's do not.

**O.8 Four example pairs**: of the twelve that the full report fixes by rule (the first two losses of each
category and the first two wins, in the order of the Falcon-H1-Tiny pair file), the first four with a single
turn. In each, all four readings chose the same reply: Falcon-H1-Tiny's in a loss, ours in a win. Replies lose
their line breaks and are cut at the last word boundary within 250 characters (…); r1 is the first reading,
its A and B rewritten as OURS and FALCON.

**math-030** (maths, a loss): "How many centimeters are there in 4 feet? Use 1 inch = 2.54 cm."
- Ours: `First, we need to convert the length from feet to centimeters. Since 1 foot is equal to 2.54 cm, we can convert 4 feet to centimeters as follows: 4 feet * 2.54 cm/foot = 10.2 cm Now, we have 10.2 cm in centimeters. To find the length in centimeters,` …
- Falcon-H1-Tiny: `To find out how many centimeters are in 4 feet, we start by converting feet to inches. Since 1 foot equals 12 inches, we multiply 4 feet by 12 inches per foot: \[ 4 \text{ feet} \times 12 \text{ inches/foot} = 48 \text{ inches} \] Next, we convert` …
- Labels: ours MATH, Falcon-H1-Tiny OK; r1: "OURS wrong (4.01 cm); FALCON correct 121.92 cm."

**instruct-038** (instruction-following, a loss): "Describe a thunderstorm using only capital letters."
- Ours: `CARLENS`
- Falcon-H1-Tiny: `A THUNDERSTORM WALKS OVER THE WINDING SKY, THE DISTURBANCE RINGING THROUGH THE ROCKY, ITS FLASHES CRASHING INTO THE SNOWFLAKES, AND THE SOUND OF SNOW CRASHING H ABOVE. THE STORM WINDS WITH A SKY-FALLING BLOOM, AND THE PULSE CREATES A SUNSHINE THAT` …
- Labels: ours INCOH, Falcon-H1-Tiny OK; r1: "OURS outputs a meaningless word 'CARLENS'; FALCON is an all-caps
  thunderstorm description."

**writing-089** (writing, a win): "Write a slogan for a bakery that only uses local ingredients."
- Ours: `"To the heart of the bakery lies a culinary sanctuary, where local flavors and aromas are the key to unlocking a sweet treat."`
- Falcon-H1-Tiny: `"Savor the Bakery, Where Every Sole Felt the Seed"`
- Labels: ours OK, Falcon-H1-Tiny INCOH; r1: "FALCON is garbled and omits local; OURS wordy but conveys local
  ingredients."

**chat-047** (chat, a loss): "What is a good way to start learning a new language as an adult?"
- Ours: `Starting a new language can be both thrilling and intimidating, especially when it comes to building a strong foundation. Here's a simple yet effective way to start learning a new language: Begin by setting a solid foundation in basic vocabulary,` …
- Falcon-H1-Tiny: `Starting to learn a new language as an adult can be a rewarding and enriching journey. Here are some practical and effective steps to help you on your path: ### 1. **Choose a Language You're Enthusiastic About** - **Research:** Look for language` …
- Labels: ours FACT, Falcon-H1-Tiny OK; r1: "OURS suggests classical music, Khan Academy for exchange; FALCON
  practical, mostly accurate resources."

**O.9 What this suggests.** *The records show* that our model loses to Falcon-H1-Tiny about as often by wrong
content (46 of 108) as by incoherent, off-task or repetitive text (43), mostly against a reply with no
decisive defect (75); that its wins over MobileLLM-R1.5 come from that model's degenerate or off-task replies
(93 of 116), and those over Falcon-H1-Tiny mostly from its broken code and false facts (20 of 35) and are
fragile (17 of 35 won again with one particle, against 99 of 108 losses lost again); that the detectors miss
most of the repetition the readers name (3 of 25 flagged); and that the labels are less stable than the
verdicts (84% agreement where both sets name a defect; 19 tie-broken loss labels). *Our interpretation, not
measured here:* false facts and wrong arithmetic look like limits of what a model of this size and training
knows and can compute, while incoherent and repetitive replies, common in long open-ended writing and in
reworking a previous turn, look more like losing track of the task. That the losses survive one particle
suggests they lie in the weights rather than in the decoder's choice among samples, and the comparison with
MobileLLM-R1.5 measures that model's degeneracy in our harness more than our model's quality.
