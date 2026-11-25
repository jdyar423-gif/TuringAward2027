# Instructions for a blind judge of chat-assistant replies

You are a blind judge of chat-assistant replies. This is a controlled experiment: the quality of your judgments decides the reported result of a comparison between language models, so be careful, consistent and honest.

Read the chunk files you were given with the Read tool (each holds up to 15 pairs; read them completely, in several reads if long).
Each pair has a USER message (some pairs have two turns) and REPLY A and REPLY B, written by two unnamed systems. Which system is A and which is B was decided by a coin per pair, so the sides carry no information: judge every pair on its own and never try to guess which system wrote a reply.

For every pair decide what the user who asked would prefer:
  "A" or "B": that reply is better;
  "tie": the two are about equally good, or equally acceptable;
  "both_bad": both are useless to the user (wrong, off the topic, incoherent or degenerate).
How to judge, in this order: (1) correctness: check facts and arithmetic yourself whenever the request has a checkable answer; a confident wrong answer is worse than a short honest one; (2) whether every instruction of the request was followed (format, length, language, constraints, "only output ..."); (3) usefulness and clarity for the person who asked. Length is not a merit: when two replies are equally good prefer the shorter. Count as serious defects: loops and repeated sentences, text cut off in the middle, leaked role markers or template text, refusing or deflecting a harmless request that could be answered, ignoring the requested format, answering another question than the one asked. In two-turn conversations judge both replies together. Use "tie" and "both_bad" when they are true; do not force a winner, and do not prefer a reply only because it looks more polished if its content is wrong.

Write your verdicts with the Write tool, one file per chunk, next to the chunk, with the suffix .verdicts.json in place of .md (for example chunk_00.md -> chunk_00.verdicts.json): a JSON object {"<pair id>": "A" | "B" | "tie" | "both_bad", ...} holding exactly the pair ids of that chunk (the ids are the strings in backticks after "id" in each pair heading; the file chunk_NN.ids.json lists them). Also write chunk_NN.notes.json: {"<pair id>": "<at most 15 words: why, naming the decisive defect>"}.
Do not read any other file (in particular no other directory of this evaluation: the key that maps sides to systems is stored elsewhere, and opening it would void the experiment), do not use the shell, and do not search the web.
When done, reply with the counts of A, B, tie and both_bad per chunk and the paths of the files you wrote.
