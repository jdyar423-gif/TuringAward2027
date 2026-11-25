#!/usr/bin/env bash
# Fetch every shard of the two MobileLLM control sets from the three runtimes, merge them into the
# rating files and print the coverage.  usage: judge_cover.sh
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
A=final-prefopt-vs-mobilellm-k6lp; B=final-prefopt-vs-mobilellm-k6ds
for SES in bes-cpu bes-cpu2 bes-cpu3; do bash $S/scripts/judge_shard.sh fetch $SES testq5 $A >/dev/null 2>&1; bash $S/scripts/judge_shard.sh fetch $SES testq6 $B >/dev/null 2>&1; done
bash $S/scripts/judge_shard.sh merge testq5 $A 2>&1 | tail -n 1 | cut -c1-200
bash $S/scripts/judge_shard.sh merge testq6 $B 2>&1 | tail -n 1 | cut -c1-200
