#!/usr/bin/env bash
# Copy the paper, its figures, data, analysis scripts and the cited evidence files from the
# scratchpads into docs/paper/ of the repository (re-runnable; does not commit).
# usage: paper_sync.sh [--with-test]   (--with-test: also the evaluation records of the test split,
#        only once every blind reading of the test pairs is finished)
set -u
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
R=/home/user/turingaward2027; D=$R/docs/paper; P=$N/paper; E=$D/evidence
with_test=0; [ "${1:-}" = "--with-test" ] && with_test=1
mkdir -p "$D/figures" "$D/data" "$D/analysis" "$E/N" "$E/S" "$E/W"
for f in paper.md refs.md related_work.md CLAIMS.md OUTLINE.md tables.md FIGURES.md FACTS_2.md; do [ -e "$P/$f" ] && cp "$P/$f" "$D/$f"; done
for f in "$P"/REVIEW_*.md "$P"/RESPONSE_*.md "$P"/paper.zh-TW.md; do [ -e "$f" ] && cp "$f" "$D/"; done
cp "$P"/figures/*.svg "$P"/figures/*.png "$P"/figures/*.py "$P"/figures/*.sh "$D/figures/" 2>/dev/null
cp "$P"/data/*.csv "$P"/data/README.md "$D/data/" 2>/dev/null
[ -d "$P/data/_src" ] && tar -czf "$D/data/_src.tar.gz" -C "$P/data" _src
for d in lead2_work review1_work lead3_work lead4_work lead5_work review3_work; do
  [ -d "$P/$d" ] || continue; mkdir -p "$D/analysis/$d"
  find "$P/$d" -maxdepth 1 -type f \( -name "*.py" -o -name "*.json" -o -name "*.txt" -o -name "*.md" -o -name "*.sh" -o -name "*.log" \) -size -600k -exec cp {} "$D/analysis/$d/" \;
  if [ -d "$P/$d/scripts" ]; then mkdir -p "$D/analysis/$d/scripts"; find "$P/$d/scripts" -maxdepth 1 -type f \( -name "*.py" -o -name "*.sh" \) -size -600k -exec cp {} "$D/analysis/$d/scripts/" \;; fi
done
# the translation's rule file and checker; the operator's scripts of the evaluation after the run
mkdir -p "$D/zh" && cp "$P/zh/GLOSSARY.md" "$D/zh/" 2>/dev/null; cp "$S/scripts/zh_check.py" "$D/zh/" 2>/dev/null
mkdir -p "$E/S/scripts"
for f in blind_eval.py judge_driver.py judge_remote.sh judge_shard.py judge_shard.sh judge_shard_merge.py controls_driver.py controls_slice_driver.py ctl_local.py ctl_assemble.py k1_driver.py endgame_driver.py endgame_testgen.sh final_pairs.sh final_score.sh score2.sh neutral_score.sh study_score.sh cmp_exports.py pack_dir.py fetch_pack.sh paper_sync.sh; do [ -e "$S/scripts/$f" ] && cp "$S/scripts/$f" "$E/S/scripts/"; done
cp "$N/audit_19_10.md" "$N/final_eval/JUDGE_INSTRUCTIONS.md" "$E/" 2>/dev/null
# evidence: the scratch files the claims ledger cites, under the ledger's own prefixes
cp "$N/OPERATIONS.md" "$N/midrun_eval_1.json" "$N/heldout_v9.jsonl" "$N/openings.json" "$N/deflection_check2.json" "$E/N/" 2>/dev/null
cp "$N"/midrun_sample_step*.md "$E/N/" 2>/dev/null
mkdir -p "$E/N/graft_rehearsal" && cp "$N/graft_rehearsal/graft.log" "$E/N/graft_rehearsal/" 2>/dev/null
cp "$N"/wf/*.md "$N"/wf/chip_kernels_0745.txt "$E/W/" 2>/dev/null
for w in "$N"/wf/v9_*_work; do [ -e "$w/estimate.py" ] && mkdir -p "$E/W/$(basename "$w")" && cp "$w/estimate.py" "$E/W/$(basename "$w")/"; done
cp "$S"/v9_meas*.txt "$S/graft_out.txt" "$S/deflection_out.txt" "$S/tpu_intervals.txt" "$S/runlog_peek.txt" "$S/openings_out.txt" "$S/graft_test.log" "$S/graft_dry_out.txt" "$E/S/" 2>/dev/null
# the blind studies: published pairs, keys, ratings, reports, the judges' chunk files and verdicts
X=$E/N/study; mkdir -p "$X/proxy"
for part in pairs keys ratings reports; do
  mkdir -p "$X/proxy/$part"
  for item in "$N/study/proxy/$part"/*; do
    b=$(basename "$item")
    case "$b" in final-*|ref-*) [ "$with_test" = 1 ] || continue ;; esac
    cp -r "$item" "$X/proxy/$part/"
  done
done
if [ -d "$N/study/proxy/reports/ab" ]; then
  [ "$with_test" = 1 ] || rm -f "$X/proxy/reports/ab"/final-* "$X/proxy/reports/ab"/ref-*
fi
for st in s1 s2; do
  for j in judge judge2; do [ -d "$N/study/$st/$j" ] && mkdir -p "$X/$st" && cp -r "$N/study/$st/$j" "$X/$st/"; done
  mkdir -p "$X/$st/raw"; for v in "$N/study/$st/raw"/*/; do b=$(basename "$v"); mkdir -p "$X/$st/raw/$b"; cp "$v/replies.jsonl" "$v/testgen.json" "$X/$st/raw/$b/" 2>/dev/null; done
done
if [ "$with_test" = 1 ]; then
  mkdir -p "$X/proxy/outputs" && cp -r "$N/study/proxy/outputs"/* "$X/proxy/outputs/"
  [ -d "$N/final_eval" ] && cp -r "$N/final_eval" "$E/N/"
  # the failure analysis (labels made with the key open), the judge's shard records, machine facts
  mkdir -p "$D/analysis/analysis_fail" "$E/N/judge_remote" "$E/N/peek"
  for f in failure_modes.md failure_modes.template.md appendix_O.md section7_paragraph.md codebook.json labels.json numbers.json tables.md examples.md labels_manual_falcon.tsv labels_manual_mobilellm.tsv; do [ -e "$P/analysis_fail/$f" ] && cp "$P/analysis_fail/$f" "$D/analysis/analysis_fail/"; done
  cp "$P"/analysis_fail/*.py "$D/analysis/analysis_fail/" 2>/dev/null
  for t in "$N"/judge_remote/*/; do b=$(basename "$t"); mkdir -p "$E/N/judge_remote/$b"; cp "$t"/*.log "$t"/ranges.txt "$E/N/judge_remote/$b/" 2>/dev/null; [ -d "$t/shards" ] && cp -r "$t/shards" "$E/N/judge_remote/$b/"; done
  cp "$N"/peek/lscpu_*.txt "$E/N/peek/" 2>/dev/null; cp "$S/neutral_map.txt" "$S/cpu4_up.log" "$E/S/" 2>/dev/null
fi
cat > "$E/README.md" <<'TXT'
# Evidence files cited by `../CLAIMS.md`

The claims ledger names its sources with three prefixes that pointed into the scratch directories
of the session that ran the experiment. Those directories do not outlive the session; the cited
files are copied here under the same prefixes:

- `N/...`  the session scratchpad (operations log, held-out evaluations, the blind studies:
  `N/study/proxy/{pairs,keys,ratings,reports}` are the published blinded pair sets, their keys,
  the rating records and the score reports; `N/study/s1|s2/judge*` are the chunk files the readers
  saw with their verdict and note files);
- `S/...`  the first scratchpad (chip measurement outputs, the TPU time ledger);
- `W/...`  `N/wf/`: the reports of the sub-agent workflows (speed studies, reviews, rehearsals).

Large inputs (snapshots, checkpoints, compiler dumps, the tokenised data) are not here; where a
claim rests on one of them the ledger says so. The evaluation records of the test split are added
only after every blind reading of the test pairs is finished.
TXT
du -sh "$D" "$E" | tr '\n' ' '; echo; find "$D" -type f -size +2M -printf '%s %p\n' | sort -n | tail -n 5
