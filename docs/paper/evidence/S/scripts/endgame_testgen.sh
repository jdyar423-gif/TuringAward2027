#!/usr/bin/env bash
# Start the generation of the 200 test replies on the runtime's host cores from the run's newest
# weights (what scripts/run_24h.py would export now), CPU only, next to the post-training stages.
# usage: endgame_testgen.sh TAG MINUTES [extra gen_test_replies options]
#   TAG: trained | prefopt | final ...; output /content/bes_cli/testgen_TAG (testgen.tar.gz at its end)
# then:  endgame_testgen.sh --fetch TAG   (downloads and unpacks into $N/testgen/TAG, merges into $N/proxy_final)
set -u
export PATH=/root/.local/bin:$PATH GOOGLE_APPLICATION_CREDENTIALS=/root/.config/colab-cli-auth/adc_user.json COLAB_CLI_ARGS=--auth=adc
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
if [ "$1" = "--fetch" ]; then
  tag="$2"; D=$N/testgen/$tag; mkdir -p "$D/out"
  timeout 60 colab --auth=adc download -s bes24h /content/bes_cli/testgen_$tag/testgen.log "$D/testgen.log" </dev/null >/dev/null 2>&1; tail -n 2 "$D/testgen.log" | cut -c1-220
  rm -f "$D/testgen.tar.gz"; timeout 600 colab --auth=adc download -s bes24h /content/bes_cli/testgen_$tag/testgen.tar.gz "$D/testgen.tar.gz" </dev/null >/dev/null 2>&1 || { echo "no tarball yet"; exit 1; }
  tar -xzf "$D/testgen.tar.gz" -C "$D/out" && ls "$D/out" | tr '\n' ' '; echo
  exit 0
fi
tag="$1"; minutes="$2"; shift 2
date +%s > "$S/presence_pause.txt"; timeout 60 colab --auth=adc upload -s bes24h "$S/presence_pause.txt" /content/bes_cli/presence_pause </dev/null >/dev/null 2>&1
timeout 60 colab --auth=adc upload -s bes24h "$S/scripts/endgame_driver.py" /content/bes_cli/endgame_driver.py </dev/null >/dev/null 2>&1; sleep 4
TAG=tg$tag bash "$S/scripts/exec_detached.sh" bes24h "$S/tg${tag}_nonce.txt" 3 --env "BES_CMD=/content/bes_cli/endgame_driver.py $tag $minutes $*"
