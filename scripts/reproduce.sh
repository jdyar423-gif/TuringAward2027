#!/usr/bin/env bash
# End-to-end reproduction of the final system (CPU, ~1.5 h on 4 cores).
set -euo pipefail
cd "$(dirname "$0")/.."
VOCAB=${VOCAB:-4096}
NAME=${NAME:-final}
python3 prepare.py --vocab "$VOCAB"
# 1) neural core: 1.28e14 total budget, 2.0e13 reserved for test-time work, 3% of train articles held out
python3 train.py --name "$NAME" --vocab "$VOCAB" --budget 1.28e14 --reserve 2e13 --holdout 0.03 \
  --ngram_orders 2,3 --smear 1 --attn_gate sparse --xsa 1 ${TRAIN_ARGS:-}
# 2) full system: count memory + gate fitted on held-out train articles + score-first dynamic eval on test
python3 run_system.py --run "$NAME" ${SYSTEM_ARGS:-}
