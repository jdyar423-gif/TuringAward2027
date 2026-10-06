#!/usr/bin/env bash
# End-to-end reproduction of the final system (CPU; ~1 h training + ~30 min system eval on 4 cores).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-final}
python3 prepare.py --vocab 4096
# 1) compute core + hashed n-gram memory.
#    Total budget 1.28e14; 2.0e13 reserved for everything after pre-training (gate fitting on held-out
#    articles + test-time training); 3% of train articles held out (never seen by the network).
python3 train.py --name "$NAME" --vocab 4096 --budget 1.28e14 --reserve 2e13 --holdout 0.03 \
  --d 256 --n_layer 6 --n_head 4 --batch 16 --seq 256 \
  --ngram_orders 2,3 --smear 1 --attn_gate sparse --xsa 1 \
  --lr_muon 0.03 --lr_emb 0.3 --lr_ngram 0.4 --lr_head 0.008 --threads 4
# 2) full system: count memory (K=8 global orders, 6 document orders), gate fitted on the held-out
#    articles, score-first dynamic evaluation (update every 128 tokens, SGD lr 0.15), budget audit.
python3 run_system.py --run "$NAME" --splits valid,test --threads 4 --tag _final
