"""Where every source lives. Edit here if a source moves; every script imports this."""
import os

PAPER = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper"
N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"   # new scratchpad
S = "/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"                   # old scratchpad
REPO = "/home/user/turingaward2027"

FIG = os.path.join(PAPER, "figures")
DATA = os.path.join(PAPER, "data")
SRC = os.path.join(DATA, "_src")                 # records extracted from the snapshots (extract_records.py)
PYLIB = os.path.join(FIG, "_work", "pylib")      # private matplotlib install

# snapshots (tar in parts) and their extracted records
SNAP_FINAL = os.path.join(N, "snap_final_main16")    # the stopped final run, step 29,722 (fixed)
SNAP_LIVE = os.path.join(N, "snap_main")             # newest snapshot of the live (v9) run (grows)
SRC_FINAL = os.path.join(SRC, "final", "run")
SRC_LIVE = os.path.join(SRC, "live", "run")

# chip measurements and logs
CHIP_KERNELS = os.path.join(N, "wf", "chip_kernels_0745.txt")
V9_MEAS = {"take": os.path.join(S, "v9_measA.txt"), "sorted": os.path.join(S, "v9_measB_sorted.txt"),
           "onehot": os.path.join(S, "v9_measC_onehot.txt"), "take, switches": os.path.join(S, "v9_measD.txt")}
MIDEVAL_LOG = os.path.join(S, "mideval_last.log")           # held-out loss at checkpoints (scripts/midrun_eval.py)
HELDOUT_V9 = os.path.join(N, "heldout_v9.jsonl")           # automatic held-out evaluations of new v9 checkpoints
MICRO_OUT = os.path.join(S, "micro_out.txt")
FLAGS_OUT = [os.path.join(S, "flags_out.txt"), os.path.join(S, "flags_out2.txt")]
TUNE_OUT = os.path.join(REPO, "docs", "tpu_profile", "tpu_tune.v6e.txt")
FINAL_B_LOG = os.path.join(REPO, "docs", "tpu_profile", "final_b.run.log")
MONITOR_LOG = os.path.join(S, "monitor.log")
TPU_INTERVALS = os.path.join(S, "tpu_intervals.txt")
MEASURE_LOGS = [os.path.join(S, f) for f in ("measure.log", "measure2.log", "measure3.log", "v9_measA.log",
                                              "resume_v9.log")]

# data-rule scans
OPENINGS = os.path.join(N, "openings.json")
DEFLECTION = os.path.join(N, "deflection_check2.json")

T0 = 1795455991.9461858        # t0 of run main16 (train/run.json): 2026-11-23 17:46:31.946 UTC
TOKENS_PER_STEP = 262144       # optim.tokens_per_step (bes/config.py); ledger tokens / steps = 262,144
