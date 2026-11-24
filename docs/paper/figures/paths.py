"""Where every source lives and where the outputs go. Edit here if a source moves; every script
imports this.

Outputs follow the scripts, not a fixed path: PAPER is the parent of the directory that holds this
file (or $PAPER_DIR), so a copy of figures/ elsewhere writes its own data/, figures and _work/ and
never the paper's. $PAPER_FIGURES_DATA moves data/ (and the extracted records in data/_src with
it), $PAPER_FIGURES_OUT the figure files. Inputs are read-only: the snapshots, logs and chip outputs
at fixed paths in the two scratchpads and the repository, and the paper's own analyses
(lead2_work/, review1_work/), taken from PAPER if it has them, else from the paper directory
PAPER_HOME (read only).
"""
import os

N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"   # new scratchpad
S = "/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"                   # old scratchpad
REPO = "/home/user/turingaward2027"
PAPER_HOME = os.path.join(N, "paper")             # the paper's directory (inputs only, never written from a copy)

FIG = os.path.dirname(os.path.abspath(__file__))                   # the scripts (this directory)
PAPER = os.path.abspath(os.environ.get("PAPER_DIR") or os.path.dirname(FIG))
DATA = os.path.abspath(os.environ.get("PAPER_FIGURES_DATA") or os.path.join(PAPER, "data"))
FIG_OUT = os.path.abspath(os.environ.get("PAPER_FIGURES_OUT") or FIG)                # figure files
SRC = os.path.join(DATA, "_src")                  # records extracted from the snapshots (extract_records.py)
WORK = os.path.join(FIG, "_work")                 # build log and the font cache
_pylib_here = os.path.join(WORK, "pylib")
PYLIB = (os.environ.get("PAPER_FIGURES_PYLIB")    # private matplotlib install (the venv has none); read only
         or (_pylib_here if os.path.isdir(_pylib_here) else os.path.join(PAPER_HOME, "figures", "_work", "pylib")))


def paper_input(rel):
    """A read-only input of the paper directory (lead2_work/..., review1_work/...): PAPER's copy if it
    exists, else PAPER_HOME's."""
    here = os.path.join(PAPER, rel)
    return here if os.path.exists(here) else os.path.join(PAPER_HOME, rel)


# snapshots (tar in parts) and their extracted records
SNAP_FINAL = os.path.join(N, "snap_final_main16")    # the stopped final run, step 29,722 (fixed)
SNAP_LIVE = os.environ.get("PAPER_SNAP_LIVE") or os.path.join(N, "snap_main")   # newest snapshot of the live
#                                                     (v9) run (grows); the variable is for tests only
SRC_FINAL = os.path.join(SRC, "final", "run")
SRC_LIVE = os.path.join(SRC, "live", "run")
GRAFT_CKPT = os.path.join(SRC_FINAL, "train", "ckpt", "ckpt-000026-step-000029722", "manifest.json")

# chip measurements and logs
CHIP_KERNELS = os.path.join(N, "wf", "chip_kernels_0745.txt")
V9_MEAS = {"take": os.path.join(S, "v9_measA.txt"), "sorted": os.path.join(S, "v9_measB_sorted.txt"),
           "onehot": os.path.join(S, "v9_measC_onehot.txt"), "take, switches": os.path.join(S, "v9_measD.txt")}
MICRO_OUT = os.path.join(S, "micro_out.txt")
FLAGS_OUT = [os.path.join(S, "flags_out.txt"), os.path.join(S, "flags_out2.txt")]
TUNE_OUT = os.path.join(REPO, "docs", "tpu_profile", "tpu_tune.v6e.txt")
FINAL_B_LOG = os.path.join(REPO, "docs", "tpu_profile", "final_b.run.log")
MONITOR_LOG = os.path.join(S, "monitor.log")
TPU_INTERVALS = os.path.join(S, "tpu_intervals.txt")
MEASURE_LOGS = [os.path.join(S, f) for f in ("measure.log", "measure2.log", "measure3.log", "v9_measA.log",
                                              "resume_v9.log")]
# how a whole step was timed (the output files do not record the number of calls; R/scripts/...)
STEP_TIMING = {
    "tpu_kernels": "mean of --step-calls whole steps run back to back after one warm call "
                   "(scripts/tpu_kernels.py, default 2; the count used is not in the output file)",
    "tpu_micro": "mean of --calls whole steps run back to back after one warm call "
                 "(scripts/tpu_micro.py, default 3; the count used is not in the output file)",
}

# held-out loss of checkpoints (export check rows: web 64, chat 64, proxy 12; float32, dense attention)
MIDRUN_EVAL = os.path.join(N, "midrun_eval_1.json")        # runtime: final 26,890 / 28,501 / 29,722, v9 to 31,912
HELDOUT_V9 = os.path.join(N, "heldout_v9.jsonl")           # runtime: automatic evaluations of new v9 checkpoints
HELDOUT_LOCAL = paper_input(os.path.join("lead2_work", "heldout_analysis.json"))   # local CPU evaluations (Table 7, G1)

# the paper's own analyses (second-pass lead author)
PREDICTOR = paper_input(os.path.join("lead2_work", "predictor_test.json"))   # Tables 4 and 5
NUMBERS = paper_input(os.path.join("lead2_work", "numbers.json"))            # the paper's live numbers; its
#                                                                              newest step is the default cut-off
GRAFT_GRAD = paper_input(os.path.join("lead2_work", "graft_grad.json"))      # gradient check at the graft (6.4)
GRAFT_CHECK_LOGS = [  # (file, where it ran): the graft tool's --check output (Table 6)
    (os.path.join(S, "graft_test.log"), "development container (snapshot)"),
    (os.path.join(S, "graft_dry_out.txt"), "runtime host CPU, next to the trainer"),
    (paper_input(os.path.join("review1_work", "graftcheck", "graft_check_29722.log")),
     "development container (re-run at HEAD 8a86bcb)"),
]

# blind comparisons (bes.eval.proxy reports): development sets read so far; the final test sets are
# added by name (make_data.py --blind final-<tag>-vs-falcon,final-<tag>-vs-mobilellm)
BLIND_DIR = os.path.join(N, "study", "proxy", "reports", "ab")
BLIND_DEV = ["s1-default-vs-falcon--opus-1", "s1-default-vs-mobilellm--opus-1",
             "s2-default-vs-falcon--judge-opus", "s2-default-vs-mobilellm--judge-opus"]
BLIND_DEFAULT_SOURCE = "judge-opus"          # rating source appended to a name given without one

# data-rule scans
OPENINGS = os.path.join(N, "openings.json")
DEFLECTION = os.path.join(N, "deflection_check2.json")

T0 = 1795455991.9461858        # t0 of run main16 (train/run.json): 2026-11-23 17:46:31.946 UTC
TOKENS_PER_STEP = 262144       # optim.tokens_per_step (bes/config.py); ledger tokens / steps = 262,144


def rel(path):
    """A path for the CSVs' source columns: relative to the paper directory when inside it."""
    path = os.path.normpath(path)
    for root in (PAPER, PAPER_HOME):
        if path.startswith(root + os.sep):
            return os.path.relpath(path, root)
    return path
