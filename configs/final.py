"""Final 24-hour configuration (TPU v6e), frozen after the evidence gates.

Decisions taken from measurements (docs/CHANGES.md, docs/E2_ABLATIONS.md, docs/TPU_NOTES.md):

* ``remat=True``: without rematerialisation the default micro-batch does not fit in device
  memory; the start-up benchmark also times the plan without it at half-size micro-batches.
* ``clock_planes=0`` and ``use_conf=False``: the structural RoPE clocks (+0.3% loss when removed)
  and the confidence channel (+0.7-0.8%) stayed under the pre-registered 1% bar for new
  components in the micro ablations, so they are off. The candidate tape (+11% overall, +19% on
  assistant tokens), the registers and the candidate input features (each about +1.1% on chat)
  and the per-loop value embeddings (+1.5%) are kept; the attention sink has outside evidence
  and was neutral, so it stays.
* ``late_from=0.67``: the last third of training runs at 2048 tokens, as the plan states.
* Training keeps the self-evidence masses at their calibrated level; only the decoder
  down-weights them (``DecodeCfg.self_evidence_weight``): a model trained with the down-weighting
  learns to undo it.
"""
from bes.config import Cfg, DataCfg, ModelCfg, RunCfg

CFG = Cfg(
    model=ModelCfg(attn_impl="splash", act_dtype="bfloat16", remat=True,
                   clock_planes=0, use_conf=False),
    data=DataCfg(late_from=0.67),
    run=RunCfg(name="final", out_dir="runs/final"),
)
