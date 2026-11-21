"""Safe configuration: configs/final.py with every feature path switched off at run time.

The model is the plain looped transformer (``bes.model.model.switches_off``: registers, candidate
input and output law, confidence channel, sink, clocks and every candidate kind off) and no
training row carries a library passage. The switches are runtime inputs, so the parameter tree
and the compiled step are those of the final configuration: it is the canary's reference and
the last rung of the fallback ladder (``bes.train.supervisor.LADDER``), and a full run with it is
the most conservative way to spend the 24 hours.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_FINAL = load_config(str(Path(__file__).with_name("final.py")))

CFG = dataclasses.replace(
    _FINAL,
    data=dataclasses.replace(_FINAL.data, library_frac_base=0.0, library_frac_chat=0.0),
    run=dataclasses.replace(_FINAL.run, name="safe", out_dir="runs/safe",
                            switches="reg:0 cand_in:0 cand_out:0 conf:0 sink:0 clock:0 kind:0"),
)
