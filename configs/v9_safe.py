"""Safe v9 configuration: configs/v9.py with every feature path switched off at run time.

The pattern of ``configs/safe.py``: the model is the plain looped transformer
(``bes.model.model.switches_off``: registers, candidate input and output law, confidence
channel, sink, clocks, token gains, selective attention, the attention output gate, the token
neurons and every candidate kind off) and no training row carries a library passage (as in v9).
The switches are runtime inputs, so the parameter tree and the compiled step are those of
``configs/v9.py``.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_V9 = load_config(str(Path(__file__).with_name("v9.py")))

CFG = dataclasses.replace(
    _V9,
    run=dataclasses.replace(
        _V9.run, name="v9_safe", out_dir="runs/v9_safe",
        switches="reg:0 cand_in:0 cand_out:0 conf:0 sink:0 clock:0 gain:0 sel:0 gate:0 page:0 "
                 "kind:0"),
)
