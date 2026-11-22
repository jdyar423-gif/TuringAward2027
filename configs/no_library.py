"""Alternative without the written library: its budget goes to wider per-loop value embeddings.

Everything else is the final configuration (configs/final.py). The micro ablations saw no gain
from wider value embeddings at their scale; this variant exists for the case where the library
is not wanted at all, and only satisfies the standard reading of the active count.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_FINAL = load_config(str(Path(__file__).with_name("final.py")))

CFG = dataclasses.replace(
    _FINAL,
    model=dataclasses.replace(_FINAL.model, ve_dim=512),
    data=dataclasses.replace(_FINAL.data, use_library=False, library_frac_base=0.0,
                             library_frac_chat=0.0),
    decode=dataclasses.replace(_FINAL.decode, use_library=False),
    run=dataclasses.replace(_FINAL.run, name="no_library", out_dir="runs/no_library"),
)
