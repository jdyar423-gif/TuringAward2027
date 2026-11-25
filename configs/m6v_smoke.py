"""Tiny CPU rehearsal of the six-hour continuation (``configs/m6v.py``), in two passes::

    BES_PROFILE=m6 python scripts/run_24h.py --config configs/m6v_smoke.py --allow-cpu \\
        --budget-minutes 9 --out runs/m6v_smoke_a
    BES_PROFILE=m6 python scripts/run_24h.py --config configs/m6v_smoke.py --allow-cpu \\
        --budget-minutes 12 --out runs/m6v_smoke_b \\
        --override run.init_from=runs/m6v_smoke_a/final

The first pass trains the smoke model with the token pages from its initialisation and exports
it; the second starts from that export (``run.init_from``) with the continuation's learning-rate
settings, as the chip run starts from ``release/main16-v9-before-dpo``. The stages, the mixture,
the answers library and the speed switches are those of ``configs/m6v.py``; the model, the
shapes and the data are those of ``configs/smoke.py`` with token pages of smoke size.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_M6V = load_config(str(Path(__file__).with_name("m6v.py")))
_SMOKE = load_config(str(Path(__file__).with_name("m6_smoke.py")))

CFG = dataclasses.replace(
    _SMOKE,
    model=dataclasses.replace(_SMOKE.model, gain_o=True, tok_neurons=4, tok_dim=16, table_grad="take"),
    data=dataclasses.replace(_SMOKE.data, exemplar_floor=_M6V.data.exemplar_floor,
                             exemplar_floor_frac=_M6V.data.exemplar_floor_frac),
    optim=dataclasses.replace(_SMOKE.optim, lr_mult=_M6V.optim.lr_mult, tokens_before=2.0e6,
                              final_lr_frac=_M6V.optim.final_lr_frac, warmup_frac=_M6V.optim.warmup_frac),
    run=dataclasses.replace(_SMOKE.run, name="m6v_smoke", out_dir="runs/m6v_smoke",
                            fast="relu2=1 ce=0 flat=1 rows=1 tokn=1", ship_library="select"),
)
