# BES-100M-A20M

A language model with at most 100M stored numbers and at most 20M numbers read per generated
token, designed to be trained from scratch on one TPU v6e within 24 hours.

- `docs/PLAN.md` — the approved plan (Traditional Chinese): principle, mathematics, architecture,
  recipe, evidence gates and an honest assessment of novelty and odds.
- `docs/SPEC.md`, `docs/SPEC_MODEL_TRAIN.md` — engineering contracts between modules.
- `bes/` — the package; `configs/` — run configurations; `tests/` — unit tests.

Status: under construction.
