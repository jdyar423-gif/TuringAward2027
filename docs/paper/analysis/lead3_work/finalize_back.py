"""Renumbering of main-text tables and figures referenced from the back matter (third draft: Figure 2 of
the second-pass layout moved to Appendix F as Figure F2; Table 5 moved to Appendix G as Table G1)."""
b = open('back_v3_pre_M.md').read()
m = open('appendix_M.md').read()
def rep(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (n, old[:100])
    return s.replace(old, new)
b = rep(b, "refusal or deflection (Table 9). [TBD: the share", "refusal or deflection (Table 8). [TBD: the share")
b = rep(b, "chip files as in Table 3 and `docs/CHANGES.md` §19.5; Figure 4b).", "chip files as in Table 3 and `docs/CHANGES.md` §19.5; Figure 3b).")
b = rep(b, "`graft_grad.py` (Section 6.4), `exact_dev.py` and `opp_bpb.py` (Table 9), `deflection_labels.py`", "`graft_grad.py` (Section 6.4), `exact_dev.py` and `opp_bpb.py` (Table 8), `deflection_labels.py`")
b = rep(b, "`micrograft.py` (Section 6.6), `readers.py` (Table 8),", "`micrograft.py` (Section 6.6, Table 6), `readers.py` (Table 7),")
m = rep(m, "sign test and the reader diagnostics are as in Table 8 (from our side;", "sign test and the reader diagnostics are as in Table 7 (from our side;")
open('back_v3.md', 'w').write(b)
open('appendix_M_v3.md', 'w').write(m)
print('ok')
