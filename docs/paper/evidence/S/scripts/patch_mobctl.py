"""Fill the placeholders of MobileLLM-R1.5's two K = 6 control arms in the paper (English and
Chinese) from N/peek/mob_ctl_results.json (mob_ctl_results.py: the registered judge) and
N/peek/mob_ctl_exact.json (mob_ctl_exact.py: the exact scores).

    JUDGE_END=HH:MM JUDGE_FACTS=path.json python patch_mobctl.py EN_IN ZH_IN EN_OUT ZH_OUT
    (the inputs are not modified; DRY=1 allows an incomplete pass of the judge, for a trial)
"""
import json
import os
import re
import sys

N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"
R = json.load(open(N + "/peek/mob_ctl_results.json"))
X = json.load(open(N + "/peek/mob_ctl_exact.json"))
en_in, zh_in, en_out, zh_out = sys.argv[1:5]
A, B = R["k6lp"], R["k6ds"]
DRY = bool(os.environ.get("DRY"))
for r in (A, B):
    assert DRY or (r["records"] == 200 and r["all"]["n"] == 200), "the judge's pass is not complete"
JUDGE_END = os.environ.get("JUDGE_END", "HH:MM")
# facts of the judge's pass on the two sets (judge_pass_facts.py): streams, seconds per pair, pairs judged twice
F = json.load(open(os.environ["JUDGE_FACTS"])) if os.environ.get("JUDGE_FACTS") else {"seconds_mean": 0.0, "twice": 0, "streams": 9, "twice_detail": {"pairs": 0, "same_verdict": 0}}
TD = F["twice_detail"]
assert DRY or (TD["pairs"] == F["twice"] and TD["same_verdict"] == TD["pairs"]), "the wording says the two records of every pair judged twice agree"
assert DRY or os.environ.get("JUDGE_FACTS"), "JUDGE_FACTS is required for the final patch"
assert DRY or JUDGE_END != "HH:MM"


def pct(x):
    return f"{100 * x:.1f}%"


def fp(x):
    return f"{x:+.3f}".replace("-", "−")


def pp(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def peq(p):
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def p2(p):
    """two significant digits, as the other rows of Tables 7, M1 and M11"""
    if p < 0.0001:
        return "< 0.0001"
    s = f"{float(f'{p:.2g}'):.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def row7(label, r, judge):
    a, s = r["all"], r["189"]
    return (f"| {label} | {judge} | 200 | {a['wlt']} | {a['score']} {a['ci']} | {a['decisive']} {a['dci']} | {p2(a['p_raw'])} | "
            f"{pct(r['agree'])}; {fp(r['first_pos'])} | {s['wlt']}; {s['score']} |")


def rowm1(label, r, judge):
    a = r["all"]
    return (f"| {label} | {judge} | 200 | {a['wlt']} | {a['score']} {a['ci']} | {a['decisive']} {a['dci']} | {p2(a['p_raw'])} | "
            f"{pct(r['agree'])} | {fp(r['first_pos'])} |")


def rowm11(label, r, judge):
    s = r["189"]
    return f"| {label} | {judge} | 189 | {s['wlt']} | {s['score']} {s['ci']} | {p2(s['p_raw'])} |"


def rowm13(a_label, b_label, r):
    v, w = r["vs_released_all"], r["vs_released_189"]
    md = f"{v['mean_diff']:+.4f}".replace("-", "−")
    return (f"| {a_label} | {b_label} | {v['a_better']}; {v['b_better']}; {v['same']} | {md} | {pp(v['p'])} | "
            f"{w['a_better']}; {w['b_better']}; {w['same']}, {peq(w['p'])} |")


def lit(text, old, new, name):
    n = text.count(old)
    assert n == 1, f"{name}: {n} matches"
    return text.replace(old, new)


def rx(text, pattern, new, name):
    out, n = re.subn(pattern, lambda m: new, text)
    assert n == 1, f"{name}: {n} matches"
    return out


def add_cols(text, anchor, labels_cells, name):
    """append cells to the rows, named by their labels, of the table whose header contains `anchor`
    (the table ends at the first blank line; each label must match one of its lines)"""
    assert text.count(anchor) == 1, f"{name}: anchor: {text.count(anchor)} matches"
    i = text.index(anchor)
    j = text.index("\n\n", i)
    seg = text[i:j]
    for label, cells in labels_cells:
        m = [x for x in re.finditer(r"(?m)^" + re.escape("| " + label + " |") + r".*$", seg)]
        assert len(m) == 1, f"{name}: row {label!r}: {len(m)} matches"
        line = m[0].group(0)
        assert line.endswith("|")
        seg = seg[:m[0].start()] + line + " " + " | ".join(cells) + " |" + seg[m[0].end():]
    return text[:i] + seg + text[j:]


(w1, l1), (w2, l2) = (A["all"]["w"], A["all"]["l"]), (B["all"]["w"], B["all"]["l"])
sA, sB, ciA, ciB = A["all"]["score"], B["all"]["score"], A["all"]["ci"], B["all"]["ci"]
vA, vB = A["vs_released_all"], B["vs_released_all"]
mdA, mdB = vA["mean_diff_3"], vB["mean_diff_3"]   # three decimals from the exact fraction, a half to even
xa, xb, xr = X["k6lp"]["all"], X["k6ds"]["all"], X["released"]["all"]
xa9, xb9 = X["k6lp"]["189"], X["k6ds"]["189"]
# what the wording below states; a result that breaks one of these needs other words
assert w1 > l1 and w2 > l2, "the judge no longer decides more pairs for our model"
assert float(sA) < 0.635 and float(sB) < 0.635 and vA["mean_diff"] > 0 and vB["mean_diff"] > 0, "our lead is not smaller against the arms"
assert (xa["degenerate"], xb["degenerate"], xr["degenerate"]) == (64, 51, 158)
assert (xa["math_correct"], xb["math_correct"], xa["con_all_ok"], xb["con_all_ok"]) == (12, 11, 9, 8)
assert (xa["prompts_with_degenerate"], xb["prompts_with_degenerate"]) == (60, 48)
r4 = lambda v: f"{v:.3f}"
t8 = [f"{xa['math_correct']} ({xa9['math_correct']}); {xb['math_correct']} ({xb9['math_correct']})",
      f"{xa['con_all_ok']} ({xa9['con_all_ok']}); {xb['con_all_ok']} ({xb9['con_all_ok']})",
      f"{xa['degenerate']} ({xa['degenerate_and_truncated']}); {xb['degenerate']} ({xb['degenerate_and_truncated']})",
      f"{xa['words_upper_median']}; {xb['words_upper_median']}"]
assert t8 == ["12 (11); 11 (10)", "9 (8); 8 (7)", "64 (50); 51 (47)", "108; 105"], t8


def m4_cells(x):
    return [str(x["math_correct"]), str(x["con_all_ok"]), str(x["single_ok"]), str(x["degenerate"]), str(x["degenerate_and_truncated"]),
            f"{x['loop']}; {x['repeated_lines']}", f"{x['empty']}; {x['unclosed_fence']}; {x['role_leak']}", f"{x['refusal']}; {x['deflection']}",
            f"{x['words_upper_median']}; {x['words_mean']:.1f}", r4(x["rep4_mean"])]


m4a, m4b = m4_cells(xa), m4_cells(xb)
sig = lambda v: "sign test " + peq(v["p"])
JQ = "`qwen3-4b-2507` (registered), no Opus reading"
JQZ = "`qwen3-4b-2507`（註冊評審），無 Opus 判讀"

# ------------------------------------------------------------------ English
t = open(en_in, encoding="utf-8").read()
t = lit(t, "Still [TBD]: MobileLLM-R1.5's two K = 6\ncontrol arms, in generation (60 of 200 prompts at 22:02 UTC; expected after 00:30 UTC on 25 November), and the\nauthor list.",
        "MobileLLM-R1.5's two K = 6\ncontrol arms were added on 25 November (generated by 00:25 UTC; judged by the registered judge and the exact\nscorers only, with no Opus reading: Section 7.2, deviation 6). Still open: the author list.", "status")
t = lit(t, " (MobileLLM-R1.5's six-sample arms: [TBD]); the pre-registered judge,", "; the pre-registered judge,", "abstract 1")
t = lit(t, "63% degenerate under MobileLLM-R1.5's recommended preset in our harness; our edge in meeting constraints (38",
        "63% degenerate under MobileLLM-R1.5's recommended preset in our harness: given six samples per reply, 25% or\n"
        f"20% do, and the pre-registered judge, the only reader of those two arms, prefers our model by less ({w1} to\n"
        f"{l1} and {w2} to {l2}; score {sA} and {sB} against 0.635); our edge in meeting constraints (38", "abstract 2")
t = lit(t, "including two control arms that give Falcon-H1-Tiny six\n   samples (MobileLLM-R1.5's: [TBD]);",
        "including control arms that give each opponent six\n   samples (Falcon-H1-Tiny's two read by the Opus readers, MobileLLM-R1.5's two by the registered judge only);", "intro")
t = rx(t, r"MobileLLM-R1\.5's two\n   arms do not exist yet \[TBD: [^\]]*\]\.",
       "MobileLLM-R1.5's two\n   arms were generated later (complete at 00:25 UTC on 25 November: three CPU runtimes and the development\n"
       "   container generated disjoint slices of the prompt order and their caches were merged; 48 samples generated\n"
       "   twice, on two runtimes, were identical) and have no Opus reading: the user asked at 00:07 UTC that no further\n"
       f"   agents be used, so they were judged by the registered judge (complete at {JUDGE_END} UTC) and the exact scorers\n"
       "   only, and their rows are not of the same kind as the Opus rows of the other arms.", "deviation 6")
t = rx(t, r"\| official vs MobileLLM-R1\.5, K = 6, own log-probability or our selector \| Opus \| – \| \[TBD: [^\]]*\] \|  \|  \|  \|  \|  \|",
       row7("official vs MobileLLM-R1.5, K = 6, own log-probability", A, JQ) + "\n" + row7("official vs MobileLLM-R1.5, K = 6, our selector", B, JQ), "table 7")
t = lit(t, "| MobileLLM-R1.5 `neutral` | MobileLLM-R1.5, K = 6 (both rules) |", "| MobileLLM-R1.5 `neutral` | MobileLLM-R1.5, K = 6: own log-probability; our selector |", "table 8 head")
for key, cell in zip(("maths answers correct, of 30 (of 28)", "prompts with every constraint met, of 48 (of 44)",
                      "degenerate replies, of 252 (of them cut at the token limit)", "words per reply, upper median (the 127th of 252)"), t8):
    m = re.search(re.escape("| " + key + " |") + r"([^\n]*)\| \[TBD\] \|", t)
    assert m, key
    t = t[:m.start()] + "| " + key + " |" + m.group(1) + "| " + cell + " |" + t[m.end():]
t = lit(t, "defeat by the readers is largely a defeat of those replies. No model gave a stock",
        "defeat by the readers is largely a defeat of those replies. **Given six samples per reply it degenerates far\n"
        f"less**: {xa['degenerate']} of 252 replies kept by its own log-probability and {xb['degenerate']} kept by our selector ({xa['prompts_with_degenerate']} and {xb['prompts_with_degenerate']} prompts),\n"
        f"with its maths and constraint counts where they were ({xa['math_correct']} and {xb['math_correct']} of 30; {xa['con_all_ok']} and {xb['con_all_ok']} of 48: our selector does not\n"
        "lift its constraints as it lifts Falcon-H1-Tiny's). These two arms were read by the registered judge only\n"
        f"(deviation 6). It still decides more pairs for our model ({w1} to {l1} and {w2} to {l2}; score {sA} {ciA} and\n"
        f"{sB} {ciB}, against 0.635 for the released preset), and prompt by prompt our score is lower against them\n"
        f"than against the released preset, by {mdA} on average ({sig(vA)}) and by {mdB} ({peq(vB['p'])}; Table M13): part of\n"
        "our lead over MobileLLM-R1.5 is a lead over its single sample. No model gave a stock", "7.6 sentence")
t = lit(t, "(with one, 99 to 16; MobileLLM-R1.5's\nsix-sample arms [TBD]);",
        f"(with one, 99 to 16; against MobileLLM-R1.5\ngiven six samples, by the registered judge alone, {w1} to {l1} and {w2} to {l2}, a smaller lead);", "7.10")
t = lit(t, "and MobileLLM-R1.5's\n   control arms are [TBD];", "and MobileLLM-R1.5's\n   six-sample control arms, against which our lead is smaller, were read by the registered judge only;", "section 8")
t = lit(t, "(59 of 64; MobileLLM-R1.5's six-sample arms: [TBD]);",
        f"(59 of 64; {w1} of {w1 + l1} and {w2} of {w2 + l2} against MobileLLM-R1.5 given six samples per reply, where its score for our model falls\nfrom 0.635 to {sA} and {sB});", "conclusion")
t = lit(t, "judged twice, the forward stream's record kept). A fourth CPU runtime was refused",
        "judged twice, the forward stream's record kept). MobileLLM-R1.5's two control sets were judged on 25 November,\n"
        f"00:28 to {JUDGE_END} UTC, on the three CPU runtimes: two 4-thread streams each over index ranges of the pair order\n"
        "and, from 01:39 as cores came free, three helper streams that took a range from its far end\n"
        f"({F['seconds_mean']:.1f} s per pair on average; {F['twice']} pairs were judged by two streams, a helper on another runtime not seeing\n"
        f"that runtime's records: the two records give the same verdict on all {TD['same_verdict']}, and the one merged first was kept).\n"
        "The three CPU runtimes were given back at 01:56 UTC; the development container had restarted once more at\n"
        "01:33 UTC, which did not touch the streams. A fourth CPU runtime was refused", "appendix operations")
t = lit(t, "from the replies, and the placeholder rows of MobileLLM-R1.5's K = 6 arms).",
        "from the replies, and the rows and columns of MobileLLM-R1.5's K = 6 arms, added on 25 November from the outputs\nof `mob_ctl_results.py` and `mob_ctl_exact.py`).", "M preamble 1")
t = lit(t, "final-prefopt-vs-mobilellm: 200 records, last written 22:35:20).",
        "final-prefopt-vs-mobilellm: 200 records, last written 22:35:20); its pass on MobileLLM-R1.5's two control sets "
        f"(final-prefopt-vs-mobilellm-k6lp and -k6ds, 200 records each) ended at {JUDGE_END} UTC on 25 November.", "M preamble 2")
t = lit(t, "`-vs-falcon-k6ds`, `-vs-mobilellm`, `final-k1-vs-falcon`,", "`-vs-falcon-k6ds`, `-vs-mobilellm`, `-vs-mobilellm-k6lp`, `-vs-mobilellm-k6ds`, `final-k1-vs-falcon`,", "M1 caption")
t = rx(t, r"\| MobileLLM-R1\.5, K = 6 \(both rules\) \| – \| \[TBD: [^\]]*\] \| \| \| \| \| \| \|",
       rowm1("MobileLLM-R1.5, K = 6 kept by its own log-probability", A, "`qwen3-4b-2507` (registered), full pass; no Opus reading") + "\n" +
       rowm1("MobileLLM-R1.5, K = 6 kept by our selector", B, "`qwen3-4b-2507` (registered), full pass; no Opus reading"), "M1")
t = lit(t, "| MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` |\n|---|---|---|---|---|---|---|---|---|\n| maths answers correct (of 30) |",
        "| MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` | MobileLLM-R1.5 K = 6, own log-prob. | MobileLLM-R1.5 K = 6, our selector |\n|---|---|---|---|---|---|---|---|---|---|---|\n| maths answers correct (of 30) |", "M4 head")
t = add_cols(t, "| MobileLLM-R1.5 K = 6, own log-prob. | MobileLLM-R1.5 K = 6, our selector |", list(zip(("maths answers correct (of 30)", "constrained prompts with every constraint met (of 48)", "single constraints met (of 56)",
                          "degenerate replies (of 252)", "… cut at the token limit", "… looping tail; repeated lines",
                          "… empty; unclosed code fence; role markers", "stock refusals; stock deflections (of 252)",
                          "words per reply: upper median (the 127th of 252); mean", "mean repeated-4-gram rate"), zip(m4a, m4b))), "M4")
t = lit(t, "MobileLLM-R1.5's two K = 6\narms: [TBD] (Section 7.2, deviation 6).",
        "The columns of MobileLLM-R1.5's two K = 6\narms (Section 7.2, deviation 6) were added after the table was built: they are recounted from the per-turn\n"
        "records of the two runs' reports in `study/proxy/reports/exact/` by `mob_ctl_exact.py`, which reproduces the\n"
        f"`recommended` column from that run's report; at least one degenerate reply on {xa['prompts_with_degenerate']} and {xb['prompts_with_degenerate']} of the 200 prompts (139\nunder the released preset).", "M4 note")
t = lit(t, "| MobileLLM-R1.5, K = 6 (both rules) | – | [TBD] | | | |",
        rowm11("MobileLLM-R1.5, K = 6, own log-probability", A, "`qwen3-4b-2507`") + "\n" + rowm11("MobileLLM-R1.5, K = 6, our selector", B, "`qwen3-4b-2507`"), "M11")
t = lit(t, "MobileLLM-R1.5 `recommended` 11 and 8; MobileLLM-R1.5 `neutral` 10 and 8 | – | – |",
        f"MobileLLM-R1.5 `recommended` 11 and 8; MobileLLM-R1.5 `neutral` 10 and 8; MobileLLM-R1.5 K = 6, own log-prob. {xa9['math_correct']} and {xa9['con_all_ok']}; "
        f"MobileLLM-R1.5 K = 6, our selector {xb9['math_correct']} and {xb9['con_all_ok']} | – | – |", "M11 exact")
t = lit(t, "as `python -m bes.eval.proxy compare --a A --b B --source judge-opus`; a prompt is better",
        "as `python -m bes.eval.proxy compare --a A --b B --source judge-opus`, and with `--source judge-qwen3-4b-2507-3605803b` for the two rows marked registered judge, `mob_ctl_results.py`; a prompt is better", "M13 caption")
t = lit(t, "| official vs released MobileLLM-R1.5 | official vs MobileLLM-R1.5 K = 6, own log-probability or our selector | [TBD] | | | |",
        rowm13("official vs released MobileLLM-R1.5 (registered judge)", "official vs MobileLLM-R1.5 K = 6, own log-probability (registered judge)", A) + "\n" +
        rowm13("official vs released MobileLLM-R1.5 (registered judge)", "official vs MobileLLM-R1.5 K = 6, our selector (registered judge)", B), "M13")
t = lit(t, "   the two official test sets (rating files complete at 22:11 and 22:35 UTC), no other test set; the second registered judge,",
        "   the two official test sets (rating files complete at 22:11 and 22:35 UTC) and, on 25 November, of\n"
        "   MobileLLM-R1.5's two control sets (deviation 6), no other test set; the second registered judge,", "deviation 1a")
t = lit(t, "read every test pair twice on exchanged sides", "read every test pair but those of MobileLLM-R1.5's two control sets twice on exchanged sides", "deviation 1b")
t = lit(t, "(Section 7.5); the other\n   test sets were read under named titles only.", "(Section 7.5); the other\n   test sets the readers saw were read under named titles only.", "deviation 1c")
t = lit(t, "Table M4, a misattributed count and claims of blindness without their caveats; those in the paper are corrected.\n",
        "Table M4, a misattributed count and claims of blindness without their caveats; those in the paper are corrected.\n"
        "After that check, on 25 November, the rows, columns and sentences of MobileLLM-R1.5's two K = 6 control arms\n"
        "were written into both language versions by one script (`patch_mobctl.py`) from the outputs of\n"
        "`mob_ctl_results.py`, `mob_ctl_exact.py` and `judge_pass_facts.py`. No agent reviewed them (the user had asked\n"
        "that no further agents be used), so these numbers have the scripts' checks only: the recount reproduces the\n"
        "published column of the released preset, the summary reproduces the harness's score reports and its `compare`\n"
        "command, and every rating record carries its pair set's commitment.\n", "appendix L")
open(en_out, "w", encoding="utf-8").write(t)
print("EN: [TBD] left:", len(re.findall(r"\[TBD", t)))

# ------------------------------------------------------------------ Chinese
z = open(zh_in, encoding="utf-8").read()
z = lit(z, "仍為 [TBD] 的項目：MobileLLM-R1.5 的兩個 K = 6\n對照組，仍在生成中（22:02 UTC 時已生成 200 題中的 60 題；預計於 11 月 25 日 00:30 UTC 之後完成），以及作者名單。",
        "MobileLLM-R1.5 的兩個 K = 6\n對照組已於 11 月 25 日補上（00:25 UTC 生成完成；只由註冊評審與精確計分器評判，沒有 Opus 判讀：第 7.2 節，偏離 6）。尚未補上的項目：作者名單。", "zh status")
z = lit(z, "（MobileLLM-R1.5 的六樣本組：[TBD]）；事先註冊的評審", "；事先註冊的評審", "zh abstract 1")
z = lit(z, "在我們的評測框架中以 MobileLLM-R1.5 建議的預設集生成時有 63% 退化；我們在滿足限制上的優勢（48 題中有",
        "在我們的評測框架中以 MobileLLM-R1.5 建議的預設集生成時有 63% 退化：每則回覆取六個樣本時，退化的是 25% 或\n"
        f"20%，而事先註冊的評審（這兩組唯一的讀者）偏好我們模型的幅度變小（{w1} 比 {l1} 與 {w2} 比 {l2}；分數 {sA} 與 {sB}，原為\n0.635）；我們在滿足限制上的優勢（48 題中有", "zh abstract 2")
z = lit(z, "包括兩個讓 Falcon-H1-Tiny\n   取六個樣本的對照組（MobileLLM-R1.5 的：[TBD]）；",
        "包括讓每個對手各取六個樣本的對照組\n   （Falcon-H1-Tiny 的兩組由 Opus 讀者判讀，MobileLLM-R1.5 的兩組只由註冊評審評判）；", "zh intro")
z = rx(z, r"MobileLLM-R1\.5 的兩組尚不存在 \[TBD: [^\]]*\]。",
       "MobileLLM-R1.5 的兩組是之後才生成的（於 11 月 25 日 00:25 UTC 完成：三個 CPU 執行階段與開發容器分別生成題目順序中互不重疊的切片，再合併快取；"
       "在兩個執行階段上各生成一次的 48 個樣本，文字全部相同），而且沒有 Opus 判讀：使用者在 00:07 UTC 要求不再使用任何代理，"
       f"因此它們只由註冊評審（於 {JUDGE_END} UTC 評完）與精確計分器評判，它們的列與其他各組的 Opus 列不是同一類。", "zh deviation 6")
z = rx(z, r"\| 正式模型對 MobileLLM-R1\.5，K = 6，自身對數機率或我們的選擇器 \| Opus \| – \| \[TBD: [^\]]*\] \|  \|  \|  \|  \|  \|",
       row7("正式模型對 MobileLLM-R1.5，K = 6，自身對數機率", A, JQZ) + "\n" + row7("正式模型對 MobileLLM-R1.5，K = 6，我們的選擇器", B, JQZ), "zh table 7")
z = lit(z, "| MobileLLM-R1.5 `neutral` | MobileLLM-R1.5，K = 6（兩種規則） |", "| MobileLLM-R1.5 `neutral` | MobileLLM-R1.5，K = 6：自身對數機率；我們的選擇器 |", "zh table 8 head")
i = z.index("| MobileLLM-R1.5 `neutral` | MobileLLM-R1.5，K = 6：自身對數機率；我們的選擇器 |")
j = z.index("\n\n", i)
seg = z[i:j]
assert seg.count("| [TBD] |") == 4, seg.count("| [TBD] |")
for c in t8:
    seg = seg.replace("| [TBD] |", "| " + c + " |", 1)
z = z[:i] + seg + z[j:]
z = lit(z, "它在讀者判讀中的落敗，很大程度上是這些回覆的落敗。",
        "它在讀者判讀中的落敗，很大程度上是這些回覆的落敗。**每則回覆取六個樣本時，它的退化少得多**："
        f"以自身對數機率保留時，252 則回覆中有 {xa['degenerate']} 則退化，以我們的選擇器保留時有 {xb['degenerate']} 則（{xa['prompts_with_degenerate']} 與 {xb['prompts_with_degenerate']} 道題目），"
        f"而它的數學與限制數字留在原處（30 道中 {xa['math_correct']} 與 {xb['math_correct']} 道；48 道中 {xa['con_all_ok']} 與 {xb['con_all_ok']} 道：我們的選擇器沒有像提升 Falcon-H1-Tiny 那樣提升它滿足限制的題數）。"
        f"這兩組只由註冊評審判讀（偏離 6）。它仍把較多配對判給我們的模型（{w1} 比 {l1} 與 {w2} 比 {l2}；分數 {sA} {ciA} 與 {sB} {ciB}，對發布預設集為 0.635），"
        f"而逐題來看，我們對這兩組的分數低於對發布預設集的分數，平均低 {mdA}（符號檢定 {peq(vA['p'])}）與 {mdB}（{peq(vB['p'])}；表 M13）："
        "我們對 MobileLLM-R1.5 的領先，有一部分是對它單一樣本的領先。", "zh 7.6 sentence")
z = lit(z, "（用一個粒子時為 99 比 16；MobileLLM-R1.5 的六樣本組 [TBD]）",
        f"（用一個粒子時為 99 比 16；對上取六個樣本的 MobileLLM-R1.5，只有註冊評審判讀，為 {w1} 比 {l1} 與 {w2} 比 {l2}，領先較小）", "zh 7.10")
z = lit(z, "而 MobileLLM-R1.5 的對照組為 [TBD]；", "而 MobileLLM-R1.5 的六樣本對照組（對上它們，我們的領先較小）只由註冊評審判讀；", "zh section 8")
z = lit(z, "（64 個中的 59 個；MobileLLM-R1.5 的六樣本組：[TBD]）",
        f"（64 個中的 59 個；對上每則回覆取六個樣本的 MobileLLM-R1.5，則是 {w1 + l1} 個中的 {w1} 個與 {w2 + l2} 個中的 {w2} 個，它給我們模型的分數從 0.635 降到 {sA} 與 {sB}）", "zh conclusion")
z = lit(z, "保留正向串流的紀錄）。第四個 CPU 執行階段遭供應商拒絕",
        f"保留正向串流的紀錄）。MobileLLM-R1.5 的兩個對照配對集於 11 月 25 日 00:28 至 {JUDGE_END} UTC 在三個 CPU 執行階段上評審："
        "每個執行階段兩條 4 執行緒串流，各負責配對順序中的一段索引範圍；01:39 起，隨著核心空出，另有三條輔助串流從某一段範圍的另一端往回評"
        f"（平均每對 {F['seconds_mean']:.1f} s；有 {F['twice']} 對被兩條串流各評了一次，因為位在另一個執行階段上的輔助串流看不到該執行階段的紀錄："
        f"這 {TD['same_verdict']} 對的兩筆紀錄判定全部相同，保留先合併的那一筆）。三個 CPU 執行階段於 01:56 UTC 歸還；"
        "開發容器在 01:33 UTC 又重新啟動了一次，沒有影響這些串流。第四個 CPU 執行階段遭供應商拒絕", "zh appendix operations")
z = lit(z, "以及 MobileLLM-R1.5 各 K = 6 組的佔位列）。", "以及 MobileLLM-R1.5 各 K = 6 組的列與欄，於 11 月 25 日依 `mob_ctl_results.py` 與 `mob_ctl_exact.py` 的輸出補上）。", "zh M preamble 1")
z = lit(z, "final-prefopt-vs-mobilellm：200 筆紀錄，最後寫入於 22:35:20）。",
        f"final-prefopt-vs-mobilellm：200 筆紀錄，最後寫入於 22:35:20）；它對 MobileLLM-R1.5 兩個對照配對集（final-prefopt-vs-mobilellm-k6lp 與 -k6ds，各 200 筆紀錄）的評審於 11 月 25 日 {JUDGE_END} UTC 結束。", "zh M preamble 2")
z = lit(z, "`-vs-falcon-k6ds`、`-vs-mobilellm`、`final-k1-vs-falcon`、", "`-vs-falcon-k6ds`、`-vs-mobilellm`、`-vs-mobilellm-k6lp`、`-vs-mobilellm-k6ds`、`final-k1-vs-falcon`、", "zh M1 caption")
z = rx(z, r"\| MobileLLM-R1\.5，K = 6（兩種規則） \| – \| \[TBD: [^\]]*\] \| \| \| \| \| \| \|",
       rowm1("MobileLLM-R1.5，K = 6，以自身對數機率保留", A, "`qwen3-4b-2507`（註冊評審），全部評完；無 Opus 判讀") + "\n" +
       rowm1("MobileLLM-R1.5，K = 6，以我們的選擇器保留", B, "`qwen3-4b-2507`（註冊評審），全部評完；無 Opus 判讀"), "zh M1")
z = lit(z, "| MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` |\n|---|---|---|---|---|---|---|---|---|\n| 數學答案正確（共 30 題） |",
        "| MobileLLM-R1.5 `recommended` | MobileLLM-R1.5 `neutral` | MobileLLM-R1.5 K = 6，自身對數機率 | MobileLLM-R1.5 K = 6，我們的選擇器 |\n|---|---|---|---|---|---|---|---|---|---|---|\n| 數學答案正確（共 30 題） |", "zh M4 head")
z = add_cols(z, "| MobileLLM-R1.5 K = 6，自身對數機率 | MobileLLM-R1.5 K = 6，我們的選擇器 |", list(zip(("數學答案正確（共 30 題）", "每項限制都滿足的帶限制題目（共 48 題）", "滿足的單項限制（共 56 項）", "退化回覆（共 252 則）", "… 在 token 上限處截斷",
                          "… 結尾陷入循環；重複的行", "… 空白；未閉合的程式碼圍欄；角色標記", "制式拒答；制式迴避（共 252 則）",
                          "每則回覆字數：上中位數（252 則中的第 127 則）；平均", "平均 4-gram 重複率"), zip(m4a, m4b))), "zh M4")
z = lit(z, "MobileLLM-R1.5 的兩個 K = 6 組：[TBD]（第 7.2 節，偏離 6）。",
        "MobileLLM-R1.5 兩個 K = 6 組（第 7.2 節，偏離 6）的欄是在表建好之後才補上的：由 `mob_ctl_exact.py` 從這兩個 run 在 `study/proxy/reports/exact/` 的報告的逐輪紀錄重新計數，"
        f"同一支腳本也從 `recommended` 那個 run 的報告重現了它的欄；200 道題目中分別有 {xa['prompts_with_degenerate']} 與 {xb['prompts_with_degenerate']} 道至少有一則退化回覆（發布預設集為 139 道）。", "zh M4 note")
z = lit(z, "| MobileLLM-R1.5，K = 6（兩種規則） | – | [TBD] | | | |",
        rowm11("MobileLLM-R1.5，K = 6，自身對數機率", A, "`qwen3-4b-2507`") + "\n" + rowm11("MobileLLM-R1.5，K = 6，我們的選擇器", B, "`qwen3-4b-2507`"), "zh M11")
z = lit(z, "MobileLLM-R1.5 `recommended`：11 與 8；MobileLLM-R1.5 `neutral`：10 與 8 | – | – |",
        f"MobileLLM-R1.5 `recommended`：11 與 8；MobileLLM-R1.5 `neutral`：10 與 8；MobileLLM-R1.5 K = 6，自身對數機率：{xa9['math_correct']} 與 {xa9['con_all_ok']}；"
        f"MobileLLM-R1.5 K = 6，我們的選擇器：{xb9['math_correct']} 與 {xb9['con_all_ok']} | – | – |", "zh M11 exact")
z = lit(z, "等同 `python -m bes.eval.proxy compare --a A --b B --source judge-opus`；若我們",
        "等同 `python -m bes.eval.proxy compare --a A --b B --source judge-opus`，標明註冊評審的兩列則用 `--source judge-qwen3-4b-2507-3605803b`（`mob_ctl_results.py`）；若我們", "zh M13 caption")
z = lit(z, "| 正式模型對已發布的 MobileLLM-R1.5 | 正式模型對 MobileLLM-R1.5 K = 6，自身對數機率或我們的選擇器 | [TBD] | | | |",
        rowm13("正式模型對已發布的 MobileLLM-R1.5（註冊評審）", "正式模型對 MobileLLM-R1.5 K = 6，自身對數機率（註冊評審）", A) + "\n" +
        rowm13("正式模型對已發布的 MobileLLM-R1.5（註冊評審）", "正式模型對 MobileLLM-R1.5 K = 6，我們的選擇器（註冊評審）", B), "zh M13")
z = lit(z, "以及兩個正式測試配對集的全部 200 個配對（評分檔案於 22:11 與 22:35 UTC 完成），沒有判讀其他測試配對集；",
        "以及兩個正式測試配對集的全部 200 個配對（評分檔案於 22:11 與 22:35 UTC 完成），並於 11 月 25 日判讀了 MobileLLM-R1.5 兩個對照配對集的全部配對（偏離 6），沒有判讀其他測試配對集；", "zh deviation 1a")
z = lit(z, "將每個測試配對以左右互換的方式判讀兩次（附錄 J）", "將 MobileLLM-R1.5 兩個對照配對集以外的每個測試配對以左右互換的方式判讀兩次（附錄 J）", "zh deviation 1b")
z = lit(z, "（第 7.5 節）；其他測試配對集只在具名標題下判讀過。", "（第 7.5 節）；讀者判讀過的其他測試配對集，只在具名標題下判讀過。", "zh deviation 1c")
z = lit(z, "以及沒有附上但書的盲測宣稱；凡出現在本文中的都已更正。\n",
        "以及沒有附上但書的盲測宣稱；凡出現在本文中的都已更正。"
        "在那次查核之後，11 月 25 日，MobileLLM-R1.5 兩個 K = 6 對照組的列、欄與句子，由一支腳本（`patch_mobctl.py`）依 `mob_ctl_results.py`、`mob_ctl_exact.py` 與 `judge_pass_facts.py` 的輸出寫入兩個語言版本。"
        "沒有任何代理審查過這些內容（使用者已要求不再使用代理），因此這些數字只經過腳本的檢查：重新計數重現了發布預設集已公布的那一欄，摘要重現了評測框架的分數報告與其 `compare` 指令的結果，而每一筆評分紀錄都帶有其配對集的承諾值。\n", "zh appendix L")
open(zh_out, "w", encoding="utf-8").write(z)
print("ZH: [TBD] left:", len(re.findall(r"\[TBD", z)))
