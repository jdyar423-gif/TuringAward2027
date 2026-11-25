"""After the registered judge's pass on MobileLLM-R1.5's two K = 6 control arms: write the results
into docs/CHANGES.md (section 19.10), README.md (both languages), the paper's FACTS_2.md and
CLAIMS.md. Reads N/peek/mob_ctl_results.json, mob_ctl_exact.json, judge_pass_facts.json and the two
score reports.
    JUDGE_END=HH:MM NOW=HH:MM python finalize_docs.py [--check]     (--check: no file is written)
"""
import json
import os
import sys

N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"
REPO = "/home/user/turingaward2027"
SRC = "judge-qwen3-4b-2507-3605803b"
CHECK = "--check" in sys.argv
END = os.environ.get("JUDGE_END", "HH:MM")
NOW = os.environ.get("NOW", "HH:MM")
R = json.load(open(N + "/peek/mob_ctl_results.json"))
X = json.load(open(N + "/peek/mob_ctl_exact.json"))
F = json.load(open(N + "/peek/judge_pass_facts.json"))
A, B = R["k6lp"], R["k6ds"]
rep = {t: json.load(open(f"{N}/study/proxy/reports/ab/final-prefopt-vs-mobilellm-{t}--{SRC}.json")) for t in ("k6lp", "k6ds")}
if not CHECK:
    assert A["records"] == 200 and B["records"] == 200 and END != "HH:MM" and NOW != "HH:MM"
    assert all(s["all_pairs_rated"] for s in F["sets"].values())


def peq(p):
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def pz(p):
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def fpz(x):
    return f"{x:+.3f}".replace("-", "−")


def wlt(r, sep="／"):
    a = r["all"]
    return f"{a['w']}{sep}{a['l']}{sep}{a['t']}"


def cats(rp, sep="／"):
    zh = {"chat": "對話", "code": "程式", "instruct": "指令", "math": "數學", "writing": "寫作"}
    return "、".join(f"{zh[k]} {v['wins']}{sep}{v['losses']}{sep}{v['ties']}" for k, v in rp["by_category"].items())


def cats_en(rp):
    return ", ".join(f"{k} {v['wins']} / {v['losses']} / {v['ties']}" for k, v in rp["by_category"].items())


def turns(rp, sep="／"):
    s, m = rp["by_turns"]["single_turn"], rp["by_turns"]["multi_turn"]
    return f"單輪 {s['wins']}{sep}{s['losses']}{sep}{s['ties']}、兩輪 {m['wins']}{sep}{m['losses']}{sep}{m['ties']}"


def edit(path, pairs):
    t = open(path, encoding="utf-8").read()
    for name, old, new in pairs:
        n = t.count(old)
        assert n == 1, f"{path}: {name}: {n} matches"
        t = t.replace(old, new)
    if not CHECK:
        open(path, "w", encoding="utf-8").write(t)
    print(("checked " if CHECK else "wrote ") + path, len(pairs), "edits")


a, b = A["all"], B["all"]
a9, b9 = A["189"], B["189"]
vA, vB = A["vs_released_all"], B["vs_released_all"]
wA, wB = A["vs_released_189"], B["vs_released_189"]
xa, xb = X["k6lp"]["all"], X["k6ds"]["all"]
assert a["w"] > a["l"] and b["w"] > b["l"] and vA["mean_diff"] > 0 and vB["mean_diff"] > 0
sp = F["seconds_mean"]
TD = F["twice_detail"]
assert TD["pairs"] == F["twice"] and TD["same_verdict"] == TD["pairs"] and F["meta_commitments_match"]

# ---------------------------------------------------------------- docs/CHANGES.md, section 19.10
c_ctl_old = "這兩組的盲測由註冊的自動評審來判（見節末）；使用者在 00:07 指示不再使用子代理，所以這兩組沒有 Opus 讀者的判讀。"
c_ctl_new = ("這兩組的盲測只由註冊的自動評審來判（使用者在 00:07 指示不再使用子代理，所以沒有 Opus 讀者的判讀）："
             f"**{wlt(A)}**（分數 {a['score']} {a['ci']}）與 **{wlt(B)}**（{b['score']} {b['ci']}），同一位評審對原設定判的是 59／5／136（0.635）。"
             f"逐題配對，我方對這兩組的分數比對原設定低 {vA['mean_diff_3']}（{vA['a_better']} 題較高、{vA['b_better']} 題較低，{pz(vA['p'])}）與 {vB['mean_diff_3']}"
             f"（{vB['a_better']} 比 {vB['b_better']}，{pz(vB['p'])}）：對 MobileLLM 的領先有一部分來自「我方 6 個樣本、對方 1 個樣本」，讓它也取 6 個樣本之後我方仍然領先，幅度變小。")
c_judge_old = "留主檔（正向那一路）的那一筆；`judge_pair` 本身、模型檔、提示與兩個順序都不變，只有排程不同；每筆紀錄仍帶配對集的承諾值與該對的摘要）。"
c_judge_new = (c_judge_old + "\n\n"
               f"**自動評審對 MobileLLM-R1.5 兩個 K = 6 對照組（25 日 {END} UTC 判完，各 200 對）。** 以它自己的對數機率挑：**{wlt(A)}**，分數 {a['score']} {a['ci']}，"
               f"決定性勝率 {a['decisive']} {a['dci']}，{pz(a['p_raw'])}；兩個順序一致 {100 * A['agree']:.1f}%，對首位回覆的偏好 {fpz(A['first_pos'])}；{A['clipped']} 對含被截斷的回覆。"
               f"分類：{cats(rep['k6lp'])}；{turns(rep['k6lp'])}。以我方的選擇器挑：**{wlt(B)}**，分數 {b['score']} {b['ci']}，決定性勝率 {b['decisive']} {b['dci']}，{pz(b['p_raw'])}；"
               f"兩個順序一致 {100 * B['agree']:.1f}%，對首位回覆的偏好 {fpz(B['first_pos'])}；{B['clipped']} 對含被截斷的回覆。分類：{cats(rep['k6ds'])}；{turns(rep['k6ds'])}。"
               f"只算 189 題時是 {a9['w']}／{a9['l']}／{a9['t']}（{a9['score']}）與 {b9['w']}／{b9['l']}／{b9['t']}（{b9['score']}）。"
               f"與同一位評審對原設定的結果逐題配對（200 題）：我方分數在原設定那組較高的有 {vA['a_better']} 題、較低 {vA['b_better']} 題、相同 {vA['same']} 題（平均差 {vA['mean_diff']:+.4f}，{pz(vA['p'])}），"
               f"以及 {vB['a_better']}、{vB['b_better']}、{vB['same']} 題（{vB['mean_diff']:+.4f}，{pz(vB['p'])}）；189 題是 {wA['a_better']}／{wA['b_better']}／{wA['same']}（{pz(wA['p'])}）與 {wB['a_better']}／{wB['b_better']}／{wB['same']}（{pz(wB['p'])}）。"
               f"判讀方式：三台 CPU 執行階段各兩路、每路 4 執行緒，六路各判配對順序中的一段（[0, 67)、[67, 134)、[134, 200)），01:39 起先空出來的核心再開反向的輔助路、從某一段的尾端往回判（共 {F['streams']} 個分段檔）；"
               f"平均每對 {sp:.1f} s；{F['twice']} 對被兩路各判了一次（輔助路在另一台執行階段上時看不到那一台的紀錄），這 {TD['same_verdict']} 對的兩筆紀錄判定全部相同，留先合併的那一筆（{TD['kept_forward']} 筆來自正向、{TD['kept_helper']} 筆來自輔助路）。"
               "每筆紀錄帶的承諾值與配對集的相符。這兩組沒有 Opus 讀者的判讀，所以它們的數字只能與自動評審對原設定的 59／5／136 相比，不能與 Opus 讀者的 116／15／69 相比。")
c_concl_old = "註冊的自動評審給出同樣的方向（對 Falcon 12 比 61，對 MobileLLM 59 比 5）。"
c_concl_new = (c_concl_old + f"讓 MobileLLM 也取 6 個樣本之後（只有自動評審判讀），我方仍然領先、幅度變小：{a['w']} 比 {a['l']} 與 {b['w']} 比 {b['l']}，分數從 0.635 降到 {a['score']} 與 {b['score']}。")
c_open_old_head = "**尚未完成的項目（25 日 00:35 UTC；完成後直接改寫本節相應的段落）。** 註冊的自動評審對 MobileLLM-R1.5 兩個 K = 6 對照組的判讀（各 200 對，六路分段，00:31 開始，預計 01:40 前後完成）。"
c_open_new = (f"**25 日補上的項目（{NOW} UTC）。** 註冊的自動評審對 MobileLLM-R1.5 兩個 K = 6 對照組的判讀已完成（00:28 開始、{END} 判完；結果寫在上面「對照組說明了什麼」、「事先註冊的自動評審」與「誠實的結論」三段），"
              "論文中英兩版的相應表格與段落也已補上（表 7、表 8、表 M1、M4、M11、M13，第 7.2、7.6、7.10、8、9 節）。開發容器在 25 日 01:33 又重啟了一次：草稿目錄與憑證都在，保活程序重開，執行階段上的評審不受影響。"
              "三台 CPU 執行階段在 01:56 歸還（當時還在跑的幾路只會重判已有紀錄的配對）。至此本節沒有未完成的項目；論文裡唯一留白的是作者名單。")
changes = [("control paragraph", c_ctl_old, c_ctl_new), ("judge paragraph", c_judge_old, c_judge_new),
           ("conclusion", c_concl_old, c_concl_new), ("open items", c_open_old_head, c_open_new)]
edit(REPO + "/docs/CHANGES.md", changes)

# ---------------------------------------------------------------- README.md
r_zh_old = "MobileLLM-R1.5 在這個評測框架、它的建議設定下有 62.7% 的回覆是退化的，那一項領先的門檻不高；"
r_zh_new = (r_zh_old + f"讓它也取 6 個樣本再挑一則時，退化降到 25.4% 與 20.2%，事先註冊的自動評審（這兩組唯一的評審）仍偏好本模型、幅度變小（{wlt(A)} 與 {wlt(B)}，它對原設定判的是 59／5／136）；")
r_en_old = ("62.7% of MobileLLM-R1.5's replies to these prompts are degenerate under its recommended preset in\n"
            "this harness, so that lead clears a low bar. Giving Falcon")
r_en_new = ("62.7% of MobileLLM-R1.5's replies to these prompts are degenerate under its recommended preset in\n"
            "this harness, so that lead clears a low bar. Given six samples per reply it degenerates in 25.4% or\n"
            "20.2% of them, and the pre-registered automatic judge, the only reader of those two arms, still\n"
            f"prefers our model, by less ({wlt(A, ' / ')} and {wlt(B, ' / ')}, where it gave 59 / 5 / 136 against\n"
            "the recommended preset). Giving Falcon")
edit(REPO + "/README.md", [("zh", r_zh_old, r_zh_new), ("en", r_en_old, r_en_new)])

# ---------------------------------------------------------------- paper/FACTS_2.md
facts = f"""
## Added {NOW} UTC (25 Nov): the registered judge's FULL pass on MobileLLM-R1.5's two control arms
- Source `{SRC}`, both orders, strict agreement; 200 records in each rating file, each carrying its
  pair set's commitment; streams started {F['first_start']} UTC, both sets complete at {END} UTC
  ({F['streams']} shard files on three CPU runtimes: six forward streams over [0,67), [67,134), [134,200)
  of each set, then from 01:39 three reverse helpers on the cores that came free); {sp:.1f} s per pair
  on average over the 400 kept records; {F['twice']} pairs judged by two streams (a helper on another
  runtime does not see that runtime's records): the two records give the same verdict on all
  {TD['same_verdict']} (identical probabilities in both orders on {TD['same_probs']}); the record merged first was kept
  ({TD['kept_forward']} from a forward stream, {TD['kept_helper']} from a helper). The three CPU runtimes were given back at
  01:56 UTC (the streams still running would only have re-judged rated pairs). No Opus reading of
  these two sets.
- K = 6 kept by its own log-probability (`final-prefopt-vs-mobilellm-k6lp`): **{wlt(A, ' / ')}**, score
  {a['score']} {a['ci']}, decisive win rate {a['decisive']} {a['dci']}, sign test {peq(a['p_raw'])}; orders agree
  {100 * A['agree']:.1f}%, first-position preference {fpz(A['first_pos'])}, {A['clipped']} pairs clipped; categories
  {cats_en(rep['k6lp'])}; 189 prompts {a9['wlt']}, score {a9['score']} {a9['ci']}, {peq(a9['p_raw'])}.
- K = 6 kept by our selector (`final-prefopt-vs-mobilellm-k6ds`): **{wlt(B, ' / ')}**, score {b['score']} {b['ci']},
  decisive win rate {b['decisive']} {b['dci']}, sign test {peq(b['p_raw'])}; orders agree {100 * B['agree']:.1f}%,
  first-position preference {fpz(B['first_pos'])}, {B['clipped']} pairs clipped; categories
  {cats_en(rep['k6ds'])}; 189 prompts {b9['wlt']}, score {b9['score']} {b9['ci']}, {peq(b9['p_raw'])}.
- The same judge on the released preset: 59 / 5 / 136, score 0.635 [0.566, 0.699] (88 pairs clipped).
  Paired, per prompt, our score against the released preset minus our score against the arm
  (`mob_ctl_results.py`; the same test as `python -m bes.eval.proxy compare`): own log-probability
  {vA['a_better']} higher, {vA['b_better']} lower, {vA['same']} same, mean {vA['mean_diff']:+.4f}, sign test {peq(vA['p'])} (189 prompts: {wA['a_better']}; {wA['b_better']};
  {wA['same']}, {peq(wA['p'])}); our selector {vB['a_better']}, {vB['b_better']}, {vB['same']}, mean {vB['mean_diff']:+.4f}, {peq(vB['p'])} (189: {wB['a_better']}; {wB['b_better']}; {wB['same']},
  {peq(wB['p'])}).
- Reading: given six samples per reply MobileLLM-R1.5 degenerates far less (64 and 51 of 252 replies
  against 158) and the registered judge still decides more pairs for our model, by less: part of
  the lead over MobileLLM-R1.5 is a lead over its single sample. These rows have no Opus reading,
  so they compare with the judge's 59 / 5 / 136, not with the readers' 116 / 15 / 69.
- The development container restarted at 01:33 UTC (scratch directories and credentials kept; the
  keep-awake task restarted at 01:34); the judge's streams on the runtimes were not touched.
- The paper (English and Chinese) is patched by `patch_mobctl.py` from `mob_ctl_results.json`,
  `mob_ctl_exact.json` and `judge_pass_facts.json` (in `evidence/N/peek/`); only the author list
  stays [TBD].
"""
fpath = N + "/paper/FACTS_2.md"
ft = open(fpath, encoding="utf-8").read()
old = "No Opus reading of these two sets (the user's instruction of 00:07). [TBD: result about 01:40.]"
assert ft.count(old) == 1
ft = ft.replace(old, "No Opus reading of these two sets (the user's instruction of 00:07). (Result: the section added below.)")
assert "the registered judge's FULL pass on MobileLLM-R1.5's two control arms" not in ft
if not CHECK:
    open(fpath, "w", encoding="utf-8").write(ft.rstrip("\n") + "\n" + facts)
print(("checked " if CHECK else "wrote ") + fpath)

# ---------------------------------------------------------------- paper/CLAIMS.md
rows = [
    ("C577", "7.2, I", "MobileLLM-R1.5's two K = 6 arms: how they were generated",
     "complete 00:12 (own log-probability) and 00:25 UTC (our selector) on 25 November; disjoint slices of the prompt order on three CPU runtimes and the development container, caches merged, runs assembled without generating; 48 of 48 samples generated on two runtimes identical",
     "`FACTS_2.md` 00:20 and 00:40 (25 Nov); `S/scripts/controls_slice_driver.py`, `ctl_local.py`, `ctl_assemble.py`; the runs' outputs in `N/study/proxy/outputs/mobilellm-f680f0533368-qwen3/`"),
    ("C578", "7.6, Tables 8, M4, M11", "exact scores of the two arms",
     f"own log-probability: maths {xa['math_correct']} of 30, all constraints {xa['con_all_ok']} of 48, single {xa['single_ok']} of 56, degenerate {xa['degenerate']} of 252 ({xa['truncated']} cut), words {xa['words_upper_median']}; {xa['words_mean']:.1f}, repeated-4-gram {xa['rep4_mean']:.4f}, {xa['prompts_with_degenerate']} prompts with a degenerate reply; our selector: {xb['math_correct']}, {xb['con_all_ok']}, {xb['single_ok']}, {xb['degenerate']} ({xb['truncated']}), {xb['words_upper_median']}; {xb['words_mean']:.1f}, {xb['rep4_mean']:.4f}, {xb['prompts_with_degenerate']}; on the 189 prompts maths {X['k6lp']['189']['math_correct']} and {X['k6ds']['189']['math_correct']} of 28, constraints {X['k6lp']['189']['con_all_ok']} and {X['k6ds']['189']['con_all_ok']} of 44",
     "`N/study/proxy/reports/exact/mobilellm-f680f0533368-qwen3__recommended-32f0c6d864-s0-test.json`, `…-aca173fa2c-s0-test.json` (`per_prompt`); `S/scripts/mob_ctl_exact.py` → `N/peek/mob_ctl_exact.json` (reproduces the `recommended` column)"),
    ("C579", "abstract, 7.6, 7.10, 9, Tables 7, M1", "the registered judge on the arm kept by its own log-probability",
     f"{a['wlt']}; score {a['score']} {a['ci']}; decisive {a['decisive']} {a['dci']}; sign test {peq(a['p_raw'])}; orders agree {100 * A['agree']:.1f}%; first-position {fpz(A['first_pos'])}; {A['clipped']} pairs clipped",
     f"`N/study/proxy/reports/ab/final-prefopt-vs-mobilellm-k6lp--{SRC}.json`; `S/scripts/mob_ctl_results.py` → `N/peek/mob_ctl_results.json`"),
    ("C580", "abstract, 7.6, 7.10, 9, Tables 7, M1", "the registered judge on the arm kept by our selector",
     f"{b['wlt']}; score {b['score']} {b['ci']}; decisive {b['decisive']} {b['dci']}; sign test {peq(b['p_raw'])}; orders agree {100 * B['agree']:.1f}%; first-position {fpz(B['first_pos'])}; {B['clipped']} pairs clipped",
     f"`N/study/proxy/reports/ab/final-prefopt-vs-mobilellm-k6ds--{SRC}.json`; `mob_ctl_results.json`"),
    ("C581", "Tables 7, M11", "the two arms on the 189 untouched prompts",
     f"{a9['wlt']}, score {a9['score']} {a9['ci']}, {peq(a9['p_raw'])}; {b9['wlt']}, score {b9['score']} {b9['ci']}, {peq(b9['p_raw'])}",
     "the reports' `per_prompt_score` without the 11 seen ids (`mob_ctl_results.py`)"),
    ("C582", "abstract, 7.6, 8, 9, Table M13", "our score against the released preset and against each arm, per prompt, registered judge",
     f"own log-probability: {vA['a_better']} higher against the released preset, {vA['b_better']} lower, {vA['same']} same, mean {vA['mean_diff']:+.4f}, sign test {peq(vA['p'])} (189: {wA['a_better']}; {wA['b_better']}; {wA['same']}, {peq(wA['p'])}); our selector: {vB['a_better']}; {vB['b_better']}; {vB['same']}, {vB['mean_diff']:+.4f}, {peq(vB['p'])} (189: {wB['a_better']}; {wB['b_better']}; {wB['same']}, {peq(wB['p'])})",
     f"`mob_ctl_results.json` (`vs_released_all`, `vs_released_189`); `python -m bes.eval.proxy compare --a final-prefopt-vs-mobilellm --b final-prefopt-vs-mobilellm-k6lp --source {SRC}` and `--b …-k6ds`"),
    ("C583", "7.2, I, M", "how and when the judge's pass on the two arms ran",
     f"streams started {F['first_start']} UTC on 25 November, both sets complete at {END} UTC; {F['streams']} shard files on three CPU runtimes (six forward streams over three index ranges of each set, three reverse helpers from 01:39 as cores came free); {sp:.1f} s per pair over the 400 kept records; {F['twice']} pairs judged by two streams, the same verdict in both records on all {TD['same_verdict']}, the record merged first kept ({TD['kept_forward']} forward, {TD['kept_helper']} helper); every record carries its pair set's commitment; runtimes given back 01:56 UTC; no Opus reading (the user's instruction of 00:07 UTC)",
     "`N/judge_remote/testq5`, `testq6` (`ranges.txt`, `shards/`, logs); `S/scripts/judge_shard.py`, `judge_shard_merge.py`, `judge_pass_facts.py` → `N/peek/judge_pass_facts.json`; `R/docs/CHANGES.md` §19.10"),
]
cpath = N + "/paper/CLAIMS.md"
ct = open(cpath, encoding="utf-8").read()
anchor = [ln for ln in ct.split("\n") if ln.startswith("| C576 |")]
assert len(anchor) == 1 and "| C577 |" not in ct
new_rows = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows)
ct = ct.replace(anchor[0], anchor[0] + "\n" + new_rows)
old = "Final pass (22:50 UTC): authors; MobileLLM-R1.5's two K = 6 control arms (in generation, 60 of 200 prompts at 22:02; C572). Everything else"
assert ct.count(old) == 1
ct = ct.replace(old, f"After the final pass (25 November, {NOW} UTC): the authors only. MobileLLM-R1.5's two K = 6 control arms are in (C577–C583: generated by 00:25 UTC, "
                     "exact scores, the registered judge's full pass; no Opus reading).\n\n"
                     "Final pass (22:50 UTC): authors; MobileLLM-R1.5's two K = 6 control arms (then in generation, 60 of 200 prompts at 22:02; C572). Everything else")
tail_anchor = [ln for ln in ct.split("\n") if ln.startswith("| M1 caption, Table 8 caption, I, J, E3, L (rows 15, 16, 18, 19, 21, 23, 30) |")]
assert len(tail_anchor) == 1
ct = ct.replace(tail_anchor[0], tail_anchor[0] + "\n| C550, C572, C452 (25 November, after the pass) | MobileLLM-R1.5's K = 6 control arms [TBD], with placeholder rows | "
                "generated, scored and judged by the registered judge only (C577–C583); the rows and columns of Tables 7, 8, M1, M4, M11, M13 and the sentences of the abstract, §1, §7.2, §7.6, §7.10, §8, §9 filled by `patch_mobctl.py` |")
if not CHECK:
    open(cpath, "w", encoding="utf-8").write(ct)
print(("checked " if CHECK else "wrote ") + cpath, len(rows), "rows")
