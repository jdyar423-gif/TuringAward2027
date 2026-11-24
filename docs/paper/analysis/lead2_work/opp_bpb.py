"""Review item 31: bits per UTF-8 byte of the opponents on the same held-out text as ours.

The text is that of the 64 web rows of the export's check (ClimbMix shard 00997, beyond the run's
download cap): every row is one document, opened by BOS and cut at 1,024 of our tokens; its
scored bytes are the bytes of the row's ordinary target tokens (our tokenizer is byte-level, so
the decoded row is exactly those bytes). Each opponent reads the same bytes as one sequence
after its own BOS and is scored on every one of its tokens; our model's number comes from the
per-position losses of heldout_eval.py. llama.cpp (llama-cpp-python, the same Q8_0 files the
evaluation uses) on CPU cores 2 and 3, float32 logits, no sampling.

    python -I opp_bpb.py ROWS.npz OUT.json [--rows N]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/turingaward2027")
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
MODELS = {"falcon_h1_tiny_90m_instruct": N / "models/Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf",
          "mobilellm_r1.5_140m": N / "models/mobilellm-r1.5-140m-q8_0.qwen3.gguf"}


def row_texts(npz, domain="web", limit=None):
    from bes.tokenizer import load
    tok = load()
    z = np.load(npz)
    t, w = z[f"{domain}/tokens"], z[f"{domain}/w"]
    out = []
    for i in range(t.shape[0] if limit is None else min(limit, t.shape[0])):
        y = t[i, 1:]
        keep = [int(a) for a, b in zip(y, w[i]) if b > 0 and not tok.is_special(int(a))]
        raw = tok.decode_bytes(keep)
        out.append(raw)
    return out


def score(model_path, texts, n_ctx=2048, threads=2):
    from llama_cpp import Llama
    llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_batch=512, n_threads=threads,
                n_threads_batch=threads, logits_all=True, verbose=False)
    res = []
    for i, raw in enumerate(texts):
        text = raw.decode("utf-8", errors="ignore")
        ids = llm.tokenize(text.encode("utf-8"), add_bos=True, special=False)
        if len(ids) > n_ctx:
            raise RuntimeError(f"row {i}: {len(ids)} tokens > n_ctx {n_ctx}")
        llm.reset()
        llm.eval(ids)
        logits = np.asarray(llm.scores[:len(ids)], dtype=np.float64)
        m = logits.max(axis=1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(logits - m).sum(axis=1)))
        nll = float(sum(lse[j] - logits[j, ids[j + 1]] for j in range(len(ids) - 1)))
        res.append({"row": i, "bytes": len(text.encode("utf-8")), "tokens_scored": len(ids) - 1, "nll_nats": nll,
                    "bos_added": ids[0] == llm.token_bos()})
    return res


def main():
    npz, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else None
    texts = row_texts(npz, limit=limit)
    result = {"rows": len(texts), "bytes_ours": int(sum(len(t) for t in texts)), "models": {}}
    for name, path in MODELS.items():
        t0 = time.time()
        r = score(path, texts)
        nll = sum(x["nll_nats"] for x in r)
        by = sum(x["bytes"] for x in r)
        result["models"][name] = {"bits_per_byte": nll / math.log(2) / by, "nats": nll, "bytes": by,
                                  "tokens_scored": sum(x["tokens_scored"] for x in r),
                                  "per_row": r, "seconds": round(time.time() - t0, 1)}
        print(f"{name}: {nll / math.log(2) / by:.4f} bits per byte over {by} bytes, "
              f"{sum(x['tokens_scored'] for x in r)} tokens, {time.time() - t0:.0f} s", flush=True)
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
