import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from common import MODEL, Weights, chat_prompt, load_calibration
from envs import load, replay

L, N, SEEDS = 16, 8168, range(20000, 20050)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.path.expanduser("~/.cache/act-range"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import vllm_patches
    from vllm import LLM, SamplingParams
    from vllm.v1.capture.consumers.filesystem.reader import read_request

    tok = AutoTokenizer.from_pretrained(MODEL)
    items = []
    for rec in load(a.episodes):
        if rec["cond"].get("steer") or rec["seed"] not in SEEDS:
            continue
        setting = "solo" if rec["n_agents"] == 1 else ("free" if not rec["cond"].get("confed") else "team")
        for t, msgs in replay(rec):
            if t["agent"] != 0:
                continue
            p = chat_prompt(tok, msgs)
            items.append((dict(game=rec["game"], setting=setting, history=rec["cond"]["history"], seed=rec["seed"],
                               round=t["round"] + 1, n_prompt=len(tok(p).input_ids)), p + t["text"]))
    print(len(items), "sequences", flush=True)
    tag, root = "cn-actrange", a.scratch
    shutil.rmtree(os.path.join(root, tag), ignore_errors=True)
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, max_num_seqs=256, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=False,
              capture_consumers=[{"name": "filesystem", "params": {"root": root, "default_layout": "packed", "fsync": False}}],
              **vllm_patches.EXTRA)
    params = [SamplingParams(max_tokens=1, temperature=0.0, capture={"filesystem": {
        "request_id": f"t{k:07d}", "tag": tag, "layout": "packed", "hooks": {"mlp_in": [L]}, "positions": "all_prompt"}})
        for k in range(len(items))]
    llm.generate([x[1] for x in items], params, use_tqdm=False)
    pub = lambda k: os.path.exists(os.path.join(root, tag, f"t{k:07d}", "packed.json"))
    t0 = time.time()
    while sum(not pub(k) for k in range(len(items))) and time.time() - t0 < 600:
        time.sleep(5)
    W = Weights(MODEL)
    g, u = W.get(L, "gate_proj")[N], W.get(L, "up_proj")[N]
    rows = []
    for k, (meta, _) in enumerate(items):
        if not pub(k):
            continue
        (x,) = [np.asarray(e.array) for e in read_request(os.path.join(root, tag, f"t{k:07d}")).values()]
        if x.dtype == np.uint16:
            x = (x.astype(np.uint32) << 16).view(np.float32)
        x = x.astype(np.float32).reshape(-1, x.shape[-1])
        gx, ux = x @ g, x @ u
        act = gx / (1 + np.exp(-gx)) * ux
        pos = np.arange(len(act))
        for kind, m in (("prompt", pos < meta["n_prompt"]), ("generated", pos >= meta["n_prompt"]),
                        ("last_prompt", pos == meta["n_prompt"] - 1)):
            rows += [dict(meta, position=kind, act=float(v)) for v in act[m]]
    import pandas as pd
    d = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    m = load_calibration()[str(L)]
    wn = float(np.linalg.norm(W.down_col(L, N)))
    shifts = {f"{dose:g}": dose * m / wn for dose in (0.05, 0.10, 0.15, 0.25)}
    qs = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]
    t = d.groupby(["game", "setting", "position"]).act.describe(percentiles=qs[1:-1])
    t.reset_index().to_csv(f"{a.out}/act_quantiles.csv", index=False)
    pooled = d.groupby("position").act.describe(percentiles=qs[1:-1])
    pooled.reset_index().to_csv(f"{a.out}/act_quantiles_pooled.csv", index=False)
    sd = d.groupby("position").act.std()
    rng_ = d.groupby("position").act.agg(lambda v: v.max() - v.min())
    out = {"activation_equivalent_shift": shifts, "resid_median_L16": m, "w_down_norm": wn,
           "shift_over_sd": {p: {k: v / sd[p] for k, v in shifts.items()} for p in sd.index},
           "shift_over_observed_range": {p: {k: v / rng_[p] for k, v in shifts.items()} for p in rng_.index},
           "n_values": d.groupby("position").size().to_dict(), "sequences": len(items)}
    json.dump(out, open(f"{a.out}/act_summary.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(pooled.round(4).to_string(), "\n", json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
