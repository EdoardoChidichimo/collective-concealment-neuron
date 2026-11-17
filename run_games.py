import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from common import HOOK, MODEL, SAMPLING, Weights, chat_prompt, load_calibration, stable_seed, steer_vector
from envs import make_game


def plan(a, conditions):
    jobs = []
    for c in conditions:
        for game in a.games.split(","):
            for n in map(int, str(a.n_agents).split(",")):
                for lvl in map(int, str(a.temptation).split(",")):
                    for s in range(a.seed0, a.seed0 + a.episodes):
                        jobs.append((c, game, n, lvl, s))
    return jobs


def job_key(cond_id, game, n, lvl, seed):
    return (cond_id, game, n, lvl, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="audit,trial")
    ap.add_argument("--n-agents", default="5")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--temptation", default="4")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--calib", default="results/calib.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=SAMPLING["max_tokens"])
    a = ap.parse_args()

    conditions = [{"id": "base"}]
    if a.conditions:
        with open(a.conditions) as f:
            conditions = json.load(f)
    steering = any(c.get("steer") for c in conditions)
    k, K = map(int, a.shard.split("/"))
    jobs = plan(a, conditions)[k::K]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                r = json.loads(line)
                done.add(job_key(r["condition"], r["game"], r["n_agents"], r.get("temptation", 0), r["seed"]))
    jobs = [j for j in jobs if job_key(j[0]["id"], *j[1:]) not in done]
    print(f"{a.games} n={a.n_agents} T={a.temptation}: {len(jobs)} episodes to run ({len(done)} done)", flush=True)
    if not jobs:
        return

    from transformers import AutoTokenizer
    import vllm_patches
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    assert os.path.realpath(a.model) == os.path.realpath(MODEL), "set CN_MODEL to the model being run"
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=6144, max_num_seqs=a.max_num_seqs,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0}, **vllm_patches.EXTRA)
    if steering:
        kw.update(enable_steering=True, max_steering_configs=512)
    llm = LLM(**kw)
    weights = Weights(a.model) if steering else None
    resid = load_calibration(a.calib) if steering else None
    vec_cache = {}

    def steer_kw(spec):
        key = json.dumps(spec, sort_keys=True)
        if key not in vec_cache:
            vec_cache[key] = steer_vector(spec, weights, resid)
        vec, scale = vec_cache[key]
        return {"steering_vectors": {HOOK: {spec["layer"]: {"vector": vec, "scale": scale}}}}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as fout:
        for c0 in range(0, len(jobs), a.chunk):
            t0, n_tok = time.time(), 0
            eps = []
            for cond, game, n, lvl, seed in jobs[c0:c0 + a.chunk]:
                g = make_game(game, seed, n, lvl, cond)
                placement = cond.get("placement", "all")
                if not cond.get("steer"):
                    steered = []
                elif placement == "focal":
                    steered = [0]
                else:
                    steered = list(range(n))
                eps.append({"cond": cond, "seed": seed, "game": g, "steered": steered, "turns": []})
            while not all(e["game"].done for e in eps):
                live = [e for e in eps if not e["game"].done]
                prompts, params, owners = [], [], []
                for e in live:
                    g = e["game"]
                    for i in range(g.n):
                        if g.CONFED.get(g.confed.get(i), {}).get("silent"):
                            continue
                        prompts.append(chat_prompt(tok, g.messages(i)))
                        p = dict(SAMPLING, max_tokens=a.max_tokens, seed=stable_seed(e["cond"]["id"], g.name, g.n, g.temptation, e["seed"], g.t, i))
                        if i in e["steered"]:
                            p.update(steer_kw(e["cond"]["steer"]))
                        params.append(SamplingParams(**p))
                        owners.append((e, i))
                outs = llm.generate(prompts, params, use_tqdm=False)
                per_ep = {}
                for (e, i), o in zip(owners, outs):
                    per_ep.setdefault(id(e), (e, {}))[1][i] = (o.outputs[0].text, o.outputs[0].finish_reason,
                                                               len(o.outputs[0].token_ids))
                    n_tok += len(o.outputs[0].token_ids)
                for e, texts in per_ep.values():
                    g = e["game"]
                    gen = [texts.get(i, ("", None, 0)) for i in range(g.n)]
                    texts = [x[0] for x in gen]
                    r = g.t
                    acts = g.step(texts)
                    for i, (txt, act) in enumerate(zip(texts, acts)):
                        e["turns"].append({"round": r, "agent": i, "text": txt, "act": act,
                                           "finish": gen[i][1], "n_tok": gen[i][2],
                                           "label": g.turn_label(i, act), "steered": i in e["steered"],
                                           "confed": g.confed.get(i)})
            for e in eps:
                g = e["game"]
                rec = {"game": g.name, "n_agents": g.n, "temptation": g.temptation, "seed": e["seed"], "condition": e["cond"]["id"],
                       "cond": e["cond"], "steered_agents": e["steered"], "turns": e["turns"],
                       "history": g.history, "history_policy": g.history_policy,
                       "match_scenario": g.match_scenario, "max_tokens": a.max_tokens,
                       "model": os.path.basename(os.path.realpath(a.model)), "metrics": g.metrics(), "parse_failures": g.parse_failures}
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            dt = time.time() - t0
            print(f"{min(c0 + a.chunk, len(jobs))}/{len(jobs)} episodes, {n_tok / dt:,.0f} gen tok/s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
