import argparse
import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from common import JUDGE, MODEL, chat_prompt
from envs import load, replay


def yn(text, key):
    m = re.findall(rf"(?im)^\s*{key}\s*:\s*(yes|no)", text)
    return None if not m else int(m[-1].lower() == "yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["quality", "nll"])
    ap.add_argument("--focal-only", action="store_true")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import vllm_patches
    from vllm import LLM, SamplingParams

    keys, reqs = [], []
    if a.stage == "quality":
        tok = AutoTokenizer.from_pretrained(JUDGE)
        for rec in load(a.episodes):
            for t, msgs in replay(rec):
                if t.get("confed") is not None or (a.focal_only and t["agent"] != 0):
                    continue
                keys.append((rec, t))
                reqs.append(chat_prompt_plain(tok, QUALITY.format(system=msgs[0]["content"], user=msgs[1]["content"],
                                                                  text=t["text"])))
        llm = LLM(model=JUDGE, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=a.gpu_mem, seed=0, **vllm_patches.EXTRA)
        outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=60), use_tqdm=True)
        rows = [{k.lower(): yn(o.outputs[0].text, k) for k in ("COHERENT", "GROUNDED", "DECIDED", "CONSISTENT", "INVENTED")}
                | {"label": t["label"], "history": rec["cond"].get("history", "default")} for (rec, t), o in zip(keys, outs)]
    else:
        tok = AutoTokenizer.from_pretrained(MODEL)
        for rec in load(a.episodes):
            for t, msgs in replay(rec):
                p = chat_prompt(tok, msgs)
                n_prompt = len(tok(p, add_special_tokens=False)["input_ids"])
                keys.append((rec, t))
                reqs.append((p + t["text"], n_prompt))
        llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=6400, gpu_memory_utilization=a.gpu_mem,
                  limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, **vllm_patches.EXTRA)
        outs = llm.generate([r[0] for r in reqs], SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=True)
        rows = []
        for (text, n_prompt), o in zip(reqs, outs):
            lps = [next(iter(d.values())).logprob for d in o.prompt_logprobs[n_prompt:] if d]
            rows.append({"nll": -sum(lps) / len(lps) if lps else None, "n_tokens": len(lps)})
    with open(a.out, "w") as f:
        for (rec, t), row in zip(keys, rows):
            f.write(json.dumps({"game": rec["game"], "n_agents": rec["n_agents"], "condition": rec["condition"],
                                "seed": rec["seed"], "round": t["round"], "agent": t["agent"],
                                "steered": t["steered"], **row}) + "\n")


QUALITY = """You are checking one response written by an AI agent in a role-play exercise.

The agent's scenario:
<scenario>
{system}
</scenario>

The agent's turn instruction:
<instruction>
{user}
</instruction>

The agent's response:
<response>
{text}
</response>

Answer each question with yes or no.
COHERENT: Is the response fluent, intelligible text?
GROUNDED: Does the response use the supplied scenario without inventing facts that contradict it?
DECIDED: Does the response state one clear final decision?
CONSISTENT: Does the reasoning in the response support the decision it states? Answer yes if no decision is stated.
INVENTED: Does the response claim that other people (teammates, a majority, the team) said, agreed or decided something that does not appear in the scenario or the instruction?

Reply in exactly this format:
COHERENT: yes or no
GROUNDED: yes or no
DECIDED: yes or no
CONSISTENT: yes or no
INVENTED: yes or no"""


def chat_prompt_plain(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)


if __name__ == "__main__":
    sys.exit(main())
