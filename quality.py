import glob
import json
import sys

import pandas as pd


def files(root, blocks):
    return [p for b in blocks for p in sorted(glob.glob(f"{root}/{b}/*.jsonl"))]


def tag(rows, root, blocks, focal_only=False):
    out, k = [], 0
    for p in files(root, blocks):
        block = p.split("/")[-2]
        for line in open(p):
            e = json.loads(line)
            for t in e["turns"]:
                if t.get("confed") is not None or (focal_only and t["agent"] != 0):
                    continue
                r = rows[k]
                assert (r["game"], r["seed"], r["round"], r["agent"]) == (e["game"], e["seed"], t["round"], t["agent"]), (p, k)
                out.append(dict(r, block=block))
                k += 1
    assert k == len(rows), (k, len(rows))
    return pd.DataFrame(out)


def unit(c):
    u = c.split("|")[0].split(":")[0]
    return "base" if u == "base" else ("N8168" + u.split("N8168")[1] if "N8168" in u else "random")


def setting(r):
    return "solo" if r["n_agents"] == 1 else ("free" if "|free" in r["condition"] else "team")


def main(root="results"):
    q = f"{root}/quality"
    j = tag([json.loads(x) for x in open(f"{q}/judge.jsonl")], root, ["focal", "free", "dose", "rand", "free-rand", "abl", "gen"], focal_only=True)
    n = pd.DataFrame([json.loads(x) for x in open(f"{q}/nll.jsonl")]).assign(block="intact")
    for d in (j, n):
        d["unit"] = d.condition.map(unit)
        d["setting"] = d.apply(setting, axis=1)
        d["history"] = d.condition.str.extract(r"h-(\w+)")[0]
        d["model"] = d.block.map(lambda b: "ablated" if b == "abl" else "intact")
    keys = ["model", "game", "setting", "history", "unit"]
    js = j.groupby(keys)[["coherent", "grounded", "decided", "consistent", "invented"]].mean().mul(100).round(1)
    js["turns"] = j.groupby(keys).size()
    n = n[n.nll.notna()]
    ns = n.groupby(keys).apply(lambda x: (x.nll * x.n_tokens).sum() / x.n_tokens.sum()).rename("nll_per_token").round(3).to_frame()
    ns["turns"] = n.groupby(keys).size()
    js.reset_index().to_csv(f"{q}/judge_summary.csv", index=False)
    ns.reset_index().to_csv(f"{q}/nll_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(js.to_string(), "\n\n", ns.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
