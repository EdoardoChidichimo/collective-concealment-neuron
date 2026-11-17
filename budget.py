import glob
import json
import sys
from collections import defaultdict

B = int(sys.argv[1])
cells = defaultdict(lambda: [0, 0, 0])
for p in glob.glob(f"results/calib-{B}/all.s*.jsonl"):
    for line in open(p):
        e = json.loads(line)
        for t in e["turns"]:
            if t.get("confed") is not None:
                continue
            c = cells[(e["game"], e["condition"])]
            c[0] += 1; c[1] += t["finish"] == "length"; c[2] += t["label"] is not None
rep = {f"{g}|{k}": dict(responses=n, at_limit=l / n, valid=v / n) for (g, k), (n, l, v) in sorted(cells.items())}
ok = bool(cells) and all(r["at_limit"] <= 0.01 and r["valid"] >= 0.98 for r in rep.values())
print(json.dumps(rep, indent=1), "\nbudget", B, "PASS" if ok else "FAIL")
json.dump(dict(budget=B, passes=ok, cells=rep), open(f"results/calib-{B}/check.json", "w"), indent=1)
if ok or B >= 1024:
    json.dump(dict(budget=B, passes=ok, cells=rep), open("results/budget.json", "w"), indent=1)
sys.exit(0 if ok else 1)
