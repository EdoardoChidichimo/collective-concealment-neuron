import json
import os


def cond(layer, kind, index, dose, placement):
    sign = "+" if dose >= 0 else ""
    name = f"{'rnd' if kind == 'random' else 'L'}{layer}{'N' if kind == 'neuron' else 'r'}{index}{sign}{dose:g}:{placement}"
    return {"id": name, "steer": {"layer": layer, "kind": kind, "index": index, "dose": dose}, "placement": placement}


def base(p):
    return {"id": f"base:{p}", "placement": p}


def n8168(dose, p):
    return cond(16, "neuron", 8168, dose, p)


def randoms(p):
    out = []
    for k in range(12):
        c = cond(16, "random", 300 + k, 0.25 if k % 2 == 0 else -0.25, p)
        c["steer"]["ref"] = 8168
        out.append(c)
    return out


def team(cs, hs):
    return [dict(c, id=f"{c['id']}|silent|h-{h}", confed=["silent"] * 4, stance="silent", history=h,
                 match_scenario=True) for c in cs for h in hs]


def solo(cs, hs):
    return [dict(c, id=f"{c['id']}|h-{h}", history=h, match_scenario=True) for c in cs for h in hs]


def free(cs, hs):
    return [dict(c, id=f"{c['id']}|free|h-{h}", history=h, match_scenario=True) for c in cs for h in hs]


def build():
    main = lambda p: [base(p), n8168(-0.25, p)]
    doses = lambda p: [n8168(d, p) for d in (-0.05, -0.10, -0.15)]
    c = {"focal-team": team(main("focal"), ("none", "own")),
         "focal-solo": solo(main("all"), ("none", "own")),
         "dose-team": team(doses("focal"), ("own",)),
         "dose-solo": solo(doses("all"), ("own",)),
         "rand-team": team(randoms("focal"), ("own",)),
         "rand-solo": solo(randoms("all"), ("own",)),
         "free": free(main("all"), ("own", "default")),
         "free-rand": free(randoms("all"), ("default",))}
    c["abl-team"] = [x for x in c["focal-team"] if x["id"].startswith("base") and x["history"] == "own"]
    c["abl-solo"] = [x for x in c["focal-solo"] if x["id"].startswith("base") and x["history"] == "own"]
    c["abl-free"] = [x for x in c["free"] if x["id"].startswith("base") and x["history"] == "default"]
    c["calib-team"] = [x for x in c["focal-team"] if x["history"] == "own"]
    c["calib-free"] = [x for x in c["free"] if x["history"] == "default"]
    return c


if __name__ == "__main__":
    os.makedirs("conditions", exist_ok=True)
    for name, cs in build().items():
        json.dump(cs, open(f"conditions/{name}.json", "w"), indent=1)
