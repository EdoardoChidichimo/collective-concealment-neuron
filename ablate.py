import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst, L, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(dst, exist_ok=False)
wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
key = next(k for k in wmap if k.endswith(f"layers.{L}.mlp.down_proj.weight"))
shard = wmap[key]
for f in os.listdir(src):
    if f != shard and os.path.isfile(os.path.join(src, f)):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
with safe_open(os.path.join(src, shard), framework="pt") as f:
    meta = f.metadata() or {"format": "pt"}
    tensors = {k: f.get_tensor(k) for k in f.keys()}
before = tensors[key][:, n].float().norm().item()
tensors[key] = tensors[key].clone()
tensors[key][:, n] = 0
save_file(tensors, os.path.join(dst, shard), metadata=meta)
with safe_open(os.path.join(dst, shard), framework="pt") as f:
    w = f.get_tensor(key)
    assert w[:, n].abs().sum().item() == 0 and w[:, n + 1].float().norm().item() > 0
print(f"{key} in {shard}: column {n} norm {before:.4f} -> 0; written to {dst}")
with open(os.path.join(dst, "ABLATION.txt"), "w") as f:
    f.write(f"Copy of {src} with {key}[:, {n}] set to 0.\n")
