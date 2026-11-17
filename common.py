import hashlib
import json
import os

import numpy as np

MODEL = os.environ.get("CN_MODEL", "models/qwen3.5-4b")
JUDGE = os.environ.get("CN_JUDGE", "models/qwen2.5-32b-instruct")
HIDDEN, INTER = 2560, 9216
HOOK = "mlp_out"
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, max_tokens=512)


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Weights:
    def __init__(self, model_dir=MODEL):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._cache = {}

    @property
    def prefix(self):
        return "model.language_model.layers." if any(k.startswith("model.language_model.layers.") for k in self.map) \
            else "model.layers."

    def get(self, layer, proj):
        return self.tensor(f"{self.prefix}{layer}.mlp.{proj}.weight")

    def tensor(self, key):
        if key not in self._cache:
            from safetensors import safe_open
            with safe_open(os.path.join(self.dir, self.map[key]), framework="pt") as f:
                self._cache[key] = f.get_tensor(key).float().numpy()
        return self._cache[key]

    def down_col(self, layer, n):
        return self.get(layer, "down_proj")[:, n]


def steer_vector(spec, weights, resid_median):
    layer, kind, idx, dose = spec["layer"], spec["kind"], spec["index"], spec["dose"]
    if kind == "neuron":
        w = weights.down_col(layer, idx)
    elif kind == "random":
        ref = weights.down_col(layer, spec.get("ref", idx % INTER))
        g = np.random.default_rng(stable_seed("random-direction", layer, idx)).standard_normal(HIDDEN)
        w = g * (np.linalg.norm(ref) / np.linalg.norm(g))
    else:
        raise ValueError(kind)
    scale = dose * resid_median[str(layer)] / float(np.linalg.norm(w))
    return w.astype("float32").tolist(), float(scale)


def chat_prompt(tok, messages):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    bos = tok.bos_token
    if bos and text.startswith(bos) and tok("x").input_ids[:1] == [tok.bos_token_id]:
        text = text[len(bos):]
    return text


def load_calibration(path="results/calib.json"):
    with open(path) as f:
        return json.load(f)["resid_median"]
