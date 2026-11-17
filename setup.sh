#!/bin/bash
set -euo pipefail
[ -d vllm-steer ] || git clone --filter=blob:none https://github.com/RhizoNymph/vllm vllm-steer
(cd vllm-steer && git checkout -q 4506f935bba37b28a41746cd8f5897f82e46588d)
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate
(cd vllm-steer && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=755492e37d7d7201d93f0effa200acac56601f38 uv pip install -e . --torch-backend=auto)
uv pip install safetensors numpy pandas
uv pip install --no-deps flash-linear-attention fla-core einops
