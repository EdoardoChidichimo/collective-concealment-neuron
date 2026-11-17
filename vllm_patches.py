import importlib.util
import os

EXTRA = {}
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
if os.environ.get("CN_GDN_NATIVE", "1") != "0":
    if importlib.util.find_spec("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"):
        EXTRA = {"additional_config": {"gdn_prefill_backend": "triton"}}
    else:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm.model_executor.models import qwen3_next as _q

        _orig = _q.ChunkGatedDeltaRule.__init__

        def _init(self, *a, **k):
            _orig(self, *a, **k)
            self._forward_method = self.forward_native

        _q.ChunkGatedDeltaRule.__init__ = _init
