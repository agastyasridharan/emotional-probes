"""
Diagnose whether ProbedModel.load() actually loads in bf16 or silently in fp32.

The batch benchmark showed ~87 GiB/card RESIDENT for Llama-3.3-70B over 3 cards
(~261 GiB total) — that is the FLOAT32 footprint (70.6B*4=263 GiB), not bf16
(~131 GiB). If confirmed, dtype=bfloat16 is not taking effect in this
transformers version. This script confirms the current dtype + memory, then
reloads with explicit torch_dtype=bfloat16 to verify the fix and the memory drop.
"""

import gc
import inspect

import torch
from transformers import AutoModelForCausalLM

from emotion_probes.config import Config
from emotion_probes.models.language_model import ProbedModel

ndev = torch.cuda.device_count()


def resident():
    return [round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in range(ndev)]


sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
print("transformers", __import__("transformers").__version__,
      "| from_pretrained has 'dtype':", "dtype" in sig.parameters,
      "| has 'torch_dtype':", "torch_dtype" in sig.parameters, flush=True)

cfg = Config()
print("Config.dtype =", cfg.dtype, "| device_map =", cfg.device_map, flush=True)

# --- 1) current code path (ProbedModel.load passes dtype=torch.bfloat16) ----
m = ProbedModel(cfg).load()
pdt = next(m.model.parameters()).dtype
print("CURRENT_CODE   param_dtype=%s  model.dtype=%s  resident_GB/card=%s"
      % (pdt, getattr(m.model, "dtype", None), resident()), flush=True)

del m
gc.collect()
torch.cuda.empty_cache()
print("freed; resident_GB/card now %s" % resident(), flush=True)

# --- 2) explicit torch_dtype=bfloat16 --------------------------------------
mdl = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
print("TORCH_DTYPE    param_dtype=%s  model.dtype=%s  resident_GB/card=%s"
      % (next(mdl.parameters()).dtype, getattr(mdl, "dtype", None), resident()), flush=True)

print("DTYPE_CHECK_DONE", flush=True)
