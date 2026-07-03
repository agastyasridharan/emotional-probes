"""
models — the ONLY part of the project that touches torch / transformers / a GPU.

Everything model-specific is hidden behind :class:`ProbedModel`, which wraps any
HuggingFace causal language model and exposes exactly what the experiments need:

* read the residual stream at chosen layers (per-token, or averaged from a
  token offset onward),
* map characters to tokens exactly (via the tokenizer's offset mapping),
* read the next-token logits (for the preference experiment),
* project a direction through the unembedding (the "logit lens"),
* steer the residual stream by adding an emotion vector, and
* generate text (for self-generation of datasets and for behavior evals).

Because all of this lives in one class, swapping the model under study is a
one-line config change (Issue #1) and the layer is chosen by fraction-of-depth
so it is correct for any model size (Issue #4).
"""

from emotion_probes.models.language_model import ProbedModel

__all__ = ["ProbedModel"]
