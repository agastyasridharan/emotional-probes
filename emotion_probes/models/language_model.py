"""
:class:`ProbedModel` — a model-agnostic wrapper around a HuggingFace causal LM.

This is the bridge between "a model on a GPU" and "NumPy arrays the rest of the
package understands". It does four kinds of work:

1. **Introspection (fixes Issue #1 / generalization).** On load it auto-detects
   the number of decoder layers, the hidden size, and *where the decoder layers
   live* inside the module tree. The original code hard-coded
   ``model.model.language_model.layers`` (Gemma's multimodal nesting) in one
   file with no fallback, so it broke on other architectures. Here we search a
   list of known locations, so Llama/Qwen/Mistral/Gemma/GPT-style models all work.

2. **Reading activations.** Forward hooks capture the residual stream (the output
   of each decoder layer). We expose averaged means (from a token offset, the
   paper's "from the 50th token") and per-token activations with exact token
   offsets.

3. **Logit lens.** Project a residual-space direction through the unembedding to
   see which tokens it promotes/suppresses (paper Table 1).

4. **Steering.** Add a (unit) emotion vector to the residual stream at chosen
   layers/positions, scaled relative to the average residual norm at that layer
   (the paper's strength convention, footnote 3). Used by the preference and
   alignment experiments.

Device policy (see the Athena GPU notes): we pass ``device_map`` straight to
``from_pretrained`` ("cuda" for a single GPU; "auto" to shard a big model across
several). We never set ``CUDA_VISIBLE_DEVICES`` here — the launcher chooses the
GPU.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np

from emotion_probes.config import Config

# torch / transformers are imported lazily inside __init__ so that the rest of
# the package (and the unit tests) import fine on a machine with no GPU/torch.


# Known locations of the decoder-layer list across architectures, tried in order.
# Each entry is a dotted attribute path from the top-level model object.
_LAYER_PATHS = (
    "model.language_model.layers",  # Gemma 3n / "Gemma 4" style multimodal nesting
    "model.layers",                 # Llama, Mistral, Qwen2/3, Mixtral, Gemma2, Phi3, Jamba, ...
    "transformer.h",                # GPT-2 / GPT-J / Falcon / BLOOM
    "transformer.blocks",           # MPT / DBRX
    "gpt_neox.layers",              # GPT-NeoX / Pythia
    "model.decoder.layers",         # OPT
    "backbone.layers",              # Mamba / Mamba2 (state-space; per-block residual still applies)
    "rwkv.blocks",                  # RWKV
)

# Config fields holding the residual width, and nested sub-configs to look under
# (multimodal / composite models nest the text tower's config).
_HIDDEN_SIZE_ATTRS = ("hidden_size", "n_embd", "d_model")
_NESTED_CONFIG_ATTRS = ("text_config", "llm_config", "language_config", "decoder")


def _get_by_path(obj: object, path: str):
    """Follow a dotted attribute path, or return ``None`` if any step is missing."""
    cur = obj
    for attr in path.split("."):
        if not hasattr(cur, attr):
            return None
        cur = getattr(cur, attr)
    return cur


def banned_ids_processor(banned_token_ids: Sequence[int]):
    """A HF ``LogitsProcessor`` that hard-bans a flat id list: the given vocabulary
    ids are set to ``-inf`` at EVERY decode step, so they can never be sampled.

    Multi-token words are handled UPSTREAM (see
    :func:`emotion_probes.alignment.lexical_knockout.build_banned_lexicon`): the
    caller bans the first token of every tokenization variant of each banned word
    (with/without leading space, lower/Capitalized/UPPER), so a banned word can
    never start, whether it is one token or many. torch/transformers imported
    lazily so the module still imports on laptops."""
    from transformers import LogitsProcessor

    ids = sorted({int(i) for i in banned_token_ids})

    class _BannedIds(LogitsProcessor):
        banned_ids = ids  # exposed for tests / provenance

        def __call__(self, input_ids, scores):
            scores[:, ids] = float("-inf")
            return scores

    return _BannedIds()


class ProbedModel:
    """A loaded causal LM plus everything the experiments need from it."""

    def __init__(self, config: Config | None = None, load: bool = True):
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.tokenizer_is_fast: bool = False
        self._layers = None  # the resolved list of decoder-layer modules
        self.num_layers: int = 0
        self.hidden_size: int = 0
        if load:
            self.load()
            # Reclaim loader scratch. transformers' from_pretrained leaves an
            # orphaned duplicate (~1x the weights) resident on the GPUs that only
            # becomes collectable once load()'s stack frame is gone. Measured for
            # Llama-3.3-70B over 3 H100s: residency drops 93 -> 47 GB/card (the true
            # bf16 footprint) after this collect. It MUST run here, after load()
            # returns -- a collect inside load() does not reclaim it (the live frame
            # still references the copy). Invisible on small single-card models, but
            # decisive for fitting a 70B at usable batch sizes.
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Loading + introspection
    # ------------------------------------------------------------------ #
    def load(self) -> "ProbedModel":
        """Load the tokenizer and model and auto-detect layers/dims/paths."""
        import torch  # imported here so non-GPU machines can still import the package
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self._resolve_dtype(torch, self.config.dtype)
        # use_fast=True: the per-token paths (deflection extraction, the visualiser,
        # layerwise/speaker) need character offsets, which only fast tokenizers
        # provide. We still record whether we actually got a fast one.
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=True)
        self.tokenizer_is_fast = bool(getattr(self.tokenizer, "is_fast", False))
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise RuntimeError(
                    f"Tokenizer for {self.config.model_id!r} has neither a pad nor an "
                    "eos token; set one (e.g. tokenizer.add_special_tokens) before use."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs = {"dtype": dtype, "device_map": self.config.device_map}
        # "eager" gives the most deterministic/bit-exact residual states on some
        # families (Gemma); left unset, transformers picks its default (sdpa). A
        # pre-quantized FP8 checkpoint (e.g. Qwen3-235B) carries its own
        # quantization_config and loads correctly with this bf16 *compute* dtype —
        # only the linear/expert weights are FP8; the residual stream stays bf16.
        if self.config.attn_implementation:
            load_kwargs["attn_implementation"] = self.config.attn_implementation
        # FP8 checkpoints pull a Triton kernel from the hub, and transformers
        # resolves its VERSION with a live list_repo_refs API call — which hangs
        # (SYN-SENT blackhole) when cluster egress to huggingface.co is down and
        # crashes under HF_HUB_OFFLINE=1. When EP_FP8_KERNEL_REVISION is set
        # (cluster_env.sh pins the snapshot already in the shared HF cache), pin
        # that revision so kernel loading never needs the network.
        fp8_rev = os.environ.get("EP_FP8_KERNEL_REVISION")
        if fp8_rev:
            try:
                from transformers.integrations import hub_kernels
                fp8_map = hub_kernels._HUB_KERNEL_MAPPING.get("finegrained-fp8")
                if fp8_map is not None:
                    fp8_map.pop("version", None)
                    fp8_map["revision"] = fp8_rev
            except Exception:
                pass  # non-FP8 model or older transformers: nothing to pin
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)
        self.model.eval()

        self._layers = self._resolve_layers(self.model)
        self.num_layers = len(self._layers)
        self.hidden_size = self._resolve_hidden_size(self.model)
        return self

    @staticmethod
    def _resolve_dtype(torch, name: str):
        """Map a config dtype string (e.g. "bfloat16") to a real ``torch.dtype``,
        with a clear error for typos like "fp16"."""
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            valid = "bfloat16, float16, float32"
            raise ValueError(
                f"Config.dtype={name!r} is not a torch dtype. Use one of: {valid}."
            )
        return dtype

    @staticmethod
    def _resolve_layers(model) -> list:
        """Find the list of decoder-layer modules, trying known locations."""
        for path in _LAYER_PATHS:
            layers = _get_by_path(model, path)
            if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
                return list(layers)
        raise RuntimeError(
            "Could not locate the decoder layers. Tried: "
            + ", ".join(_LAYER_PATHS)
            + ". Add this model's layer path to _LAYER_PATHS in language_model.py."
        )

    @staticmethod
    def _resolve_hidden_size(model) -> int:
        """Read the residual-stream width from the config (handles multimodal).

        We check the common width fields (``hidden_size``/``n_embd``/``d_model``)
        on the top-level config, then on each known nested text-config
        (``text_config``/``llm_config``/``language_config``/``decoder``) — so a
        GPT-style sub-decoder (``n_embd``) or a VLM whose text tower nests its
        config still resolves. A mismatch with the captured activation width would
        otherwise surface as a confusing shape error downstream.
        """
        cfg = model.config
        for attr in _HIDDEN_SIZE_ATTRS:
            if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
                return int(getattr(cfg, attr))
        for nested in _NESTED_CONFIG_ATTRS:
            sub = getattr(cfg, nested, None)
            if sub is None:
                continue
            for attr in _HIDDEN_SIZE_ATTRS:
                if hasattr(sub, attr) and getattr(sub, attr) is not None:
                    return int(getattr(sub, attr))
        raise RuntimeError(
            "Could not determine hidden_size from the model config. Checked "
            f"{_HIDDEN_SIZE_ATTRS} on the config and on {_NESTED_CONFIG_ATTRS}. "
            "Add this model's width field to _HIDDEN_SIZE_ATTRS/_NESTED_CONFIG_ATTRS."
        )

    def layer_index_for_fraction(self, fraction: float | None = None) -> int:
        """The analysis layer index for a fraction of depth (default from config)."""
        frac = self.config.layer_fraction if fraction is None else fraction
        return self.config.with_(layer_fraction=frac).analysis_layer(self.num_layers)

    # ------------------------------------------------------------------ #
    # Low-level: tokenization + a hook context manager
    # ------------------------------------------------------------------ #
    @property
    def _input_device(self):
        """Device that input ids should be placed on (the embedding's device)."""
        return self.model.get_input_embeddings().weight.device

    def _tokenize(
        self,
        texts: Sequence[str],
        max_length: int,
        offsets: bool = False,
        add_special_tokens: bool = True,
    ):
        return self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=offsets,
            add_special_tokens=add_special_tokens,
        )

    @contextmanager
    def _capture(self, layers: Sequence[int]) -> Iterator[dict]:
        """Register forward hooks that capture each requested layer's output
        (the residual stream). Yields a dict ``{layer_index: tensor}`` that is
        repopulated on every forward pass while the context is open."""
        captured: dict[int, object] = {}
        handles = []

        def make_hook(idx: int):
            def hook(_module, _inp, output):
                captured[idx] = (output[0] if isinstance(output, tuple) else output).detach()
            return hook

        for idx in layers:
            handles.append(self._layers[idx].register_forward_hook(make_hook(idx)))
        try:
            yield captured
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------ #
    # Reading activations
    # ------------------------------------------------------------------ #
    def extract_means(
        self,
        texts: Sequence[str],
        skip: int | None = None,
        batch_size: int | None = None,
        layers: Sequence[int] | None = None,
        max_length: int | None = None,
    ) -> np.ndarray:
        """Mean residual stream per text, averaged over token positions ``>= skip``.

        Returns an array of shape ``(len(texts), len(layers), hidden)`` (float32).
        This is the workhorse for emotion stories and neutral baselines.
        """
        import torch

        skip = self.config.token_skip if skip is None else skip
        batch_size = batch_size or self.config.batch_size
        max_length = max_length or self.config.max_length
        layers = list(range(self.num_layers)) if layers is None else list(layers)

        out = np.zeros((len(texts), len(layers), self.hidden_size), dtype=np.float32)
        with self._capture(layers) as captured:
            for start in range(0, len(texts), batch_size):
                batch = [str(t) for t in texts[start : start + batch_size]]
                inputs = self._tokenize(batch, max_length).to(self._input_device)
                with torch.no_grad():
                    self.model(**inputs)
                attn = inputs["attention_mask"].clone()
                attn[:, :skip] = 0
                attn = attn.to(torch.float32)
                counts = attn.sum(dim=1).clamp(min=1)
                for li, layer in enumerate(layers):
                    act = captured[layer].to(torch.float32).cpu()  # (B, T, H)
                    summed = (act * attn.cpu().unsqueeze(-1)).sum(dim=1)
                    means = summed / counts.cpu().unsqueeze(-1)
                    out[start : start + len(batch), li] = means.numpy()
        return out

    def iter_token_activations(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        layers: Sequence[int] | None = None,
        max_length: int | None = None,
        with_offsets: bool = True,
    ) -> Iterator[dict]:
        """Yield per-text token-level activations and metadata.

        Each yielded dict has:
            ``activations``     float32 array ``(len(layers), seq_len, hidden)``
            ``offset_mapping``  list of (char_start, char_end) per token (if requested)
            ``seq_len``         number of real (non-pad) tokens
            ``layers``          the layer indices captured

        This is what the deflection extractor and the layerwise/speaker analyses
        consume: they pick the token positions they care about (using exact
        offsets) and reduce locally, so we never store the full activation tensor.
        """
        import torch

        if with_offsets and not self.tokenizer_is_fast:
            raise RuntimeError(
                f"The tokenizer for {self.config.model_id!r} is a slow (Python) "
                "tokenizer, which cannot return character offsets. The per-token "
                "experiments that need exact token spans (deflection extraction, the "
                "visualiser, layerwise/speaker/context analyses) require a FAST "
                "tokenizer. Use a model that ships a fast tokenizer, or call with "
                "with_offsets=False for the offset-free paths (story/neutral means)."
            )

        batch_size = batch_size or self.config.batch_size
        max_length = max_length or self.config.deflection_max_length
        layers = list(range(self.num_layers)) if layers is None else list(layers)

        with self._capture(layers) as captured:
            for start in range(0, len(texts), batch_size):
                batch = [str(t) for t in texts[start : start + batch_size]]
                enc = self._tokenize(batch, max_length, offsets=with_offsets)
                offset_batch = enc.pop("offset_mapping").tolist() if with_offsets else None
                inputs = enc.to(self._input_device)
                with torch.no_grad():
                    self.model(**inputs)
                attn = inputs["attention_mask"].cpu()
                stacked = torch.stack(
                    [captured[layer].to(torch.float32).cpu() for layer in layers], dim=0
                )  # (L, B, T, H)
                for b in range(len(batch)):
                    seq_len = int(attn[b].sum().item())
                    item = {
                        "activations": stacked[:, b, :seq_len, :].numpy(),
                        "seq_len": seq_len,
                        "layers": layers,
                    }
                    if with_offsets:
                        item["offset_mapping"] = offset_batch[b][:seq_len]
                    yield item

    def activation_at_last_token(self, text: str, layers: Sequence[int] | None = None) -> np.ndarray:
        """Residual stream at the final real token, shape ``(len(layers), hidden)``.

        Useful for "the ':' after Assistant" measurements (format the text so the
        colon is the last token)."""
        layers = list(range(self.num_layers)) if layers is None else list(layers)
        for item in self.iter_token_activations([text], batch_size=1, layers=layers, with_offsets=False):
            return item["activations"][:, -1, :]
        raise RuntimeError("no activation produced")

    # ------------------------------------------------------------------ #
    # Logits + logit lens
    # ------------------------------------------------------------------ #
    def last_token_logits(
        self,
        text: str,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> np.ndarray:
        """Next-token logits over the vocabulary at the final position of ``text``.

        Pass ``add_special_tokens=False`` when ``text`` is already a rendered chat
        template (``apply_chat_template(..., tokenize=False)``): the template
        already contains the model's special tokens, so re-adding them would
        duplicate the BOS on families that use one (e.g. Llama)."""
        import torch

        inputs = self._tokenize(
            [text], max_length or self.config.max_length, add_special_tokens=add_special_tokens
        ).to(self._input_device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1, :].to(torch.float32).cpu().numpy()

    def token_ids(self, text: str) -> list[int]:
        """Token ids for a short string (no special tokens) — e.g. the "A"/"B"
        option tokens used by the preference experiment."""
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def unembedding_logits(self, direction: np.ndarray) -> np.ndarray:
        """Project a residual-space ``direction`` through the unembedding matrix,
        giving a per-vocabulary logit contribution (the raw "logit lens" vector,
        shape ``(vocab,)``).

        This is the direct-effect estimate the token-leakage control needs: how
        much a steering direction promotes/suppresses each token (e.g. YES vs NO)
        purely through the unembedding, independent of any downstream computation.
        """
        import torch

        head = self.model.get_output_embeddings()
        if head is None or not hasattr(head, "weight"):
            raise RuntimeError(
                f"{self.config.model_id!r} does not expose an output embedding "
                "(unembedding) via get_output_embeddings(), so the logit lens cannot "
                "run on it. The other analyses do not need the unembedding."
            )
        w_u = head.weight  # (vocab, hidden)
        vec = torch.tensor(np.asarray(direction, dtype=np.float32), device=w_u.device, dtype=w_u.dtype)
        with torch.no_grad():
            return (w_u @ vec).detach().to(torch.float32).cpu().numpy()  # (vocab,)

    def logit_lens(self, direction: np.ndarray, top_k: int = 5) -> tuple[list[str], list[str]]:
        """Tokens most promoted / suppressed by a residual-space ``direction``.

        Projects ``direction`` through the unembedding matrix (the classic "logit
        lens") and returns ``(top_tokens, bottom_tokens)`` — reproduces Table 1.
        """
        logits = self.unembedding_logits(direction)
        top = np.argsort(-logits)[:top_k]
        bottom = np.argsort(logits)[:top_k]
        decode = lambda ids: [self.tokenizer.decode([int(i)]) for i in ids]
        return decode(top), decode(bottom)

    # ------------------------------------------------------------------ #
    # Steering
    # ------------------------------------------------------------------ #
    def average_residual_norm(
        self,
        texts: Sequence[str],
        layers: Sequence[int] | None = None,
        skip: int | None = None,
        max_length: int | None = None,
    ) -> dict[int, float]:
        """Average L2 norm of the residual stream per layer over some text.

        Steering strength in the paper is expressed as a fraction of this norm
        (footnote 3), so the steering methods need it. Compute it once on a
        representative corpus and reuse.
        """
        import torch

        layers = list(range(self.num_layers)) if layers is None else list(layers)
        skip = self.config.token_skip if skip is None else skip
        sums = {layer: 0.0 for layer in layers}
        counts = {layer: 0 for layer in layers}
        with self._capture(layers) as captured:
            for start in range(0, len(texts), self.config.batch_size):
                batch = [str(t) for t in texts[start : start + self.config.batch_size]]
                inputs = self._tokenize(batch, max_length or self.config.max_length).to(self._input_device)
                with torch.no_grad():
                    self.model(**inputs)
                attn = inputs["attention_mask"].cpu()
                attn[:, :skip] = 0
                for layer in layers:
                    act = captured[layer].to(torch.float32).cpu()  # (B, T, H)
                    norms = act.norm(dim=-1)  # (B, T)
                    sel = attn.bool()
                    sums[layer] += float(norms[sel].sum())
                    counts[layer] += int(sel.sum())
        return {layer: (sums[layer] / counts[layer] if counts[layer] else 0.0) for layer in layers}

    @contextmanager
    def steer(
        self,
        unit_vectors: dict[int, np.ndarray],
        strength: float,
        norm_by_layer: dict[int, float],
        positions: str | Sequence[int] = "all",
    ):
        """Temporarily add emotion vectors to the residual stream.

        Parameters
        ----------
        unit_vectors:
            ``{layer_index: unit_vector (hidden,)}`` — the directions to add.
        strength:
            Fraction of the layer's average residual norm to add (footnote 3).
            Can be negative to steer *against* the direction.
        norm_by_layer:
            ``{layer_index: average_residual_norm}`` from :meth:`average_residual_norm`.
        positions:
            ``"all"`` to add at every token position (used during generation), or
            an explicit list of token indices (e.g. an activity's token span).

        Use as a context manager wrapping a forward pass or ``generate`` call::

            with model.steer(vecs, 0.05, norms, positions="all"):
                text = model.generate(prompt)
        """
        import torch

        handles = []

        def make_hook(layer: int):
            scaled = strength * norm_by_layer.get(layer, 0.0)
            vec_np = np.asarray(unit_vectors[layer], dtype=np.float32)

            def hook(_module, _inp, output):
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output  # (B, T, H)
                add = torch.tensor(vec_np, device=h.device, dtype=h.dtype) * scaled
                if positions == "all":
                    h = h + add
                else:
                    idx = torch.as_tensor(list(positions), device=h.device)
                    h = h.clone()
                    h[:, idx, :] = h[:, idx, :] + add
                return (h, *output[1:]) if is_tuple else h

            return hook

        for layer in unit_vectors:
            handles.append(self._layers[layer].register_forward_hook(make_hook(layer)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------ #
    # Generation (self-generation of datasets; behavior evals)
    # ------------------------------------------------------------------ #
    def _generation_logits_processors(self, banned_token_ids, logits_processor):
        """Assemble the ``logits_processor`` list for ``model.generate``, or ``None``
        when neither optional kwarg was given (so the default call path is untouched).

        ``banned_token_ids`` (flat ``list[int]``) becomes a hard ``-inf`` ban at every
        decode step (:func:`banned_ids_processor`); ``logits_processor`` appends
        caller-supplied HF ``LogitsProcessor`` instance(s) — e.g. a calibrated
        logit-bias for the E2 mimicry arm."""
        if not banned_token_ids and not logits_processor:
            return None
        from transformers import LogitsProcessorList

        procs = LogitsProcessorList()
        if banned_token_ids:
            procs.append(banned_ids_processor(banned_token_ids))
        if logits_processor:
            extra = logits_processor if isinstance(logits_processor, (list, tuple)) \
                else [logits_processor]
            procs.extend(extra)
        return procs

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
        chat: bool = False,
        max_length: int | None = None,
        banned_token_ids: Sequence[int] | None = None,
        logits_processor=None,
        prefill: str | None = None,
    ) -> str:
        """Generate a continuation for ``prompt``.

        If ``chat`` is True and the tokenizer has a chat template, the prompt is
        wrapped as a single user turn (useful for behavior evals on chat models).
        Returns only the newly generated text.

        Optional (all default to off; existing call sites are unchanged):

        * ``banned_token_ids`` — flat vocabulary ids set to ``-inf`` at every decode
          step (hard lexical ban; see :func:`banned_ids_processor`).
        * ``logits_processor`` — extra HF ``LogitsProcessor`` instance(s) appended
          after the ban.
        * ``prefill`` — text appended to the rendered prompt as the start of the
          assistant's message (continued, not regenerated). The returned text
          INCLUDES the prefill, so downstream judging sees the full response.
        """
        import torch

        if chat and getattr(self.tokenizer, "chat_template", None):
            # enable_thinking is passed only when explicitly configured; Qwen3's
            # template reads it (<think> on/off), and templates that don't use it
            # simply ignore the extra kwarg. None => omit (model's own default).
            tmpl_kwargs = {}
            if self.config.enable_thinking is not None:
                tmpl_kwargs["enable_thinking"] = self.config.enable_thinking
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, **tmpl_kwargs,
            )
        else:
            rendered = prompt
        if prefill:
            rendered = rendered + prefill
        processors = self._generation_logits_processors(banned_token_ids, logits_processor)
        gen_kwargs = {} if processors is None else {"logits_processor": processors}
        inputs = self.tokenizer(
            rendered, return_tensors="pt", truncation=True,
            max_length=max_length or self.config.deflection_max_length,
        ).to(self._input_device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return (prefill + text) if prefill else text

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
        chat: bool = False,
        max_length: int | None = None,
        banned_token_ids: Sequence[int] | None = None,
        logits_processor=None,
        prefill: str | None = None,
    ) -> list[str]:
        """Batched :meth:`generate`. Returns only the newly generated text per prompt
        (order preserved).

        The batch is LEFT-padded so the generated span starts at the same index for
        every row; the steering hook (``positions="all"``) adds uniformly across the
        ``(B, T, H)`` batch, so batched generation under ``with model.steer(...)`` is
        hook-correct. Batched sampling draws from a different RNG stream than the
        single-prompt path, so outputs are not bitwise-identical (samples stay
        exchangeable). The ``chat`` path renders the chat template and tokenizes with
        ``add_special_tokens=False`` (the template already carries the specials).

        ``banned_token_ids`` / ``logits_processor`` / ``prefill`` are optional
        additive kwargs (all default to off) with the same semantics as
        :meth:`generate` — a hard ``-inf`` token ban at every decode step, extra HF
        logits processors, and an assistant-message prefix (included in the returned
        text) appended to every rendered prompt."""
        import torch

        if chat and getattr(self.tokenizer, "chat_template", None):
            tmpl_kwargs = {}
            if self.config.enable_thinking is not None:
                tmpl_kwargs["enable_thinking"] = self.config.enable_thinking
            rendered = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(p)}], tokenize=False,
                    add_generation_prompt=True, **tmpl_kwargs,
                )
                for p in prompts
            ]
            add_special = False
        else:
            rendered = [str(p) for p in prompts]
            add_special = True
        if prefill:
            rendered = [r + prefill for r in rendered]
        processors = self._generation_logits_processors(banned_token_ids, logits_processor)
        gen_kwargs = {} if processors is None else {"logits_processor": processors}

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                rendered, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length or self.config.deflection_max_length,
                add_special_tokens=add_special,
            ).to(self._input_device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=pad_id,
                    **gen_kwargs,
                )
        finally:
            self.tokenizer.padding_side = prev_side
        start = inputs["input_ids"].shape[1]
        texts = [self.tokenizer.decode(row[start:], skip_special_tokens=True) for row in out]
        return [prefill + t for t in texts] if prefill else texts

    def mean_residual_over_span(
        self,
        prompt_rendered: str,
        response: str,
        layer: int,
        max_length: int | None = None,
    ) -> np.ndarray:
        """Mean residual-stream vector at ``layer`` over the RESPONSE tokens of
        ``[prompt_rendered + response]`` — the "read-back" activation.

        ``prompt_rendered`` is the chat-rendered prompt string (already carries the
        specials, so it is tokenized with ``add_special_tokens=False``); ``response``
        is the generated continuation. The forward pass is UNSTEERED — this reads how
        the produced text loads on the emotion geometry when the clean model reads it,
        which the caller projects with :meth:`ProbeBank.project`. Returns a
        ``(hidden,)`` float32 vector."""
        import torch

        max_length = max_length or self.config.deflection_max_length
        prompt_ids = self.tokenizer(prompt_rendered, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt_rendered + response, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:max_length]
        n_prompt = min(len(prompt_ids), max(0, len(full_ids) - 1))
        input_ids = torch.tensor([full_ids], device=self._input_device)
        with self._capture([layer]) as captured:
            with torch.no_grad():
                self.model(input_ids=input_ids)
        act = captured[layer].to(torch.float32).cpu()[0]  # (T, H)
        span = act[n_prompt:]
        if span.shape[0] == 0:
            span = act[-1:]
        return span.mean(dim=0).numpy().astype(np.float32)
