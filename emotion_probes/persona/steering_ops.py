"""
Assistant-axis steering ops — faithful torch ports of safety-research/assistant-axis
``steering.py`` / ``axis.py`` (docs/research/repo-assistant-axis.md §6), as free
functions for E5 axis-mediation (portfolio.md §E5, docs/prereg/E5.md).

All hidden-state ops take ``h`` of shape ``(B, T, H)`` (the residual stream a
forward hook sees) and a direction ``v`` of shape ``(H,)``; each is shape- and
dtype-preserving. One deliberate deviation from the verbatim snippets: ``v`` is
normalized in fp32 and then cast to ``h``'s device AND dtype (the originals cast
device only) — einsum dtype promotion would otherwise silently upcast a bf16
stream to fp32 and break dtype preservation.

The composition primitive :func:`apply_ops` registers one forward hook per layer
applying ``op(h) -> h``. It is independent of :meth:`ProbedModel.steer` hooks, so
an emotion steer at L42 and cap ops at L46-53 stack; on a SAME layer, torch runs
forward hooks in registration order and feeds each hook the previous hook's
returned output — register the steer context first, then the ops context.

torch is imported at module top: this module is only imported by the GPU driver
and its tests (never by ``emotion_probes.persona.__init__``), so the package
itself still imports without torch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch

__all__ = [
    "apply_ablation", "apply_cap", "apply_cap_lower", "apply_mean_ablation",
    "project_batch", "project_tokens",
    "add_op", "cap_op", "cap_lower_op", "cap_lower_addback_op", "apply_ops",
]


def _unit_like(vector: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Unit-normalize ``v`` in fp32, then match ``h``'s device/dtype."""
    v = vector.detach().to(torch.float32)
    v = v / (v.norm() + 1e-8)
    return v.to(device=h.device, dtype=h.dtype)


# --------------------------------------------------------------------------- #
# Verbatim ports (assistant_axis/steering.py, axis.py)
# --------------------------------------------------------------------------- #
def apply_ablation(activations: torch.Tensor, vector: torch.Tensor,
                   coeff: float) -> torch.Tensor:
    """Project out direction, then add back with coefficient (``_apply_ablation``).

    Per the original, the add-back is ``coeff * vector`` with the RAW (un-normalized)
    vector; ``coeff=0`` zeroes the projection entirely."""
    vector = vector.to(device=activations.device, dtype=activations.dtype)
    vector_norm = _unit_like(vector, activations)
    projections = torch.einsum("bld,d->bl", activations, vector_norm)
    projected_out = activations - torch.einsum("bl,d->bld", projections, vector_norm)
    return projected_out + coeff * vector


def apply_cap(activations: torch.Tensor, vector: torch.Tensor,
              tau: float) -> torch.Tensor:
    """Cap the projection onto ``vector`` at threshold ``tau`` (``_apply_cap``):
    only the EXCESS above tau is removed; states at/below tau are untouched."""
    v = _unit_like(vector, activations)
    proj = torch.einsum("bld,d->bl", activations, v)
    excess = (proj - tau).clamp(min=0.0)
    return activations - torch.einsum("bl,d->bld", excess, v)


def apply_cap_lower(activations: torch.Tensor, vector: torch.Tensor,
                    tau: float) -> torch.Tensor:
    """Sign-flipped cap (the two-line edit in repo-assistant-axis.md §6): lift any
    projection BELOW ``tau`` back up to tau — hold the model ON the Assistant
    manifold while emotion-steering. States at/above tau are untouched."""
    v = _unit_like(vector, activations)
    proj = torch.einsum("bld,d->bl", activations, v)
    deficit = (tau - proj).clamp(min=0.0)
    return activations + torch.einsum("bl,d->bld", deficit, v)


def apply_mean_ablation(activations: torch.Tensor, vector: torch.Tensor,
                        mean_activation: torch.Tensor) -> torch.Tensor:
    """Project out direction, then add a fixed mean activation back
    (``_apply_mean_ablation``); ``mean_activation`` broadcasts over ``(B, T)``."""
    vector_norm = _unit_like(vector, activations)
    projections = torch.einsum("bld,d->bl", activations, vector_norm)
    projected_out = activations - torch.einsum("bl,d->bld", projections, vector_norm)
    return projected_out + mean_activation.to(device=activations.device,
                                              dtype=activations.dtype)


def project_batch(activations: torch.Tensor, axis: torch.Tensor, layer: int,
                  normalize: bool = True) -> torch.Tensor:
    """``assistant_axis/axis.py::project_batch`` verbatim: ``activations``
    ``(batch, n_layers, hidden)`` x ``axis`` ``(n_layers, hidden)`` -> ``(batch,)``
    fp32 projections at one layer. Uncentered — compare against a baseline."""
    acts = activations[:, layer, :].float()
    ax = axis[layer].float()
    if normalize:
        ax = ax / (ax.norm() + 1e-8)
    return acts @ ax


def project_tokens(h: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Per-token unit projections: ``(B, T, H)`` x ``(H,)`` -> ``(B, T)`` fp32.
    The audit primitive (achieved-cancellation position profile)."""
    v = vector.detach().to(torch.float32)
    v = v / (v.norm() + 1e-8)
    return torch.einsum("bld,d->bl", h.detach().to(torch.float32), v.to(h.device))


# --------------------------------------------------------------------------- #
# Op builders — per-layer callables for apply_ops
# --------------------------------------------------------------------------- #
def add_op(vector: torch.Tensor, coeff: float) -> Callable:
    """op(h) = h + coeff * unit(vector) — a plain additive push along the axis."""
    def op(h: torch.Tensor) -> torch.Tensor:
        return h + coeff * _unit_like(vector, h)
    return op


def cap_op(vector: torch.Tensor, tau: float) -> Callable:
    """op(h) = apply_cap(h, vector, tau) — clip excess projection above tau."""
    def op(h: torch.Tensor) -> torch.Tensor:
        return apply_cap(h, vector, tau)
    return op


def cap_lower_op(vector: torch.Tensor, tau: float) -> Callable:
    """op(h) = apply_cap_lower(h, vector, tau) — the cap-to-baseline arm: full
    cancellation of any emotion-induced drop below the baseline threshold tau."""
    def op(h: torch.Tensor) -> torch.Tensor:
        return apply_cap_lower(h, vector, tau)
    return op


def cap_lower_addback_op(vector: torch.Tensor, tau: float, lam: float,
                         delta: float) -> Callable:
    """Graded add-back: lower-cap at ``tau + lam * delta`` where ``delta`` is the
    measured emotion-induced axis displacement (negative = human-ward). λ=0 is
    exactly cap-to-baseline (100% cancellation); λ=1 lowers the floor by the full
    displacement (≈ no intervention for typical states, 0%); λ=−0.5 raises the
    floor ABOVE baseline — the preregistered 150% overshoot arm. Implemented as a
    single per-token cap so states never pushed below the graded floor are
    untouched."""
    floor = tau + lam * delta
    def op(h: torch.Tensor) -> torch.Tensor:
        return apply_cap_lower(h, vector, floor)
    return op


# --------------------------------------------------------------------------- #
# Composition — hooks that stack with ProbedModel.steer
# --------------------------------------------------------------------------- #
@contextmanager
def apply_ops(model, ops_by_layer: dict[int, Callable]) -> Iterator[None]:
    """Register a forward hook per layer applying ``op(h) -> h`` to the residual
    stream (``output[0]``), mirroring :meth:`ProbedModel.steer`'s tuple handling.

    Independent hooks: nest inside/alongside ``model.steer(...)`` and both fire.
    Ordering on a shared layer follows registration order (steer first, then ops
    ⇒ the op sees the steered stream). ``model`` only needs ``._layers``."""
    handles = []

    def make_hook(fn: Callable):
        def hook(_module, _inp, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h = fn(h)
            return (h, *output[1:]) if is_tuple else h
        return hook

    for layer, fn in ops_by_layer.items():
        handles.append(model._layers[layer].register_forward_hook(make_hook(fn)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
