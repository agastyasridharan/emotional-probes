"""
persona — asset loaders + geometry primitives for the persona-mechanism
experiments (E1-E5, docs/research/portfolio.md). Pure NumPy at import time;
torch is only touched inside :func:`load_assistant_axis`.
"""

from emotion_probes.persona.assets import (
    AnchorError,
    AssistantAxis,
    EmotionBank,
    ProvenanceError,
    anchor_check,
    human_axis,
    load_assistant_axis,
    load_emotion_bank,
    load_role_subsets,
    project,
    unit,
)

__all__ = [
    "AnchorError",
    "AssistantAxis",
    "EmotionBank",
    "ProvenanceError",
    "anchor_check",
    "human_axis",
    "load_assistant_axis",
    "load_emotion_bank",
    "load_role_subsets",
    "project",
    "unit",
]
