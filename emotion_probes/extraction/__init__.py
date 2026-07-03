"""
extraction — run the model over the datasets to produce residual-stream activations.

This is the bridge from raw text to NumPy arrays that :mod:`emotion_probes.core`
turns into vectors. Three extractors, all built on :class:`ProbedModel`:

* :class:`StoryActivationExtractor`    — per-emotion mean activations (the main signal).
* :class:`NeutralActivationExtractor`  — per-text means on neutral text (PCA baseline);
                                         used for both neutral stories and neutral dialogues.
* :class:`DeflectionActivationExtractor` — scenario / masking-speaker means for deflection,
                                         using EXACT character->token spans (fixes Issue #7)
                                         and checkpointing (for the cluster's 12h policy).

Output layout (all NumPy ``.npz``, portable, no torch needed to read):
    activations/stories.npz         {means: (L, E, H), emotions: [...]}
    activations/neutral_stories.npz {activations: (L, N, H)}
    activations/neutral_dialogues.npz {activations: (L, N, H)}
    activations/deflection.npz      target/scenario/displayed/pair means + emotion lists
"""

from emotion_probes.extraction.extractor import (
    DeflectionActivationExtractor,
    NeutralActivationExtractor,
    StoryActivationExtractor,
)

__all__ = [
    "StoryActivationExtractor",
    "NeutralActivationExtractor",
    "DeflectionActivationExtractor",
]
