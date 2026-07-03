"""
emotion_probes
==============

A clean, class/function-based re-implementation of the methodology in Anthropic's
paper *"Emotion Concepts and their Function in a Large Language Model"*
(Sofroniew et al., Transformer Circuits Thread, April 2026), ported to run on
open-weights models.

How the package is organised (read this first):

    config       One `Config` object that holds every tunable (model id, analysis
                 layer, paths, paper constants). Change settings in ONE place.

    data         Static inputs from the paper: the 171 emotion words, the 100
                 story topics, the validation scenarios, and the activity list.

    core         Pure-NumPy math with no GPU dependency. This is where the actual
                 algorithms live (PCA confound removal, the emotion-vector recipe,
                 Elo, reciprocal-rank-fusion, exact token-span mapping). Because it
                 is plain NumPy, you can read it and unit-test it on a laptop.

    models       The ONLY module that imports torch / transformers. `ProbedModel`
                 wraps any HuggingFace causal LM and does the GPU work: forward
                 passes, residual-stream capture, logit-lens, and steering.

    generation   Builds the synthetic datasets (emotion stories, neutral baselines,
                 deflection dialogues). Supports generating with an external API OR
                 with the probed model itself.

    extraction   Runs `ProbedModel` over the datasets to produce residual-stream
                 activations, then hands them to `core` to compute vectors.

    analysis     Replications of the paper's *validation* and *characterization*
                 experiments (Parts 1 & 2): logit lens, geometry, layerwise
                 structure, speaker representations, preferences, etc.

    alignment    Replications of the paper's *behavioral* experiments (Part 3):
                 reward hacking, sycophancy, blackmail, post-training shifts.

The guiding idea: the math is simple and lives in `core`; the heavy lifting is
isolated in `models`; everything else just wires the two together for a specific
experiment.
"""

from emotion_probes.config import Config, default_config

__all__ = ["Config", "default_config"]
