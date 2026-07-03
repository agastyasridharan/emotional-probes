"""
generation — build the synthetic datasets the probes are extracted from.

* :mod:`prompts`        the (verbatim) generation prompts.
* :mod:`backends`       where text comes from: an external API, or the probed
                        model itself (Issue #2 self-generation).
* :mod:`generators`     the four dataset generators (emotion stories, neutral
                        stories, neutral dialogues, deflection dialogues), with
                        ``one_story_per_call`` support (Issue #6).
* :mod:`pair_selection` choose (target, displayed) deflection pairs (Issue #5).

Run a generator from the command line, e.g.::

    python -m emotion_probes.generation.generators emotion_stories
    python -m emotion_probes.generation.pair_selection
"""

# Note: importing the submodules requires the generation dependencies
# (pydantic-ai, tenacity, pandas). They are imported lazily by callers so that
# the rest of the package works without them.

__all__ = ["prompts", "backends", "generators", "pair_selection"]
