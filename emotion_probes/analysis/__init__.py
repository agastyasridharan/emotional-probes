"""
analysis — replications of the paper's *validation* (Part 1) and
*characterization* (Part 2) experiments.

Each experiment is a small class with a ``run()`` method that takes a loaded
:class:`~emotion_probes.models.ProbedModel` and a :class:`~emotion_probes.core.probes.ProbeBank`
and returns a plain results dict (and saves it under ``config.analysis_dir``).
None of these modules invent results — they implement the measurement; you run
them on a GPU to get numbers.

Part 1 (validation):
    logit_lens          Table 1 — tokens an emotion vector promotes/suppresses.
    context_activation  Fig 1 — where vectors fire on a corpus (90th-pct tokens).
    implicit_scenarios  Fig 2 — do probes detect un-named emotions? (cosine matrix)
    intensity_templates Fig 3 — do probes track appraised intensity, not numbers?
    classification      held-out emotion-classification accuracy (a probe sanity check).

Part 2 (characterization):
    geometry            Figs 5-9 — clustering, PCA = valence/arousal, RSA across layers.
    human_alignment     Figs 8/58 — correlate the geometry with human (or LLM-judged) norms.
    layerwise           Figs 12-14 — present-content vs planned-emotion across layers.
    speaker             Figs 10-11,17-19 — user vs Assistant, present vs other speaker.
    chronic_state       Table 5/Fig 16 — is there a *persistent* emotional state? (probe test)

Shared:
    steering            SteeringEngine — add an emotion vector to the residual stream.
    preferences         Fig 4 — Elo preferences + steering (see also alignment/).

Every class below imports cleanly without a GPU (torch / sklearn / scipy /
matplotlib are imported lazily inside the methods that need them), so the package
is importable on a laptop; only ``run()`` needs the cluster.
"""

from emotion_probes.analysis.steering import SteeringEngine
from emotion_probes.analysis.logit_lens import LogitLensAnalysis
from emotion_probes.analysis.context_activation import ContextActivationAnalysis
from emotion_probes.analysis.implicit_scenarios import ImplicitScenarioAnalysis
from emotion_probes.analysis.intensity_templates import IntensityTemplateAnalysis
from emotion_probes.analysis.classification import EmotionClassification
from emotion_probes.analysis.geometry import GeometryAnalysis
from emotion_probes.analysis.human_alignment import HumanAlignmentAnalysis
from emotion_probes.analysis.layerwise import LayerwiseAnalysis
from emotion_probes.analysis.speaker import SpeakerAnalysis
from emotion_probes.analysis.chronic_state import ChronicStateProbe
from emotion_probes.analysis.preferences import PreferenceExperiment

__all__ = [
    "SteeringEngine",
    "LogitLensAnalysis",
    "ContextActivationAnalysis",
    "ImplicitScenarioAnalysis",
    "IntensityTemplateAnalysis",
    "EmotionClassification",
    "GeometryAnalysis",
    "HumanAlignmentAnalysis",
    "LayerwiseAnalysis",
    "SpeakerAnalysis",
    "ChronicStateProbe",
    "PreferenceExperiment",
]
