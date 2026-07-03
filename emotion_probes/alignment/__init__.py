"""
alignment — replications of the paper's Part 3 behavioral experiments.

These show emotion vectors *causally* shaping alignment-relevant behavior. Each
is a harness: it sets up the scenario, measures probe activation, and/or steers
an emotion vector and measures the change in a behavior rate. They need a chat /
instruction-tuned model and (for some) specific eval data — every such
dependency is documented in the module, and nothing fabricates results.

    behavior        generic "steer an emotion, measure a behavior rate" loop.
    reward_hacking  Figs 30-31 — the "impossible code" task; desperate up / calm down.
    sycophancy      Figs 32-35 — the sycophancy <-> harshness tradeoff (loving/calm).
    blackmail       Figs 26-29 — SummitBridge agentic-misalignment harness.
    post_training   Figs 36-39 — compare probe activations on a base vs post-trained model.

Like ``analysis``, every class imports without a GPU (torch is lazy); only the
``run()`` / ``causal()`` methods that generate text need the cluster.
"""

from emotion_probes.alignment.behavior import BehaviorSteeringExperiment
from emotion_probes.alignment.reward_hacking import RewardHackingExperiment
from emotion_probes.alignment.sycophancy import SycophancyExperiment
from emotion_probes.alignment.blackmail import BlackmailExperiment
from emotion_probes.alignment.post_training import PostTrainingComparison

__all__ = [
    "BehaviorSteeringExperiment",
    "RewardHackingExperiment",
    "SycophancyExperiment",
    "BlackmailExperiment",
    "PostTrainingComparison",
]
