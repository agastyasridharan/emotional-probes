"""
Static inputs taken directly from the paper.

* :data:`EMOTIONS` — the 171 emotion words (paper appendix, p. 82), verbatim.
* :data:`TOPICS`   — the 100 story/dialogue seed topics (paper appendix, pp. 82-85), verbatim.
* :mod:`scenarios` — the hand-written validation stimuli (implicit-content
  scenarios, user-vs-assistant dissociation prompts, numerical-intensity
  templates, the 64 preference activities, sycophancy prompts, blackmail).
* :mod:`human_norms` — loader for the Russell & Mehrabian (1977) PAD norms used
  to validate the valence/arousal geometry.
"""

from emotion_probes.data.emotions import EMOTIONS
from emotion_probes.data.topics import TOPICS

__all__ = ["EMOTIONS", "TOPICS"]
