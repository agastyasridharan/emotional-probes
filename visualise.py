"""
Emotion Visualisation Server — thin launcher.

The implementation now lives in the package
(:mod:`emotion_probes.visualisation`); this top-level script is kept so the
familiar workflow still works:

    # forward a port from the GPU box, then:
    python visualise.py            # serves http://localhost:<port>

It loads the model named in the config, the emotion vectors from
``config.emotion_vectors_path``, and (if present) the deflection vectors, then
serves the token-heatmap UI. Pass ``--model`` / ``--port`` / ``--host`` to
override. See :mod:`emotion_probes.visualisation.server` for details.
"""

from emotion_probes.visualisation.server import main

if __name__ == "__main__":
    main()
