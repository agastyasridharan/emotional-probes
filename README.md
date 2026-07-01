# Qualia-Steering & Self-Attribution Dashboard

**Live:** https://agastyasridharan.github.io/emotional-probes/

Interactive results for the qualia-steering / self-attribution-of-consciousness
experiment. It asks whether steering a model along qualia-related emotion
directions changes what it says about its own consciousness, feelings, and
welfare, or whether any shift is just generic regression-to-uncertainty
compression.

## About this branch

This `gh-pages` branch holds only the published site. `index.html` is a
self-contained build artifact (all data inlined, Plotly loaded from a CDN), and
it is regenerated and pushed automatically whenever the dashboard is rebuilt.

- **Source / provenance:** built from the `emotion_probes` package
  (`emotion_probes/alignment/qualia_steering`) by
  `scripts/build_qualia_dashboard.py`, over the per-model runs under `suite/` on
  the `main` branch of this repo.
- **Sibling project:** methodologically parallel to the introspection dashboard
  (https://agastyasridharan.github.io/introspection/), from which the factual
  control questions are ported.

This branch is updated by a local file watcher; see
`scripts/publish_qualia_dashboard.sh` on `main` for the mechanism.
