"""
:class:`VisualiserServer` — the Flask app that serves the emotion heatmap UI.

This is the package-native replacement for the old top-level ``visualise.py``. It
keeps the front-end (``template.HTML_TEMPLATE``) byte-for-byte and moves all the
logic onto the package:

* the model + layer resolution come from :class:`~emotion_probes.models.ProbedModel`
  (so it works on any architecture, not just Gemma — fixes the hard-coded layer
  path the old server had);
* the per-token scoring comes from :class:`TokenActivationScorer`;
* the emotion / deflection vectors are loaded from the standard pipeline outputs
  (``config.emotion_vectors_path`` / ``config.deflection_vectors_path``), instead
  of bespoke ``.pt`` files.

The default layer is now ``config.analysis_layer(num_layers)`` (the paper's ~2/3
depth) rather than a hard-coded ``23`` — but you can still pick any layer in the
dropdown, which now spans the model's real depth.

Flask is imported lazily so the module (and the rest of the package) imports on a
machine without Flask installed; you only need it to actually run the server.

    python -m emotion_probes.visualisation.server          # or: python visualise.py
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.core.vectors import DeflectionVectors
from emotion_probes.models import ProbedModel
from emotion_probes.visualisation.clusters import PROBE_MODES
from emotion_probes.visualisation.scoring import ScoreResult, TokenActivationScorer
from emotion_probes.visualisation.template import HTML_TEMPLATE


class VisualiserServer:
    """Loads the model + vectors and serves the token-heatmap web UI."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.model: ProbedModel | None = None
        self.scorer: TokenActivationScorer | None = None
        self.num_layers: int = 0
        self.default_layer: int = 0

    # ------------------------------------------------------------------ #
    # Resource loading
    # ------------------------------------------------------------------ #
    def load(self) -> "VisualiserServer":
        """Load the model, the emotion vectors, and (if present) the deflection
        vectors, and build the scorer. Call once before serving."""
        print(f"Loading emotion vectors from {self.config.emotion_vectors_path} ...")
        expression_bank = ProbeBank.load(self.config.emotion_vectors_path)
        print(f"  {expression_bank.num_emotions} emotions x {expression_bank.num_layers} layers")

        deflection_bank: ProbeBank | None = None
        if self.config.deflection_vectors_path.exists():
            print(f"Loading deflection vectors from {self.config.deflection_vectors_path} ...")
            deflection_bank = DeflectionVectors.load(self.config.deflection_vectors_path).target
            print(f"  {deflection_bank.num_emotions} deflection (target) emotions")
        else:
            print("  (no deflection vectors found — deflection mode disabled)")

        print(f"Loading model ({self.config.model_id}) ...")
        self.model = ProbedModel(self.config)
        self.num_layers = self.model.num_layers
        self.default_layer = self.config.analysis_layer(self.num_layers)
        expression_bank.check_against(self.model, where="visualiser expression vectors")
        if deflection_bank is not None:
            deflection_bank.check_against(self.model, where="visualiser deflection vectors")
        self.scorer = TokenActivationScorer(self.model, expression_bank, deflection_bank)
        print(f"Ready. {self.num_layers} layers; default analysis layer = {self.default_layer}.")
        return self

    # ------------------------------------------------------------------ #
    # Flask app
    # ------------------------------------------------------------------ #
    def create_app(self):
        """Build the Flask application (Flask imported lazily)."""
        from flask import Flask, render_template_string, request

        app = Flask(__name__)
        server = self

        @app.route("/", methods=["GET", "POST"])
        def index():
            context = server._build_context(request, render_template_string)
            return context

        return app

    def run(self, host: str = "0.0.0.0", port: int | None = None) -> None:
        """Load resources (if needed) and start serving."""
        if self.scorer is None:
            self.load()
        port = self.config.visualiser_port if port is None else port
        self.create_app().run(host=host, port=port)

    # ------------------------------------------------------------------ #
    # Request handling
    # ------------------------------------------------------------------ #
    def _build_context(self, request, render_template_string):
        """Parse the request, score the text, and render the page."""
        fields = self._empty_fields()

        if request.method == "POST":
            text = request.form.get("text", "").strip()
            layer = self._parse_layer(request.form.get("layer"))
            probe_mode = request.form.get("probe_mode", "expression")
            if probe_mode not in PROBE_MODES:
                probe_mode = "expression"
            fields.update(text=text, selected_layer=layer, selected_probe_mode=probe_mode)

            if text:
                result = self.scorer.score(text, layer, probe_mode)
                fields.update(self._render_fields(result, probe_mode))

        return render_template_string(
            HTML_TEMPLATE,
            num_layers=self.num_layers,
            model_label=self.config.model_id,
            probe_modes=PROBE_MODES,
            probe_mode_label=PROBE_MODES.get(fields["selected_probe_mode"], "Expression"),
            **fields,
        )

    def _empty_fields(self) -> dict[str, Any]:
        """The default (GET / no-text) template fields."""
        return {
            "text": None,
            "tokens": None,
            "emotion_ranking": None,
            "all_scores_json": None,
            "emotions_json": None,
            "tokens_json": None,
            "cluster_scores_json": None,
            "expr_scores_json": None,
            "defl_scores_json": None,
            "expr_emotions_json": None,
            "defl_emotions_json": None,
            "selected_emotion": None,
            "selected_layer": self.default_layer,
            "selected_probe_mode": "expression",
        }

    def _parse_layer(self, raw: str | None) -> int:
        """Parse + clamp the requested layer to ``[0, num_layers)``."""
        try:
            layer = int(raw)
        except (TypeError, ValueError):
            return self.default_layer
        if not (0 <= layer < self.num_layers):
            return self.default_layer
        return layer

    @staticmethod
    def _render_fields(result: ScoreResult, probe_mode: str) -> dict[str, Any]:
        """Turn a :class:`ScoreResult` into the JSON blobs the template expects.

        Mirrors the original server: the primary (selected-mode) scores feed the
        initial heatmap, and BOTH spaces are emitted as JSON so the front-end can
        switch modes without another forward pass.
        """
        use_deflection = probe_mode == "deflection" and result.defl_emotions is not None
        if use_deflection:
            emotions = result.defl_emotions
            token_scores = result.defl_token_scores
        else:
            emotions = result.expr_emotions
            token_scores = result.expr_token_scores

        all_scores = [token_scores[e] for e in emotions]
        emotion_ranking = [{"emotion": e, "avg_score": ""} for e in result.ranked_emotions]

        fields: dict[str, Any] = {
            "tokens": result.tokens,
            "emotion_ranking": emotion_ranking,
            "selected_emotion": result.ranked_emotions[0] if result.ranked_emotions else None,
            "all_scores_json": json.dumps(all_scores),
            "emotions_json": json.dumps(emotions),
            "tokens_json": json.dumps(result.tokens),
            "cluster_scores_json": json.dumps(result.cluster_scores),
            "expr_scores_json": json.dumps([result.expr_token_scores[e] for e in result.expr_emotions]),
            "expr_emotions_json": json.dumps(result.expr_emotions),
        }
        if result.defl_token_scores and result.defl_emotions:
            fields["defl_scores_json"] = json.dumps(
                [result.defl_token_scores[e] for e in result.defl_emotions]
            )
            fields["defl_emotions_json"] = json.dumps(result.defl_emotions)
        return fields


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion token-heatmap visualiser (Flask).")
    parser.add_argument("--model", help="override config.model_id")
    parser.add_argument("--port", type=int, help="port to serve on (default: config.visualiser_port)")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    config = Config()
    if args.model:
        config = config.with_(model_id=args.model)
    VisualiserServer(config).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
