"""
Logit-lens analysis (paper Table 1).

Part 1 of the paper validates the emotion vectors by asking a simple question:
*what does an emotion direction "say" when you read it straight off the model's
own vocabulary?* The classic **logit lens** answers this by pushing a
residual-stream direction through the model's unembedding matrix and reading the
resulting per-token logits. Tokens with the highest logit are the ones the
direction *promotes*; tokens with the lowest are the ones it *suppresses*.

For a faithful emotion vector this lands on exactly the words you'd expect — the
paper's headline example is::

    Desperate -> desperate, urgent, bankrupt, ...

This module takes each emotion's unit vector at the analysis layer
(``config.analysis_layer(model.num_layers)``), runs :meth:`ProbedModel.logit_lens`
on it, and saves the resulting table of promoted/suppressed tokens.

No results are fabricated: the actual token strings come from the loaded model,
so this must be run on the GPU box where the model is available.
"""

from __future__ import annotations

import json

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models import ProbedModel


class LogitLensAnalysis:
    """Replicate Table 1: tokens each emotion vector promotes / suppresses."""

    def __init__(self, model: ProbedModel, bank: ProbeBank, config: Config):
        """
        Parameters
        ----------
        model:  the loaded :class:`ProbedModel` (its unembedding does the lens).
        bank:   the emotion vectors to read off the vocabulary.
        config: paths and the analysis-layer fraction.
        """
        self.model = model
        self.bank = bank
        self.bank.check_against(self.model, where="logit_lens")
        self.config = config

    def run(
        self,
        emotions: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, dict[str, list[str]]]:
        """Build the logit-lens table.

        For each emotion, take its unit vector at the analysis layer and ask the
        model which vocabulary tokens that direction most promotes (``"top"``)
        and most suppresses (``"bottom"``).

        Parameters
        ----------
        emotions:
            Which emotions to include. ``None`` (default) uses every emotion in
            the bank, in bank order.
        top_k:
            How many tokens to keep on each side (paper shows ~5).

        Returns
        -------
        dict
            ``{emotion: {"top": [token, ...], "bottom": [token, ...]}}``.
        """
        layer = self.config.analysis_layer(self.model.num_layers)
        emotions = list(self.bank.emotions) if emotions is None else list(emotions)

        table: dict[str, dict[str, list[str]]] = {}
        for emotion in emotions:
            index = self.bank.index_of(emotion)
            direction = self.bank.vectors[layer, index]  # (hidden,) unit vector
            top, bottom = self.model.logit_lens(direction, top_k=top_k)
            table[emotion] = {"top": top, "bottom": bottom}

        self._save(table, layer)
        self._print_examples(table)
        return table

    def _save(self, table: dict[str, dict[str, list[str]]], layer: int) -> None:
        """Write the table to ``config.analysis_dir / "logit_lens.json"``."""
        self.config.analysis_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.analysis_dir / "logit_lens.json"
        payload = {"layer": layer, "table": table}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Saved logit-lens table for {len(table)} emotions -> {path}")

    def _print_examples(self, table: dict[str, dict[str, list[str]]], how_many: int = 3) -> None:
        """Pretty-print a few rows in the paper's ``Emotion -> tok, tok, ...`` style."""
        for emotion in list(table)[:how_many]:
            promoted = ", ".join(token.strip() for token in table[emotion]["top"])
            print(f"{emotion} -> {promoted}")


def main() -> None:
    """CLI: load the model + emotion vectors, build and save the logit-lens table."""
    import argparse

    parser = argparse.ArgumentParser(description="Logit-lens analysis (paper Table 1).")
    parser.add_argument("--top-k", type=int, default=5, help="tokens kept per side")
    parser.add_argument(
        "--emotions",
        nargs="*",
        default=None,
        help="subset of emotions (default: all in the bank)",
    )
    args = parser.parse_args()

    config = Config()
    bank = ProbeBank.load(config.emotion_vectors_path)
    model = ProbedModel(config)
    LogitLensAnalysis(model, bank, config).run(emotions=args.emotions, top_k=args.top_k)


if __name__ == "__main__":
    main()
