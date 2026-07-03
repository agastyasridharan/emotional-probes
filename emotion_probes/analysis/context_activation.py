"""
Context-activation analysis (paper Figure 1).

Figure 1 shows that the emotion vectors are not just abstract directions: when
you slide them across a real piece of text, they *light up on the words that
carry the emotion*. The paper visualises this by colouring each token by its
probe score; the tokens that glow brightest for "anger" are the angry ones, and
so on.

This module reproduces that result quantitatively. For every text it reads the
per-token residual stream at the analysis layer
(``config.analysis_layer(model.num_layers)``), projects each token onto every
emotion vector (:meth:`ProbeBank.project`), and then, per emotion:

1. collects the probe score of every token in the whole corpus,
2. finds the score threshold at the requested percentile (default 90th), and
3. returns the ``top_snippets`` highest-scoring tokens, each with the token
   string, its score, and a small surrounding context window — the textual
   analogue of Figure 1's highlighted spans.

No results are fabricated: the activations come from the loaded model, so this
must run on the GPU box. Token strings and context windows are recovered exactly
from the tokenizer's character offsets, so the snippets line up with the source
text.
"""

from __future__ import annotations

import json

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models import ProbedModel


class ContextActivationAnalysis:
    """Replicate Figure 1: where each emotion vector fires across a corpus."""

    def __init__(self, model: ProbedModel, bank: ProbeBank, config: Config):
        """
        Parameters
        ----------
        model:  the loaded :class:`ProbedModel` (reads per-token activations).
        bank:   the emotion vectors used to score each token.
        config: paths and the analysis-layer fraction.
        """
        self.model = model
        self.bank = bank
        self.bank.check_against(self.model, where="context_activation")
        self.config = config

    def run(
        self,
        texts: list[str],
        emotions: list[str] | None = None,
        percentile: float = 90.0,
        top_snippets: int = 10,
        context_chars: int = 60,
    ) -> dict:
        """Find the highest-scoring tokens for each emotion across ``texts``.

        Parameters
        ----------
        texts:
            The corpus to scan, one string per document.
        emotions:
            Which emotions to report. ``None`` (default) uses every emotion in
            the bank.
        percentile:
            Tokens are reported only if their score is at or above this
            percentile of the emotion's score distribution over the whole corpus
            (paper highlights the strongly-activating tokens; default 90th).
        top_snippets:
            How many of the above-threshold tokens to return per emotion, highest
            score first.
        context_chars:
            How many characters of surrounding text to include on each side of a
            token to form its context window.

        Returns
        -------
        dict
            ``{"layer": int, "percentile": float, "emotions": {emotion: {...}}}``
            where each emotion maps to its score ``threshold`` and a list of
            ``snippets`` (``{text_index, token, score, context}``).
        """
        layer = self.config.analysis_layer(self.model.num_layers)
        emotions = list(self.bank.emotions) if emotions is None else list(emotions)
        emotion_indices = [self.bank.index_of(emotion) for emotion in emotions]

        # One flat record per token across the whole corpus, plus a parallel
        # score array of shape (num_tokens, num_selected_emotions).
        token_records: list[dict] = []
        score_rows: list[np.ndarray] = []

        for text_index, item in enumerate(
            self.model.iter_token_activations(texts, layers=[layer], with_offsets=True)
        ):
            activations = item["activations"][0]            # (seq_len, hidden), layer axis = 0
            offsets = item["offset_mapping"]                # list of (char_start, char_end)
            text = texts[text_index]

            all_scores = self.bank.project(activations, layer)        # (seq_len, num_emotions)
            selected_scores = all_scores[:, emotion_indices]          # (seq_len, len(emotions))

            for position, (char_start, char_end) in enumerate(offsets):
                token_records.append(
                    self._token_record(text, text_index, char_start, char_end, context_chars)
                )
                score_rows.append(selected_scores[position])

        result = self._summarize(
            emotions, token_records, score_rows, layer, percentile, top_snippets
        )
        self._save(result)
        return result

    def _token_record(
        self,
        text: str,
        text_index: int,
        char_start: int,
        char_end: int,
        context_chars: int,
    ) -> dict:
        """Recover a token's string and surrounding context from character offsets."""
        token = text[char_start:char_end]
        window_start = max(0, char_start - context_chars)
        window_end = min(len(text), char_end + context_chars)
        context = text[window_start:window_end]
        return {"text_index": text_index, "token": token, "context": context}

    def _summarize(
        self,
        emotions: list[str],
        token_records: list[dict],
        score_rows: list[np.ndarray],
        layer: int,
        percentile: float,
        top_snippets: int,
    ) -> dict:
        """Threshold each emotion at ``percentile`` and keep its top snippets."""
        per_emotion: dict[str, dict] = {}
        if not score_rows:
            for emotion in emotions:
                per_emotion[emotion] = {"threshold": 0.0, "snippets": []}
            return {
                "layer": layer,
                "percentile": percentile,
                "num_tokens": 0,
                "emotions": per_emotion,
            }

        scores = np.asarray(score_rows, dtype=np.float64)  # (num_tokens, len(emotions))
        for column, emotion in enumerate(emotions):
            emotion_scores = scores[:, column]
            threshold = float(np.percentile(emotion_scores, percentile))
            above = np.flatnonzero(emotion_scores >= threshold)
            # Highest score first, then keep at most `top_snippets`.
            ranked = above[np.argsort(-emotion_scores[above])][:top_snippets]
            snippets = [
                {
                    "text_index": token_records[i]["text_index"],
                    "token": token_records[i]["token"],
                    "score": float(emotion_scores[i]),
                    "context": token_records[i]["context"],
                }
                for i in ranked
            ]
            per_emotion[emotion] = {"threshold": threshold, "snippets": snippets}

        return {
            "layer": layer,
            "percentile": percentile,
            "num_tokens": int(scores.shape[0]),
            "emotions": per_emotion,
        }

    def _save(self, result: dict) -> None:
        """Write the result to ``config.analysis_dir / "context_activation.json"``."""
        self.config.analysis_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.analysis_dir / "context_activation.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(
            f"Saved context-activation snippets for {len(result['emotions'])} emotions "
            f"over {result['num_tokens']} tokens -> {path}"
        )


def main() -> None:
    """CLI: scan a corpus (one text per line of a file) for emotion activations."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Context-activation analysis (paper Figure 1).")
    parser.add_argument("corpus", type=Path, help="text file, one document per line")
    parser.add_argument("--percentile", type=float, default=90.0, help="score percentile cutoff")
    parser.add_argument("--top-snippets", type=int, default=10, help="snippets kept per emotion")
    parser.add_argument(
        "--emotions",
        nargs="*",
        default=None,
        help="subset of emotions (default: all in the bank)",
    )
    args = parser.parse_args()

    texts = [line for line in args.corpus.read_text().splitlines() if line.strip()]

    config = Config()
    bank = ProbeBank.load(config.emotion_vectors_path)
    model = ProbedModel(config)
    ContextActivationAnalysis(model, bank, config).run(
        texts,
        emotions=args.emotions,
        percentile=args.percentile,
        top_snippets=args.top_snippets,
    )


if __name__ == "__main__":
    main()
