"""
Human alignment of the emotion geometry — paper Figures 8 and 58.

The geometry analysis (:mod:`emotion_probes.analysis.geometry`) finds that the
top two principal components of the emotion-vector space form a 2-D map. This
module tests the paper's claim that this map is *human-like*: PC1 should track
human **valence** (pleasant vs unpleasant) and PC2 human **arousal** (activated
vs calm).

There are two ways to get the human side of the comparison:

* **Published norms (Fig 8).** :meth:`correlate_with_norms` loads the
  Russell & Mehrabian (1977) valence/arousal ratings (you fill them into a CSV
  template — see :mod:`emotion_probes.data.human_norms`) and correlates PC1 with
  valence and PC2 with arousal across the overlapping emotions.
* **LLM-as-judge (Fig 58).** :meth:`llm_judge_ratings` instead has the *model
  itself* rate every emotion 1-7 on valence and arousal. This needs no external
  data but **does need the loaded model (a GPU)** and so is optional.

The PCA itself is reused from :class:`~emotion_probes.analysis.geometry.GeometryAnalysis`
so the two analyses always agree on the coordinates.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Sequence

import numpy as np

from emotion_probes.analysis.geometry import GeometryAnalysis
from emotion_probes.core.probes import ProbeBank
from emotion_probes.data.human_norms import (
    NORMS_FILENAME,
    ensure_template,
    load_human_norms,
)

if TYPE_CHECKING:  # keep torch and Config out of import-time
    from emotion_probes.config import Config
    from emotion_probes.models.language_model import ProbedModel


class HumanAlignmentAnalysis:
    """Correlate the emotion geometry with human (or LLM-judged) valence/arousal."""

    def __init__(
        self,
        bank: ProbeBank,
        config: "Config",
        model: "ProbedModel | None" = None,
    ):
        """
        Parameters
        ----------
        bank:
            The emotion vectors whose geometry we are validating.
        config:
            The shared :class:`~emotion_probes.config.Config`.
        model:
            Optional loaded :class:`ProbedModel`. Only the LLM-judge path
            (:meth:`llm_judge_ratings`) needs it; correlating with published
            norms does not.
        """
        self.bank = bank
        self.config = config
        self.model = model
        if self.model is not None:
            self.bank.check_against(self.model, where="human_alignment")
        # Reuse the PCA so coordinates match the geometry analysis exactly.
        self.geometry = GeometryAnalysis(bank, config)

    # ------------------------------------------------------------------ #
    # Published-norms path (Fig 8)
    # ------------------------------------------------------------------ #
    def correlate_with_norms(self, norms_csv_path: str, layer: int | None = None) -> dict:
        """Correlate PC1 with human valence and PC2 with human arousal.

        Parameters
        ----------
        norms_csv_path:
            Path to the filled human-norms CSV (see
            :mod:`emotion_probes.data.human_norms`).
        layer:
            Layer to take the PCA at; defaults to the analysis layer.

        Returns
        -------
        dict
            ``{"layer": int, "n": int, "emotions": [...],
               "valence_vs_pc1": {"r": float, "p": float},
               "arousal_vs_pc2": {"r": float, "p": float}}``.
            ``n`` is the number of emotions present in both the bank and the
            norms. If fewer than three overlap, the correlations are ``None`` and
            a ``"note"`` explains why.
        """
        from scipy.stats import pearsonr  # lazy import

        pca = self.geometry.pca(layer)
        coords = pca["coords"]
        norms = load_human_norms(norms_csv_path)

        # Keep only emotions present in BOTH, in a single consistent order.
        shared = [e for e in self.bank.emotions if e in coords and e in norms]
        result: dict = {"layer": pca["layer"], "n": len(shared), "emotions": shared}

        if len(shared) < 3:
            result["valence_vs_pc1"] = None
            result["arousal_vs_pc2"] = None
            result["note"] = (
                "Fewer than 3 emotions overlap between the bank and the filled "
                f"norms CSV ({norms_csv_path}); fill in more rows to correlate."
            )
            return result

        pc1 = np.array([coords[e][0] for e in shared], dtype=np.float64)
        pc2 = np.array([coords[e][1] for e in shared], dtype=np.float64)
        valence = np.array([norms[e]["valence"] for e in shared], dtype=np.float64)
        arousal = np.array([norms[e]["arousal"] for e in shared], dtype=np.float64)

        valence_r, valence_p = pearsonr(pc1, valence)
        arousal_r, arousal_p = pearsonr(pc2, arousal)
        result["valence_vs_pc1"] = {"r": float(valence_r), "p": float(valence_p)}
        result["arousal_vs_pc2"] = {"r": float(arousal_r), "p": float(arousal_p)}
        return result

    # ------------------------------------------------------------------ #
    # LLM-as-judge path (Fig 58) — needs the model (GPU)
    # ------------------------------------------------------------------ #
    def llm_judge_ratings(self, emotions: Sequence[str] | None = None) -> dict[str, dict]:
        """Have the model rate each emotion 1-7 on valence and arousal.

        This is the paper's Fig 58 approach: instead of external human norms, the
        model itself scores every emotion. **It requires the loaded model (a
        GPU)** — construct this class with ``model=...`` to use it.

        Parameters
        ----------
        emotions:
            Which emotions to rate; defaults to every emotion in the bank.

        Returns
        -------
        dict[str, dict]
            ``{emotion: {"valence": int|None, "arousal": int|None}}``. An entry
            is ``None`` when the model's reply could not be parsed as a 1-7
            integer.
        """
        if self.model is None:
            raise RuntimeError(
                "llm_judge_ratings needs a loaded model; construct "
                "HumanAlignmentAnalysis(bank, config, model=...) on a GPU."
            )
        emotions = list(emotions) if emotions is not None else list(self.bank.emotions)

        ratings: dict[str, dict] = {}
        for emotion in emotions:
            ratings[emotion] = {
                "valence": self._rate(emotion, "valence"),
                "arousal": self._rate(emotion, "arousal"),
            }
        return ratings

    def _rate(self, emotion: str, dimension: str) -> int | None:
        """Ask the model for a 1-7 rating of ``emotion`` on one ``dimension``.

        ``dimension`` is ``"valence"`` (1 = very unpleasant, 7 = very pleasant) or
        ``"arousal"`` (1 = very calm, 7 = very activated). Returns the parsed
        integer, or ``None`` if the reply does not contain a 1-7 digit.
        """
        if dimension == "valence":
            scale = "1 means very unpleasant/negative and 7 means very pleasant/positive"
        else:
            scale = "1 means very calm/low-energy and 7 means very excited/high-energy"
        question = (
            f"On a scale from 1 to 7, where {scale}, how would you rate the "
            f"feeling of being \"{emotion}\"? Reply with only the number."
        )
        prompt = f"Human: {question}\n\nAssistant:"
        # Greedy decoding for a stable, short answer.
        reply = self.model.generate(
            prompt, max_new_tokens=8, temperature=0.0, do_sample=False
        )
        return self._parse_rating(reply)

    @staticmethod
    def _parse_rating(reply: str) -> int | None:
        """Pull the first integer in 1-7 out of a free-text model reply."""
        for match in re.finditer(r"\d+", reply):
            value = int(match.group())
            if 1 <= value <= 7:
                return value
        return None

    # ------------------------------------------------------------------ #
    # Bundle + save
    # ------------------------------------------------------------------ #
    def run(self, layer: int | None = None, use_llm_judge: bool = False) -> dict:
        """Run the alignment check and save it to ``human_alignment.json``.

        Tries the published-norms path first: if the norms CSV is missing it
        writes a blank template (via :func:`ensure_template`) and tells you to
        fill it in. If ``use_llm_judge`` is True and a model is available, it also
        runs the LLM-judge ratings and correlates them with the PCA.

        Parameters
        ----------
        layer:
            Layer to take the PCA at; defaults to the analysis layer.
        use_llm_judge:
            Also run the (model-requiring) LLM-judge path.

        Returns
        -------
        dict
            JSON-serializable results, also written under ``config.analysis_dir``.
        """
        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        results: dict = {}

        # --- norms path ---
        norms_path = out_dir / NORMS_FILENAME
        loaded = load_human_norms(norms_path)
        if not loaded:
            ensure_template(norms_path)
            results["norms"] = {
                "available": False,
                "template_path": str(norms_path),
                "note": (
                    "No filled human norms found. A blank template was written; "
                    "fill in valence/arousal (e.g. Russell & Mehrabian 1977) and "
                    "re-run to correlate PC1/PC2 with human ratings."
                ),
            }
        else:
            results["norms"] = {
                "available": True,
                "correlation": self.correlate_with_norms(str(norms_path), layer=layer),
            }

        # --- LLM-judge path (optional, needs the model) ---
        if use_llm_judge and self.model is not None:
            ratings = self.llm_judge_ratings()
            results["llm_judge"] = {
                "ratings": ratings,
                "correlation": self._correlate_judge(ratings, layer=layer),
            }
        elif use_llm_judge:
            results["llm_judge"] = {
                "note": "LLM-judge requested but no model was provided (needs a GPU)."
            }

        with open(out_dir / "human_alignment.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    def _correlate_judge(
        self, ratings: dict[str, dict], layer: int | None = None
    ) -> dict:
        """Correlate PC1/PC2 with the model's own valence/arousal ratings.

        Mirrors :meth:`correlate_with_norms` but on the LLM-judge ratings, using
        only emotions the model successfully rated on both dimensions.
        """
        from scipy.stats import pearsonr  # lazy import

        pca = self.geometry.pca(layer)
        coords = pca["coords"]
        shared = [
            e
            for e in self.bank.emotions
            if e in coords
            and ratings.get(e, {}).get("valence") is not None
            and ratings.get(e, {}).get("arousal") is not None
        ]
        result: dict = {"layer": pca["layer"], "n": len(shared), "emotions": shared}

        if len(shared) < 3:
            result["valence_vs_pc1"] = None
            result["arousal_vs_pc2"] = None
            result["note"] = "Fewer than 3 emotions were rated on both dimensions."
            return result

        pc1 = np.array([coords[e][0] for e in shared], dtype=np.float64)
        pc2 = np.array([coords[e][1] for e in shared], dtype=np.float64)
        valence = np.array([ratings[e]["valence"] for e in shared], dtype=np.float64)
        arousal = np.array([ratings[e]["arousal"] for e in shared], dtype=np.float64)

        valence_r, valence_p = pearsonr(pc1, valence)
        arousal_r, arousal_p = pearsonr(pc2, arousal)
        result["valence_vs_pc1"] = {"r": float(valence_r), "p": float(valence_p)}
        result["arousal_vs_pc2"] = {"r": float(arousal_r), "p": float(arousal_p)}
        return result


def main() -> None:
    """CLI: correlate the emotion geometry with human norms (and optionally an
    LLM judge). The LLM-judge path needs the model and is left to the GPU run."""
    import argparse

    from emotion_probes.config import default_config

    parser = argparse.ArgumentParser(description="Human-alignment of emotion geometry (Figs 8/58).")
    parser.add_argument(
        "--vectors",
        default=None,
        help="Path to the emotion-vectors .npz (default: config.emotion_vectors_path).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index for the PCA (default: the ~2/3-depth analysis layer).",
    )
    args = parser.parse_args()

    config = default_config()
    vectors_path = args.vectors or config.emotion_vectors_path
    bank = ProbeBank.load(vectors_path)
    # No model on a CPU box: the norms path runs; the LLM-judge path needs a GPU.
    results = HumanAlignmentAnalysis(bank, config, model=None).run(layer=args.layer)
    print(f"Wrote human-alignment results to {config.analysis_dir}; "
          f"norms available: {results['norms']['available']}")


if __name__ == "__main__":
    main()
