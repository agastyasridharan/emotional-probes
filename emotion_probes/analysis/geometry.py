"""
Emotion-space geometry — paper Figures 5-9 (Part 2 characterization).

Once we have a unit emotion vector for every emotion at every layer, we can ask
what the *space* of emotions looks like. The paper finds a strikingly human-like
structure, which this module measures four ways:

* **Cosine matrix (Fig 5/6)** — how similar each pair of emotion vectors is.
  Related emotions (``furious``/``enraged``) point in nearly the same direction.
* **Clustering (Fig 7)** — k-means on the vectors recovers intuitive emotion
  families (joy-like, fear-like, sadness-like, ...).
* **PCA (Fig 8)** — the top two principal components of the emotion vectors form
  a 2-D map; the paper reports that PC1 tracks **valence** (pleasant vs
  unpleasant) and PC2 tracks **arousal** (activated vs calm).
* **RSA across layers (Fig 9)** — representational similarity analysis: build a
  cosine matrix at several depths, then compare those matrices to each other.
  A high second-order similarity means the emotion geometry is stable across
  layers.

This analysis needs only a :class:`ProbeBank` — no model and no GPU. It runs on
the saved emotion vectors, so it is cheap to re-run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from emotion_probes.core import linalg
from emotion_probes.core.probes import ProbeBank

if TYPE_CHECKING:  # avoid a hard import; Config is only used for typing/paths
    from emotion_probes.config import Config


class GeometryAnalysis:
    """Measure the geometry of the emotion-vector space (paper Figs 5-9)."""

    def __init__(self, bank: ProbeBank, config: "Config"):
        """
        Parameters
        ----------
        bank:
            The emotion vectors to analyse (one unit vector per layer/emotion).
        config:
            The shared :class:`~emotion_probes.config.Config`; supplies the
            analysis layer and the output directory.
        """
        self.bank = bank
        self.config = config

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_layer(self, layer: int | None) -> int:
        """Default to the paper's ~2/3-depth analysis layer when ``layer`` is None."""
        if layer is None:
            return self.config.analysis_layer(self.bank.num_layers)
        return layer

    def _rsa_layers(self) -> list[int]:
        """A set of evenly spaced layers (including the first and last) for the
        across-layers RSA in :meth:`rsa_across_layers`."""
        num_layers = self.bank.num_layers
        num_points = min(num_layers, 12)  # keep the RSA matrix small and readable
        return [int(round(i)) for i in np.linspace(0, num_layers - 1, num_points)]

    # ------------------------------------------------------------------ #
    # The four measurements
    # ------------------------------------------------------------------ #
    def cosine_matrix(self, layer: int | None = None) -> np.ndarray:
        """Pairwise cosine similarity of every emotion vector at one layer.

        Parameters
        ----------
        layer:
            Layer index; defaults to the analysis layer.

        Returns
        -------
        np.ndarray
            Symmetric ``(num_emotions, num_emotions)`` matrix; entry ``(i, j)``
            is the cosine similarity between emotions ``i`` and ``j``.
        """
        layer = self._resolve_layer(layer)
        return linalg.cosine_similarity_matrix(self.bank.layer_vectors(layer))

    def cluster(self, k: int = 10, layer: int | None = None) -> dict:
        """Group emotions into ``k`` clusters with k-means (paper Fig 7).

        Parameters
        ----------
        k:
            Number of clusters to form.
        layer:
            Layer index; defaults to the analysis layer.

        Returns
        -------
        dict
            ``{"layer": int, "k": int,
               "assignments": {emotion: cluster_id},
               "members": {cluster_id: [emotion, ...]}}``.
        """
        from sklearn.cluster import KMeans  # lazy: keep sklearn out of import time

        layer = self._resolve_layer(layer)
        vectors = self.bank.layer_vectors(layer)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(vectors)

        assignments: dict[str, int] = {}
        members: dict[int, list[str]] = {cluster_id: [] for cluster_id in range(k)}
        for emotion, label in zip(self.bank.emotions, labels):
            cluster_id = int(label)
            assignments[emotion] = cluster_id
            members[cluster_id].append(emotion)
        return {"layer": layer, "k": k, "assignments": assignments, "members": members}

    def pca(self, layer: int | None = None) -> dict:
        """Project the emotion vectors onto their top two principal components.

        The paper (Fig 8) reports that PC1 lines up with human **valence** and
        PC2 with human **arousal**, so the 2-D coordinates are an interpretable
        emotion map.

        Parameters
        ----------
        layer:
            Layer index; defaults to the analysis layer.

        Returns
        -------
        dict
            ``{"layer": int,
               "coords": {emotion: [pc1, pc2]},
               "explained_variance_ratio": [pc1_ratio, pc2_ratio]}``.
        """
        from sklearn.decomposition import PCA  # lazy import

        layer = self._resolve_layer(layer)
        vectors = self.bank.layer_vectors(layer)
        pca = PCA(n_components=2, random_state=0)
        projected = pca.fit_transform(vectors)  # (num_emotions, 2)

        coords = {
            emotion: [float(projected[i, 0]), float(projected[i, 1])]
            for i, emotion in enumerate(self.bank.emotions)
        }
        ratio = [float(r) for r in pca.explained_variance_ratio_]
        return {"layer": layer, "coords": coords, "explained_variance_ratio": ratio}

    def rsa_across_layers(self) -> np.ndarray:
        """Representational similarity analysis across depth (paper Fig 9).

        For each of an evenly spaced set of layers we build that layer's cosine
        matrix (the "first-order" emotion geometry). We then compare those
        matrices to one another by flattening each into a vector and taking the
        cosine similarity between every pair — the "second-order" similarity. A
        high value means the emotion geometry is consistent at those two depths.

        Returns
        -------
        np.ndarray
            ``(num_rsa_layers, num_rsa_layers)`` second-order similarity matrix.
            Use :meth:`_rsa_layers` for the matching layer indices.
        """
        layers = self._rsa_layers()
        # Flatten each layer's cosine matrix into one long vector (one row each).
        flattened = np.stack(
            [self.cosine_matrix(layer).ravel() for layer in layers], axis=0
        )
        return linalg.cosine_similarity_matrix(flattened)

    # ------------------------------------------------------------------ #
    # Bundle + save
    # ------------------------------------------------------------------ #
    def run(self, k: int = 10, layer: int | None = None) -> dict:
        """Run all four measurements, save JSON + plots, and return the results.

        Parameters
        ----------
        k:
            Number of clusters for :meth:`cluster`.
        layer:
            Layer to analyse for the cosine matrix, clustering and PCA; defaults
            to the analysis layer. (The RSA always spans several layers.)

        Returns
        -------
        dict
            JSON-serializable results dict, also written to
            ``config.analysis_dir/geometry.json``.
        """
        layer = self._resolve_layer(layer)
        cosine = self.cosine_matrix(layer)
        clusters = self.cluster(k=k, layer=layer)
        pca = self.pca(layer)
        rsa_layers = self._rsa_layers()
        rsa = self.rsa_across_layers()

        results = {
            "layer": layer,
            "emotions": list(self.bank.emotions),
            "cosine_matrix": cosine.tolist(),
            "clusters": clusters,
            "pca": pca,
            "rsa_layers": rsa_layers,
            "rsa_matrix": rsa.tolist(),
        }

        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "geometry.json", "w") as f:
            json.dump(results, f, indent=2)

        self._plot(cosine, pca, rsa, rsa_layers)
        return results

    def _plot(
        self,
        cosine: np.ndarray,
        pca: dict,
        rsa: np.ndarray,
        rsa_layers: list[int],
    ) -> None:
        """Save the cosine heatmap, the PCA scatter, and the RSA heatmap.

        Matplotlib is imported lazily so the module loads without it; if it is
        not installed we skip plotting and keep the JSON output.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless: write files, never open a window
            import matplotlib.pyplot as plt
        except ImportError:
            return

        out_dir = self.config.analysis_dir

        # --- cosine heatmap (Fig 5/6) ---
        fig, ax = plt.subplots(figsize=(10, 9))
        image = ax.imshow(cosine, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title("Emotion-vector cosine similarity")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "geometry_cosine.png", dpi=150)
        plt.close(fig)

        # --- PCA scatter (Fig 8): PC1 ~ valence, PC2 ~ arousal ---
        coords = pca["coords"]
        ratio = pca["explained_variance_ratio"]
        xs = [coords[emotion][0] for emotion in self.bank.emotions]
        ys = [coords[emotion][1] for emotion in self.bank.emotions]
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.scatter(xs, ys, s=12, alpha=0.7)
        for emotion, x, y in zip(self.bank.emotions, xs, ys):
            ax.annotate(emotion, (x, y), fontsize=6, alpha=0.8)
        ax.set_xlabel(f"PC1 (valence?)  -  {ratio[0]:.1%} var")
        ax.set_ylabel(f"PC2 (arousal?)  -  {ratio[1]:.1%} var")
        ax.set_title("Emotion space (top 2 principal components)")
        ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.axvline(0.0, color="grey", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "geometry_pca.png", dpi=150)
        plt.close(fig)

        # --- RSA heatmap (Fig 9) ---
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(rsa, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(rsa_layers)))
        ax.set_yticks(range(len(rsa_layers)))
        ax.set_xticklabels(rsa_layers, rotation=90, fontsize=7)
        ax.set_yticklabels(rsa_layers, fontsize=7)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        ax.set_title("RSA: cross-layer similarity of emotion geometry")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "geometry_rsa.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """CLI: build the geometry analysis from saved emotion vectors."""
    import argparse

    from emotion_probes.config import default_config

    parser = argparse.ArgumentParser(description="Emotion-space geometry (paper Figs 5-9).")
    parser.add_argument(
        "--vectors",
        default=None,
        help="Path to the emotion-vectors .npz (default: config.emotion_vectors_path).",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of k-means clusters.")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index to analyse (default: the ~2/3-depth analysis layer).",
    )
    args = parser.parse_args()

    config = default_config()
    vectors_path = args.vectors or config.emotion_vectors_path
    bank = ProbeBank.load(vectors_path)
    results = GeometryAnalysis(bank, config).run(k=args.k, layer=args.layer)
    print(f"Analysed layer {results['layer']} of {bank.num_layers}; "
          f"PC1/PC2 var = {results['pca']['explained_variance_ratio']}; "
          f"wrote outputs to {config.analysis_dir}")


if __name__ == "__main__":
    main()
