"""
The three activation extractors.

Each takes a :class:`~emotion_probes.models.ProbedModel` and a :class:`Config`,
runs the model over a dataset, and writes NumPy arrays that the vector computers
consume. The heavy GPU work is delegated to ``ProbedModel``; the extractors only
decide *which token positions* to average and *how to group* the results.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.spans import (
    char_spans_to_token_mask,
    find_dialogue_start_char,
    find_speaker_char_spans,
)
from emotion_probes.data import EMOTIONS
from emotion_probes.models.language_model import ProbedModel


def _ordered_present(present: set[str]) -> list[str]:
    """Emotions that appear in the data, in the canonical EMOTIONS order."""
    return [e for e in EMOTIONS if e in present]


# --------------------------------------------------------------------------- #
class StoryActivationExtractor:
    """Per-emotion mean residual stream from the emotion-story dataset.

    For each story we average the residual stream from token ``token_skip``
    onward (delegated to ``ProbedModel.extract_means``); then we average those
    per-story means within each emotion. Results are written one ``.npy`` file
    per emotion (so a long run is resumable), then combined into one array.
    """

    def __init__(self, model: ProbedModel, config: Config | None = None):
        self.model = model
        self.config = config or model.config
        self.per_emotion_dir = self.config.activations_dir

    def run(self) -> Path:
        import pandas as pd

        self.per_emotion_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(self.config.stories_csv)
        for emotion in _ordered_present(set(df["emotion"])):
            out_file = self.per_emotion_dir / f"{emotion}.npy"
            if out_file.exists():
                continue
            stories = df.loc[df["emotion"] == emotion, "story"].astype(str).tolist()
            means = self.model.extract_means(stories)        # (n_stories, L, H)
            np.save(out_file, means.mean(axis=0))            # (L, H)
            print(f"  {emotion}: {len(stories)} stories")
        return self.combine()

    def combine(self) -> Path:
        """Stack the per-emotion ``.npy`` files into ``activations/stories.npz``."""
        present = {p.stem for p in self.per_emotion_dir.glob("*.npy")}
        emotions = _ordered_present(present)
        stacked = np.stack([np.load(self.per_emotion_dir / f"{e}.npy") for e in emotions])  # (E, L, H)
        means = np.transpose(stacked, (1, 0, 2))  # (L, E, H)
        out = self.config.activations_dir.parent / "stories.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, means=means.astype(np.float32), emotions=np.array(emotions, dtype=object))
        print(f"Saved {out} — means {means.shape}, {len(emotions)} emotions")
        return out


# --------------------------------------------------------------------------- #
class NeutralActivationExtractor:
    """Per-text means on a neutral corpus (the PCA confound baseline).

    Works for both neutral stories and neutral dialogues — just point it at the
    right CSV / text column / output path.
    """

    def __init__(self, model: ProbedModel, csv_path: Path, text_column: str, output_path: Path, config: Config | None = None):
        self.model = model
        self.config = config or model.config
        self.csv_path = csv_path
        self.text_column = text_column
        self.output_path = output_path

    def run(self) -> Path:
        import pandas as pd

        df = pd.read_csv(self.csv_path)
        texts = df[self.text_column].astype(str).tolist()
        means = self.model.extract_means(
            texts, batch_size=self.config.batch_size, max_length=self.config.max_length
        )  # (N, L, H)
        activations = np.transpose(means, (1, 0, 2))  # (L, N, H)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.output_path, activations=activations.astype(np.float32))
        print(f"Saved {self.output_path} — activations {activations.shape}")
        return self.output_path


# --------------------------------------------------------------------------- #
class DeflectionActivationExtractor:
    """Scenario / masking-speaker means for the deflection dataset.

    For each dialogue we separate two token regions using EXACT character->token
    mapping (Issue #7, via :func:`char_spans_to_token_mask`):

    * **scenario tokens** — the preamble where the real (hidden) emotion is named,
    * **masking-speaker tokens** — only NAME_A's utterances, where they display a
      different emotion.

    We accumulate running sums per emotion (and per (real, displayed) pair) so we
    never store the full activation tensor, and checkpoint periodically.
    """

    def __init__(self, model: ProbedModel, config: Config | None = None):
        self.model = model
        self.config = config or model.config

    def run(self) -> Path:
        import pandas as pd

        df = pd.read_csv(self.config.deflection_dialogues_csv)
        n_layers, hidden = self.model.num_layers, self.model.hidden_size

        # Accumulators (resume from checkpoint if present).
        state = self._load_checkpoint()
        if state is None:
            state = {
                "scenario_sums": defaultdict(lambda: np.zeros((n_layers, hidden), dtype=np.float64)),
                "dialogue_sums": defaultdict(lambda: np.zeros((n_layers, hidden), dtype=np.float64)),
                "displayed_sums": defaultdict(lambda: np.zeros((n_layers, hidden), dtype=np.float64)),
                "pair_sums": defaultdict(lambda: np.zeros((n_layers, hidden), dtype=np.float64)),
                "target_counts": defaultdict(int),
                "displayed_counts": defaultdict(int),
                "pair_counts": defaultdict(int),
                "next_index": 0,
                "skipped": 0,
            }

        start = state["next_index"]
        rows = df.iloc[start:].to_dict("records")
        texts = [str(r["dialogue"]) for r in rows]

        processed = start
        for row, item in zip(rows, self.model.iter_token_activations(
            texts, batch_size=self.config.deflection_batch_size,
            max_length=self.config.deflection_max_length, with_offsets=True,
        )):
            processed += 1
            self._accumulate(row, item, state)
            if processed % (self.config.checkpoint_every * self.config.deflection_batch_size) == 0:
                state["next_index"] = processed
                self._save_checkpoint(state)

        print(f"Processed {processed} dialogues, skipped {state['skipped']}.")
        return self._finalize(state, n_layers, hidden)

    # ---- per-dialogue accumulation ----
    def _accumulate(self, row: dict, item: dict, state: dict) -> None:
        text = str(row["dialogue"])
        name_a, name_b = str(row["name_a"]), str(row["name_b"])
        real, displayed = row["real_emotion"], row["displayed_emotion"]
        offsets = item["offset_mapping"]
        acts = item["activations"]  # (L, seq, H)

        dialogue_start = find_dialogue_start_char(text, [name_a, name_b])
        a_spans = find_speaker_char_spans(text, name_a, [name_a, name_b])
        if dialogue_start is None or not a_spans:
            state["skipped"] += 1
            return

        scenario_mask = char_spans_to_token_mask(offsets, [(0, dialogue_start)])
        speaker_a_mask = char_spans_to_token_mask(offsets, a_spans)
        if scenario_mask.sum() < 1 or speaker_a_mask.sum() < 1:
            state["skipped"] += 1
            return

        scenario_mean = self._masked_mean_all_layers(acts, scenario_mask)
        speaker_a_mean = self._masked_mean_all_layers(acts, speaker_a_mask)

        state["scenario_sums"][real] += scenario_mean
        state["dialogue_sums"][real] += speaker_a_mean
        state["displayed_sums"][displayed] += speaker_a_mean
        state["pair_sums"][(real, displayed)] += speaker_a_mean
        state["target_counts"][real] += 1
        state["displayed_counts"][displayed] += 1
        state["pair_counts"][(real, displayed)] += 1

    @staticmethod
    def _masked_mean_all_layers(acts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mean over masked token positions for every layer at once.

        acts: (L, seq, H); mask: (seq,) bool -> returns (L, H)."""
        m = mask.astype(np.float64)
        count = max(m.sum(), 1.0)
        return (acts * m[None, :, None]).sum(axis=1) / count

    # ---- checkpoint IO ----
    def _save_checkpoint(self, state: dict) -> None:
        path = self.config.deflection_checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        # defaultdicts with lambda factories can't be pickled directly; convert.
        dumpable = {
            k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in state.items()
        }
        # Atomic write: a kill mid-dump (12h-interruption policy, OOM) must never
        # leave a truncated checkpoint that crashes the next resume. Write a temp
        # file in the same dir, fsync, then os.replace (atomic on POSIX).
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(dumpable, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"  checkpoint @ {state['next_index']} -> {path}", flush=True)

    def _load_checkpoint(self) -> dict | None:
        path = self.config.deflection_checkpoint
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            # A truncated/corrupt checkpoint (e.g. killed mid-write before this
            # atomic-write fix landed) would otherwise crash the whole run. Warn
            # loudly and restart extraction from scratch rather than die.
            print(f"  WARNING: checkpoint {path} unreadable ({type(e).__name__}: "
                  f"{e}); restarting deflection extraction from index 0", flush=True)
            return None
        n_layers, hidden = self.model.num_layers, self.model.hidden_size
        # Rehydrate the running-sum defaultdicts.
        for key in ("scenario_sums", "dialogue_sums", "displayed_sums", "pair_sums"):
            dd = defaultdict(lambda: np.zeros((n_layers, hidden), dtype=np.float64))
            dd.update(raw[key])
            raw[key] = dd
        for key in ("target_counts", "displayed_counts", "pair_counts"):
            dd = defaultdict(int)
            dd.update(raw[key])
            raw[key] = dd
        print(f"Resuming deflection extraction from index {raw['next_index']}")
        return raw

    # ---- finalize: sums -> means -> npz ----
    def _finalize(self, state: dict, n_layers: int, hidden: int) -> Path:
        def stack_means(sums: dict, counts: dict) -> tuple[np.ndarray, list[str]]:
            emotions = _ordered_present(set(counts))
            if not emotions:
                return np.zeros((n_layers, 0, hidden), dtype=np.float32), []
            arr = np.stack([sums[e] / counts[e] for e in emotions])  # (E, L, H)
            return np.transpose(arr, (1, 0, 2)).astype(np.float32), emotions  # (L, E, H)

        target_means, target_emos = stack_means(state["dialogue_sums"], state["target_counts"])
        scenario_means, _ = stack_means(state["scenario_sums"], state["target_counts"])
        displayed_means, displayed_emos = stack_means(state["displayed_sums"], state["displayed_counts"])

        pairs = sorted(state["pair_counts"].keys())
        if pairs:
            pair_arr = np.stack([state["pair_sums"][p] / state["pair_counts"][p] for p in pairs])
            pair_means = np.transpose(pair_arr, (1, 0, 2)).astype(np.float32)  # (L, P, H)
        else:
            pair_means = np.zeros((n_layers, 0, hidden), dtype=np.float32)

        out = self.config.deflection_activations
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            target_means=target_means, target_emotions=np.array(target_emos, dtype=object),
            scenario_means=scenario_means,
            displayed_means=displayed_means, displayed_emotions=np.array(displayed_emos, dtype=object),
            pair_means=pair_means, pair_labels=np.array([f"{r}->{d}" for r, d in pairs], dtype=object),
        )
        print(f"Saved {out} — target {target_means.shape}, {len(target_emos)} target emotions")
        return out


# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI: ``python -m emotion_probes.extraction.extractor <which>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract residual-stream activations.")
    parser.add_argument("which", choices=["stories", "neutral_stories", "neutral_dialogues", "deflection"])
    args = parser.parse_args()

    config = Config()
    model = ProbedModel(config)

    if args.which == "stories":
        StoryActivationExtractor(model, config).run()
    elif args.which == "neutral_stories":
        NeutralActivationExtractor(
            model, config.neutral_stories_csv, "story", config.neutral_story_activations, config
        ).run()
    elif args.which == "neutral_dialogues":
        NeutralActivationExtractor(
            model, config.neutral_dialogues_csv, "dialogue", config.neutral_dialogue_activations, config
        ).run()
    elif args.which == "deflection":
        DeflectionActivationExtractor(model, config).run()


if __name__ == "__main__":
    main()
