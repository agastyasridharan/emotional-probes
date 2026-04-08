"""
Extract residual stream activations from the 239,400 deflection dialogues.

For each dialogue we split at the scenario→dialogue boundary and compute
separate means for:
  - scenario tokens (where the real emotion is named)
  - dialogue tokens (where the masking speaker displays a different emotion)

Activations are accumulated into running sums per (real_emotion, layer) and
per (displayed_emotion, layer), then saved as means. This avoids storing
the full 98 GB of per-dialogue activations.

Output: /mnt/ssd/data/deflection_activations.pt
    {
        "target_dialogue_means":    {emotion: {layer: tensor(2560)}},
        "target_scenario_means":    {emotion: {layer: tensor(2560)}},
        "displayed_dialogue_means": {emotion: {layer: tensor(2560)}},
        "target_counts":            {emotion: int},
        "displayed_counts":         {emotion: int},
    }

Usage (on the VM):
    ~/venv/bin/python3 extract_deflection_activations.py
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B"
INPUT_PATH = Path("/mnt/ssd/data/deflection_dialogues.csv")
OUTPUT_PATH = Path("/mnt/ssd/data/deflection_activations.pt")
CHECKPOINT_PATH = Path("/mnt/ssd/data/deflection_checkpoint.pt")
CHECKPOINT_EVERY = 500  # save every N batches
NUM_LAYERS = 42
BATCH_SIZE = 8


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def find_dialogue_start(text, name_a, name_b):
    """Find the character position where dialogue begins (first speaker turn)."""
    markers = []
    for name in (name_a, name_b):
        for sep in (f"\n\n{name}:", f"\n{name}:"):
            pos = text.find(sep)
            if pos != -1:
                markers.append(pos)
    return min(markers) if markers else None


def find_speaker_a_spans(text, name_a, name_b):
    """Find character spans of NAME_A's utterances only (the masking speaker)."""
    spans = []
    prefix_a = f"{name_a}:"
    i = 0
    while True:
        start = text.find(prefix_a, i)
        if start == -1:
            break
        content_start = start + len(prefix_a)
        # End at next speaker turn or end of text.
        next_turn = len(text)
        for name in (name_a, name_b):
            for sep in (f"\n\n{name}:", f"\n{name}:"):
                pos = text.find(sep, content_start)
                if pos != -1 and pos < next_turn:
                    next_turn = pos
        spans.append((content_start, next_turn))
        i = content_start
    return spans


def estimate_scenario_tokens(tokenizer, text, dialogue_char_start):
    """Estimate how many tokens the scenario portion uses."""
    scenario_text = text[:dialogue_char_start]
    scenario_ids = tokenizer(scenario_text, add_special_tokens=False)["input_ids"]
    return len(scenario_ids)


def estimate_char_to_token_spans(tokenizer, text, char_spans):
    """Convert character-level spans to approximate token-level spans.

    Tokenises text up to each span boundary to estimate token positions.
    Returns list of (token_start, token_end) tuples.
    """
    token_spans = []
    for char_start, char_end in char_spans:
        tok_start = len(tokenizer(text[:char_start], add_special_tokens=False)["input_ids"])
        tok_end = len(tokenizer(text[:char_end], add_special_tokens=False)["input_ids"])
        token_spans.append((tok_start, tok_end))
    return token_spans


def main():
    print(f"Loading tokenizer + model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = get_layers(model)
    assert len(layers) == NUM_LAYERS

    print(f"Reading dialogues from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df)} dialogues")

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = act.detach()

        return hook_fn

    hooks = [layers[i].register_forward_hook(make_hook(i)) for i in range(NUM_LAYERS)]

    # Running accumulators — restore from checkpoint if available.
    start_batch = 0
    target_dialogue_sums = defaultdict(lambda: torch.zeros(NUM_LAYERS, 2560))
    target_scenario_sums = defaultdict(lambda: torch.zeros(NUM_LAYERS, 2560))
    displayed_dialogue_sums = defaultdict(lambda: torch.zeros(NUM_LAYERS, 2560))
    pair_dialogue_sums = defaultdict(lambda: torch.zeros(NUM_LAYERS, 2560))
    target_counts = defaultdict(int)
    displayed_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    skipped = 0

    if CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
        target_dialogue_sums.update(ckpt["target_dialogue_sums"])
        target_scenario_sums.update(ckpt["target_scenario_sums"])
        displayed_dialogue_sums.update(ckpt["displayed_dialogue_sums"])
        pair_dialogue_sums.update(ckpt["pair_dialogue_sums"])
        target_counts.update(ckpt["target_counts"])
        displayed_counts.update(ckpt["displayed_counts"])
        pair_counts.update(ckpt["pair_counts"])
        skipped = ckpt["skipped"]
        start_batch = ckpt["next_batch"]
        print(f"  Resuming from batch {start_batch}")

    all_starts = list(range(0, len(df), BATCH_SIZE))
    try:
        for batch_idx, start in enumerate(tqdm(all_starts[start_batch:], desc="Extracting", initial=start_batch, total=len(all_starts))):
            batch_df = df.iloc[start : start + BATCH_SIZE]
            texts = batch_df["dialogue"].tolist()
            real_emotions = batch_df["real_emotion"].tolist()
            displayed_emotions = batch_df["displayed_emotion"].tolist()
            names_a = batch_df["name_a"].tolist()
            names_b = batch_df["name_b"].tolist()

            # Find boundaries and speaker-A spans for each item.
            batch_info = []
            for text, na, nb in zip(texts, names_a, names_b):
                text = str(text)
                na, nb = str(na), str(nb)
                char_start = find_dialogue_start(text, na, nb)
                if char_start is None:
                    batch_info.append(None)
                    continue
                n_scenario = estimate_scenario_tokens(tokenizer, text, char_start)
                # Find NAME_A's utterance spans (character-level), then map to tokens.
                a_char_spans = find_speaker_a_spans(text, na, nb)
                a_token_spans = estimate_char_to_token_spans(tokenizer, text, a_char_spans)
                batch_info.append((n_scenario, a_token_spans))

            inputs = tokenizer(
                [str(t) for t in texts],
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                model(**inputs)

            attn_mask = inputs["attention_mask"].float()  # (B, L)

            for b in range(len(batch_df)):
                if batch_info[b] is None:
                    skipped += 1
                    continue

                n_scenario, a_token_spans = batch_info[b]
                if n_scenario < 5 or not a_token_spans:
                    skipped += 1
                    continue

                real_em = real_emotions[b]
                disp_em = displayed_emotions[b]
                seq_len = int(attn_mask[b].sum().item())
                L = attn_mask.shape[1]

                # Scenario mask: tokens [1, n_scenario+1) (skip BOS).
                scenario_start = 1
                dialogue_start = min(n_scenario + 1, seq_len)
                if dialogue_start >= seq_len - 1:
                    skipped += 1
                    continue

                scenario_mask = torch.zeros(L, device="cpu")
                scenario_mask[scenario_start:dialogue_start] = 1.0

                # Speaker-A dialogue mask: only NAME_A's utterance tokens.
                # +1 offset for BOS token.
                speaker_a_mask = torch.zeros(L, device="cpu")
                for tok_start, tok_end in a_token_spans:
                    ts = min(tok_start + 1, seq_len)  # +1 for BOS
                    te = min(tok_end + 1, seq_len)
                    speaker_a_mask[ts:te] = 1.0

                scenario_count = scenario_mask.sum().clamp(min=1)
                speaker_a_count = speaker_a_mask.sum().clamp(min=1)

                if speaker_a_count < 2:
                    skipped += 1
                    continue

                for layer_idx in range(NUM_LAYERS):
                    act = captured[layer_idx][b].float().cpu()  # (L, 2560)

                    scenario_mean = (act * scenario_mask.unsqueeze(-1)).sum(dim=0) / scenario_count
                    speaker_a_mean = (act * speaker_a_mask.unsqueeze(-1)).sum(dim=0) / speaker_a_count

                    target_scenario_sums[real_em][layer_idx] += scenario_mean
                    target_dialogue_sums[real_em][layer_idx] += speaker_a_mean
                    displayed_dialogue_sums[disp_em][layer_idx] += speaker_a_mean
                    pair_dialogue_sums[(real_em, disp_em)][layer_idx] += speaker_a_mean

                target_counts[real_em] += 1
                pair_counts[(real_em, disp_em)] += 1
                displayed_counts[disp_em] += 1

            # Periodic checkpoint.
            if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
                torch.save(
                    {
                        "target_dialogue_sums": dict(target_dialogue_sums),
                        "target_scenario_sums": dict(target_scenario_sums),
                        "displayed_dialogue_sums": dict(displayed_dialogue_sums),
                        "pair_dialogue_sums": dict(pair_dialogue_sums),
                        "target_counts": dict(target_counts),
                        "displayed_counts": dict(displayed_counts),
                        "pair_counts": dict(pair_counts),
                        "skipped": skipped,
                        "next_batch": start_batch + batch_idx + 1,
                    },
                    CHECKPOINT_PATH,
                )

    finally:
        for h in hooks:
            h.remove()

    print(f"Skipped {skipped} dialogues (no boundary found or too short)")
    print("Computing final means...")

    # Convert sums to means.
    target_dialogue_means = {}
    target_scenario_means = {}
    displayed_dialogue_means = {}

    for emotion in target_counts:
        c = target_counts[emotion]
        target_dialogue_means[emotion] = {
            layer: target_dialogue_sums[emotion][layer] / c for layer in range(NUM_LAYERS)
        }
        target_scenario_means[emotion] = {
            layer: target_scenario_sums[emotion][layer] / c for layer in range(NUM_LAYERS)
        }

    for emotion in displayed_counts:
        c = displayed_counts[emotion]
        displayed_dialogue_means[emotion] = {
            layer: displayed_dialogue_sums[emotion][layer] / c for layer in range(NUM_LAYERS)
        }

    pair_dialogue_means = {}
    for pair in pair_counts:
        c = pair_counts[pair]
        pair_dialogue_means[pair] = {
            layer: pair_dialogue_sums[pair][layer] / c for layer in range(NUM_LAYERS)
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_dialogue_means": target_dialogue_means,
            "target_scenario_means": target_scenario_means,
            "displayed_dialogue_means": displayed_dialogue_means,
            "pair_dialogue_means": pair_dialogue_means,
            "target_counts": dict(target_counts),
            "displayed_counts": dict(displayed_counts),
            "pair_counts": dict(pair_counts),
        },
        OUTPUT_PATH,
    )

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"  {len(target_counts)} target emotions, {len(displayed_counts)} displayed emotions")
    print(f"  Sample counts: {list(target_counts.items())[:5]}")


if __name__ == "__main__":
    main()
