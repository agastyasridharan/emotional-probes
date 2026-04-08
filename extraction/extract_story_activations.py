"""Extract activations at all 42 layers for all 171 emotions from Gemma 4 E4B."""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B"
DATA_DIR = Path.home() / "data" / "stories"
ACTIVATIONS_DIR = Path.home() / "data" / "activations_all_layers"
NUM_LAYERS = 42
MIN_TOKEN_OFFSET = 50
BATCH_SIZE = 32

ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()

layers = model.model.language_model.layers

# Hook all 42 layers
captured = {}
hooks = []
for i in range(NUM_LAYERS):
    def make_hook(idx):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            captured[idx] = act.detach().cpu()
        return hook_fn
    hooks.append(layers[i].register_forward_hook(make_hook(i)))

# Load all stories grouped by emotion
print("Loading stories...")
stories_by_emotion = defaultdict(list)
for path in DATA_DIR.glob("*.json"):
    data = json.loads(path.read_text())
    for story in data["stories"]:
        stories_by_emotion[data["emotion"]].append(story)

emotions = sorted(stories_by_emotion.keys())
print(f"Loaded {len(emotions)} emotions, {sum(len(s) for s in stories_by_emotion.values())} stories total\n")

start = time.time()
for emotion in emotions:
    out_file = ACTIVATIONS_DIR / f"{emotion}.pt"
    if out_file.exists():
        print(f"  Skipping {emotion}")
        continue

    stories = stories_by_emotion[emotion]
    layer_activations = defaultdict(list)

    for i in tqdm(range(0, len(stories), BATCH_SIZE), desc=emotion, leave=True):
        batch_stories = stories[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_stories, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(model.device)

        attention_mask = inputs["attention_mask"].cpu()

        with torch.no_grad():
            model(**inputs)

        for layer_idx in range(NUM_LAYERS):
            act = captured[layer_idx]
            for j in range(act.shape[0]):
                seq_len = attention_mask[j].sum().item()
                if seq_len < MIN_TOKEN_OFFSET + 10:
                    continue
                avg = act[j, MIN_TOKEN_OFFSET:seq_len, :].mean(dim=0)
                layer_activations[layer_idx].append(avg.float())

    result = {idx: torch.stack(vecs) for idx, vecs in layer_activations.items() if vecs}
    torch.save(result, out_file)
    elapsed = time.time() - start
    print(f"  {emotion}: {len(layer_activations[0])} stories, 42 layers | {elapsed / 60:.1f}m elapsed")

for h in hooks:
    h.remove()

total = time.time() - start
print(f"\nDone. {total / 60:.1f} minutes total.")
