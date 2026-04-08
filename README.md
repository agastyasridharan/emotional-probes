# Emotional Probes

Extract emotion and emotion-deflection probes from large language models.

Built from the methodology described in Anthropic's ["Emotion Concepts and their Function in a Large Language Model"](https://transformer-circuits.pub/2026/emotions/index.html) (Sofroniew et al., April 2026).

![Emotion Visualiser Demo](assets/demo.gif)

## Overview

This project provides tools to:

1. **Generate synthetic datasets** for 171 emotion concepts — emotional stories, neutral baselines, and emotion-deflection dialogues
2. **Extract residual stream activations** from these datasets through a target model (Gemma 4 E4B)
3. **Compute emotion probes** — linear directions in activation space that detect expressed and suppressed emotions
4. **Visualise** emotion probe activations on arbitrary text

## Dataset

The generated datasets are available on HuggingFace: [ryancodrai/emotion-probes](https://huggingface.co/datasets/ryancodrai/emotion-probes)

```python
from datasets import load_dataset
ds = load_dataset("ryancodrai/emotion-probes", data_files="expression/stories.parquet")
```

## Project Structure

```
agents/                         Dataset generation
  story/                        205k emotional stories (171 emotions × 100 topics × 12)
  neutral_story/                1.2k emotionally neutral stories (PCA baseline)
  neutral_dialogue/             1.2k neutral Person/AI dialogues (PCA baseline)
  deflection_story/             239k emotion-deflection dialogues

extraction/                     Activation extraction & vector computation
  extract_story_activations.py
  extract_neutral_story_activations.py
  extract_neutral_dialogue_activations.py
  extract_deflection_activations.py
  compute_expression_vectors.py
  compute_deflection_vectors.py

visualise.py                    Flask visualiser with expression/deflection modes
agent.py                        Base agent class (pydantic-ai)
```

## Setup

```bash
pip install pydantic-ai tenacity tqdm pandas torch transformers flask
```

## Usage

### 1. Generate datasets

From the repo root:

```bash
# Emotional stories (requires Gemini API key)
python -m agents.story.agent

# Neutral stories
python -m agents.neutral_story.agent

# Neutral dialogues
python -m agents.neutral_dialogue.agent

# Deflection pair selection (requires pre-computed expression vectors)
python agents/deflection_story/select_pairs.py

# Deflection dialogues
python -m agents.deflection_story.agent
```

Each agent skips existing files, so re-running is safe.

### 2. Extract activations

Run on a GPU machine with the model loaded:

```bash
# Story activations (83GB output, ~6 hours on A100)
python extraction/extract_story_activations.py

# Neutral story activations (~30 seconds)
python extraction/extract_neutral_story_activations.py

# Neutral dialogue activations (~30 seconds)
python extraction/extract_neutral_dialogue_activations.py

# Deflection activations (~6 hours, supports checkpointing)
python extraction/extract_deflection_activations.py
```

### 3. Compute vectors

```bash
# Expression vectors (reads 83GB activations, applies PCA confound removal)
python extraction/compute_expression_vectors.py

# Deflection vectors (applies neutral dialogue PCA + expression-space orthogonalisation)
python extraction/compute_deflection_vectors.py
```

### 4. Visualise

```bash
python visualise.py
```

Opens a Flask server on port 8080. Paste any text to see per-token emotion probe activations with:

- Expression and deflection probe modes
- Emotion groups (Fear, Anger, Sadness, Disgust, Surprise, Joy, Guilt, Shame + alignment-relevant groups)
- Checkbox multi-select for custom emotion combinations
- Drag-to-select token spans with ranked emotion analysis
- Layer selection (0–41)

## How It Works

### Expression probes

For each of 171 emotions, we generate stories where a character experiences that emotion (without naming it). We extract residual stream activations from a target model, average across stories per emotion, subtract the global mean, project out confound directions from neutral text (PCA, 50% variance), and unit-normalise. The resulting vectors detect when an emotion is being openly expressed.

### Deflection probes

We generate dialogues where a character masks one emotion with another. We extract activations on the masking speaker's tokens, apply the same difference-of-means recipe, then additionally orthogonalise against the expression vector space (99% variance). The resulting vectors detect when an emotion is contextually present but being suppressed — a distinct signal from expression, and potentially useful for alignment monitoring.

## Reference

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., et al. (2026). *Emotion Concepts and their Function in a Large Language Model.* Transformer Circuits Thread. https://transformer-circuits.pub/2026/emotions/index.html

## Author

Ryan Codrai — [GitHub](https://github.com/RyanCodrai) · [LinkedIn](https://linkedin.com/in/ryan-codrai)
