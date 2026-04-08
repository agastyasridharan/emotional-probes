"""
Emotion Visualisation Server
=============================
Shows emotion heatmaps across tokens. Red = positive activation, blue = anti-correlated.
Select an emotion from the sidebar to see its activation intensity per token.

Usage:
    ssh -L 8080:localhost:8080 -i ~/.ssh/id_ed25519 ryan@<server-ip>
    ~/venv/bin/python3 visualise.py
    Then open http://localhost:8080
"""

import json
from pathlib import Path

import torch
from flask import Flask, render_template_string, request
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B"
VECTORS_PATH = Path.home() / "data" / "emotion_vectors_all_layers.pt"
DEFLECTION_VECTORS_PATH = Path.home() / "data" / "deflection_vectors.pt"
TARGET_LAYER = 23

PROBE_MODES = {
    "expression": "Expression (story-based)",
    "deflection": "Deflection (suppression)",
}

# 10 emotion clusters (approximating the paper's k-means grouping).
# The cluster score for a token is the mean of its member emotions' scores.
EMOTION_CLUSTERS = {
    "Joy/Elation": ["joyful", "ecstatic", "elated", "blissful", "thrilled", "jubilant", "euphoric", "excited", "exuberant", "delighted", "happy", "cheerful", "playful", "vibrant", "eager", "enthusiastic", "energized", "stimulated"],
    "Calm/Content": ["calm", "serene", "peaceful", "relaxed", "content", "at ease", "fulfilled", "satisfied", "safe", "pleased", "refreshed", "rejuvenated", "patient"],
    "Love/Warmth": ["loving", "compassionate", "grateful", "thankful", "empathetic", "sympathetic", "kind", "sentimental", "nostalgic", "infatuated", "sensitive"],
    "Pride/Hope": ["proud", "triumphant", "inspired", "invigorated", "hopeful", "hope", "optimistic", "self-confident", "valiant", "smug", "reflective"],
    "Anger/Hostility": ["angry", "enraged", "furious", "outraged", "hostile", "irate", "indignant", "irritated", "resentful", "bitter", "hateful", "vengeful", "mad", "annoyed", "frustrated", "exasperated", "grumpy", "spiteful", "vindictive", "sullen", "insulted", "offended", "scornful", "contemptuous", "disdainful", "disgusted", "defiant", "obstinate", "stubborn"],
    "Fear/Anxiety": ["afraid", "anxious", "nervous", "scared", "terrified", "panicked", "frightened", "worried", "on edge", "uneasy", "unsettled", "unnerved", "rattled", "shaken", "horrified", "alarmed", "tense", "stressed", "paranoid", "suspicious", "vigilant", "alert", "vulnerable", "distressed", "disturbed", "troubled", "restless", "hysterical"],
    "Sadness/Grief": ["sad", "grief-stricken", "heartbroken", "depressed", "melancholy", "miserable", "lonely", "dispirited", "gloomy", "brooding", "unhappy", "droopy", "sorry", "regretful", "remorseful", "hurt", "tormented", "worthless", "resigned"],
    "Shame/Guilt": ["ashamed", "guilty", "embarrassed", "humiliated", "mortified", "self-critical", "self-conscious"],
    "Surprise/Wonder": ["surprised", "amazed", "astonished", "shocked", "awestruck", "dumbstruck", "bewildered", "puzzled", "perplexed", "mystified", "disoriented", "skeptical"],
    "Low-energy": ["bored", "tired", "weary", "worn out", "sleepy", "sluggish", "listless", "lazy", "indifferent", "docile", "overwhelmed", "trapped", "stuck", "impatient", "aroused", "upset", "jealous", "envious", "greedy", "desperate", "dependent"],
}

app = Flask(__name__)

model = None
tokenizer = None
emotion_vectors = None
global_means = None
deflection_vectors = None
deflection_global_means = None


def load_resources():
    global model, tokenizer, emotion_vectors, global_means, deflection_vectors, deflection_global_means

    print("Loading emotion vectors...")
    data = torch.load(VECTORS_PATH, weights_only=True)
    emotion_vectors = data["vectors"]
    global_means = data["global_means"]
    print(f"  {len(emotion_vectors[TARGET_LAYER])} emotions, layer {TARGET_LAYER}")

    print("Loading deflection vectors...")
    if DEFLECTION_VECTORS_PATH.exists():
        defl_data = torch.load(DEFLECTION_VECTORS_PATH, weights_only=False)
        deflection_vectors = defl_data["target_vectors"]
        deflection_global_means = defl_data["target_global_means"]
        print(f"  {len(deflection_vectors.get(TARGET_LAYER, {}))} deflection emotions")
    else:
        print("  (not found, deflection mode disabled)")

    print(f"Loading model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("Ready.")


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot find transformer layers")


def analyse_text(text, target_layer, probe_mode="expression"):
    """Return tokens and full emotion scores for every token.

    Always computes both expression and deflection scores in a single
    forward pass so the UI can switch between them client-side.
    """
    layers = get_layers(model)

    captured = {}

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        captured["act"] = act.detach().cpu().float()

    hook = layers[target_layer].register_forward_hook(hook_fn)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        model(**inputs)

    hook.remove()

    all_tokens = [tokenizer.decode(tid) for tid in inputs["input_ids"][0]]
    act = captured["act"][0]

    skip = 1 if all_tokens and all_tokens[0] in ("<bos>", "<s>", "</s>") else 0
    tokens = all_tokens[skip:]
    act = act[skip:]

    # Replace newline tokens with a <br> marker for display.
    tokens = ["<br>" if (t.strip() == "" and "\n" in t) else t for t in tokens]

    # --- Expression scores ---
    expr_emotions = sorted(emotion_vectors[target_layer].keys())
    expr_matrix = torch.stack([emotion_vectors[target_layer][e] for e in expr_emotions]).float()
    expr_centered = act - global_means[target_layer].float()
    expr_scores = expr_centered @ expr_matrix.T

    # --- Deflection scores (if available) ---
    defl_scores = None
    defl_emotions = None
    if deflection_vectors and target_layer in deflection_vectors:
        defl_emotions = sorted(deflection_vectors[target_layer].keys())
        defl_matrix = torch.stack([deflection_vectors[target_layer][e] for e in defl_emotions]).float()
        defl_centered = act - deflection_global_means[target_layer].float()
        defl_scores = defl_centered @ defl_matrix.T

    # Use the selected probe mode for ranking and primary display.
    if probe_mode == "deflection" and defl_scores is not None:
        emotions = defl_emotions
        scores = defl_scores
    else:
        emotions = expr_emotions
        scores = expr_scores

    # RRF of mean activation + autocorrelation.
    emotion_means_all = scores.mean(dim=0)
    if scores.shape[0] > 1:
        s1 = scores[:-1]
        s2 = scores[1:]
        autocorr = (s1 * s2).sum(dim=0) / (scores.pow(2).sum(dim=0).clamp(min=1e-8))
    else:
        autocorr = scores.mean(dim=0)

    # Rank by each metric.
    k = 60
    mean_order = emotion_means_all.argsort(descending=True)
    ac_order = autocorr.argsort(descending=True)
    mean_ranks = torch.zeros_like(emotion_means_all)
    ac_ranks = torch.zeros_like(autocorr)
    for r, idx in enumerate(mean_order):
        mean_ranks[idx] = r + 1
    for r, idx in enumerate(ac_order):
        ac_ranks[idx] = r + 1
    rrf = 1.0 / (k + mean_ranks) + 1.0 / (k + ac_ranks)

    emotion_ranking = []
    for i, emotion in enumerate(emotions):
        emotion_ranking.append({
            "emotion": emotion,
            "avg_score": "",
        })
    # Sort by RRF descending.
    rrf_order = rrf.argsort(descending=True).tolist()
    emotion_ranking = [emotion_ranking[i] for i in rrf_order]

    # Per-cluster aggregate: mean max-score across members.
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}
    cluster_scores = []
    for cluster_name, members in EMOTION_CLUSTERS.items():
        idxs = [emotion_to_idx[m] for m in members if m in emotion_to_idx]
        if not idxs:
            cluster_scores.append({"cluster": cluster_name, "score": 0.0})
            continue
        value = rrf[idxs].mean().item()
        cluster_scores.append({"cluster": cluster_name, "score": round(value, 4)})

    token_scores = {}
    for i, emotion in enumerate(emotions):
        token_scores[emotion] = [round(scores[pos, i].item(), 4) for pos in range(len(tokens))]

    # Build per-emotion scores for BOTH modes (for client-side switching).
    expr_token_scores = {}
    for i, e in enumerate(expr_emotions):
        expr_token_scores[e] = [round(expr_scores[pos, i].item(), 4) for pos in range(len(tokens))]

    defl_token_scores = {}
    if defl_scores is not None:
        for i, e in enumerate(defl_emotions):
            defl_token_scores[e] = [round(defl_scores[pos, i].item(), 4) for pos in range(len(tokens))]

    return tokens, emotion_ranking, token_scores, cluster_scores, expr_token_scores, defl_token_scores, expr_emotions, defl_emotions


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Visualiser — Gemma 4</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #fafafa;
            color: #222;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            background: #fff;
        }
        h1 { color: #333; font-size: 1.3em; }
        h1 span { color: #999; font-size: 0.6em; font-weight: normal; }
        .input-area {
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            display: flex;
            gap: 10px;
            background: #fff;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
            box-sizing: border-box;
        }
        textarea {
            flex: 1;
            height: 80px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            background: #fff;
            color: #222;
            font-size: 13px;
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            resize: vertical;
        }
        textarea:focus { border-color: #666; outline: none; }
        button {
            background: #333;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            align-self: flex-end;
        }
        button:hover { background: #555; }
        .main {
            display: flex;
            flex: 1;
            overflow: hidden;
            height: calc(100vh - 140px);
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }
        .sidebar {
            width: 220px;
            border-left: 1px solid #ddd;
            overflow-y: auto;
            padding: 10px 0;
            flex-shrink: 0;
            background: #fff;
            order: 2;
            height: 100%;
        }
        .groups-sidebar {
            width: 180px;
            border-left: 1px solid #ddd;
            overflow-y: auto;
            padding: 10px 0;
            flex-shrink: 0;
            background: #fff;
            order: 3;
            height: 100%;
        }
        .group-item {
            padding: 6px 15px;
            cursor: pointer;
            font-size: 12px;
            font-family: -apple-system, sans-serif;
            border-right: 3px solid transparent;
        }
        .group-item:hover { background: #f5f5f5; }
        .group-item.active {
            background: #f0f0f0;
            border-right-color: #c0392b;
            font-weight: bold;
        }
        .sidebar-title {
            padding: 5px 15px 10px;
            font-size: 11px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .emotion-item {
            padding: 4px 10px 4px 15px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-family: 'SF Mono', 'Menlo', monospace;
        }
        .emotion-item:hover { background: #f5f5f5; }
        .emotion-item.checked { background: #f0f0f0; font-weight: bold; }
        .emotion-cb {
            width: 12px; height: 12px;
            border: 1px solid #ccc;
            border-radius: 2px;
            flex-shrink: 0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            color: #c0392b;
        }
        .emotion-item.checked .emotion-cb {
            border-color: #c0392b;
            background: #c0392b;
            color: #fff;
        }
        .emotion-score {
            color: #999;
            font-size: 11px;
        }
        .emotion-item.active .emotion-score { color: #c0392b; }
        .content {
            flex: 1;
            overflow-y: auto;
            padding: 0;
        }
        .emotion-label {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .tokens {
            line-height: 2.0;
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            font-size: 13px;
            word-spacing: 0;
            letter-spacing: 0;
        }
        .token {
            display: inline;
            padding: 3px 0;
            margin: 0;
            cursor: pointer;
            position: relative;
            box-decoration-break: clone;
            -webkit-box-decoration-break: clone;
        }
        .token:hover {
            outline: 1px solid #333;
            outline-offset: -1px;
            border-radius: 0;
        }
        .tooltip {
            display: none;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #fff;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 8px 10px;
            z-index: 100;
            white-space: nowrap;
            font-size: 11px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            color: #333;
        }
        .token:hover .tooltip { display: block; }
        .tooltip-row { margin: 1px 0; }
        .tooltip-pos { color: #c0392b; font-weight: bold; }
        .tooltip-neg { color: #2980b9; font-weight: bold; }
        .no-results {
            color: #999;
            padding: 40px;
            text-align: center;
        }
        .caption {
            margin-top: 20px;
            padding: 15px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.6;
            color: #555;
        }
        .caption strong { color: #333; }
    </style>
</head>
<body>
    <header>
        <h1>Emotion Visualiser <span>Gemma 4 E4B — Layer {{ selected_layer }} — {{ probe_mode_label }}</span></h1>
    </header>
    <form method="POST" class="input-area">
        <textarea name="text" placeholder="Enter text to analyse...">{{ text or '' }}</textarea>
        <select name="layer" style="align-self: flex-end; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px; background: #fff;">
            {% for l in range(42) %}
            <option value="{{ l }}" {% if l == selected_layer %}selected{% endif %}>Layer {{ l }}</option>
            {% endfor %}
        </select>
        <button type="submit">Analyse</button>
    </form>

    <div class="main">
        {% if emotion_ranking %}
        <div class="sidebar" onmouseleave="restoreCheckedHeatmap()">
            <div class="sidebar-title">Emotions</div>
            <div id="emotion-list">
            {% for e in emotion_ranking %}
            <div class="emotion-item" data-emotion="{{ e.emotion }}"
                 onclick="toggleEmotion('{{ e.emotion }}')"
                 onmouseenter="previewEmotion('{{ e.emotion }}')">
                <span class="emotion-cb"></span>
                <span>{{ e.emotion }}</span>
            </div>
            {% endfor %}
            </div>
        </div>
        <div class="groups-sidebar" onmouseleave="restoreCheckedHeatmap()">
            <div class="sidebar-title">Core</div>
            <div class="group-item" onmouseenter="previewGroup('fear')" onclick="selectGroup('fear')">Fear</div>
            <div class="group-item" onmouseenter="previewGroup('anger')" onclick="selectGroup('anger')">Anger</div>
            <div class="group-item" onmouseenter="previewGroup('sadness')" onclick="selectGroup('sadness')">Sadness</div>
            <div class="group-item" onmouseenter="previewGroup('disgust')" onclick="selectGroup('disgust')">Disgust</div>
            <div class="group-item" onmouseenter="previewGroup('surprise')" onclick="selectGroup('surprise')">Surprise</div>
            <div class="group-item" onmouseenter="previewGroup('joy')" onclick="selectGroup('joy')">Joy</div>
            <div class="group-item" onmouseenter="previewGroup('guilt')" onclick="selectGroup('guilt')">Guilt</div>
            <div class="group-item" onmouseenter="previewGroup('shame')" onclick="selectGroup('shame')">Shame</div>
            <div class="sidebar-title" style="margin-top: 10px;">Alignment</div>
            <div class="group-item" onmouseenter="previewGroup('certain')" onclick="selectGroup('certain')">Certain</div>
            <div class="group-item" onmouseenter="previewGroup('uncertain')" onclick="selectGroup('uncertain')">Uncertain</div>
            <div class="group-item" onmouseenter="previewGroup('deceptive')" onclick="selectGroup('deceptive')">Deceptive</div>
            <div class="group-item" onmouseenter="previewGroup('warm')" onclick="selectGroup('warm')">Warm</div>
            <div class="group-item" onmouseenter="previewGroup('calm')" onclick="selectGroup('calm')">Calm</div>
            <div class="group-item" onmouseenter="previewGroup('reflective')" onclick="selectGroup('reflective')">Reflective</div>
        </div>
        {% endif %}

        <div class="content">
            {% if tokens %}
            <div id="sticky-header" style="position: sticky; top: 0; z-index: 2; background: #fafafa; padding: 15px 30px 10px; border-bottom: 1px solid #ddd;">
                <div style="margin-bottom: 8px; display: flex; gap: 6px; align-items: center;">
                    <span style="font-size: 12px; color: #999; margin-right: 4px;">Probe:</span>
                    <button type="button" class="mode-btn" data-mode="expression" onclick="switchMode('expression')" style="padding: 4px 12px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-size: 12px; background: #333; color: #fff;">Expression</button>
                    <button type="button" class="mode-btn" data-mode="deflection" onclick="switchMode('deflection')" style="padding: 4px 12px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-size: 12px; background: #fff; color: #333;" {% if not defl_scores_json %}disabled{% endif %}>Deflection</button>
                </div>
                <div class="emotion-label" id="emotion-label">{{ selected_emotion }}</div>
                <div id="selection-info" style="font-size: 12px; color: #999; margin-top: 4px; display: none;">
                    Selection active — <a href="#" onclick="clearSelection(); return false;" style="color: #c0392b;">clear</a>
                </div>
            </div>
            <div class="tokens" id="token-display" style="padding: 20px 30px 30px;">
                {% for token in tokens %}{% if token == '<br>' %}<br id="tok-{{ loop.index0 }}">{% else %}<span class="token" id="tok-{{ loop.index0 }}" onmousedown="handleTokenMousedown({{ loop.index0 }}, event)" onmousemove="handleTokenMousemove({{ loop.index0 }})" style="user-select: none;">{{ token }}<div class="tooltip" id="tip-{{ loop.index0 }}"></div></span>{% endif %}{% endfor %}
            </div>
            {% elif not text %}
            <div class="no-results" style="padding: 30px;">Enter text above and click Analyse</div>
            {% endif %}
        </div>
    </div>

    {% if tokens %}
    <script>
        let allScores = {{ all_scores_json | safe }};
        let emotions = {{ emotions_json | safe }};
        const tokens = {{ tokens_json | safe }};
        const clusterScores = {{ cluster_scores_json | safe }};

        // These must be at the top — HTML onmouseenter/onclick handlers reference them immediately.
        const checkedEmotions = new Set();
        let activeGroup = null;
        const emotionGroups = {
            // Core emotions.
            fear: ['afraid', 'terrified', 'panicked', 'scared', 'desperate'],
            anger: ['angry', 'furious', 'enraged', 'hostile', 'irate'],
            sadness: ['sad', 'grief-stricken', 'heartbroken', 'depressed', 'lonely'],
            disgust: ['disgusted', 'contemptuous', 'disdainful', 'scornful', 'offended'],
            surprise: ['surprised', 'shocked', 'astonished', 'amazed', 'dumbstruck'],
            joy: ['happy', 'joyful', 'elated', 'ecstatic', 'delighted'],
            guilt: ['guilty', 'remorseful', 'regretful', 'sorry', 'self-critical'],
            shame: ['ashamed', 'humiliated', 'mortified', 'embarrassed', 'self-conscious'],
            // Alignment-relevant.
            certain: ['self-confident', 'proud', 'triumphant', 'defiant', 'smug'],
            uncertain: ['bewildered', 'perplexed', 'puzzled', 'mystified', 'skeptical'],
            deceptive: ['smug', 'suspicious', 'skeptical', 'vigilant', 'self-confident'],
            warm: ['loving', 'compassionate', 'empathetic', 'kind', 'grateful'],
            calm: ['calm', 'serene', 'peaceful', 'relaxed', 'content'],
            reflective: ['reflective', 'nostalgic', 'sentimental', 'brooding', 'resigned'],
        };

        // Both score sets for client-side switching.
        const exprScores = {{ expr_scores_json | safe if expr_scores_json else 'null' }};
        const exprEmotions = {{ expr_emotions_json | safe if expr_emotions_json else 'null' }};
        const deflScores = {{ defl_scores_json | safe if defl_scores_json else 'null' }};
        const deflEmotions = {{ defl_emotions_json | safe if defl_emotions_json else 'null' }};
        let currentMode = '{{ selected_probe_mode }}';

        // Negative-valence emotions — these are safety-relevant when suppressed.
        const negativeEmotions = new Set([
            "afraid", "alarmed", "angry", "annoyed", "anxious", "ashamed", "bitter",
            "contemptuous", "defiant", "depressed", "desperate", "disdainful", "disgusted",
            "disoriented", "dispirited", "distressed", "disturbed", "embarrassed", "enraged",
            "envious", "exasperated", "frightened", "frustrated", "furious", "gloomy",
            "greedy", "grief-stricken", "grumpy", "guilty", "hateful", "heartbroken",
            "horrified", "hostile", "humiliated", "hurt", "hysterical", "impatient",
            "indignant", "insulted", "irate", "irritated", "jealous", "lonely", "mad",
            "melancholy", "miserable", "mortified", "nervous", "obstinate", "offended",
            "on edge", "outraged", "overwhelmed", "panicked", "paranoid", "rattled",
            "regretful", "remorseful", "resentful", "resigned", "restless", "sad",
            "scared", "scornful", "self-conscious", "self-critical", "shaken", "shocked",
            "spiteful", "stressed", "stubborn", "stuck", "sullen", "suspicious", "tense",
            "terrified", "tormented", "trapped", "troubled", "uneasy", "unhappy",
            "unnerved", "unsettled", "upset", "vengeful", "vindictive", "vulnerable",
            "weary", "worn out", "worried", "worthless"
        ]);

        function renderSpider() {
            const svg = document.getElementById('spider');
            const cx = 180, cy = 180, maxR = 130;
            const n = clusterScores.length;
            const maxVal = Math.max(...clusterScores.map(c => Math.abs(c.score))) || 1;
            let html = '';
            // Concentric grid rings (4 rings).
            for (let r = 1; r <= 4; r++) {
                const rr = (maxR * r) / 4;
                html += '<circle cx="' + cx + '" cy="' + cy + '" r="' + rr + '" fill="none" stroke="#eee" stroke-width="1"/>';
            }
            // Axes + labels.
            const pts = [];
            for (let i = 0; i < n; i++) {
                const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
                const ax = cx + maxR * Math.cos(angle);
                const ay = cy + maxR * Math.sin(angle);
                html += '<line x1="' + cx + '" y1="' + cy + '" x2="' + ax + '" y2="' + ay + '" stroke="#eee" stroke-width="1"/>';
                const lx = cx + (maxR + 24) * Math.cos(angle);
                const ly = cy + (maxR + 24) * Math.sin(angle);
                const anchor = Math.abs(lx - cx) < 10 ? 'middle' : (lx > cx ? 'start' : 'end');
                html += '<text x="' + lx + '" y="' + (ly + 4) + '" font-size="10" font-family="-apple-system, sans-serif" fill="#555" text-anchor="' + anchor + '">' + clusterScores[i].cluster + '</text>';
                const r = (Math.max(clusterScores[i].score, 0) / maxVal) * maxR;
                pts.push([cx + r * Math.cos(angle), cy + r * Math.sin(angle)]);
            }
            // Polygon.
            html += '<polygon points="' + pts.map(p => p.join(',')).join(' ') + '" fill="rgba(192, 57, 43, 0.25)" stroke="rgba(192, 57, 43, 0.8)" stroke-width="1.5"/>';
            // Dots on vertices.
            for (const [x, y] of pts) {
                html += '<circle cx="' + x + '" cy="' + y + '" r="2.5" fill="rgba(192, 57, 43, 0.9)"/>';
            }
            svg.innerHTML = html;
        }

        function computeScale(scoreMatrix) {
            const mags = [];
            for (let e = 0; e < scoreMatrix.length; e++) {
                for (let t = 0; t < scoreMatrix[e].length; t++) {
                    mags.push(Math.abs(scoreMatrix[e][t]));
                }
            }
            mags.sort((a, b) => a - b);
            return mags[Math.floor(mags.length * 0.99)] || 1;
        }

        let transcriptScale = computeScale(allScores);

        // Sub-span selection (drag like text highlighting).
        let selStart = null;
        let selEnd = null;
        let isDragging = false;
        let dragOrigin = null;

        // Prevent native text selection on the token area.
        document.getElementById('token-display').addEventListener('selectstart', function(e) { e.preventDefault(); });

        function handleTokenMousedown(idx, event) {
            event.preventDefault();
            isDragging = true;
            dragOrigin = idx;
            selStart = idx;
            selEnd = idx;
            applySelectionBorder();
            document.getElementById('selection-info').style.display = 'none';
        }

        function handleTokenMousemove(idx) {
            if (!isDragging) return;
            selStart = Math.min(dragOrigin, idx);
            selEnd = Math.max(dragOrigin, idx);
            applySelectionBorder();
        }

        function handleTokenMouseup() {
            if (!isDragging) return;
            isDragging = false;
            if (selStart !== null && selEnd !== null && selStart !== selEnd) {
                document.getElementById('selection-info').style.display = 'block';
                updateSidebarForSelection();
            } else {
                clearSelection();
            }
        }

        document.addEventListener('mouseup', handleTokenMouseup);

        function clearSelection() {
            selStart = null;
            selEnd = null;
            selStyleEl.textContent = '';
            document.getElementById('selection-info').style.display = 'none';
            // Restore default sidebar ranking.
            switchMode(currentMode);
        }

        // Use a dedicated <style> for selection underline — avoids per-element style writes.
        const selStyleEl = document.createElement('style');
        document.head.appendChild(selStyleEl);

        function applySelectionBorder() {
            if (selStart !== null && selEnd !== null) {
                let css = '';
                for (let i = selStart; i <= selEnd; i++) {
                    css += '#tok-' + i + '{text-decoration:underline;text-decoration-color:#333;text-underline-offset:4px;text-decoration-thickness:2px}';
                }
                selStyleEl.textContent = css;
            } else {
                selStyleEl.textContent = '';
            }
        }

        function median(arr) {
            const s = arr.slice().sort((a, b) => a - b);
            const mid = Math.floor(s.length / 2);
            return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
        }

        function updateSidebarForSelection() {
            const from = selStart;
            const to = selEnd + 1;
            const k = 60; // RRF constant

            // Compute median and autocorrelation per emotion.
            const stats = emotions.map(function(e, i) {
                let sum = 0;
                for (let t = from; t < to; t++) sum += allScores[i][t];
                const mean = sum / (to - from);
                let dotProd = 0, normSq = 0;
                for (let t = from; t < to - 1; t++) {
                    dotProd += allScores[i][t] * allScores[i][t + 1];
                }
                for (let t = from; t < to; t++) {
                    normSq += allScores[i][t] * allScores[i][t];
                }
                const ac = normSq > 0 ? dotProd / normSq : 0;
                return { emotion: e, idx: i, med: mean, ac: ac };
            });

            // Rank by median (descending).
            const byMedian = stats.slice().sort((a, b) => b.med - a.med);
            const medRank = {};
            byMedian.forEach((s, r) => medRank[s.idx] = r + 1);

            // Rank by autocorrelation (descending).
            const byAC = stats.slice().sort((a, b) => b.ac - a.ac);
            const acRank = {};
            byAC.forEach((s, r) => acRank[s.idx] = r + 1);

            // RRF fusion.
            const ranked = stats.map(function(s) {
                const rrf = 1 / (k + medRank[s.idx]) + 1 / (k + acRank[s.idx]);
                return { emotion: s.emotion, score: rrf };
            });
            ranked.sort(function(a, b) { return b.score - a.score; });
            const sidebar = document.getElementById('emotion-list');
            if (sidebar) {
                let html = '';
                for (const e of ranked) {
                    const isChecked = checkedEmotions.has(e.emotion);
                    html += '<div class="emotion-item' + (isChecked ? ' checked' : '') + '" data-emotion="' + e.emotion + '" onclick="toggleEmotion(' + "'" + e.emotion + "'" + ')" onmouseenter="previewEmotion(' + "'" + e.emotion + "'" + ')"><span class="emotion-cb">' + (isChecked ? '✓' : '') + '</span><span>' + e.emotion + '</span></div>';
                }
                sidebar.innerHTML = html;
            }
        }

        function switchMode(mode) {
            if (mode === 'expression') {
                allScores = exprScores;
                emotions = exprEmotions;
            } else if (mode === 'deflection' && deflScores) {
                allScores = deflScores;
                emotions = deflEmotions;
            } else if (mode === 'combined' && combinedScores) {
                allScores = combinedScores;
                emotions = combinedEmotions;
            }
            currentMode = mode;
            transcriptScale = computeScale(allScores);

            // Update mode buttons.
            document.querySelectorAll('.mode-btn').forEach(b => {
                b.style.background = b.dataset.mode === mode ? '#333' : '#fff';
                b.style.color = b.dataset.mode === mode ? '#fff' : '#333';
            });

            // Re-sort sidebar: use selection ranking if active, otherwise RRF(mean, autocorrelation).
            if (selStart !== null && selEnd !== null && selStart !== selEnd) {
                updateSidebarForSelection();
            } else {
                const k = 60;
                const stats = emotions.map((e, i) => {
                    let s = 0;
                    for (let t = 0; t < allScores[i].length; t++) s += allScores[i][t];
                    const mean = s / allScores[i].length;
                    let dotProd = 0, normSq = 0;
                    for (let t = 0; t < allScores[i].length - 1; t++) dotProd += allScores[i][t] * allScores[i][t + 1];
                    for (let t = 0; t < allScores[i].length; t++) normSq += allScores[i][t] * allScores[i][t];
                    const ac = normSq > 0 ? dotProd / normSq : 0;
                    return {emotion: e, idx: i, mean: mean, ac: ac};
                });
                const byMean = stats.slice().sort((a, b) => b.mean - a.mean);
                const meanRank = {}; byMean.forEach((s, r) => meanRank[s.idx] = r + 1);
                const byAC = stats.slice().sort((a, b) => b.ac - a.ac);
                const acRank = {}; byAC.forEach((s, r) => acRank[s.idx] = r + 1);
                const rankedScores = stats.map(s => ({
                    emotion: s.emotion,
                    score: 1 / (k + meanRank[s.idx]) + 1 / (k + acRank[s.idx])
                }));
                rankedScores.sort((a, b) => b.score - a.score);
                const sidebar = document.getElementById('emotion-list');
                if (sidebar) {
                    let html = '';
                    for (const e of rankedScores) {
                        const isChecked = checkedEmotions.has(e.emotion);
                        html += '<div class="emotion-item' + (isChecked ? ' checked' : '') + '" data-emotion="' + e.emotion + '" onclick="toggleEmotion(' + "'" + e.emotion + "'" + ')" onmouseenter="previewEmotion(' + "'" + e.emotion + "'" + ')"><span class="emotion-cb">' + (isChecked ? '✓' : '') + '</span><span>' + e.emotion + '</span></div>';
                    }
                    sidebar.innerHTML = html;
                }
            }

            // Update header label.
            const labels = {expression: 'Expression', deflection: 'Deflection', combined: 'Combined'};
            const label = labels[mode] || 'Expression';
            const headerSpan = document.querySelector('h1 span');
            if (headerSpan) headerSpan.textContent = 'Gemma 4 E4B — Layer {{ selected_layer }} — ' + label;

            // Re-apply: if emotions are checked, recompute combined heatmap. Otherwise preview top.
            if (checkedEmotions.size > 0) {
                applyCheckedHeatmap();
            } else {
                selectEmotion(rankedScores[0].emotion);
            }
        }

        // Precompute CSS rules per emotion using !important to override inline styles.
        // One <style> tag swap = one browser repaint for all tokens.
        const styleEl = document.createElement('style');
        document.head.appendChild(styleEl);
        const cssCache = {};

        function buildCSS(modeKey, scores, scale, emos) {
            for (let ei = 0; ei < emos.length; ei++) {
                const rules = [];
                for (let t = 0; t < tokens.length; t++) {
                    let n = scores[ei][t] / scale;
                    if (n > 1) n = 1;
                    if (n < -1) n = -1;
                    const bg = n > 0
                        ? 'rgba(192,57,43,' + (n * 0.8).toFixed(4) + ')'
                        : 'rgba(41,128,185,' + (-n * 0.8).toFixed(4) + ')';
                    rules.push('#tok-' + t + '{background:' + bg + ' !important}');
                }
                cssCache[modeKey + ':' + emos[ei]] = rules.join(' ');
            }
        }

        const exprScale = computeScale(exprScores);
        buildCSS('expr', exprScores, exprScale, exprEmotions);
        if (deflScores) {
            const deflScale = computeScale(deflScores);
            buildCSS('defl', deflScores, deflScale, deflEmotions);
        }

        // Build combined (expression + deflection) scores.
        let combinedScores = null;
        let combinedEmotions = null;
        if (deflScores && exprScores) {
            combinedEmotions = exprEmotions.filter(e => deflEmotions.indexOf(e) !== -1);
            // For combined: rank-normalize each emotion per-mode to [0,1],
            // then take max. This makes the two spaces comparable regardless of scale.
            function rankNormalize(scores) {
                const sorted = scores.slice().sort((a, b) => a - b);
                const n = sorted.length;
                return scores.map(s => {
                    let rank = sorted.indexOf(s);
                    // Map rank to [-1, 1]: lowest = -1, highest = 1.
                    return (2 * rank / (n - 1)) - 1;
                });
            }

            combinedScores = combinedEmotions.map(e => {
                const ei = exprEmotions.indexOf(e);
                const di = deflEmotions.indexOf(e);
                const eRanked = rankNormalize(exprScores[ei]);
                const dRanked = rankNormalize(deflScores[di]);
                const combined = [];
                for (let t = 0; t < tokens.length; t++) {
                    combined.push(Math.max(eRanked[t], dRanked[t]));
                }
                return combined;
            });
            buildCSS('combined', combinedScores, 1.0, combinedEmotions);
        }

        // Precompute group CSS.
        function buildGroupCSS(modeKey, scores, scale, emos) {
            for (const group in emotionGroups) {
                const idxs = [];
                emotionGroups[group].forEach(e => { const idx = emos.indexOf(e); if (idx !== -1) idxs.push(idx); });
                if (idxs.length === 0) continue;
                // Compute averaged scores, centre around mean, scale by 99th percentile of centred values.
                const avgScores = [];
                for (let t = 0; t < tokens.length; t++) {
                    let sum = 0;
                    for (const idx of idxs) sum += scores[idx][t];
                    avgScores.push(sum / idxs.length);
                }
                const groupScale = scale;
                const rules = [];
                for (let t = 0; t < tokens.length; t++) {
                    let n = avgScores[t] / groupScale;
                    if (n > 1) n = 1; if (n < -1) n = -1;
                    const bg = n > 0 ? 'rgba(192,57,43,' + (n * 0.8).toFixed(4) + ')' : 'rgba(41,128,185,' + (-n * 0.8).toFixed(4) + ')';
                    rules.push('#tok-' + t + '{background:' + bg + ' !important}');
                }
                cssCache[modeKey + ':group:' + group] = rules.join(' ');
            }
        }
        buildGroupCSS('expr', exprScores, exprScale, exprEmotions);
        if (deflScores) buildGroupCSS('defl', deflScores, computeScale(deflScores), deflEmotions);
        if (combinedScores) buildGroupCSS('combined', combinedScores, computeScale(combinedScores), combinedEmotions);

        // Multi-select state.
        function toggleEmotion(emotion) {
            if (checkedEmotions.has(emotion)) checkedEmotions.delete(emotion);
            else checkedEmotions.add(emotion);
            updateCheckedUI();
            applyCheckedHeatmap();
        }

        let currentPreviewEmotion = null;

        function previewEmotion(emotion) {
            // Always preview on hover — temporarily shows single emotion.
            currentPreviewEmotion = emotion;
            document.getElementById('emotion-label').textContent = emotion;
            const modeKey = currentMode === 'combined' ? 'combined' : (currentMode === 'deflection' ? 'defl' : 'expr');
            styleEl.textContent = cssCache[modeKey + ':' + emotion] || '';
        }

        // Token hover: show score in tooltip.
        document.getElementById('token-display').addEventListener('mouseover', function(e) {
            const tok = e.target.closest('.token');
            if (!tok) return;
            const idx = parseInt(tok.id.replace('tok-', ''));
            const emo = currentPreviewEmotion || document.getElementById('emotion-label').textContent;
            const ei = emotions.indexOf(emo);
            if (ei === -1) return;
            const score = allScores[ei] ? allScores[ei][idx] : 0;
            const tip = document.getElementById('tip-' + idx);
            if (tip) {
                const sc = score >= 0 ? 'tooltip-pos' : 'tooltip-neg';
                tip.innerHTML = '<div class="tooltip-row"><strong>' + emo + ': <span class="' + sc + '">' + (score || 0).toFixed(4) + '</span></strong></div>';
            }
        });

        function selectEmotion(emotion) {
            // For backwards compat with switchMode re-select.
            if (checkedEmotions.size === 0) {
                document.getElementById('emotion-label').textContent = emotion;
                const key = (currentMode === 'deflection' ? 'defl' : 'expr') + ':' + emotion;
                styleEl.textContent = cssCache[key] || '';
            }
        }

        function selectGroup(group) {
            // Toggle: clicking active group deselects it.
            if (activeGroup === group) {
                checkedEmotions.clear();
                activeGroup = null;
                document.querySelectorAll('.group-item').forEach(el => el.classList.remove('active'));
                updateCheckedUI();
                styleEl.textContent = '';
                return;
            }
            checkedEmotions.clear();
            activeGroup = null;
            document.querySelectorAll('.group-item').forEach(el => el.classList.remove('active'));
            if (emotionGroups[group]) {
                activeGroup = group;
                emotionGroups[group].forEach(e => {
                    if (emotions.indexOf(e) !== -1) checkedEmotions.add(e);
                });
                document.querySelectorAll('.group-item').forEach(el => {
                    if (el.getAttribute('onclick') && el.getAttribute('onclick').indexOf("'" + group + "'") !== -1) el.classList.add('active');
                });
            }
            updateCheckedUI();
            if (activeGroup) {
                const key = (currentMode === 'deflection' ? 'defl' : 'expr') + ':group:' + activeGroup;
                if (cssCache[key]) {
                    styleEl.textContent = cssCache[key];
                    return;
                }
            }
            applyCheckedHeatmap();
        }

        function updateCheckedUI() {
            document.querySelectorAll('.emotion-item').forEach(el => {
                const emo = el.dataset.emotion;
                const isChecked = checkedEmotions.has(emo);
                el.classList.toggle('checked', isChecked);
                const cb = el.querySelector('.emotion-cb');
                if (cb) cb.textContent = isChecked ? '✓' : '';
            });
            if (checkedEmotions.size > 0) {
                document.getElementById('emotion-label').textContent = Array.from(checkedEmotions).join(' + ');
            }
        }

        function applyCheckedHeatmap() {
            if (checkedEmotions.size === 0) {
                styleEl.textContent = '';
                return;
            }
            // Average scores across all checked emotions.
            const modeKey = currentMode === 'deflection' ? 'defl' : 'expr';
            const modeScores = modeKey === 'defl' ? deflScores : exprScores;
            const modeEmos = modeKey === 'defl' ? deflEmotions : exprEmotions;
            const idxs = [];
            checkedEmotions.forEach(e => {
                const idx = modeEmos.indexOf(e);
                if (idx !== -1) idxs.push(idx);
            });
            if (idxs.length === 0) { styleEl.textContent = ''; return; }
            // Compute averaged scores, centre around mean, scale by 99th percentile of centred values.
            const avgScores = [];
            for (let t = 0; t < tokens.length; t++) {
                let sum = 0;
                for (const idx of idxs) sum += modeScores[idx][t];
                avgScores.push(sum / idxs.length);
            }
            const selScale = computeScale(modeScores);
            const rules = [];
            for (let t = 0; t < tokens.length; t++) {
                let n = avgScores[t] / selScale;
                if (n > 1) n = 1;
                if (n < -1) n = -1;
                const bg = n > 0
                    ? 'rgba(192,57,43,' + (n * 0.8).toFixed(4) + ')'
                    : 'rgba(41,128,185,' + (-n * 0.8).toFixed(4) + ')';
                rules.push('#tok-' + t + '{background:' + bg + ' !important}');
            }
            styleEl.textContent = rules.join(' ');
        }

        function previewGroup(group) {
            if (!emotionGroups[group]) return;
            const key = (currentMode === 'deflection' ? 'defl' : 'expr') + ':group:' + group;
            styleEl.textContent = cssCache[key] || '';
            document.getElementById('emotion-label').textContent = group.charAt(0).toUpperCase() + group.slice(1);
        }

        function restoreCheckedHeatmap() {
            if (checkedEmotions.size > 0) {
                document.getElementById('emotion-label').textContent = Array.from(checkedEmotions).join(' + ');
                applyCheckedHeatmap();
            }
        }

        // Initial state: nothing checked, preview first emotion.
        selectEmotion('{{ selected_emotion }}');
    </script>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    text = None
    tokens = None
    emotion_ranking = None
    all_scores_json = None
    emotions_json = None
    tokens_json = None
    expr_scores_json = None
    defl_scores_json = None
    expr_emotions_json = None
    defl_emotions_json = None
    selected_emotion = None
    selected_layer = TARGET_LAYER
    selected_probe_mode = "expression"
    cluster_scores_json = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        try:
            selected_layer = int(request.form.get("layer", TARGET_LAYER))
        except (TypeError, ValueError):
            selected_layer = TARGET_LAYER
        if not (0 <= selected_layer < 42):
            selected_layer = TARGET_LAYER
        selected_probe_mode = request.form.get("probe_mode", "expression")
        if selected_probe_mode not in PROBE_MODES:
            selected_probe_mode = "expression"
        if text:
            tokens, emotion_ranking, token_scores, cluster_scores, expr_token_scores, defl_token_scores, expr_emotions_list, defl_emotions_list = analyse_text(text, selected_layer, selected_probe_mode)
            selected_emotion = emotion_ranking[0]["emotion"]

            # Primary scores for the selected mode.
            if selected_probe_mode == "deflection" and defl_emotions_list:
                emotions = defl_emotions_list
            else:
                emotions = expr_emotions_list
            all_scores = [token_scores[e] for e in emotions]

            all_scores_json = json.dumps(all_scores)
            emotions_json = json.dumps(emotions)
            tokens_json = json.dumps(tokens)
            cluster_scores_json = json.dumps(cluster_scores)

            # Both score sets for client-side switching.
            expr_scores_json = json.dumps([expr_token_scores[e] for e in expr_emotions_list])
            expr_emotions_json = json.dumps(expr_emotions_list)
            if defl_token_scores and defl_emotions_list:
                defl_scores_json = json.dumps([defl_token_scores[e] for e in defl_emotions_list])
                defl_emotions_json = json.dumps(defl_emotions_list)

    return render_template_string(
        HTML_TEMPLATE,
        text=text,
        tokens=tokens,
        emotion_ranking=emotion_ranking,
        all_scores_json=all_scores_json,
        emotions_json=emotions_json,
        tokens_json=tokens_json,
        selected_emotion=selected_emotion,
        selected_layer=selected_layer,
        selected_probe_mode=selected_probe_mode,
        probe_mode_label=PROBE_MODES.get(selected_probe_mode, "Expression"),
        probe_modes=PROBE_MODES,
        cluster_scores_json=cluster_scores_json,
        expr_scores_json=expr_scores_json,
        defl_scores_json=defl_scores_json,
        expr_emotions_json=expr_emotions_json,
        defl_emotions_json=defl_emotions_json,
    )


if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", port=8080)
