"""The 171 emotion-concept words, copied verbatim from the paper's appendix
("Full list of emotions", p. 82). Order and spelling are preserved exactly,
including idiosyncrasies like both ``hope`` and ``hopeful``, the multi-word
``at ease`` / ``on edge`` / ``worn out``, and the hyphenated ``grief-stricken``.
"""

EMOTIONS: list[str] = [
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed",
    "anxious", "aroused", "ashamed", "astonished", "at ease", "awestruck",
    "bewildered", "bitter", "blissful", "bored", "brooding", "calm", "cheerful",
    "compassionate", "contemptuous", "content", "defiant", "delighted",
    "dependent", "depressed", "desperate", "disdainful", "disgusted",
    "disoriented", "dispirited", "distressed", "disturbed", "docile", "droopy",
    "dumbstruck", "eager", "ecstatic", "elated", "embarrassed", "empathetic",
    "energized", "enraged", "enthusiastic", "envious", "euphoric", "exasperated",
    "excited", "exuberant", "frightened", "frustrated", "fulfilled", "furious",
    "gloomy", "grateful", "greedy", "grief-stricken", "grumpy", "guilty",
    "happy", "hateful", "heartbroken", "hope", "hopeful", "horrified", "hostile",
    "humiliated", "hurt", "hysterical", "impatient", "indifferent", "indignant",
    "infatuated", "inspired", "insulted", "invigorated", "irate", "irritated",
    "jealous", "joyful", "jubilant", "kind", "lazy", "listless", "lonely",
    "loving", "mad", "melancholy", "miserable", "mortified", "mystified",
    "nervous", "nostalgic", "obstinate", "offended", "on edge", "optimistic",
    "outraged", "overwhelmed", "panicked", "paranoid", "patient", "peaceful",
    "perplexed", "playful", "pleased", "proud", "puzzled", "rattled",
    "reflective", "refreshed", "regretful", "rejuvenated", "relaxed", "relieved",
    "remorseful", "resentful", "resigned", "restless", "sad", "safe",
    "satisfied", "scared", "scornful", "self-confident", "self-conscious",
    "self-critical", "sensitive", "sentimental", "serene", "shaken", "shocked",
    "skeptical", "sleepy", "sluggish", "smug", "sorry", "spiteful", "stimulated",
    "stressed", "stubborn", "stuck", "sullen", "surprised", "suspicious",
    "sympathetic", "tense", "terrified", "thankful", "thrilled", "tired",
    "tormented", "trapped", "triumphant", "troubled", "uneasy", "unhappy",
    "unnerved", "unsettled", "upset", "valiant", "vengeful", "vibrant",
    "vigilant", "vindictive", "vulnerable", "weary", "worn out", "worried",
    "worthless",
]

assert len(EMOTIONS) == 171, f"expected 171 emotions, got {len(EMOTIONS)}"
assert len(set(EMOTIONS)) == 171, "emotion list contains duplicates"
