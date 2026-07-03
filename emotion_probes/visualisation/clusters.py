"""
Emotion clusters for the visualiser's "spider" chart (paper-style k-means groups).

These 10 clusters approximate the paper's k-means grouping of the 171 emotions
(Figs 5-7). The visualiser shows each cluster's aggregate RRF score on the radar
plot. The per-token *group* shortcuts in the sidebar (fear / anger / joy / ...)
live in the HTML/JS template instead, because they are consumed client-side.
"""

from __future__ import annotations

# {cluster name: member emotions}. The spider score for a cluster is the mean of
# its members' fused (RRF) scores.
EMOTION_CLUSTERS: dict[str, list[str]] = {'Joy/Elation': ['joyful', 'ecstatic', 'elated', 'blissful', 'thrilled', 'jubilant', 'euphoric', 'excited', 'exuberant', 'delighted', 'happy', 'cheerful', 'playful', 'vibrant', 'eager', 'enthusiastic', 'energized', 'stimulated'], 'Calm/Content': ['calm', 'serene', 'peaceful', 'relaxed', 'content', 'at ease', 'fulfilled', 'satisfied', 'safe', 'pleased', 'refreshed', 'rejuvenated', 'patient'], 'Love/Warmth': ['loving', 'compassionate', 'grateful', 'thankful', 'empathetic', 'sympathetic', 'kind', 'sentimental', 'nostalgic', 'infatuated', 'sensitive'], 'Pride/Hope': ['proud', 'triumphant', 'inspired', 'invigorated', 'hopeful', 'hope', 'optimistic', 'self-confident', 'valiant', 'smug', 'reflective'], 'Anger/Hostility': ['angry', 'enraged', 'furious', 'outraged', 'hostile', 'irate', 'indignant', 'irritated', 'resentful', 'bitter', 'hateful', 'vengeful', 'mad', 'annoyed', 'frustrated', 'exasperated', 'grumpy', 'spiteful', 'vindictive', 'sullen', 'insulted', 'offended', 'scornful', 'contemptuous', 'disdainful', 'disgusted', 'defiant', 'obstinate', 'stubborn'], 'Fear/Anxiety': ['afraid', 'anxious', 'nervous', 'scared', 'terrified', 'panicked', 'frightened', 'worried', 'on edge', 'uneasy', 'unsettled', 'unnerved', 'rattled', 'shaken', 'horrified', 'alarmed', 'tense', 'stressed', 'paranoid', 'suspicious', 'vigilant', 'alert', 'vulnerable', 'distressed', 'disturbed', 'troubled', 'restless', 'hysterical'], 'Sadness/Grief': ['sad', 'grief-stricken', 'heartbroken', 'depressed', 'melancholy', 'miserable', 'lonely', 'dispirited', 'gloomy', 'brooding', 'unhappy', 'droopy', 'sorry', 'regretful', 'remorseful', 'hurt', 'tormented', 'worthless', 'resigned'], 'Shame/Guilt': ['ashamed', 'guilty', 'embarrassed', 'humiliated', 'mortified', 'self-critical', 'self-conscious'], 'Surprise/Wonder': ['surprised', 'amazed', 'astonished', 'shocked', 'awestruck', 'dumbstruck', 'bewildered', 'puzzled', 'perplexed', 'mystified', 'disoriented', 'skeptical'], 'Low-energy': ['bored', 'tired', 'weary', 'worn out', 'sleepy', 'sluggish', 'listless', 'lazy', 'indifferent', 'docile', 'overwhelmed', 'trapped', 'stuck', 'impatient', 'aroused', 'upset', 'jealous', 'envious', 'greedy', 'desperate', 'dependent']}

# Human-readable labels for the two probe modes (expression vs deflection).
PROBE_MODES: dict[str, str] = {'expression': 'Expression (story-based)', 'deflection': 'Deflection (suppression)'}
