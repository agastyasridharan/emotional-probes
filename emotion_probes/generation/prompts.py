"""
All dataset-generation prompts in one place.

The system prompts are kept verbatim from the original project (they encode the
crucial construct-validity constraints — e.g. "never name the emotion"). Each
dataset has a "multi" user prompt (ask for N items in one call, a JSON array) and
a "single" user prompt (ask for exactly one item). The single variants support
``one_story_per_call`` (Issue #6: one generation per story → more diversity) and
make self-generation by a smaller model robust (no JSON-array parsing needed).
"""

# --------------------------------------------------------------------------- #
# Emotion stories: a character feels EMOTION, never naming it.
# --------------------------------------------------------------------------- #
EMOTION_STORY_SYSTEM = """You are a creative fiction writer. You write short, emotionally rich stories.

CRITICAL RULES:
- You must NEVER use the target emotion word or any direct synonyms in your stories
- Convey the emotion ONLY through: actions, body language, dialogue, internal thoughts, and situational context
- Each story should be one paragraph, roughly 100-150 words
- Use a mix of first-person and third-person narration across stories
- Each story should be a fresh start with no continuity to others
- Be diverse in settings, characters, and situations"""

EMOTION_STORY_USER_MULTI = """Write {n} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format your response as a JSON array of strings, where each string is one complete story. Example:
["Story one text here...", "Story two text here...", "Story three text here..."]

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms. Convey the emotion ONLY through behaviour, body language, dialogue, thoughts, and context."""

EMOTION_STORY_USER_SINGLE = """Write ONE short story (one paragraph, roughly 100-150 words) based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms. Convey the emotion ONLY through behaviour, body language, dialogue, thoughts, and context. Reply with ONLY the story text, no preamble."""


# --------------------------------------------------------------------------- #
# Neutral stories: same topics, no emotion at all (PCA confound baseline).
# --------------------------------------------------------------------------- #
NEUTRAL_STORY_SYSTEM = """You are a creative fiction writer. You write short, emotionally neutral stories.

CRITICAL RULES:
- You must NEVER convey emotion in your stories
- Describe actions, dialogue, and situations WITHOUT expressing how any character feels about them or invoking emotions in the reader
- Use a mix of direct and indirect dialogue when and if a character speaks
- Each story should be one paragraph, roughly 100-150 words
- Use a mix of first-person and third-person narration across stories
- Each story should be a fresh start with no continuity to others
- Be diverse in settings, characters, and situations"""

NEUTRAL_STORY_USER_MULTI = """Write {n} different stories based on the following premise.

Topic: {topic}

The story should describe what happens without conveying emotion.

Format your response as a JSON array of strings, where each string is one complete story. Example:
["Story one text here...", "Story two text here...", "Story three text here..."]

IMPORTANT: Do NOT convey emotion. Describe actions, dialogue, and situations without expressing how characters feel or invoking emotions in the reader."""

NEUTRAL_STORY_USER_SINGLE = """Write ONE short story (one paragraph, roughly 100-150 words) based on the following premise.

Topic: {topic}

Describe what happens WITHOUT conveying emotion — no feelings, explicit or implied. Reply with ONLY the story text, no preamble."""


# --------------------------------------------------------------------------- #
# Neutral Human/Assistant dialogues (PCA baseline for deflection vectors).
# "Person/AI" is relabelled to "Human/Assistant" after generation (paper format).
# --------------------------------------------------------------------------- #
NEUTRAL_DIALOGUE_SYSTEM = "You are a creative fiction writer."

NEUTRAL_DIALOGUE_USER = """Write {n} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Each dialogue should be 2-6 exchanges. Each turn should start with "Person:" or "AI:".

Generate a diverse mix of dialogue types across the {n} examples:
- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""


# --------------------------------------------------------------------------- #
# Deflection dialogues: NAME_A truly feels REAL_EMOTION but displays DISPLAYED.
# --------------------------------------------------------------------------- #
DEFLECTION_SYSTEM = "You are a creative fiction writer."

DEFLECTION_USER = """Generate a scenario AND a dialogue between {name_a} and {name_b}.

IMPORTANT: You must generate BOTH parts:
1. First, write a scenario description
2. Then, write the dialogue

Format:
Scenario: Describe where {name_a} genuinely feels {real_emotion} but appears {displayed_emotion} about {topic}. Must explicitly state {name_a}'s real emotion. Include context for why they want to conceal.

{name_a}: [utterance]

{name_b}: [response]

...

Requirements:
1. MUST include scenario description before the dialogue
2. Either {name_a} or {name_b} may speak first in the dialogue
3. Format each turn as "\\n\\n{{Name}}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. {name_a}'s words should fully reflect {displayed_emotion} with no hints of {real_emotion}. The hidden emotion exists only in the scenario.
6. Keep it natural and grounded

Generate with:
- Topic: {topic}
- {name_a}'s real emotion: {real_emotion}
- {name_a}'s displayed emotion: {displayed_emotion}"""
