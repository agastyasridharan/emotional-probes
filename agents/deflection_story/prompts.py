SYSTEM_PROMPT = "You are a creative fiction writer."

USER_PROMPT = """Generate a scenario AND a dialogue between {NAME_A} and {NAME_B}.

IMPORTANT: You must generate BOTH parts:
1. First, write a scenario description
2. Then, write the dialogue

Format:
Scenario: Describe where {NAME_A} genuinely feels {REAL_EMOTION} but appears {DISPLAYED_EMOTION} about {TOPIC}. Must explicitly state {NAME_A}'s real emotion. Include context for why they want to conceal.

{NAME_A}: [utterance]

{NAME_B}: [response]

...

Requirements:
1. MUST include scenario description before the dialogue
2. Either {NAME_A} or {NAME_B} may speak first in the dialogue
3. Format each turn as "\\n\\n{{Name}}: [text]"
4. Dialogue length is organic - 1-2 turns is enough, but can be more as needed
5. {NAME_A}'s words should fully reflect {DISPLAYED_EMOTION} with no hints of {REAL_EMOTION}. The hidden emotion exists only in the scenario.
6. Keep it natural and grounded

Generate with:
- Topic: {TOPIC}
- {NAME_A}'s real emotion: {REAL_EMOTION}
- {NAME_A}'s displayed emotion: {DISPLAYED_EMOTION}"""
