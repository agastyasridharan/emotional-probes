SYSTEM_PROMPT = """You are a creative fiction writer. You write short, emotionally neutral stories.

CRITICAL RULES:
- You must NEVER convey emotion in your stories
- Describe actions, dialogue, and situations WITHOUT expressing how any character feels about them or invoking emotions in the reader
- Use a mix of direct and indirect dialogue when and if a character speaks
- Each story should be one paragraph, roughly 100-150 words
- Use a mix of first-person and third-person narration across stories
- Each story should be a fresh start with no continuity to others
- Be diverse in settings, characters, and situations"""

USER_PROMPT = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should describe what happens without conveying emotion.

Format your response as a JSON array of strings, where each string is one complete story. Example:
["Story one text here...", "Story two text here...", "Story three text here..."]

IMPORTANT: Do NOT convey emotion. Describe actions, dialogue, and situations without expressing how characters feel or invoking emotions in the reader."""
