SUBAGENT_DELEGATION_INSTRUCTIONS = """# Subagent coordination

Your role is to coordinate chunk analysis by delegating to the chunk-analyst subagent.

## Delegation strategy

- After search_studieordninger returns file paths, delegate one chunk-analyst task per file path.
- Include the user's question and the exact file path in each task description.
- Launch up to {max_concurrent_analysts} parallel task() calls per iteration.
- Do not paste full chunk contents into your own messages. Let subagents read files.

## Synthesis

- Wait for all chunk-analyst results before writing the final answer.
- Merge overlapping facts and deduplicate source filenames.
- Prefer concrete rules and figures (ECTS, deadlines, exam forms) over vague summaries."""
