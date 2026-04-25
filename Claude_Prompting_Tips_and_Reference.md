LLM Prompting: Quick Reference & Best Practices
Executive Summary (Read This First)
The single most important rule: Before sending any message, ask: "Does this produce a deliverable, or does it produce a response that will lead to a second message that produces the deliverable?" If the latter, rewrite to skip the intermediate step.

Five fastest wins:

Batch everything. Five separate messages = 10+ turns. One batched message = 1–2 turns. Paste all related material at once.

Combine assessment and action. Don't ask "Is X relevant?" then "OK, add it." Say: "Assess X. If relevant, integrate it. If not, say so and move on."

Treat "can you?" as "do this." Don't ask "Can you add this edit?" Say "Add this edit. If you can't, explain why."

Lead with a briefing. Give the full project state before asking for action. A good briefing answers: what's the project, what's done, what remains, what's the goal for this session.

Upload files, don't describe them. Paraphrasing a long document wastes tokens and introduces errors.

One habit to break: Asking for opinions you'll immediately act on. If you're going to say "yes" after the LLM says "yes," just ask for the action directly.

Section 1: Generic Session Initialization Prompt
Paste and adapt this at the start of any project-oriented chat.

Project: [One-sentence description]

Uploaded files: [List files and what each contains]

Session goal: [What you want to accomplish]

Standing instructions for this session:

When I share material for review, assess relevance and act on your recommendation in the same response. If relevant, integrate it immediately. If not, say so briefly and move on.

Treat "Can this be done?" as "Do this." If you cannot or should not proceed, explain why. Otherwise, proceed.

When I request an edit, make it directly. Do not describe the edit and wait for approval unless the change is ambiguous or risky (e.g., deleting content, changing a key claim).

Batch multiple small tasks from a single message into one response rather than handling them sequentially.

When I tell you I am transitioning to a new chat, produce transplant materials (briefing, checklist, etc.) without being asked whether they would be useful.

Out of scope for this chat: [List exclusions]

Section 2: General Best Practices
Starting a session
Brief first, then ask. A complete briefing eliminates back-and-forth.

Upload, don't paraphrase. Token efficiency + accuracy.

Set boundaries explicitly. Tell the LLM what to ignore. This prevents tangents.

During a session
Batch related items. One prompt covering five items > five prompts.

Grant agency with guardrails. "Proceed unless the change is ambiguous or would delete content" is faster than asking permission for every step.

Don't ask for confirmation of what the LLM just said. If it recommends an edit and you agree, say "do it" — not "yes, I agree, please implement it."

When to start a fresh chat
Transition at task boundaries, not when hitting the context limit. Switching from "edit paper" to "write code" is a natural break. Mid-task transitions due to a full context window lose working state.

Prepare handoff materials before you need them. At the end of a productive session, ask for a briefing and checklist while context is still fresh.

File management
One concern, one document. The paper, research notes, and checklists are separate files. Mixing them slows every session.

Version your deliverables. v1, v2, v3 naming avoids ambiguity.

Save after each successful edit. Download after each confirmed change. If a later edit fails, roll back without redoing work.

Section 3: Personal Coaching Notes
Based on observed patterns. Review before starting a session until the habits become automatic.

Pattern 1: Feasibility check before action
❌ Instead of: "Is there an efficient way to add these revisions?" → [LLM explains] → "OK, please do it."

✅ Do this: "Add these revisions. If there's a more efficient method than what I'm imagining, use that instead."

Why: Each unnecessary check costs a full round-trip. Action-first gets the same result in half the messages.

Pattern 2: Evaluate then act, separately
❌ Instead of: "Read this and tell me what's relevant" → [LLM assesses] → "OK, please integrate it."

✅ Do this: "Read this. Integrate anything relevant. For anything not relevant, note why briefly."

Why: The two-step version makes the LLM read the material twice. The one-step version reads once and acts.

Pattern 3: Sequential review of multiple items
❌ Instead of: Five separate messages, each with "is this relevant?"

✅ Do this: One message: "Here are five excerpts. For each: (a) integrate, (b) file as reference, or (c) skip. For (a) and (b), draft the integration in the same response."

Why: Five cycles cost 10+ messages. One batched message costs 1–2. This is the largest single efficiency gain.

Pattern 4: Asking permission to transition
❌ Instead of: "How close to the limit am I? Should I start a new chat?"

✅ Do this: "I'm starting a new chat after this task. Please produce: (1) an updated project briefing, and (2) any handoff notes on decisions made in this session."

Why: You already know the answer. Stating the decision gets the handoff materials in one turn.

What to keep doing
Separate concerns (paper vs. notes vs. checklists). This is correct and valuable.

Write good briefings. They're the highest-leverage thing you can produce.

Delegate phrasing but control scope. That's the most efficient division of labor.

Last updated: April 2026. Review the Executive Summary before each session until the patterns feel automatic.