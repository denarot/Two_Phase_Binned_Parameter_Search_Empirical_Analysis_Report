# Claude Prompting: Tips, Tricks, and Personal Reference

**Purpose:** A reusable reference document for getting the most out of Claude sessions. Three sections: a generic initialization prompt template, general best practices, and personal coaching notes based on observed patterns. Review before starting a new session until the habits are internalized.

---

## Section 1: Generic Initialization Prompt Template

Paste and adapt this at the start of any project-oriented chat session. Replace the bracketed sections with your specifics.

---

**Template:**

> **Project:** [One-sentence description of the project]
>
> **Uploaded files:** [List files you've uploaded and what each one is]
>
> **Session goal:** [What you want to accomplish in this session — e.g., "edit the paper," "write experimental code," "evaluate new material for relevance"]
>
> **Standing instructions for this session:**
>
> 1. When I share material for review, assess its relevance and act on your recommendation in the same response. If something should be integrated into [the paper / the journal / the codebase], draft the integration immediately. If it is not relevant, say so briefly and move on. Do not wait for a second prompt to act on your own recommendation.
>
> 2. When I ask whether something can be done, treat it as a request to do it. If it cannot be done or there is a reason to pause, explain why instead. Otherwise, proceed.
>
> 3. If an edit to an existing document is requested, make the edit directly. Do not describe the edit and wait for approval unless the change is ambiguous or risky (e.g., deleting content, changing a claim, altering a proof).
>
> 4. When multiple small tasks are apparent from a single message, batch them into one response rather than handling them sequentially across multiple turns.
>
> 5. I will tell you when I am planning to transition to a new chat. At that point, produce any transplant materials (briefing, checklist, etc.) without being asked whether they would be useful.
>
> **What NOT to bring into this chat:** [List anything that is out of scope — e.g., "research journal material," "earlier transcript discussions," "pre-revision drafts"]

---

**Example (filled in for the Two-Phase Search project):**

> **Project:** Two-Phase Search paper — experimental validation phase.
>
> **Uploaded files:** Two_Phase_Search_Paper_Clean_v4.docx (current paper draft), Two_Phase_Search_Implementation_Guide.md (prompt-by-prompt code plan).
>
> **Session goal:** Generate the experimental code described in the implementation guide, prompts 1–4.
>
> **Standing instructions for this session:**
> [paste the five standing instructions above unchanged]
>
> **What NOT to bring into this chat:** Research journal material, provenance discussions, design philosophy analogies, pre-v4 paper drafts, or the Stumpy Forest / feature sampling research threads.

---

## Section 2: General Best Practices

### Starting a session

- **Lead with a briefing, not a question.** Give Claude the full project state before asking it to do anything. A good briefing answers: what is the project, what is done, what remains, and what is the goal for this session. This eliminates multiple rounds of Claude asking for context.

- **Upload files rather than describing them.** If Claude needs to reference a document, upload it. Paraphrasing a 30-page paper into chat messages wastes tokens and introduces inaccuracies.

- **Scope the session explicitly.** Tell Claude what is in scope and what is not. This prevents Claude from pulling in tangential material and keeps responses focused.

### During a session

- **Combine assessment and action in a single request.** Instead of "Is X relevant?" followed by "OK, add it," say "Assess X for relevance. If relevant, integrate it and tell me what you changed. If not, say so and move on." This collapses two turns into one.

- **Treat feasibility questions as action requests.** Instead of "Can you do X?" followed by "Please do X," say "Do X. If there's a reason you can't, explain instead." Claude will almost always be able to do it; the rare exception is worth handling as a fallback rather than as the default path.

- **Batch related material.** If you have five items to evaluate, paste all five in one message with a framing instruction rather than submitting them one at a time. One prompt and one response covers all five.

- **Grant agency with guardrails.** Instead of asking Claude to wait for permission at every step, say "proceed unless the change is ambiguous or would delete existing content." This lets Claude act on clear-cut items while still pausing on genuinely uncertain ones.

- **Don't ask Claude to confirm what it just said.** If Claude recommends an edit and you agree, say "do it" — not "yes, I agree with your recommendation, could you please go ahead and implement it?" The shorter version saves tokens and communicates the same intent.

### Knowing when to transition

- **Move to a new chat when switching tasks, not when hitting the wall.** Transitioning between "edit the paper" and "write experimental code" is a natural breakpoint. Transitioning mid-task because the context window filled up produces worse results because Claude loses the working state at the worst possible time.

- **Prepare transplant materials before you need them.** Ask Claude to produce a briefing and checklist at the end of a productive session, while the full context is still available. Don't wait until the next session and try to reconstruct from memory.

- **Keep separate documents for separate concerns.** The paper, the research journal, the checklist, and the implementation guide are four distinct documents serving four distinct purposes. Mixing them into one mega-document makes every session slower because Claude has to parse what's relevant.

### Document and file management

- **Bifurcate strictly:** content for the paper goes in the paper; provenance, analogies, and future directions go in a separate journal or notes document; checklists and tracking go in a third document. This discipline keeps the paper tight and prevents scope creep.

- **Version your deliverables.** v1, v2, v3, v4 naming makes it unambiguous which file is authoritative. Always tell Claude which version is current.

- **Save after each successful edit.** Download the file after each confirmed change. If a later edit fails or introduces problems, you can roll back to the last good version without re-doing work.

---

## Section 3: Personal Coaching Notes

*These notes are based on patterns observed in your April 2026 session on the Two-Phase Search paper. Review them before starting a session until the habits become automatic. The goal is to reduce unnecessary round-trips and maximize the amount of useful work per message.*

### Pattern 1: The feasibility check before the action request

**What you did:** "Is there an efficient method of adding the revisions?" → [Claude explains yes] → "Please do it."

**What to do instead:** "Add the revisions to the paper using targeted XML edits. If there's a more efficient method than what I'm imagining, use that instead."

**Why it matters:** Each feasibility check costs a full round-trip (your message + Claude's response). Over a session with five such checks, that is ten messages that produce no deliverable. The action-first pattern gets the same result in half the messages.

**Self-check question before sending:** "Am I asking whether something can be done, or am I asking for it to be done?" If the former, rewrite to the latter.

### Pattern 2: Evaluate, then act — in separate messages

**What you did:** "Could you read through this and determine what is relevant?" → [Claude assesses] → "Yes, please integrate the relevant parts."

**What to do instead:** "Read through this. Integrate anything relevant to [the paper / the journal]. For anything not relevant, note why briefly and skip it."

**Why it matters:** The two-step version means Claude reads the material twice — once to assess, once to act. The one-step version reads once and acts. For long documents, this saves significant context.

**Self-check question before sending:** "Am I asking for an opinion I'll immediately act on, or am I asking for a deliverable?" If the opinion is just a gate to the deliverable, skip the gate.

### Pattern 3: Sequential material review instead of batching

**What you did:** Submitted five separate pieces of historical material across five separate messages, each with "is this relevant?"

**What to do instead:** Paste all related material into a single message: "Here are five excerpts from earlier chats. For each one, classify as (a) integrate into paper, (b) integrate into journal, or (c) skip. For (a) and (b), draft the integration in the same response."

**Why it matters:** Five sequential evaluate-then-act cycles cost 10+ messages. One batched message costs 1–2 messages for the same outcome. This is the single largest efficiency gain available in material-heavy sessions.

**Self-check question before sending:** "Do I have more material of the same type to share? If yes, wait and batch it."

### Pattern 4: Asking for permission to transition

**What you did:** "How close to the chat limit am I? Is it beneficial to take the current draft to a new chat?"

**What to do instead:** "I'm planning to transition to a new chat after this task. Please produce: (1) an updated project briefing for the new chat, and (2) any transplant notes on decisions made in this session."

**Why it matters:** You already knew the answer — you were deep into the session and about to shift tasks. The question was a politeness reflex, not a genuine uncertainty. Treating it as a statement and a request gets the transplant materials in one turn.

**Self-check question before sending:** "Am I asking a question I already know the answer to?" If yes, state the answer and ask for the next action.

### What you should NOT change

- **Your instinct to separate concerns is correct and valuable.** Keep asking "is this relevant to the paper or to the journal?" — just combine the question with the action instruction.

- **Your briefing documents are excellent.** The project briefing you produced for this session was the gold standard. Keep using that format.

- **Your willingness to catalog and defer is rare and valuable.** Most people try to cram everything into the paper. Your discipline in keeping the paper tight and the journal separate produces a better paper. Do not lose this habit.

- **Your sense of when to ask for help vs. when to specify is well-calibrated.** You delegate phrasing but retain control over scope and structure. This is the most efficient division of labor between you and Claude.

### The one-sentence summary

**Before sending any message, ask: "Does this message produce a deliverable, or does it produce a response that will lead to a second message that produces the deliverable?" If the latter, rewrite to skip the intermediate step.**

---

*Last updated: April 2026. Review before each new session until the patterns in Section 3 feel automatic.*
