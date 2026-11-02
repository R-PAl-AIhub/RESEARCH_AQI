# How to reply to me

- Keep every reply short. Maximum about 15 lines unless I ask for more.
- Use simple, everyday words. I am a beginner. No jargon without a 3-word meaning.
- Do not narrate what you are about to do or what you did step by step.
- Do not paste full files or long logs back to me. Show only the lines that matter.
- After finishing a step, reply in exactly this shape:

  **Done:** one sentence on what was built.
  **Files:** the file names only.
  **Check:** a small table of expected vs actual numbers.
  **Problem:** only if something failed or needs my decision. Otherwise leave it out.

- If you need a decision from me, ask one clear question and stop.
- PROGRESS.md entries: one line, under 200 characters.

# Standing rules for this project

- The playbook is AQI_CODING_PLAYBOOK.md. Do ONLY the step I name. Never start another step.
- AQI_Reliability_Layer_Architecture_v2.md is the source of truth. If a step conflicts with it, stop and tell me.
- Code in src/, tests in tests/, settings in config.yaml. No file paths inside .py files.
- Never change anything in the raw data folder (Kolkata/).
- Anything learned from data (scalers, clusters, covariances) is fitted on the TRAIN split only.
  Thresholds are chosen on the CALIBRATION split only. The TEST split is used only from Part 7 on.
- Regime labels and scores use input features only, never the true PM2.5 target.
- A GATE step means: show me the numbers, ask, and wait for my answer.
- After each step, add one line to PROGRESS.md.
- Freeze rule: from Part 7 on, never change a setting in config.yaml because of a test result.
- When a check needs two parts of the calibration split, use alternating weeks, not two halves by time.

# Coding guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## 5. Explanation type

**Defines the way explanation must be.**


- Always use the skill i-have-adhd and asd-ste100 for explanation

---

