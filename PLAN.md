# PLAN: Prompt Conciseness — Trust the Model More

## Context

Comparison of foundry-mcp's deep research prompts against `open_deep_research` (ODR) found several prompts that are over-prescriptive relative to what the model needs. In each case below, either the code already enforces the behavior (making verbose prompt instructions redundant) or the JSON schema already constrains the output (making prose restatements wasteful). These trims reduce token cost per research session without changing behavior.

---

## Phase 1 — Researcher Reflection Protocol

**Problem:** The researcher system prompt (`topic_research.py:189-200`) devotes 12 lines and ~200 words to a "CRITICAL: Reflection Protocol" section with 3 numbered rules, correct/incorrect pattern examples, and a first-turn exception. ODR says the same thing in one sentence.

**Why we're confident:** The code at `topic_research.py:581-615` already **enforces** this behavior — if the researcher skips `think` after a search, the loop detects it, injects a synthetic reflection prompt, and forces a retry. The verbose prompt instructions are belt-and-suspenders where the belt (code) is load-bearing.

**Changes:**

1. **Collapse the Reflection Protocol section.** Replace the 12-line `## CRITICAL: Reflection Protocol` block (lines 189-200) and the duplicate instruction in the `think` tool description (line 182) with a single clear statement in the `think` tool description:

   Before:
   ```
   ### think
   ...
   CRITICAL: You MUST call think after every web_search or extract_content before doing anything else.

   ## CRITICAL: Reflection Protocol

   After EVERY `web_search` or `extract_content` call, you MUST call `think` as your very next action before issuing another search or extraction.

   Rules:
   1. After any `web_search` or `extract_content`, your next response MUST contain ONLY a `think` call.
   2. Do NOT call `think` in the same turn as `web_search` or `extract_content`.
   3. Exception: On your FIRST turn only, you may issue multiple `web_search` calls for initial broad coverage.

   Correct pattern: search → think → search → think → ... → research_complete
   Incorrect: search → search (missing reflection between searches)
   Incorrect: search + think in same turn (think must be a separate turn)
   ```

   After:
   ```
   ### think
   ...
   After each web_search or extract_content, call think as your next action before issuing another search. On your first turn only, you may issue multiple web_search calls for initial broad coverage.
   ```

2. **Remove the duplicate in Research Strategy.** Line 219 (`You MUST use think after every search to assess your findings before deciding next steps.`) is now redundant — delete it.

3. **Simplify the Response Format note.** Line 214 (`Generally include one tool call per turn. On your first turn, you may include multiple web_search calls for initial broad coverage.`) is now covered by the think tool description. Shorten to: `Generally include one tool call per turn.`

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — prompt string only

**Net reduction:** ~160 words, ~10 lines removed from a prompt sent per-topic per-round.

---

## Phase 2 — First-Round Delegation Schema Duplication

**Problem:** The first-round delegation prompt (`supervision.py:1789-1799`) has a "Quality Guidelines" section that restates what the JSON schema already defines. The schema says `"research_topic": "Detailed paragraph-length description..."` and the guidelines repeat: `Each directive's "research_topic" MUST be a detailed paragraph (2-4 sentences) specifying: The specific topic or facet to investigate, The research approach, What the researcher should focus on and what to exclude`. Same for `perspective`, `evidence_needed`, and `priority`.

**Why we're confident:** The JSON schema in the same prompt (lines 1769-1780) already communicates the structure. Restating field descriptions in prose is pure duplication — the model reads the schema.

**Changes:**

1. **Remove the Quality Guidelines section** (lines 1789-1799). The entire block:
   ```
   Quality Guidelines:
   - Each directive's "research_topic" MUST be a detailed paragraph (2-4 sentences) specifying:
     - The specific topic or facet to investigate
     - The research approach (compare, investigate, validate, survey, etc.)
     - What the researcher should focus on and what to exclude
   - Each directive should be SPECIFIC enough to yield targeted search results
   - Directives must cover DISTINCT aspects — no two should investigate substantially the same ground
   - "perspective" should specify the angle: ...
   - "evidence_needed" should name concrete evidence types: ...
   - "priority": 1=critical ..., 2=important ..., 3=nice-to-have ...
   ```

2. **Fold the two non-redundant points into Decomposition Guidelines.** Two guidelines carry information NOT in the schema:
   - "Each directive should be SPECIFIC enough to yield targeted search results"
   - "Directives must cover DISTINCT aspects — no two should investigate substantially the same ground"

   Add these as bullets in the existing Decomposition Guidelines section.

3. **Enrich the schema's inline descriptions** slightly to absorb the priority definitions:
   ```json
   "priority": 1  // 1=critical, 2=important, 3=supplementary
   ```

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — first-round delegation prompt only

**Net reduction:** ~100 words, 11 lines removed.

---

## Phase 3 — Clarification Example Reduction

**Problem:** The clarification prompt (`clarification.py:281-289`) includes 6 examples (3 vague, 3 specific) totaling ~120 words. ODR uses zero examples — it relies on the JSON schema and a brief instruction. The examples are pedagogical but the structured output schema (`need_clarification: bool`) already constrains the decision space.

**Why we're confident:** Clarification is a binary classification task. One example pair is enough to calibrate the threshold. The model understands "too broad" vs "specific enough" without needing 3 illustrations of each.

**Changes:**

1. **Reduce from 6 examples to 2** (one vague, one specific). Keep the most illustrative pair:
   ```
   Example — needs clarification: "What's the best database?" → Missing context (use case, scale, budget)
   Example — does NOT need clarification: "Compare PostgreSQL vs MySQL for high-write OLTP workloads in 2024"
   ```

2. **Remove the remaining 4 examples** (lines 282-284 partial, 288-289).

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/clarification.py` — prompt string only

**Net reduction:** ~80 words, 4 lines removed.

---

## Phase 4 — Brief Generation Meta-Commentary

**Problem:** The brief generation prompt (`brief.py:229-230`) ends with: `"Do not include meta-commentary, greetings, or formatting markers in any field."` This instruction addresses a problem that doesn't occur with structured JSON output — models responding with `{"research_brief": "..."}` don't add greetings or meta-commentary inside the JSON value.

**Why we're confident:** The prompt already says `"Output your response as a JSON object"` and `"IMPORTANT: Return ONLY valid JSON"`. The meta-commentary instruction is defensive against a failure mode that structured output prevents by design.

**Changes:**

1. **Remove the meta-commentary sentence.** Delete: `"Do not include meta-commentary, greetings, or formatting markers in any field.\n\n"`

2. **Merge the remaining field description into the schema block.** The line `"The research_brief field should contain the complete brief as a well-structured paragraph (or two)."` can move into the schema as an inline comment, removing one paragraph break.

   Before:
   ```
   Output your response as a JSON object with this schema:
   {
     "research_brief": "The detailed, structured research brief paragraph(s)",
     "scope_boundaries": "What the research should include and exclude" or null,
     "source_preferences": "Preferred source types" or null
   }

   The research_brief field should contain the complete brief as a
   well-structured paragraph (or two). Do not include meta-commentary,
   greetings, or formatting markers in any field.

   IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text.
   ```

   After:
   ```
   Output your response as a JSON object with this schema:
   {
     "research_brief": "Complete brief as one or two well-structured paragraphs",
     "scope_boundaries": "What the research should include and exclude" or null,
     "source_preferences": "Preferred source types" or null
   }

   IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text.
   ```

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` — prompt string only

**Net reduction:** ~30 words, 3 lines removed.
