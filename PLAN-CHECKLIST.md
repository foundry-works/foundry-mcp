# PLAN-CHECKLIST: Deep Research Alignment with open_deep_research

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23

---

## Phase 1: Update Model Token Limits

- [x] **1.1** Update `model_token_limits.json` with accurate upstream values
  - GPT-4.1 variants: 1,000,000 → 1,047,576
  - Add gpt-4.1-nano (1,047,576), gpt-4o (128,000), gpt-4o-mini (128,000)
  - Add gemini-2.5-pro (1,048,576), gemini-2.5-flash (1,048,576)
  - Add gemini-1.5-pro (2,097,152)
- [x] **1.2** Update `_lifecycle.py` inline `MODEL_TOKEN_LIMITS` fallback dict to match json
- [x] **1.3** Verify `truncate_to_token_estimate` loads from json, not just hardcoded dict
- [x] **1.4** Run existing token truncation tests — confirm no regressions (66/66 passed)

---

## Phase 2: Research Brief Generation

- [ ] **2.1** Add `_build_brief_refinement_prompt()` to `planning.py`
  - Original prompt (not copied), modeled after upstream principles
  - Maximize specificity, fill unstated dimensions as open-ended
  - Prefer primary sources, preserve language preference
- [ ] **2.2** Add brief-refinement LLM call at start of `_execute_planning_async()`
  - Execute before sub-query decomposition
  - Use cheap model role (summarization or new "brief" role)
- [ ] **2.3** Store refined brief in `state.research_brief`
- [ ] **2.4** Wire refined brief as input to `_build_planning_user_prompt()`
- [ ] **2.5** Add "brief" to `_ROLE_RESOLUTION_CHAIN` in `research.py` if using new role
- [ ] **2.6** Test: ambiguous query produces more specific brief
- [ ] **2.7** Test: sub-queries are grounded in refined brief
- [ ] **2.8** Test: existing planning tests pass (additive change)

---

## Phase 3: Synthesis Prompt Engineering

- [ ] **3.1** Add language detection directive to `_build_synthesis_system_prompt()`
  - Detect language from user query, instruct report in same language
- [ ] **3.2** Add structure-adaptive directives
  - Comparison → side-by-side structure
  - Enumeration → single-section list
  - Explanation → overview + concept sections + conclusion
  - How-to → step-by-step with prerequisites
- [ ] **3.3** Add anti-pattern guardrails
  - No meta-commentary ("based on the research", "the findings show")
  - No hedging openers ("it appears that", "it seems")
  - No self-reference ("as an AI", "I found that")
- [ ] **3.4** Enforce citation format: inline `[Title](URL)` + numbered source section
- [ ] **3.5** Add query-type hint to `_build_synthesis_user_prompt()`
- [ ] **3.6** Test: non-English query produces non-English report
- [ ] **3.7** Test: comparison query produces comparison structure
- [ ] **3.8** Test: output is free of anti-pattern phrases
- [ ] **3.9** Test: citations are consistently formatted

---

## Phase 4: Message-Aware Token Recovery

- [ ] **4.1** Add `_structured_truncation()` helper to `_helpers.py`
  - Parse prompt into source/finding blocks
  - Identify block boundaries (section headers, source markers)
  - Truncate longest blocks first, preserving all block headers
- [ ] **4.2** Add quality-aware source dropping
  - If quality scores available, drop lowest-quality sources first
  - Otherwise drop longest sources first (heuristic: length ≠ quality)
- [ ] **4.3** Update retry loop in `execute_llm_call()` (lines 288-351)
  - Retry 1: structured truncation (longest blocks first)
  - Retry 2: source dropping (lowest quality or longest)
  - Retry 3: current char-based truncation (fallback)
- [ ] **4.4** Test: structured truncation preserves high-quality sources
- [ ] **4.5** Test: char-based fallback still works when structured fails
- [ ] **4.6** Test: no regression in existing token recovery tests

---

## Phase 5: Reflection Enforcement

- [ ] **5.1** Update reflection system prompt in `_topic_reflect()` (lines 395-417)
  - Hard stop: research_complete=true when 3+ sources from distinct domains
  - Continue: only if <2 relevant sources found
  - Rationale must articulate specific gap to be filled
- [ ] **5.2** Add guard: force research_complete if iteration >= max_searches
  - Regardless of reflection LLM decision
- [ ] **5.3** Add logging of reflection rationale for observability
- [ ] **5.4** Test: reflection returns research_complete after 3+ distinct-domain sources
- [ ] **5.5** Test: reflection never recommends continuing past max_searches
- [ ] **5.6** Test: rationale field is always non-empty

---

## Phase 6: Iterative Supervisor Architecture (Deferred)

_Not for immediate implementation. Evaluate after Phases 1-5 land._

- [ ] **6.1** Design supervision phase between gathering/compression and analysis
- [ ] **6.2** Implement coverage gap assessment (topics with insufficient sources)
- [ ] **6.3** Implement targeted sub-query generation for gaps
- [ ] **6.4** Add iteration budget management (tokens, API calls)
- [ ] **6.5** Integrate with phase lifecycle in `_lifecycle.py`
- [ ] **6.6** Test: supervisor identifies and fills coverage gaps
- [ ] **6.7** Test: iteration limit is respected
- [ ] **6.8** Test: budget tracking prevents runaway costs

---

## Sign-off

- [ ] All phases reviewed and approved
- [ ] Tests pass: `pytest tests/core/research/ -x`
- [ ] No regressions in existing deep research tests
- [ ] Code review completed
