# PLAN CHECKLIST: Prompt Conciseness — Trust the Model More

## Phase 1 — Researcher Reflection Protocol
- [x] 1.1 Remove `## CRITICAL: Reflection Protocol` section (lines 189-200) from `_RESEARCHER_SYSTEM_PROMPT`
- [x] 1.2 Remove duplicate `CRITICAL:` line from `think` tool description (line 182)
- [x] 1.3 Add concise reflection instruction to `think` tool description: "After each web_search or extract_content, call think as your next action before issuing another search. On your first turn only, you may issue multiple web_search calls for initial broad coverage."
- [x] 1.4 Remove redundant line 219 from Research Strategy (`You MUST use think after every search...`)
- [x] 1.5 Shorten line 214 Response Format note to `Generally include one tool call per turn.`
- [x] 1.6 Verify code enforcement at lines 581-615 still functions as the hard backstop
- [x] 1.7 Test: researcher still alternates search → think → search pattern
- [x] 1.8 Test: first-turn parallel searches still allowed
- [x] 1.9 Test: synthetic reflection injection still fires when model skips think

## Phase 2 — First-Round Delegation Schema Duplication
- [x] 2.1 Remove `Quality Guidelines:` section (lines 1789-1799) from first-round delegation prompt
- [x] 2.2 Add "Directives should be SPECIFIC enough to yield targeted search results" to Decomposition Guidelines
- [x] 2.3 Add "Directives must cover DISTINCT aspects — no two should investigate substantially the same ground" to Decomposition Guidelines
- [x] 2.4 Add priority definitions as inline comment in JSON schema: `// 1=critical, 2=important, 3=supplementary`
- [x] 2.5 Test: first-round decomposition still produces well-structured directives with diverse topics

## Phase 3 — Clarification Example Reduction
- [x] 3.1 Remove 4 of 6 examples, keep: "What's the best database?" (vague) and "Compare PostgreSQL vs MySQL for high-write OLTP workloads in 2024" (specific)
- [x] 3.2 Consolidate remaining examples into compact format (one line each)
- [x] 3.3 Test: vague queries still trigger clarification
- [x] 3.4 Test: specific queries still skip clarification

## Phase 4 — Brief Generation Meta-Commentary
- [x] 4.1 Remove "Do not include meta-commentary, greetings, or formatting markers in any field" sentence
- [x] 4.2 Move "complete brief as one or two well-structured paragraphs" into JSON schema description
- [x] 4.3 Test: brief generation output unchanged (no meta-commentary appears)
