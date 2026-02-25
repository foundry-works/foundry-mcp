# PLAN: Deep Research Prompt Alignment with open_deep_research

## Context

Comparison of foundry-mcp's deep research workflow against open_deep_research identified four prompt-level improvements that strengthen existing behavior without removing any foundry-mcp capabilities (multi-round supervision, token budgeting, content fidelity tracking, etc.).

All changes are surgical prompt text edits — no logic, models, or control flow changes.

## Phases

### Phase 1: Research brief — domain-specific source guidance

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`
**Method:** `_build_brief_system_prompt()` (line ~215)

**Current:** Generic source preference instruction:
> Specify source preferences: Bias toward primary and official sources (specifications, documentation, peer-reviewed work, government datasets, original research papers) over aggregators or secondary commentary.

**Change:** Add domain-specific examples after the generic instruction, matching ODR's `transform_messages_into_research_topic_prompt` (prompts.py:71-76):
- Product/travel research → official or primary websites, manufacturer pages, reputable e-commerce for user reviews — not aggregator sites or SEO-heavy blogs
- Academic/scientific queries → original paper or official journal publication — not survey papers or secondary summaries
- People research → LinkedIn profile or personal website
- Language-specific queries → prioritize sources published in that language

These examples are actionable guidance that flows through to researchers and improves source selection for common query types.

---

### Phase 2: Supervisor directive isolation + no-acronym instruction

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Change A — Directive isolation rationale** (line ~1842, `_build_first_round_delegation_user_prompt`):

Current:
> Each directive should be a detailed, self-contained research assignment for a specialized researcher.

Add rationale (matching ODR prompts.py:134):
> Each directive should be a detailed, self-contained research assignment for a specialized researcher — sub-agents cannot see other agents' work, so every directive must include full context.

**Change B — No-acronym instruction** (in the decomposition guidelines section, line ~1788):

Add after "Directives must cover DISTINCT aspects":
> Do NOT use acronyms or abbreviations in directive text — spell out all terms so researchers search for the correct concepts.

Matching ODR prompts.py:135. Prevents researchers from wasting budget disambiguating acronyms.

**Change C — Apply same changes to follow-up round prompt** (`_build_followup_delegation_system_prompt`, line ~1410 area):

The follow-up delegation prompt has its own directive guidelines. Apply the same no-acronym instruction there.

---

### Phase 3: Language preservation in synthesis and brief

**File A:** `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`
**Method:** `_build_synthesis_system_prompt()` (line ~606)

Add language matching instruction before the closing "IMPORTANT: Return ONLY the markdown report" line. Matching ODR's final_report_generation_prompt (prompts.py:237-239, 293-294):

```
## Language

CRITICAL: The report MUST be written in the same language as the original research query.
If the query is in English, write in English. If the query is in Chinese, write entirely in Chinese. If in Spanish, write entirely in Spanish.
The research and findings may be in English, but you must translate information to match the query language.
```

**File B:** `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`
**Method:** `_build_brief_system_prompt()` (line ~203)

Add to the brief system prompt, after the source preferences instruction:
> If the query is in a specific language, prioritize sources published in that language.

This ensures non-English queries get language-appropriate sources from the brief stage onward.

---

## Out of Scope

- Multi-query search tool (supervisor-as-orchestrator pattern already handles decomposition)
- Compression citation format (already aligned with ODR)
- System evaluation / native search features
- Logic or control flow changes
