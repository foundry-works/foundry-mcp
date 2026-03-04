# Plan: Claim Verification & Synthesis Fidelity Fixes

**Branch:** alpha
**Scope:** 4 targeted fixes addressing claim verification failure and synthesis citation accuracy revealed by deep research session `deepres-d4612c6f0474`

## Context

Session analysis showed 29/35 claims UNSUPPORTED (14% support rate) despite all cited sources having content. Root cause: verification uses compressed summaries (~2KB) instead of raw page content (~15-100KB), and the synthesis prompt lacks citation-accuracy guardrails. Additionally, `fidelity_score` is never computed.

---

## Phase 1: Use raw_content for claim verification

**Problem:** `_resolve_source_text()` returns `source.content` (Haiku-compressed ~2KB summary) first. Since all cited sources have `content` populated, `raw_content` (the actual 15-100KB page text) is never used. The compressed summaries strip specific numbers, transfer ratios, and factual details — exactly what claims reference. The multi-window truncation then operates on a 2KB summary instead of 100KB of real content, producing excerpts that don't contain the facts being verified. The verification LLM correctly returns UNSUPPORTED because the evidence simply isn't in the excerpt.

**Approach:** Flip the preference order in `_resolve_source_text()` to prefer `raw_content` over `content`. The `_multi_window_truncate()` function already handles large inputs via keyword-based window selection with an 8,000 char budget per source — it's designed for exactly this use case but was starved of input by the wrong fallback order.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`
  - Change `_resolve_source_text()` (~line 119): `return source.raw_content or source.content or source.snippet`
  - Update docstring to reflect new preference order

**Risk assessment:** Low. The `_multi_window_truncate` function already handles large text via keyword windowing with a per-source 8,000 char cap (`VERIFICATION_SOURCE_MAX_CHARS`). Token budget estimation in `_apply_token_budget` uses `min(len(text), VERIFICATION_SOURCE_MAX_CHARS)` so budget estimates remain accurate regardless of input size. No downstream consumers depend on `_resolve_source_text` returning the compressed version.

**Tests:**
- Add unit test: `_resolve_source_text` returns `raw_content` when both `content` and `raw_content` are present
- Add unit test: `_resolve_source_text` falls back to `content` when `raw_content` is None
- Add unit test: `_resolve_source_text` falls back to `snippet` when both are None
- Update any existing tests that mock source content to verify the new preference order

---

## Phase 2: Compute and store fidelity_score

**Problem:** `ClaimVerificationResult` has no `fidelity_score` field. The session data shows `fidelity_score: None` on the state. While `_token_budget.py` has a `fidelity_level_from_score()` function, there's no code computing a claim-verification fidelity score from verdict distributions.

**Approach:** Add a `fidelity_score` property to `ClaimVerificationResult` that computes a weighted score from verdict counts:
- SUPPORTED = 1.0, PARTIALLY_SUPPORTED = 0.5, UNSUPPORTED = 0.0, CONTRADICTED = 0.0
- Score = weighted_sum / claims_verified (or None if claims_verified == 0)

Store it as a computed property rather than a persisted field — it's always derivable from the verdict counts and stays consistent if claims are re-verified.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`
  - Add `fidelity_score` as a `@computed_field` (Pydantic v2) or `@property` on `ClaimVerificationResult` (~after line 1356)
- `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py`
  - Include `fidelity_score` in the deep-research-status response where claim_verification is serialized (it should serialize automatically if using `computed_field`)

**Tests:**
- Test fidelity_score with all SUPPORTED → 1.0
- Test fidelity_score with all UNSUPPORTED → 0.0
- Test fidelity_score with mix: 5 SUPPORTED, 1 PARTIALLY_SUPPORTED, 29 UNSUPPORTED → (5*1.0 + 1*0.5) / 35 ≈ 0.157
- Test fidelity_score with 0 claims_verified → None
- Test that fidelity_score serializes in model_dump() / JSON output

---

## Phase 3: Add citation-accuracy guardrails to synthesis prompt

**Problem:** The synthesis prompt's Citations section instructs format (`[Title](URL) [N]` for first use, `[N]` for subsequent) but doesn't instruct accuracy. The LLM sometimes attributes facts to the wrong source — e.g., a claim about Virgin Atlantic's ANA award chart was cited to a NerdWallet article about Chase Sapphire Preferred vs Reserve. This happens because compressed findings lose source-level attribution precision, and the LLM fills in citation numbers based on topic proximity rather than actual provenance.

**Approach:** Add a citation-accuracy instruction to the `## Citations` section of the synthesis system prompt. This should emphasize:
1. Only cite a source for a specific fact if that fact actually appears in the source's content
2. Do not guess or infer which source a fact came from — if uncertain, omit the citation rather than cite the wrong source
3. Never attribute a fact from one source to a different source based on topical similarity

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`
  - Add 3-4 lines to the `## Citations` section (~after line 1267) with accuracy instructions

**Tests:**
- Add prompt assertion test: system prompt contains citation-accuracy instruction text (keyword check)

---

## Phase 4: Log claim verification source resolution for diagnostics

**Problem:** When claims are marked UNSUPPORTED due to source content gaps or citation map misses, the only evidence is warning-level log lines. There's no structured record in the session data showing why each claim got its verdict — specifically whether the verification LLM was given useful content or was auto-assigned UNSUPPORTED due to missing source text.

**Approach:** Enrich `ClaimVerdict` details to record the source resolution outcome:
1. Add `source_resolution` field to `ClaimVerdict`: a string enum like `"full_content"`, `"compressed_only"`, `"snippet_only"`, `"no_content"`, `"citation_not_found"`
2. Set this in `_build_verification_user_prompt` / `_verify_single_claim` based on what happened during source lookup

This makes it possible to diagnose verification quality from session data alone without needing log access.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`
  - Add `source_resolution: Optional[str] = None` to `ClaimVerdict` model
- `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`
  - In `_build_verification_user_prompt`: track which content tier was used for each source
  - In `_verify_single_claim`: set `source_resolution` on the returned claim based on the best content tier that was resolved
  - When auto-assigning UNSUPPORTED (no source text available), set `source_resolution = "no_content"`

**Tests:**
- Test that `source_resolution` is set to `"full_content"` when `raw_content` is available
- Test that `source_resolution` is set to `"no_content"` when all sources lack text
- Test that `source_resolution` is set to `"citation_not_found"` when citation number is not in citation_map
- Verify `source_resolution` appears in serialized claim details
