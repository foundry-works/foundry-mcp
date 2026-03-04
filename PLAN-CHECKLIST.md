# Plan Checklist: Claim Verification & Synthesis Fidelity Fixes

## Phase 1: Use raw_content for claim verification
- [x] Change `_resolve_source_text()` in `claim_verification.py` (~line 119) to prefer `raw_content`: `return source.raw_content or source.content or source.snippet`
- [x] Update docstring to say "Falls back through raw_content → content → snippet"
- [x] Add unit test: returns `raw_content` when both `content` and `raw_content` are present
- [x] Add unit test: falls back to `content` when `raw_content` is None/empty
- [x] Add unit test: falls back to `snippet` when both `raw_content` and `content` are None/empty
- [x] Add unit test: returns None when all three are None/empty
- [x] Run existing claim verification tests — confirm no regressions from the preference flip (139 passed)

## Phase 2: Compute and store fidelity_score
- [x] Add `fidelity_score` computed property to `ClaimVerificationResult` in `models/deep_research.py` (~after line 1356)
  - [x] Weight: SUPPORTED=1.0, PARTIALLY_SUPPORTED=0.5, UNSUPPORTED=0.0, CONTRADICTED=0.0
  - [x] Return `None` if `claims_verified == 0`
  - [x] Return `float` otherwise: `(claims_supported + 0.5 * claims_partially_supported) / claims_verified`
- [x] Verify `fidelity_score` appears in `model_dump()` output (Pydantic `@computed_field` auto-serializes)
- [x] Verify action_handlers.py deep-research-status includes fidelity_score in claim_verification data (added to both status and report serialization points)
- [x] Add test: all SUPPORTED → 1.0
- [x] Add test: all UNSUPPORTED → 0.0
- [x] Add test: all CONTRADICTED → 0.0
- [x] Add test: mixed (5 SUP, 1 PARTIAL, 29 UNSUP) → ≈0.157
- [x] Add test: 0 claims_verified → None
- [x] Add test: serializes correctly in JSON output (both model_dump and model_dump_json)

## Phase 3: Add citation-accuracy guardrails to synthesis prompt
- [ ] Add citation-accuracy lines to `## Citations` section in `_build_synthesis_system_prompt` (~after line 1267, before `## Language`)
  - [ ] Instruction: only cite a source for a fact if that fact appears in the source content provided
  - [ ] Instruction: do not guess citation numbers — omit the citation rather than cite the wrong source
  - [ ] Instruction: never attribute a fact from one source to a different source based on topical similarity
- [ ] Add prompt assertion test: system prompt contains the new citation-accuracy text

## Phase 4: Log claim verification source resolution for diagnostics
- [ ] Add `source_resolution: Optional[str] = None` field to `ClaimVerdict` in `models/deep_research.py`
  - [ ] Valid values: `"full_content"`, `"compressed_only"`, `"snippet_only"`, `"no_content"`, `"citation_not_found"`
- [ ] In `_build_verification_user_prompt` (~line 773): track best content tier resolved across all cited sources
  - [ ] If `raw_content` used → `"full_content"`
  - [ ] If only `content` (compressed) used → `"compressed_only"`
  - [ ] If only `snippet` used → `"snippet_only"`
  - [ ] If all citations miss the map → `"citation_not_found"`
  - [ ] Return the resolution tier alongside the prompt string (change return type to tuple or add to claim)
- [ ] In `_verify_single_claim`: set `claim.source_resolution` from the resolution tier
- [ ] When auto-assigning UNSUPPORTED (prompt is None), set `source_resolution = "no_content"`
- [ ] Add test: `source_resolution` is `"full_content"` when `raw_content` available
- [ ] Add test: `source_resolution` is `"compressed_only"` when only `content` available (after Phase 1 flip, this means `raw_content` was None)
- [ ] Add test: `source_resolution` is `"no_content"` when no source text resolves
- [ ] Add test: `source_resolution` is `"citation_not_found"` when citation number not in map
- [ ] Add test: field serializes in `ClaimVerdict.model_dump()`
