# Post-Synthesis Quality Fixes — Phase 2 Checklist

## Fix 2: Strip inline LLM-generated source lists
- [x] Add regex pattern to match inline `*Sources: [N] ...` paragraphs in `strip_llm_sources_section()`
- [x] Also match `Sources:` (non-italic) and `---` + `*Sources:` patterns
- [x] Strip matched paragraph to next `\n\n` or end of string
- [x] Clean up orphaned `---` horizontal rule preceding stripped paragraph
- [x] Add test: report with only inline `*Sources: [1] [Title](url), ...*`
- [x] Add test: report with both `## Sources` heading and inline sources paragraph
- [x] Add test: report with `---` before inline sources (both stripped)
- [x] Add test: report with no sources (no-op)
- [x] Verify existing `strip_llm_sources_section` tests still pass

## Fix 1: Preserve heading/body boundaries in correction application
- [x] Add helper `_repair_heading_boundaries(original_window, corrected_text)` in `claim_verification.py`
- [x] Regex: detect `^(#{1,6}\s+[^\n]*?[a-z0-9])([A-Z][a-z])` on same line (heading fused with body)
- [x] Also detect heading line not followed by `\n` or `\n\n`
- [x] Insert `\n\n` between heading and body text
- [x] Call helper in `_correct_single_claim()` after line 1070, before replacement
- [x] Add test: correction that concatenates heading and body → repaired
- [x] Add test: correction that preserves heading/body boundary → no change
- [x] Add test: multiple headings in single context window
- [x] Add test: heading text legitimately modified by correction

## Fix 5: Recalibrate fidelity_score
- [x] Update `fidelity_score` property in `ClaimVerificationResult` to apply `-0.5` weight for contradicted claims
- [x] Add `max(0.0, ...)` floor
- [x] Update existing test expectations for fidelity_score
- [x] Add test: contradicted claims reduce score below what unsupported would
- [x] Add test: score floors at 0.0 with many contradictions

## Fix 4: Strengthen synthesis prompt to reduce citation misattribution
- [x] In `_build_synthesis_tail()`, add 1-sentence topic summary per source in the source reference list
- [x] Derive summary from source title + first ~15 words of content (no LLM call)
- [x] Respect token budget — truncate summaries if total exceeds allocation headroom
- [x] Add explicit instruction: "verify the specific fact appears in that source's content before citing"
- [x] Add test: source summaries appear in synthesis prompt
- [x] Add test: summaries are truncated when token budget is tight

## Fix 3: Citation remapping for UNSUPPORTED claims
- [x] Add `citations_remapped: int = 0` field to `ClaimVerificationResult`
- [x] Add `remap_unsupported_citations()` function in `claim_verification.py`
- [x] For each UNSUPPORTED claim with `cited_sources`: search source contents for better match
- [x] Use LLM-based matching: send claim + candidate sources, ask which supports it
- [x] Batch claims per source to reduce LLM calls
- [x] Replace `[old_citation]` → `[new_citation]` within the claim's quote_context region only
- [x] If no source supports the claim, remove the citation (leave fact uncited)
- [x] Wire into `workflow_execution.py` after claim verification block
- [x] Add test: unsupported claim gets remapped to correct source
- [x] Add test: unsupported claim with no matching source → citation removed
- [x] Add test: partially_supported claim with citation mismatch → remapped
- [x] Add test: remapping stats tracked in verification result
- [x] Add token budget cap for remapping LLM calls
