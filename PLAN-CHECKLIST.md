# Fix Plan Checklist

## Phase 1 — Critical Fixes
- [x] 1.1 Restore `ANALYSIS`/`REFINEMENT` enum values as deprecated + add to `_SKIP_PHASES`
- [x] 1.2 Fix `deep_research_timeout` default in `from_toml_dict()` (600 → 2400)
- [x] 1.3 Validate compression output before clearing `message_history`

## Phase 2 — Security Fixes
- [ ] 2.1 Block IPv4-mapped IPv6 SSRF bypass in `_injection_protection.py` and `tavily.py`
- [ ] 2.2 Add SSRF validation to `ExtractContentTool._cap_urls`
- [ ] 2.3 Cap `WebSearchTool.queries` list length (max 10)

## Phase 3 — Major Correctness Fixes
- [ ] 3.1 Fix `cleanup_stale_tasks` classmethod → instance method
- [ ] 3.2 Subtract system prompt from supplementary notes headroom calculation
- [ ] 3.3 Break retry loop when truncation is a no-op
- [ ] 3.4 Fix compression hard truncation direction (tail → head)
- [ ] 3.5 Add context-window error handling to topic research LLM calls
