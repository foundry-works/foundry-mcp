# Plan 8 Checklist: Flatten Review Types

## Part A — Plan review tool

### A1. Constants — plan unified tool
- [ ] `tools/unified/plan.py` lines 46-52: Delete `REVIEW_TYPES` list and `REVIEW_TYPE_TO_TEMPLATE` dict; add `_PLAN_REVIEW_TEMPLATE_ID = "MARKDOWN_PLAN_REVIEW_FULL_V1"`

### A1. Constants — plan CLI
- [ ] `cli/commands/plan.py` lines 31-39: Delete `REVIEW_TYPES` list and `REVIEW_TYPE_TO_TEMPLATE` dict; add `_PLAN_REVIEW_TEMPLATE_ID = "MARKDOWN_PLAN_REVIEW_FULL_V1"`

### A2. Remove `review_type` from `perform_plan_review()`
- [ ] `tools/unified/plan.py` line 211: Remove `review_type: str = "full"` from signature
- [ ] `tools/unified/plan.py` lines 221-231: Delete `if review_type not in REVIEW_TYPES` validation block
- [ ] `tools/unified/plan.py` line 311: Remove `"review_type": review_type` from dry-run response
- [ ] `tools/unified/plan.py` line 328: `REVIEW_TYPE_TO_TEMPLATE[review_type]` → `_PLAN_REVIEW_TEMPLATE_ID`
- [ ] `tools/unified/plan.py` line 414: File naming `f"...{review_type}-{response.provider_id}.md"` → `f"...{response.provider_id}.md"`
- [ ] `tools/unified/plan.py` line 519: File naming `f"...{review_type}.md"` → `f"...-review.md"`
- [ ] `tools/unified/plan.py` line 538: Remove `"review_type": review_type` from metric labels
- [ ] `tools/unified/plan.py` line 544: Remove `"review_type": review_type` from response data

### A3. MCP handler + registration — plan tool
- [ ] `tools/unified/plan.py` `_handle_plan_review`: Remove `review_type=payload.get(...)` from `perform_plan_review()` call
- [ ] `tools/unified/plan.py` tool registration: Remove `review_type: str = "full"` from signature
- [ ] `tools/unified/plan.py` tool registration: Remove `"review_type": review_type` from payload dict

### A4. CLI — plan review command
- [ ] `cli/commands/plan.py` lines 135-141: Delete `@click.option("--type", "review_type", ...)` decorator
- [ ] `cli/commands/plan.py` line 169: Remove `review_type: Optional[str]` from function signature
- [ ] `cli/commands/plan.py` lines 188-192: Delete `if review_type is None` config default lookup block
- [ ] `cli/commands/plan.py` line 244: Remove `"review_type": review_type` from dry-run response
- [ ] `cli/commands/plan.py` line 265: `REVIEW_TYPE_TO_TEMPLATE[review_type]` → `_PLAN_REVIEW_TEMPLATE_ID`
- [ ] `cli/commands/plan.py` line 356: File naming `f"...{review_type}.md"` → `f"...-review.md"`
- [ ] `cli/commands/plan.py` line 373: Remove `"review_type": review_type` from response

---

## Part B — Spec review tool

### B1. Constants — review helpers
- [ ] `review_helpers.py` line 40: Delete `REVIEW_TYPES` list
- [ ] `review_helpers.py` lines 43-48: Replace `_REVIEW_TYPE_TO_TEMPLATE` dict with two constants: `_SPEC_REVIEW_TEMPLATE_ID` and `_SPEC_VS_PLAN_REVIEW_TEMPLATE_ID`

### B2. Simplify `_run_ai_review()`
- [ ] `review_helpers.py` line 120: Remove `review_type: str` from `_run_ai_review()` signature
- [ ] `review_helpers.py` lines 130-146: Replace effective_review_type logic + dict lookup + error block with simple `template_id = ... if plan_content else ...`
- [ ] `review_helpers.py` line 330: Replace `if effective_review_type == "spec-vs-plan"` with `if plan_content`

### B3. Spec review handler
- [ ] `review.py` lines 62-63: Remove `REVIEW_TYPES` import from review_helpers
- [ ] `review.py` lines 99-103: Remove `review_type` resolution from consultation config
- [ ] `review.py` lines 115-123: Delete `if review_type not in REVIEW_TYPES` validation block
- [ ] `review.py` lines 180-187: Delete `if review_type == "quick"` dispatch to `_run_quick_review()`
- [ ] `review.py` lines 224-231: Remove `if review_type == "full"` condition — always load plan content
- [ ] `review.py` line 236: Remove `review_type=review_type` from `_run_ai_review()` call

### B4. Remove `_run_quick_review`
- [ ] `review.py` line 66: Remove `_run_quick_review` from imports
- [ ] `review_helpers.py` lines 67-113: Delete `_run_quick_review()` function entirely

### B5. Simplify list-tools responses
- [ ] `review.py` `_handle_list_tools`: Remove `review_types=REVIEW_TYPES` from response
- [ ] `review.py` `_handle_list_plan_tools`: Simplify to list a single review tool instead of four

### B6. MCP registration + CLI
- [ ] `review.py` tool registration: Remove `review_type: Optional[str] = None` from signature
- [ ] `review.py` tool registration: Remove `"review_type": review_type` from payload dict
- [ ] `cli/commands/review.py`: Remove `REVIEW_TYPES` import from review_helpers
- [ ] `cli/commands/review.py` lines 122-128: Delete `@click.option("--type", "review_type", ...)` decorator
- [ ] `cli/commands/review.py`: Remove `review_type` from function signature and forwarding

---

## Part C — Config cleanup

- [ ] `core/llm_config/consultation.py` line 51: Remove `default_review_type: str = "full"` field
- [ ] `core/llm_config/consultation.py` line 54: Delete `VALID_REVIEW_TYPES` class constant
- [ ] `core/llm_config/consultation.py` lines 68-72: Delete `default_review_type` validation block
- [ ] `core/llm_config/consultation.py` `from_dict()`: Stop reading `default_review_type`

---

## Part D — Prompt template cleanup (optional)

### Markdown plan review prompts
- [ ] `core/prompts/markdown_plan_review.py`: Remove `MARKDOWN_PLAN_REVIEW_QUICK_V1`
- [ ] `core/prompts/markdown_plan_review.py`: Remove `MARKDOWN_PLAN_REVIEW_SECURITY_V1`
- [ ] `core/prompts/markdown_plan_review.py`: Remove `MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1`
- [ ] `core/prompts/markdown_plan_review.py`: Remove those entries from `MARKDOWN_PLAN_REVIEW_TEMPLATES` registry dict

### Spec review prompts
- [ ] `core/prompts/plan_review.py`: Remove `PLAN_REVIEW_QUICK_V1`
- [ ] `core/prompts/plan_review.py`: Remove `PLAN_REVIEW_SECURITY_V1`
- [ ] `core/prompts/plan_review.py`: Remove `PLAN_REVIEW_FEASIBILITY_V1`
- [ ] `core/prompts/plan_review.py`: Remove those entries from `PLAN_REVIEW_TEMPLATES` registry dict

---

## Part E — Documentation

- [ ] `docs/04-cli-command-reference.md`: Remove `--type` option from `plan review` section
- [ ] `docs/04-cli-command-reference.md`: Remove `--type` option from `review spec` section
- [ ] `docs/04-cli-command-reference.md`: Update examples that use `--type security` etc.
- [ ] `docs/05-mcp-tool-reference.md`: Remove `review_type` from tool parameter listings (if present)

---

## Verification

- [ ] `plan review` (MCP) works without `review_type` parameter
- [ ] `plan review` (MCP) silently ignores `review_type` if passed
- [ ] `review spec` (MCP) works without `review_type` parameter
- [ ] `review spec` (MCP) silently ignores `review_type` if passed
- [ ] `foundry plan review ./PLAN.md` (CLI) works without `--type` flag
- [ ] `foundry review spec my-spec` (CLI) works without `--type` flag
- [ ] Spec-vs-plan auto-enhancement still triggers when plan content is available
- [ ] No remaining references to `REVIEW_TYPES` (the list constant) in `src/` (except `core/review.py` QUICK_REVIEW_TYPES if retained)
- [ ] No remaining references to `"quick"`, `"security"`, `"feasibility"` as review type choices in tool/CLI code
- [ ] Run test suite — no regressions
