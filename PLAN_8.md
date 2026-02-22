# Plan 8: Flatten Review Types

**Decision:** Remove the quick/full/security/feasibility review type selection; always use the comprehensive (full) review
**Dependencies:** None — independent of Plans 5-7, but runs cleanly after Plan 7
**Risk:** Medium — touches more files than Plan 7, spans two tools (plan + review) and prompt registries

---

## Rationale

The `plan review` and `review spec` actions currently offer four review types (`quick`, `full`, `security`, `feasibility`) via a `review_type` parameter. In practice, `full` is always the default, always the right choice, and the other types are never explicitly requested. The selection mechanism adds parameter surface, validation code, CLI options, prompt template duplication, and config validation for zero user value. Remove it.

### What survives

- **Plan review:** The `MARKDOWN_PLAN_REVIEW_FULL_V1` prompt template becomes the single prompt
- **Spec review (AI):** The `PLAN_REVIEW_FULL_V1` prompt template becomes the single prompt, with the `spec-vs-plan` auto-enhancement still triggering when plan content is available
- **Spec review (quick):** The structural `quick_review()` code path is removed — `spec validate` already provides structural checks
- **Synthesis:** `SYNTHESIS_PROMPT_V1` and `FIDELITY_SYNTHESIS_PROMPT_V1` are unaffected (they are not review-type-specific)
- **Fidelity review:** Unaffected — it never used `review_type`

---

## Scope

### Part A — Plan review tool (`plan.py`)

#### A1. Remove `REVIEW_TYPES` list and `REVIEW_TYPE_TO_TEMPLATE` dict

**File: `src/foundry_mcp/tools/unified/plan.py`**
- Lines 46-52: Delete `REVIEW_TYPES` and `REVIEW_TYPE_TO_TEMPLATE`. Replace with a single constant:
  ```python
  _PLAN_REVIEW_TEMPLATE_ID = "MARKDOWN_PLAN_REVIEW_FULL_V1"
  ```

**File: `src/foundry_mcp/cli/commands/plan.py`**
- Lines 31-39: Same replacement — delete both, add single constant

#### A2. Remove `review_type` parameter from `perform_plan_review()`

**File: `src/foundry_mcp/tools/unified/plan.py`**
- Line 211: Remove `review_type: str = "full"` from signature
- Lines 221-231: Delete the `if review_type not in REVIEW_TYPES` validation block
- Line 311: Remove `"review_type": review_type` from dry-run response
- Line 328: Change `template_id = REVIEW_TYPE_TO_TEMPLATE[review_type]` → `template_id = _PLAN_REVIEW_TEMPLATE_ID`
- Line 414: Change file naming `f"{plan_name}-{review_type}-{response.provider_id}.md"` → `f"{plan_name}-{response.provider_id}.md"`
- Line 519: Change `f"{plan_name}-{review_type}.md"` → `f"{plan_name}-review.md"`
- Line 538: Remove `"review_type": review_type` from metric labels
- Line 544: Remove `"review_type": review_type` from response data

#### A3. Remove `review_type` from plan MCP handler and registration

**File: `src/foundry_mcp/tools/unified/plan.py`**
- Line 748 (`_handle_plan_review`): Remove `review_type=payload.get("review_type", "full")` from `perform_plan_review()` call
- Line 796 (tool registration): Remove `review_type: str = "full"` from signature
- Line 807: Remove `"review_type": review_type` from payload dict

#### A4. Remove `--type` from plan review CLI

**File: `src/foundry_mcp/cli/commands/plan.py`**
- Lines 135-141: Delete the `@click.option("--type", "review_type", ...)` decorator
- Line 169: Remove `review_type: Optional[str]` from `plan_review_cmd()` signature
- Lines 188-192: Delete the `if review_type is None: ...` config lookup block
- Line 244: Remove `"review_type": review_type` from dry-run response
- Line 265: Change `template_id = REVIEW_TYPE_TO_TEMPLATE[review_type]` → `template_id = _PLAN_REVIEW_TEMPLATE_ID`
- Line 356: Change `f"{plan_name}-{review_type}.md"` → `f"{plan_name}-review.md"`
- Line 373: Remove `"review_type": review_type` from response

---

### Part B — Spec review tool (`review.py` + `review_helpers.py`)

#### B1. Remove `REVIEW_TYPES` and `_REVIEW_TYPE_TO_TEMPLATE` from review helpers

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- Line 40: Delete `REVIEW_TYPES` list
- Lines 43-48: Replace `_REVIEW_TYPE_TO_TEMPLATE` dict with two constants:
  ```python
  _SPEC_REVIEW_TEMPLATE_ID = "PLAN_REVIEW_FULL_V1"
  _SPEC_VS_PLAN_REVIEW_TEMPLATE_ID = "SPEC_REVIEW_VS_PLAN_V1"
  ```

#### B2. Simplify `_run_ai_review()` — remove `review_type` parameter

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- Line 120: Remove `review_type: str` from `_run_ai_review()` signature
- Lines 130-136: Replace the effective_review_type + dict lookup with:
  ```python
  template_id = _SPEC_VS_PLAN_REVIEW_TEMPLATE_ID if plan_content else _SPEC_REVIEW_TEMPLATE_ID
  ```
- Lines 137-146: Delete the `if template_id is None` error block (no longer possible)
- Line 330: Replace `if effective_review_type == "spec-vs-plan"` with `if plan_content`

#### B3. Remove `review_type` from spec review handler

**File: `src/foundry_mcp/tools/unified/review.py`**
- Lines 62-63: Remove `REVIEW_TYPES` import from review_helpers
- Lines 99-103: Remove `review_type` resolution from config
- Lines 115-123: Delete the `if review_type not in REVIEW_TYPES` validation block
- Lines 180-187: Delete the `if review_type == "quick"` dispatch to `_run_quick_review()`
- Lines 224-231: Remove the `if review_type == "full"` condition — always try to load plan content
- Line 236: Remove `review_type=review_type` from `_run_ai_review()` call

#### B4. Remove `_run_quick_review` import and function

**File: `src/foundry_mcp/tools/unified/review.py`**
- Line 66: Remove `_run_quick_review` from imports

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- Lines 67-113: Delete `_run_quick_review()` function entirely

#### B5. Simplify `_handle_list_tools` and `_handle_list_plan_tools`

**File: `src/foundry_mcp/tools/unified/review.py`**
- Line 322: Remove `review_types=REVIEW_TYPES` from `_handle_list_tools` response
- Lines 347-376: Simplify `_handle_list_plan_tools` to list a single review tool instead of four

#### B6. Remove `review_type` from review MCP registration and CLI

**File: `src/foundry_mcp/tools/unified/review.py`**
- Line 1411: Remove `review_type: Optional[str] = None` from tool registration signature
- Line 1435: Remove `"review_type": review_type` from payload dict

**File: `src/foundry_mcp/cli/commands/review.py`**
- Line 42: Remove `REVIEW_TYPES` import from review_helpers
- Lines 122-128: Delete the `@click.option("--type", "review_type", ...)` decorator
- Remove `review_type` from `review_spec_cmd()` function signature and its forwarding
- Line 254: Remove `"review_types": REVIEW_TYPES` from list-tools response (if present in CLI)

---

### Part C — Config cleanup

**File: `src/foundry_mcp/core/llm_config/consultation.py`**
- Line 51: Remove `default_review_type: str = "full"` from `WorkflowConsultationConfig`
- Line 54: Delete `VALID_REVIEW_TYPES` class constant
- Lines 68-72: Delete the `if self.default_review_type not in self.VALID_REVIEW_TYPES` validation block
- Update `from_dict()` to stop reading `default_review_type` from config dicts

---

### Part D — Prompt template cleanup (optional, low-risk)

These prompt files retain dead templates after removing the review_type dispatch. Cleaning them up is optional but keeps the codebase tidy.

**File: `src/foundry_mcp/core/prompts/markdown_plan_review.py`**
- Remove `MARKDOWN_PLAN_REVIEW_QUICK_V1`, `MARKDOWN_PLAN_REVIEW_SECURITY_V1`, `MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1`
- Remove those entries from `MARKDOWN_PLAN_REVIEW_TEMPLATES` registry dict
- Keep `MARKDOWN_PLAN_REVIEW_FULL_V1`

**File: `src/foundry_mcp/core/prompts/plan_review.py`**
- Remove `PLAN_REVIEW_QUICK_V1`, `PLAN_REVIEW_SECURITY_V1`, `PLAN_REVIEW_FEASIBILITY_V1`
- Remove those entries from `PLAN_REVIEW_TEMPLATES` registry dict
- Keep `PLAN_REVIEW_FULL_V1` and `SYNTHESIS_PROMPT_V1`

**File: `src/foundry_mcp/core/prompts/spec_review.py`**
- No changes — `SPEC_REVIEW_VS_PLAN_V1` is still used

---

### Part E — Documentation

**File: `docs/04-cli-command-reference.md`**
- Remove `--type` option row from `plan review` parameters table
- Remove `--type` option row from `review spec` parameters table
- Update examples that use `--type security` etc.

**File: `docs/05-mcp-tool-reference.md`**
- Remove `review_type` from `plan` tool parameters (if listed)
- Remove `review_type` from `review` tool parameters (if listed)

---

## Design Notes

- **Backward compatibility for MCP tool parameters.** If a caller passes `review_type="full"` or `review_type="security"`, the parameter is simply ignored (removed from handlers). No error — the single review is always used.
- **Backward compatibility for CLI.** Removing `click.option` means `--type security` will error. Acceptable breaking change for a CLI beta.
- **The `spec-vs-plan` auto-enhancement is preserved.** It triggers based on plan content availability, not review_type. This behavior is unchanged.
- **The `quick` structural review is removed.** Users needing structural validation should use `spec validate`, which already provides this. The `_run_quick_review` code path and its `core.review.quick_review()` dependency become dead code.
- **Config's `default_review_type` becomes dead.** Any existing `foundry-mcp.toml` files with `default_review_type = "full"` in `[consultation.workflows.*]` sections will have that field silently ignored. No errors.
- **Parts A-E can be done in any order**, though doing A and B together keeps the `REVIEW_TYPES` removal atomic.
