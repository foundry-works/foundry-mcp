# LLM Ergonomics Fixes — v0.14.3

## Context

A user tested foundry-mcp tools with an LLM caller and observed several stumbles where the LLM guessed plausible but wrong parameter names or paths. Additionally, spec review results were not always persisted to disk, and the spec metadata was never updated with the review path.

These fixes also include the `plan_path` resolution improvement already in progress (relative/absolute/prefixed path normalization for `spec-create`).

---

## Fix 1: Parameter alias `update_metadata` → `custom_metadata`

**Problem:** `task(action: "update-metadata", update_metadata: {...})` fails silently — the param is `custom_metadata`, but `metadata-batch` uses `update_metadata` for the same semantic purpose.

**File:** `src/foundry_mcp/tools/unified/task_handlers/handlers_mutation.py`

**Change:** Insert 2-line remap before `validate_payload()` at ~line 506:
```python
if "update_metadata" in payload and "custom_metadata" not in payload:
    payload["custom_metadata"] = payload.pop("update_metadata")
```

---

## Fix 2: Always persist spec review to disk

**Problem:** `review(action: "spec")` only saves to `.spec-reviews/` when plan content exists. Standalone reviews are ephemeral.

**File:** `src/foundry_mcp/tools/unified/review.py`

**Changes:**
- Line 221: Remove `plan_content` and `plan_enhanced` from persistence gate — persist when `result.get("success") and not dry_run and specs_dir`
- Line 235: Make review type label dynamic (plan-enhanced vs standalone)
- Line 254: Make footer dynamic

---

## Fix 3: Write `spec_review_path` to spec metadata

**Problem:** Even when review is persisted, the path is only in the response — never stored in spec metadata for future discovery.

**File:** `src/foundry_mcp/tools/unified/review.py`

**Changes:**
- Add `update_frontmatter` to imports (line 45)
- After writing review file (line 258), call `update_frontmatter(spec_id, "spec_review_path", rel_path, specs_dir)`
- Log warning on failure but don't break the review

---

## Fix 4: Flexible `plan_path` resolution in `spec-create` (already in progress)

**Problem:** `spec-create` with `plan_path: "specs/.plans/foo.md"` fails because it resolves to `<specs_dir>/specs/.plans/...`. Only absolute paths worked by accident.

**File:** `src/foundry_mcp/core/spec/templates.py`

**Changes:** (already implemented) Replace rigid `specs_dir_path / plan_path` with `_resolve_plan_file()` that handles:
- Absolute paths → use directly, relativize for storage
- Relative from specs_dir → use as-is (canonical)
- `specs/`-prefixed → strip redundant prefix

---

## Implementation Sequence

1. Fix 1 (standalone)
2. Fix 4 tests (already coded, need to verify)
3. Fix 2 (standalone)
4. Fix 3 (depends on Fix 2)
5. Full regression test run

## Key Files

| File | Fixes |
|------|-------|
| `src/foundry_mcp/tools/unified/task_handlers/handlers_mutation.py` | 1 |
| `src/foundry_mcp/tools/unified/review.py` | 2, 3 |
| `src/foundry_mcp/core/spec/templates.py` | 4 (done) |
| `tests/unit/test_core/test_spec_plan_linkage.py` | 4 (done) |
