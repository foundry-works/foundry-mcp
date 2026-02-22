# LLM Ergonomics Fixes — Checklist

## Fix 1: `update_metadata` → `custom_metadata` alias
- [x] Add 2-line remap in `handlers_mutation.py` before `validate_payload()`
- [x] Add test: alias is remapped when `custom_metadata` absent
- [x] Add test: `custom_metadata` takes precedence when both present
- [x] Add test: normal `custom_metadata` usage unchanged
- [x] Run: `pytest tests/unit/test_core/test_spec_plan_linkage.py -v` (sanity)

## Fix 2: Always persist spec review to disk
- [x] Remove `plan_content` and `plan_enhanced` from persistence gate (line 221)
- [x] Make review type label dynamic (plan-enhanced vs standalone)
- [x] Make footer label dynamic
- [x] Add test: standalone review persisted to `.spec-reviews/`
- [x] Add test: plan-enhanced review still persisted
- [x] Add test: dry-run review NOT persisted
- [x] Add test: failed review NOT persisted
- [x] Add test: standalone markdown has correct labels

## Fix 3: Write `spec_review_path` to spec metadata
- [x] Add `update_frontmatter` import in `review.py`
- [x] Call `update_frontmatter()` after review file write
- [x] Store relative path (`.spec-reviews/{spec_id}-spec-review.md`)
- [x] Log warning on failure, don't break review
- [x] Add test: `update_frontmatter` called with correct key/value
- [x] Add test: frontmatter failure doesn't break review response
- [x] Add test: stored path is relative, not absolute

## Fix 4: Flexible `plan_path` resolution (in progress)
- [x] Implement `_resolve_plan_file()` in `templates.py`
- [x] Handle absolute paths
- [x] Handle `specs/`-prefixed paths
- [x] Handle canonical relative paths
- [x] Add test: absolute path resolves and normalizes
- [x] Add test: `specs/`-prefixed path strips and resolves
- [x] Add test: canonical relative path unchanged
- [x] Add test: absolute path outside specs_dir resolves
- [x] Run full test suite

## Final
- [x] Run: `pytest tests/ --timeout=120`
- [x] Update version to 0.14.3
- [x] Update CHANGELOG.md
- [ ] Commit, push, release
