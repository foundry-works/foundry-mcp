# ruff-pyright-ci

**Mission**: Introduce Ruff linting/formatting and Pyright type-checking to CI in a phased, low-churn rollout.

## Objective

Add consistent Ruff + Pyright tooling to local dev and CI, starting with advisory checks and progressively promoting to blocking gates as the codebase is cleaned up.

## Scope

### In Scope
- Ruff configuration in pyproject.toml (conservative ruleset: F, E, W, I, B)
- Dev dependency group for ruff + pyright
- GitHub Actions lint workflow (.github/workflows/lint.yml)
- Bulk autofix + format cleanup
- Pyright error ratchet mechanism
- .git-blame-ignore-revs for bulk-format commits

### Out of Scope
- Fixing all historical lint/type issues in the same PR as CI wiring
- Enabling high-churn rule families (UP, SIM, N, RUF, T20) — deferred
- Pre-commit hooks (future consideration after Phase 2 stable)

## Phases

### Phase 1: Tooling + Advisory CI (PR A)

**Purpose**: Wire up Ruff and Pyright config, add a CI workflow that runs checks in advisory mode (continue-on-error). No source code formatting changes.

**Tasks**:
1. **Add Ruff config to pyproject.toml** — Conservative ruleset (F, E, W, I, B), line-length=120, target-version py310. Ignore E501 (handled by formatter). Add isort config with known-first-party. Add per-file-ignores section.
   - File: `pyproject.toml`
   - Category: implementation
2. **Add dev dependency group** — Add `[project.optional-dependencies] dev` with ruff>=0.9.0,<0.10, pyright>=1.1.390, and foundry-mcp[test].
   - File: `pyproject.toml`
   - Category: implementation
3. **Create lint CI workflow (advisory)** — GitHub Actions workflow on push/PR to main+beta. Single Python 3.11 runner. Steps: ruff format --check, ruff check --statistics, pyright — all with continue-on-error: true.
   - File: `.github/workflows/lint.yml`
   - Category: implementation
4. **Verify Phase 1** — CI runs on pushes/PRs and produces Ruff/Pyright output. Workflow stays green. No source formatting churn.
   - Category: verification (fidelity)

**Verification**: CI workflow runs successfully (green) with advisory output visible in logs.

### Phase 2: Ruff Cleanup + Blocking Ruff (PR B)

**Purpose**: Auto-fix existing Ruff issues in a dedicated cleanup commit, then promote Ruff lint + format checks to blocking in CI.

**Tasks**:
1. **Bulk autofix + format** — Run `ruff check --fix`, then `ruff format`, then verify `ruff check` passes. Commit as single commit. Add commit SHA to .git-blame-ignore-revs. Resolve remaining high-signal items (F821, B904) manually.
   - Files: `src/**/*.py`, `tests/**/*.py`, `.git-blame-ignore-revs`
   - Category: implementation
2. **Make Ruff blocking in CI** — Remove continue-on-error from both Ruff format and Ruff lint steps in lint.yml. Keep Pyright advisory. Add quality job as required status check on main and beta branch protection.
   - File: `.github/workflows/lint.yml`
   - Category: implementation

**Verification**: Ruff lint + format pass cleanly. CI fails on Ruff violations. Pyright remains advisory.

### Phase 3: Pyright Ratchet + Blocking Pyright (PR C+)

**Purpose**: Incrementally reduce Pyright errors using a ratchet mechanism, then promote to blocking when error count reaches zero.

**Tasks**:
1. **Implement Pyright ratchet** — Create .pyright-threshold file with current error count (553). Update lint.yml Pyright step to compare actual errors against threshold, failing if count increases. Keep as advisory (continue-on-error: true).
   - Files: `.pyright-threshold`, `.github/workflows/lint.yml`
   - Category: implementation
2. **Make Pyright blocking** — When error count reaches zero: remove continue-on-error, remove ratchet wrapper, delete .pyright-threshold. Verify quality job already required in branch protection.
   - File: `.github/workflows/lint.yml`
   - Category: implementation

**Verification**: Pyright passes cleanly. CI fails on type errors. Ratchet prevents regression during incremental cleanup.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Bulk autofix introduces bugs | Medium | Single dedicated commit, review carefully, run tests |
| Ruff version update changes rules | Medium | Pin to >=0.9.0,<0.10 (minor range) |
| Pyright errors block unrelated PRs | High | Advisory-first, ratchet only prevents increases |
| git blame polluted by format commit | Low | .git-blame-ignore-revs excludes bulk commits |

## Key Decisions
- line-length = 120 minimizes disruption
- Advisory-first CI avoids blocking merges during baseline discovery
- Separate config/wiring from cleanup keeps reviews focused
- Pyright ratchet threshold in .pyright-threshold (not CI config)

## Success Criteria

- [ ] Phase 1: Advisory quality CI lands and runs stably
- [ ] Phase 2: Ruff checks blocking and passing on main/beta
- [ ] Phase 3: Pyright blocking and passing on main/beta
