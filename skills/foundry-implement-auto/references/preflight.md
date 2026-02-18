# Preflight Sequence

All checks must pass before session start. Any failure produces a structured error (`FoundryImplementAutoError`) with error code and remediation hint. **Never prompt the user — emit the error and EXIT immediately.**

## Contents

- [Step 0: Extract Spec ID](#step-0-extract-spec-id)
- [Step 1: Resolve Spec](#step-1-resolve-spec)
- [Step 2: Detect Action Shape](#step-2-detect-action-shape)
- [Step 3: Verify Feature Flags](#step-3-verify-feature-flags)
- [Step 4: Verify Posture Profile](#step-4-verify-posture-profile)
- [Step 5: Verify Role Authorization](#step-5-verify-role-authorization)
- [Step 6: Separation Principle](#step-6-separation-principle)
- [Environment Prerequisites](#environment-prerequisites)

## Step 0: Extract Spec ID

After skill expansion, the spec_id appears as text in the user's message. To extract it:

1. **Look in the user message** for any text following `/foundry-implement-auto` or any `<command-name>` tag. The spec_id is the first whitespace-delimited token after the skill name.
2. **Scan the full conversation turn** for a string matching a spec ID pattern (hyphenated slug, e.g., `hello-world-python-2026-02-18-001` or `hello-world-python-2026-02-18-001.json`).
3. Strip any `.json` suffix if present.

If no spec_id is found after checking all locations, emit error and EXIT:

```
FoundryImplementAutoError {
  error_code: "SPEC_ID_MISSING",
  message: "spec_id argument is required",
  remediation: "Invoke with: claude -p /foundry-implement-auto <spec-id>"
}
```

**Never list specs, offer choices, or use AskUserQuestion.** This skill is fully unattended.

Fail: `SPEC_ID_MISSING`

## Step 1: Resolve Spec

```
spec(action="find", spec_id=...)
```

Fail: `SPEC_RESOLUTION_FAILED`

## Step 2: Detect Action Shape

Probe canonical shape first, then fall back to legacy:

| Probe | Call | Non-terminal errors |
|-------|------|---------------------|
| Canonical | `task(action="session", command="list", limit=1)` | `AUTHORIZATION`, `FEATURE_DISABLED` — shape accepted but gated |
| Legacy | `task(action="session-list", limit=1)` | Same |

Non-terminal errors do not fail the probe — they indicate the shape is accepted but gated by policy.

Fail: `ACTION_SHAPE_UNSUPPORTED`

## Step 3: Verify Feature Flags

```
server(action="capabilities") → data.runtime.autonomy.enabled_now
```

Required flags:
- `autonomy_sessions == true`
- `autonomy_fidelity_gates == true`

Fail: `FEATURE_DISABLED`

## Step 4: Verify Posture Profile

Read `data.runtime.autonomy.posture_profile` from capabilities response.

| Profile | Verdict | Reason |
|---------|---------|--------|
| `unattended` | Accept (preferred) | Full gate enforcement |
| `supervised` | Accept | Human-in-the-loop provides equivalent oversight |
| `debug` | Reject | Disables gate enforcement, incompatible with unattended operation |

Fail: `POSTURE_UNSUPPORTED`

## Step 5: Verify Role Authorization

Using the detected action shape, call session list with `limit=1`.

If the response returns `AUTHORIZATION` or `FEATURE_DISABLED`, the `autonomy_runner` role either isn't set or doesn't have session access.

Fail: `AUTHORIZATION` or `ROLE_PREFLIGHT_FAILED`

## Step 6: Separation Principle

Action-shape detection (step 2) and role verification (step 5) are separate probes. Shape detection determines which action format to use. Role verification confirms the caller has permission. They must not be conflated.

## Environment Prerequisites

The server must be configured for unattended autonomous operation before invocation:

```sh
# Required: role and posture
FOUNDRY_MCP_ROLE=autonomy_runner
FOUNDRY_MCP_AUTONOMY_POSTURE=unattended

# Required: feature flags
FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_SESSIONS=true
FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_FIDELITY_GATES=true
```

The `unattended` posture applies server-side defaults automatically. The skill verifies (never sets) these:

| Setting | Unattended default | Purpose |
|---|---|---|
| `allow_lock_bypass` | `false` | Write lock cannot be bypassed |
| `allow_gate_waiver` | `false` | Gates cannot be waived |
| `enforce_required_phase_gates` | `true` | Phase transitions blocked without gate |
| `gate_policy` | `strict` | Gate failures block progression |
| `stop_on_phase_completion` | `true` | One-phase primitive enforced |
| `auto_retry_fidelity_gate` | `true` | Automatic feedback-gate retry cycle |
| `max_consecutive_errors` | `3` | Pause after 3 consecutive step failures |
| `max_fidelity_review_cycles_per_phase` | `3` | Prevent infinite gate retry spinning |

Equivalent TOML (alternative to env vars):

```toml
[feature_flags]
autonomy_sessions = true
autonomy_fidelity_gates = true

[autonomy_posture]
profile = "unattended"
```
