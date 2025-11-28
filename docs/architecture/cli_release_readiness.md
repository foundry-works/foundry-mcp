# CLI Release Readiness Plan

**Status:** Draft
**Date:** 2025-11-28
**Spec:** sdd-cli-native-parity-2025-11-27-001

## Overview

This document defines the release readiness criteria and feature flag controls for the native SDD CLI implementation in foundry-mcp.

## Feature Flag Controls

### CLI Discovery Flag

Controls visibility of the native CLI in tool discovery and capabilities manifest.

```python
# foundry_mcp/core/feature_flags.py
FeatureFlag(
    name="native_cli",
    description="Enable native SDD CLI implementation",
    state=FlagState.EXPERIMENTAL,
    default_enabled=False,
    metadata={
        "category": "cli",
        "owner": "foundry-mcp",
        "documentation": "docs/architecture/sdd_cli_runtime.md"
    }
)
```

**Lifecycle Stages:**

| Stage | State | Default | Behavior |
|-------|-------|---------|----------|
| Development | `EXPERIMENTAL` | Off | CLI available only with explicit opt-in |
| Alpha | `BETA` | Off | CLI available, opt-in encouraged for testing |
| GA | `STABLE` | On | CLI enabled by default, opt-out available |
| Post-GA | - | On | Flag removed, CLI always available |

### CLI Command Group Flags

Granular flags for individual command groups during rollout:

```python
# Command group flags
CLI_FLAGS = {
    "cli_spec_commands": FeatureFlag(
        name="cli_spec_commands",
        description="Enable spec management commands (create, validate, fix)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_task_commands": FeatureFlag(
        name="cli_task_commands",
        description="Enable task management commands (next, prepare, update)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_journal_commands": FeatureFlag(
        name="cli_journal_commands",
        description="Enable journal commands (add, list)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_lifecycle_commands": FeatureFlag(
        name="cli_lifecycle_commands",
        description="Enable lifecycle commands (activate, complete, archive)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_doc_commands": FeatureFlag(
        name="cli_doc_commands",
        description="Enable documentation commands (render, doc)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_test_commands": FeatureFlag(
        name="cli_test_commands",
        description="Enable test commands (run, discover)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_review_commands": FeatureFlag(
        name="cli_review_commands",
        description="Enable review commands (review, fidelity-review)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
}
```

### Capabilities Manifest Switches

The capabilities manifest advertises available CLI features:

```python
# foundry_mcp/core/capabilities.py
def get_cli_capabilities() -> dict:
    """Return CLI capabilities based on feature flags."""
    registry = get_flag_registry()

    return {
        "cli": {
            "enabled": registry.is_enabled("native_cli"),
            "version": "0.1.0",
            "command_groups": {
                "spec": registry.is_enabled("cli_spec_commands"),
                "task": registry.is_enabled("cli_task_commands"),
                "journal": registry.is_enabled("cli_journal_commands"),
                "lifecycle": registry.is_enabled("cli_lifecycle_commands"),
                "doc": registry.is_enabled("cli_doc_commands"),
                "test": registry.is_enabled("cli_test_commands"),
                "review": registry.is_enabled("cli_review_commands"),
            },
            "features": {
                "json_output": True,
                "color_output": registry.is_enabled("cli_color_output"),
                "progress_bars": registry.is_enabled("cli_progress_bars"),
            }
        }
    }
```

### Configuration Override

Users can override flags via environment or config file:

```bash
# Environment variable override
export FOUNDRY_CLI_ENABLED=true
export FOUNDRY_CLI_SPEC_COMMANDS=true

# Or via config file (.foundry/config.json)
{
    "feature_flags": {
        "native_cli": true,
        "cli_spec_commands": true,
        "cli_task_commands": true
    }
}
```

### CLI Entry Point Guard

The CLI entry point checks the feature flag:

```python
# foundry_mcp/cli/main.py
from foundry_mcp.core.feature_flags import get_flag_registry

@click.group()
@click.pass_context
def cli(ctx):
    """SDD CLI - Spec-Driven Development tools."""
    registry = get_flag_registry()

    if not registry.is_enabled("native_cli"):
        if not ctx.obj.get('force'):
            click.echo(
                "Native CLI is not yet enabled. "
                "Set FOUNDRY_CLI_ENABLED=true to opt-in.",
                err=True
            )
            ctx.exit(1)
```

## Rollout Strategy

### Phase 1: Internal Testing (Week 1-2)

- **Flag State:** `EXPERIMENTAL`
- **Default:** Off
- **Target:** Foundry-MCP developers only
- **Criteria:**
  - [ ] Core commands implemented (spec, task, journal)
  - [ ] Unit tests passing (>80% coverage)
  - [ ] No known critical bugs

### Phase 2: Alpha (Week 3-4)

- **Flag State:** `BETA`
- **Default:** Off
- **Target:** Opt-in early adopters
- **Criteria:**
  - [ ] All Tier 1 commands implemented
  - [ ] Integration tests passing
  - [ ] Documentation complete
  - [ ] Performance benchmarks met

### Phase 3: Beta (Week 5-6)

- **Flag State:** `BETA`
- **Default:** Off (moving to On)
- **Target:** Broader community testing
- **Criteria:**
  - [ ] All Tier 1 + Tier 2 commands implemented
  - [ ] No regressions from claude_skills CLI
  - [ ] User feedback incorporated
  - [ ] Migration guide published

### Phase 4: General Availability

- **Flag State:** `STABLE`
- **Default:** On
- **Target:** All users
- **Criteria:**
  - [ ] All release checklist items complete
  - [ ] No blocking issues for 2 weeks
  - [ ] Performance parity with claude_skills
  - [ ] Deprecation notice for claude_skills.cli.sdd

## Metrics and Monitoring

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Command success rate | >99% | Telemetry |
| P95 latency | <500ms | Telemetry |
| Error rate | <1% | Telemetry |
| User adoption | >50% of MCP users | Analytics |

### Rollback Triggers

Automatic rollback if:
- Error rate exceeds 5%
- P99 latency exceeds 2s
- Critical security vulnerability discovered
- Data corruption detected

### Telemetry Events

```python
# CLI telemetry events
TELEMETRY_EVENTS = [
    "cli.command.invoked",
    "cli.command.succeeded",
    "cli.command.failed",
    "cli.feature_flag.checked",
    "cli.rollback.triggered",
]
```

## Release Checklist

### Pre-Release Enablement Checklist

#### Code Quality
- [ ] All Tier 1 (Critical) commands implemented and tested
- [ ] Unit test coverage ≥80% for CLI modules
- [ ] Integration tests passing for all command groups
- [ ] No critical or high-severity bugs open
- [ ] Static analysis (mypy, ruff) passing
- [ ] Security audit completed for input handling

#### Documentation
- [ ] CLI command reference generated (`sdd --help` output)
- [ ] Migration guide from claude_skills.cli.sdd published
- [ ] CHANGELOG updated with CLI additions
- [ ] README updated with CLI installation instructions
- [ ] Architecture ADR reviewed and approved

#### Performance
- [ ] Benchmark results meet targets (P95 < 500ms)
- [ ] Memory usage within acceptable limits
- [ ] No memory leaks detected in long-running sessions
- [ ] Startup time < 200ms

#### Compatibility
- [ ] Python 3.10+ compatibility verified
- [ ] macOS, Linux tested
- [ ] Windows compatibility verified (or documented limitations)
- [ ] Works with existing MCP server (no conflicts)

### Communication Plan

#### Internal Communication
| Milestone | Audience | Channel | Content |
|-----------|----------|---------|---------|
| Alpha Start | Dev team | Slack #foundry-dev | Feature flag instructions, feedback form |
| Beta Start | All engineers | Email + Slack | Opt-in instructions, known limitations |
| GA Announce | All users | Blog + Email | Feature highlights, migration guide |
| Deprecation | claude_skills users | Email + Docs | Timeline, migration steps |

#### External Communication
- [ ] Blog post drafted for GA announcement
- [ ] Twitter/social media posts scheduled
- [ ] Community Discord announcement prepared
- [ ] Documentation site updated

### Readiness Signals

#### Go/No-Go Criteria

**GO Signals (all must be true):**
- [ ] All Pre-Release Checklist items complete
- [ ] Error rate < 1% in beta testing
- [ ] No critical bugs for 7 consecutive days
- [ ] Positive feedback from ≥5 beta testers
- [ ] Performance targets met
- [ ] Security review signed off

**NO-GO Signals (any blocks release):**
- [ ] Critical bug discovered in core functionality
- [ ] Security vulnerability identified
- [ ] Performance regression > 20%
- [ ] Data corruption or loss reported
- [ ] Blocking dependency issue

#### Readiness Review Meeting

Before each phase transition, conduct readiness review:

1. **Participants:** Tech Lead, QA, DevOps, Product
2. **Agenda:**
   - Review checklist completion
   - Discuss open issues
   - Review metrics from previous phase
   - Vote on proceeding to next phase
3. **Outcomes:**
   - Document decision and rationale
   - Assign owners for any blocking items
   - Set date for next review

### Post-Release Checklist

#### Immediate (Day 1)
- [ ] Monitor error rates and latency
- [ ] Check support channels for issues
- [ ] Verify telemetry is flowing
- [ ] Confirm feature flags working correctly

#### Short-term (Week 1)
- [ ] Review user feedback
- [ ] Address critical bugs
- [ ] Publish FAQ based on common questions
- [ ] Update documentation based on feedback

#### Medium-term (Month 1)
- [ ] Analyze adoption metrics
- [ ] Plan Tier 2 command implementations
- [ ] Gather enhancement requests
- [ ] Schedule claude_skills deprecation timeline

### Rollback Procedure

If rollback is needed:

1. **Immediate Actions:**
   ```bash
   # Disable CLI feature flag globally
   export FOUNDRY_CLI_ENABLED=false

   # Or update config
   foundry config set feature_flags.native_cli false
   ```

2. **Communication:**
   - Post to #foundry-status: "CLI temporarily disabled due to [issue]"
   - Email affected users with workaround (use claude_skills.cli.sdd)

3. **Investigation:**
   - Gather logs and telemetry
   - Identify root cause
   - Document in incident report

4. **Recovery:**
   - Fix issue in hotfix branch
   - Test thoroughly
   - Gradual re-enablement starting at 10%

### Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| 0.1.0-alpha | TBD | Planned | Internal testing |
| 0.1.0-beta | TBD | Planned | Opt-in beta |
| 0.1.0 | TBD | Planned | General availability |

## Related Documents

- [CLI Runtime Architecture](sdd_cli_runtime.md) - Package structure and design
- [CLI Parity Matrix](../cli_parity_matrix.md) - Command coverage analysis
- [Feature Flags Guide](../mcp_best_practices/14-feature-flags.md) - Flag patterns
