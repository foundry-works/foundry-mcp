# Core Concepts

This page defines the foundational ideas used across foundry-mcp.

## Specs as source of truth

Specs are machine-readable JSON files that describe work before it is built.
They live under `specs/` and move through lifecycle folders:

```
specs/
  pending/
  active/
  completed/
  archived/
```

Specs contain phases, tasks, dependencies, and verification requirements. Tools
read the spec state to determine the next actionable step.

## Tasks, phases, and dependencies

- **Phases** group tasks by intent (planning, implementation, testing).
- **Tasks** are the smallest units of work the system tracks.
- **Dependencies** declare order and blocking relationships between tasks.

This structure allows progressive task discovery: tools can answer "what is
next?" based on current status and dependencies.

## Journals and decisions

Journals capture decisions, tradeoffs, and task outcomes. They are intended to
be queryable context for AI assistants and future maintainers.

## Response envelopes

All tools return a consistent envelope that includes `success`, `data`, `error`,
and `meta`. This keeps clients from guessing output shape and supports
pagination and warnings.

## Unified tools and actions

foundry-mcp uses a small set of unified tools. Each tool accepts an `action`
field that switches behavior (for example, `spec` with `action=list`).

The authoritative list of tools, parameters, and actions lives in:

- `mcp/capabilities_manifest.json`

## Workspace scope

Workspace roots and specs directories are configurable. This prevents
accidental access outside the intended project and keeps tooling deterministic.

See [Configuration](06-configuration.md) for the supported settings.

## Go deeper

- Foundry motivation and philosophy: [Foundry Philosophy](concepts/foundry-philosophy.md)
- Deep research architecture: [Deep Research Workflow](concepts/deep_research_workflow.md)
