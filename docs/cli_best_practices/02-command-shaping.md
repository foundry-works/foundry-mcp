# 2. Command Design & Naming

> Mirror canonical MCP tool names so CLI commands, specs, and envelopes stay aligned.

## 2.1 Namespace Strategy

- Group subcommands under the same domains listed in [docs/codebase_standards/naming-conventions.md](../codebase_standards/naming-conventions.md) (plan, next, update, validate, provider, test, spec-mod, plan-review/pr, context).
- The `sdd --help` output MUST list namespaces in the same order as the Recommended Mapping Matrix to reinforce the mental model from the docs.

## 2.2 Canonical Names

- Every CLI operation should refer to the canonical tool name in its help text (e.g., `sdd validate check` maps to MCP `spec(action="validate")`).
- When introducing a CLI command first, reserve the canonical tool name immediately so the MCP adapter can pick it up without another rename.
- Avoid legacy aliases unless absolutely necessary for migration; if an alias exists, document the retirement timeline inside the spec and remove it within two releases.

## 2.3 Help Text & Examples

- Follow the verb–noun style (`spec-create`, `task-update`, `plan-report`) in usage strings and examples.
- Show JSON-first examples whenever a command emits data that another tool might parse.
- Include references back to the relevant best-practice document (e.g., “See [CLI Runtime](./01-cli-runtime.md) §1.2 for verbosity guidance”) so engineers know where the rules live.

## 2.4 Flags & Options

- Prefer long-form flags (`--template`, `--phase`) and keep short aliases only for frequently repeated options (`-q`, `-v`).
- Reuse shared option groups (verbosity, JSON, workspace paths) rather than redefining them per command.
- When a flag toggles a feature behind a rollout, ensure the same feature flag is declared in `mcp/capabilities_manifest.json` and discovery metadata.
