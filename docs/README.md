# Documentation

This directory is the end-user guide for foundry-mcp: how to install it, how
the workflow is organized, and how to use the CLI and MCP tools.

## Start Here

| Doc | Description |
|-----|-------------|
| [Quick Start](01-quick-start.md) | Install and run your first spec |
| [Core Concepts](02-core-concepts.md) | Specs, phases, tasks, and hierarchy |
| [Workflow Guide](03-workflow-guide.md) | Day-to-day development workflow |

## Reference

| Doc | Description |
|-----|-------------|
| [CLI Command Reference](04-cli-command-reference.md) | All CLI commands with options and examples |
| [MCP Tool Reference](05-mcp-tool-reference.md) | All MCP tools with actions and parameters |
| [Configuration](06-configuration.md) | Environment variables and config files |
| [Troubleshooting](07-troubleshooting.md) | Common issues and solutions |
| [Error Codes](reference/error-codes.md) | All error codes with remediation |

## Concepts

| Doc | Description |
|-----|-------------|
| [Foundry Philosophy](concepts/foundry-philosophy.md) | Why spec-driven development matters |
| [Spec Schema](concepts/spec-schema.md) | JSON structure of specification files |
| [Response Envelope](concepts/response-envelope.md) | Standard response format for all tools |
| [Deep Research Workflow](concepts/deep_research_workflow.md) | Multi-phase research workflow |

## Guides

| Doc | Description |
|-----|-------------|
| [Intake Guide](guides/intake.md) | Fast-capture notes system |
| [LLM Configuration](guides/llm-configuration.md) | Provider setup and troubleshooting |
| [Autonomy Supervisor Runbook](guides/autonomy-supervisor-runbook.md) | Preflight, loop, and escalation operations for autonomous sessions |

## Examples

| Doc | Description |
|-----|-------------|
| [First Run Example](examples/first-run.md) | Minimal CLI walkthrough |
| [Deep Research Examples](examples/deep-research/README.md) | Sample sessions and reports |

---

## Find by Task

| Task | See |
|------|-----|
| Create a new spec | [CLI: specs create](04-cli-command-reference.md#specs-create), [MCP: authoring](05-mcp-tool-reference.md#authoring) |
| Find next task | [CLI: tasks next](04-cli-command-reference.md#tasks-next), [MCP: task](05-mcp-tool-reference.md#task) |
| Complete a task | [CLI: tasks complete](04-cli-command-reference.md#tasks-complete), [MCP: task](05-mcp-tool-reference.md#task) |
| Validate a spec | [CLI: validate](04-cli-command-reference.md#validate), [MCP: spec](05-mcp-tool-reference.md#spec) |
| Run a review | [CLI: review spec](04-cli-command-reference.md#review-spec), [MCP: review](05-mcp-tool-reference.md#review) |
| Configure LLM | [LLM Configuration](guides/llm-configuration.md), [Troubleshooting: LLM](07-troubleshooting.md#llmai-issues) |
| Fix errors | [Error Codes](reference/error-codes.md), [Troubleshooting](07-troubleshooting.md) |

---

## CLI to MCP Quick Reference

| CLI Command | MCP Tool | Action |
|-------------|----------|--------|
| `specs find` | `spec` | `list` |
| `specs create` | `authoring` | `spec-create` |
| `tasks next` | `task` | `next` |
| `tasks complete` | `task` | `complete` |
| `tasks block` | `task` | `block` |
| `lifecycle activate` | `lifecycle` | `activate` |
| `lifecycle complete` | `lifecycle` | `complete` |
| `review spec` | `review` | `spec` |
| `review fidelity` | `review` | `fidelity` |
| `validate check` | `spec` | `validate` |
| `validate fix` | `spec` | `fix` |
| `journal add` | `journal` | `add` |
| `plan create` | `plan` | `create` |
| `plan review` | `plan` | `review` |

---

## Directory Structure

```
docs/
├── README.md                    # This file
├── 01-quick-start.md           # Getting started
├── 02-core-concepts.md         # Core concepts
├── 03-workflow-guide.md        # Workflow guide
├── 04-cli-command-reference.md # CLI reference
├── 05-mcp-tool-reference.md    # MCP tool reference
├── 06-configuration.md         # Configuration
├── 07-troubleshooting.md       # Troubleshooting
│
├── concepts/
│   ├── foundry-philosophy.md   # Foundry philosophy
│   ├── spec-schema.md          # Spec JSON structure
│   ├── response-envelope.md    # Response format
│   └── deep_research_workflow.md
│
├── guides/
│   ├── intake.md               # Intake/notes guide
│   ├── llm-configuration.md    # LLM setup
│   └── autonomy-supervisor-runbook.md  # Autonomous loop operations
│
├── examples/
│   ├── first-run.md            # First run walkthrough
│   └── deep-research/          # Research examples
│
└── reference/
    └── error-codes.md          # Error codes reference
```
