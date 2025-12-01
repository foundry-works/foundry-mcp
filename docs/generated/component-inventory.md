# foundry-mcp - Component Inventory

**Date:** 2025-12-01

## Complete Directory Structure

```
foundry-mcp/
├── benchmark_results
├── docs
│   ├── architecture
│   │   └── adr-001-cli-architecture.md
│   ├── cli_best_practices
│   │   ├── 01-cli-runtime.md
│   │   ├── 02-command-shaping.md
│   │   ├── 03-shared-services.md
│   │   ├── 04-testing-parity.md
│   │   └── README.md
│   ├── codebase_standards
│   │   ├── cli-output.md
│   │   ├── mcp_response_schema.md
│   │   └── naming-conventions.md
│   ├── concepts
│   │   └── sdd-philosophy.md
│   ├── generated
│   │   ├── architecture.md
│   │   ├── codebase.json
│   │   ├── component-inventory.md
│   │   ├── doc-generation-state.json
│   │   ├── index.md
│   │   └── project-overview.md
│   ├── guides
│   │   ├── development-guide.md
│   │   ├── llm-configuration.md
│   │   └── testing.md
│   └── mcp_best_practices
│       ├── 01-versioned-contracts.md
│       ├── 02-envelopes-metadata.md
│       ├── 03-serialization-helpers.md
│       ├── 04-validation-input-hygiene.md
│       ├── 05-observability-telemetry.md
│       ├── 06-pagination-streaming.md
│       ├── 07-error-semantics.md
│       ├── 08-security-trust-boundaries.md
│       ├── 09-spec-driven-development.md
│       ├── 10-testing-fixtures.md
│       ├── 11-ai-llm-integration.md
│       ├── 12-timeout-resilience.md
│       ├── 13-tool-discovery.md
│       ├── 14-feature-flags.md
│       ├── 15-concurrency-patterns.md
│       └── README.md
├── mcp
│   └── capabilities_manifest.json
├── samples
│   └── foundry-mcp.toml
├── src
│   └── foundry_mcp
│       ├── cli
│       │   ├── commands
│       │   ├── __init__.py
│       │   ├── __main__.py
│       │   ├── agent.py
│       │   ├── config.py
│       │   ├── context.py
│       │   ├── flags.py
│       │   ├── logging.py
│       │   ├── main.py
│       │   ├── output.py
│       │   ├── registry.py
│       │   └── resilience.py
│       ├── core
│       │   ├── providers
│       │   ├── __init__.py
│       │   ├── cache.py
│       │   ├── capabilities.py
│       │   ├── concurrency.py
│       │   ├── discovery.py
│       │   ├── docs.py
│       │   ├── feature_flags.py
│       │   ├── journal.py
│       │   ├── lifecycle.py
│       │   ├── llm_config.py
│       │   ├── llm_patterns.py
│       │   ├── llm_provider.py
│       │   ├── modifications.py
│       │   ├── naming.py
│       │   ├── observability.py
│       │   ├── pagination.py
│       │   ├── progress.py
│       │   ├── rate_limit.py
│       │   ├── rendering.py
│       │   ├── resilience.py
│       │   ├── responses.py
│       │   ├── security.py
│       │   ├── spec.py
│       │   ├── task.py
│       │   ├── testing.py
│       │   └── validation.py
│       ├── prompts
│       │   ├── __init__.py
│       │   └── workflows.py
│       ├── resources
│       │   ├── __init__.py
│       │   └── specs.py
│       ├── schemas
│       │   ├── __init__.py
│       │   └── sdd-spec-schema.json
│       ├── tools
│       │   ├── __init__.py
│       │   ├── analysis.py
│       │   ├── authoring.py
│       │   ├── context.py
│       │   ├── discovery.py
│       │   ├── docs.py
│       │   ├── documentation.py
│       │   ├── environment.py
│       │   ├── git_integration.py
│       │   ├── journal.py
│       │   ├── lifecycle.py
│       │   ├── mutations.py
│       │   ├── planning.py
│       │   ├── pr_workflow.py
│       │   ├── providers.py
│       │   ├── queries.py
│       │   ├── rendering.py
│       │   ├── reporting.py
│       │   ├── review.py
│       │   ├── spec_helpers.py
│       │   ├── tasks.py
│       │   ├── testing.py
│       │   ├── utilities.py
│       │   └── validation.py
│       ├── __init__.py
│       ├── config.py
│       └── server.py
├── tests
│   ├── contract
│   │   ├── __init__.py
│   │   ├── response_schema.json
│   │   └── test_response_schema.py
│   ├── doc_query
│   │   ├── test_codebase_query.py
│   │   └── test_scope_command.py
│   ├── fixtures
│   │   ├── context_tracker
│   │   │   └── transcript.jsonl
│   │   └── golden
│   │       ├── README.md
│   │       ├── error_not_found.json
│   │       ├── error_validation_failure.json
│   │       ├── success_specs_list.json
│   │       ├── success_task_progress.json
│   │       ├── success_test_presets.json
│   │       └── success_validation.json
│   ├── integration
│   │   ├── test_authoring_tools.py
│   │   ├── test_environment_tools.py
│   │   ├── test_fallback_integration.py
│   │   ├── test_llm_docs.py
│   │   ├── test_llm_review.py
│   │   ├── test_llm_tools.py
│   │   ├── test_mcp_smoke.py
│   │   ├── test_mcp_tools.py
│   │   ├── test_notifications_sampling.py
│   │   ├── test_prepare_task_cli.py
│   │   ├── test_sdd_cli_advanced.py
│   │   ├── test_sdd_cli_parity.py
│   │   └── test_spec_helpers.py
│   ├── llm_doc_gen
│   │   ├── test_ab_testing.py
│   │   ├── test_component_generator.py
│   │   ├── test_freshness.py
│   │   ├── test_overview_generator.py
│   │   └── test_performance_benchmark.py
│   ├── parity
│   │   └── harness
│   ├── property
│   │   ├── __init__.py
│   │   └── test_input_validation.py
│   ├── sdd_next
│   │   ├── test_context_utils.py
│   │   └── test_prepare_task_context.py
│   ├── skills
│   │   └── llm_doc_gen
│   │       ├── __init__.py
│   │       ├── test_ai_consultation.py
│   │       ├── test_architecture_generator.py
│   │       ├── test_component_generator.py
│   │       ├── test_e2e_generators.py
│   │       ├── test_e2e_orchestration.py
│   │       ├── test_overview_generator.py
│   │       └── test_workflow_engine.py
│   ├── unit
│   │   ├── test_core
│   │   │   ├── __init__.py
│   │   │   ├── test_concurrency.py
│   │   │   ├── test_discovery.py
│   │   │   ├── test_feature_flags.py
│   │   │   ├── test_journal.py
│   │   │   ├── test_lifecycle.py
│   │   │   ├── test_llm_patterns.py
│   │   │   ├── test_pagination.py
│   │   │   ├── test_rendering.py
│   │   │   ├── test_resilience.py
│   │   │   ├── test_security.py
│   │   │   ├── test_spec.py
│   │   │   ├── test_task.py
│   │   │   └── test_validation.py
│   │   ├── __init__.py
│   │   ├── test_ai_config_fallback.py
│   │   ├── test_analysis.py
│   │   ├── test_authoring.py
│   │   ├── test_consultation_limits.py
│   │   ├── test_documentation.py
│   │   ├── test_environment.py
│   │   ├── test_execute_tool_fallback.py
│   │   ├── test_golden_fixtures.py
│   │   ├── test_llm_provider.py
│   │   ├── test_mutations.py
│   │   ├── test_planning.py
│   │   ├── test_reporting.py
│   │   ├── test_review.py
│   │   ├── test_sdd_cli_core.py
│   │   ├── test_sdd_cli_runtime.py
│   │   ├── test_spec_helpers.py
│   │   └── test_utilities.py
│   ├── verification
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cli_verbosity.py
│   ├── test_doc_query_advanced_verbosity.py
│   ├── test_doc_query_json_output.py
│   ├── test_doc_query_verbosity.py
│   ├── test_output_reduction.py
│   ├── test_responses.py
│   ├── test_sdd_fidelity_review_verbosity.py
│   ├── test_sdd_next_verbosity.py
│   ├── test_sdd_plan_review_verbosity.py
│   ├── test_sdd_plan_verbosity.py
│   ├── test_sdd_pr_verbosity.py
│   ├── test_sdd_render_verbosity.py
│   ├── test_sdd_spec_mod_verbosity.py
│   ├── test_sdd_update_tasks_verbosity.py
│   ├── test_sdd_update_verbosity.py
│   ├── test_sdd_validate_verbosity.py
│   ├── test_start_helper_contracts.py
│   ├── test_support_verbosity.py
│   └── test_verbosity_regression.py
├── AGENTS.md
├── CHANGELOG.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
└── pytest.ini
```

---

### 1. Source Tree Overview

This codebase for `foundry-mcp` is organized as a Python project centered around a core library that provides Spec-Driven Development (SDD) tooling and an MCP (Multi-Agent Communication Protocol) server. The primary organizational pattern is a hybrid approach, combining logical layers (e.g., `cli`, `core`) with modules grouped by concern or domain (`prompts`, `schemas`, `tools`). Notable characteristics include a clear separation between CLI components, core business logic, and a rich set of specialized tools, along with comprehensive documentation.

### 2. Critical Directories

| Directory Path | Purpose | Contents Summary | Entry Points | Integration Notes |
| :------------- | :------ | :--------------- | :----------- | :---------------- |
| `src/foundry_mcp/` | Contains all primary Python source code for the `foundry-mcp` package. | Subdirectories for CLI, core logic, prompts, resources, schemas, and tools. Also contains top-level `config.py` and `server.py`. | `__init__.py`, `config.py`, `server.py` (for `foundry-mcp` script), `cli/` (for `foundry-cli` script). | The root of the Python package, defining the overall structure and components. |
| `src/foundry_mcp/cli/` | Houses all code related to the command-line interface. | `commands/` (individual CLI commands), `agent.py`, `config.py`, `context.py`, `flags.py`, `logging.py`, `main.py` (main CLI entry), `output.py`, `registry.py`, `resilience.py`. | `main.py` (via `foundry_mcp.cli.main:cli`), `__main__.py` (for `python -m foundry_mcp.cli`). | Directly interacts with users via the terminal, orchestrates calls to `core` and `tools`. |
| `src/foundry_mcp/core/` | Contains the core business logic, foundational services, and shared utilities for the application. | Subdirectory `providers/`, along with modules for caching, capabilities, concurrency, discovery, documentation, feature flags, journaling, lifecycle management, LLM configuration/patterns/providers, modifications, naming, observability, pagination, progress, rate limiting, rendering, resilience, responses, security, spec management, task management, testing, and validation. | No direct CLI entry points, but modules expose functions/classes used by `cli` and `tools`. | Provides the backbone functionalities. Modules here are designed to be reusable and independent of the CLI or specific tools. |
| `src/foundry_mcp/tools/` | Contains specialized modules that implement specific functionalities or "tools" used by the core logic or agents. | Subdirectory `providers/`, along with modules for analysis, authoring, context, discovery, documentation, environment, git integration, journaling, lifecycle, mutations, planning, PR workflow, queries, rendering, reporting, review, spec helpers, tasks, testing, utilities, and validation. | No direct CLI entry points. Functions/classes are called by `core` components or potentially `cli` commands. | Extends core capabilities with specific domain-oriented operations, often acting on specs or codebase elements. |
| `src/foundry_mcp/schemas/` | Stores JSON schema definitions used for data validation, particularly for Spec-Driven Development (SDD) components. | `sdd-spec-schema.json`. | No executable entry points. Schemas are imported and used by validation logic in `core` or `tools`. | Ensures data integrity and contract adherence across various parts of the system, especially for SDD specifications. |
| `docs/` | Comprehensive documentation for the project, including architecture, best practices, concepts, guides, and generated content. | Subdirectories `architecture/`, `cli_best_practices/`, `codebase_standards/`, `concepts/`, `generated/`, `guides/`, `mcp_best_practices/`. Contains various Markdown files. | No executable entry points. Serves as reference material. | Essential for understanding the project's design, conventions, and usage. `generated/` holds documentation automatically created from the codebase. |
| `tests/` | Contains all unit, integration, property, and contract tests for the `foundry-mcp` project. | Subdirectories `contract/`, `doc_query/`, `fixtures/`, `integration/`, `llm_doc_gen/`, `parity/`, `property/`, `sdd_next/`, `skills/`, `unit/`, `verification/`. Contains `pytest.ini` and many `test_*.py` files. | Executed via `pytest`. | Critical for verifying correctness and adherence to contracts. Organized by testing type and corresponding application modules. |

### 3. Entry Points

*   **Main CLI Entry Point:** The primary command-line interface for `foundry-mcp` is exposed as `foundry-cli`, which points to `src/foundry_mcp/cli/main:cli` as defined in `pyproject.toml`. It can also be run directly as a Python module: `python -m foundry_mcp.cli`.
*   **MCP Server Entry Point:** The project also includes a server component, accessible via the `foundry-mcp` script, which executes `src/foundry_mcp/server:main` as specified in `pyproject.toml`.

### 4. File Organization Patterns

*   **Naming Conventions:** The codebase primarily uses `snake_case` for Python file names, modules, functions, and variables, adhering to Python's PEP 8 guidelines. JSON schema files (e.g., `sdd-spec-schema.json`) utilize `kebab-case`.
*   **File Grouping Strategies:**
    *   **By Layer:** The top-level `src/foundry_mcp/` is divided into distinct layers like `cli` (for user interaction), `core` (for business logic and shared services), and `tools` (for specialized operations).
    *   **By Concern/Domain:** Within `core` and `tools`, files are further grouped by specific functionalities (e.g., `cache.py`, `security.py`, `validation.py`, `planning.py`). The `src/foundry_mcp/cli/commands/` directory organizes individual CLI commands.
*   **Module/Package Structure:** Standard Python package structure is employed, with `__init__.py` files present in each directory to denote packages and enable hierarchical module imports.
*   **Co-location Patterns:** Tests are co-located in the `tests/` directory, mirroring the source code structure (e.g., `tests/unit/test_core/` for `src/foundry_mcp/core/`). Documentation files are centrally located in `docs/` with logical subdirectories for various topics.

### 5. Key File

---

## Related Documentation

For additional information, see:

- `index.md` - Master documentation index
- `project-overview.md` - Project overview and summary
- `architecture.md` - Detailed architecture

---

*Generated using LLM-based documentation workflow*