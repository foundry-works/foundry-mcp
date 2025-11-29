# Foundry MCP - Component Inventory

**Date:** 2025-11-28

## Complete Directory Structure

```
foundry_mcp/
├── cli
│   ├── commands
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── dev.py
│   │   ├── docgen.py
│   │   ├── docquery.py
│   │   ├── lifecycle.py
│   │   ├── modify.py
│   │   ├── pr.py
│   │   ├── render.py
│   │   ├── review.py
│   │   ├── session.py
│   │   ├── specs.py
│   │   ├── tasks.py
│   │   ├── testing.py
│   │   └── validate.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
│   ├── config.py
│   ├── context.py
│   ├── flags.py
│   ├── logging.py
│   ├── main.py
│   ├── output.py
│   ├── registry.py
│   └── resilience.py
├── core
│   ├── __init__.py
│   ├── cache.py
│   ├── capabilities.py
│   ├── concurrency.py
│   ├── discovery.py
│   ├── docs.py
│   ├── feature_flags.py
│   ├── journal.py
│   ├── lifecycle.py
│   ├── llm_config.py
│   ├── llm_patterns.py
│   ├── llm_provider.py
│   ├── naming.py
│   ├── observability.py
│   ├── pagination.py
│   ├── progress.py
│   ├── rate_limit.py
│   ├── rendering.py
│   ├── resilience.py
│   ├── responses.py
│   ├── security.py
│   ├── spec.py
│   ├── task.py
│   ├── testing.py
│   └── validation.py
├── prompts
│   ├── __init__.py
│   └── workflows.py
├── resources
│   ├── __init__.py
│   └── specs.py
├── schemas
│   ├── __init__.py
│   └── sdd-spec-schema.json
├── tools
│   ├── __init__.py
│   ├── analysis.py
│   ├── authoring.py
│   ├── context.py
│   ├── discovery.py
│   ├── docs.py
│   ├── documentation.py
│   ├── environment.py
│   ├── git_integration.py
│   ├── journal.py
│   ├── lifecycle.py
│   ├── mutations.py
│   ├── planning.py
│   ├── pr_workflow.py
│   ├── queries.py
│   ├── rendering.py
│   ├── reporting.py
│   ├── review.py
│   ├── spec_helpers.py
│   ├── tasks.py
│   ├── testing.py
│   ├── utilities.py
│   └── validation.py
├── __init__.py
├── config.py
└── server.py
```

---

### 1. Source Tree Overview

The Foundry MCP codebase is primarily organized by **layer and module**, with a clear separation between CLI components, core logic, prompt definitions, resources, schemas, and tools. The structure is characterized by a single-part application design, rooted in `src/foundry_mcp`, with distinct top-level directories for different functional concerns.

### 2. Critical Directories

| Directory Path | Purpose | Contents Summary | Entry Points | Integration Notes |
| :------------- | :------ | :--------------- | :----------- | :---------------- |
| `src/foundry_mcp/cli` | Handles command-line interface logic, parsing arguments, and dispatching commands. | Contains `__main__.py` for CLI entry, `commands/` for specific CLI commands, and modules for agent integration, configuration, context management, logging, output, registry, and resilience. | `src/foundry_mcp/cli/__main__.py`, individual command files in `src/foundry_mcp/cli/commands/` (e.g., `cache.py`, `dev.py`, `docgen.py`, `main.py`). | Integrates heavily with `core/` for business logic and `tools/` for execution, providing the user-facing interface. |
| `src/foundry_mcp/core` | Encapsulates the core business logic, foundational services, and common utilities. | Includes modules for caching, capabilities, concurrency, discovery, documentation, feature flags, journaling, lifecycle management, LLM configuration and providers, naming conventions, observability, pagination, progress tracking, rate limiting, rendering, resilience, responses, security, specification management, tasks, testing, and validation. | None directly as entry points, but functions like `error_response`, `success_response` (from `responses.py`), `canonical_tool` (from `naming.py`), `find_specs_directory`, `load_spec` (from `spec.py`), and `with_sync_timeout` (from `resilience.py`) are frequently called by other modules. | Provides the underlying functionalities consumed by the CLI and various tools. Acts as the central hub for shared logic. |
| `src/foundry_mcp/prompts` | Stores prompt templates and definitions used for AI/LLM interactions. | Contains `workflows.py` which likely defines structured prompts or sequences for different AI-driven tasks. | None. | Used by `core/llm_provider.py` or `tools/` modules that interact with LLMs to generate responses or perform actions. |
| `src/foundry_mcp/resources` | Houses various static resources or data files required by the application. | Contains `specs.py` which might define paths or access methods for project specifications. | None. | Accessed by other modules (e.g., `core/spec.py`) to locate and load external resources. |
| `src/foundry_mcp/schemas` | Defines data schemas, particularly for validating input and output data structures. | Contains `sdd-spec-schema.json`, likely a JSON schema for validating SDD (Specification-Driven Development) specifications. | None. | Utilized by `core/validation.py` and `tools/validation.py` to enforce data integrity and consistency. |
| `src/foundry_mcp/tools` | Provides a collection of specific tools or utilities that perform distinct operations, often interacting with external systems or specialized logic. | Includes modules for analysis, authoring, context management, discovery, documentation, environment interaction, Git integration, journaling, lifecycle, mutations, planning, pull request workflows, queries, rendering, reporting, review, spec helpers, tasks, testing, and utilities. | None directly, but tools within these modules are called by CLI commands or core logic to perform discrete tasks. | These tools are typically invoked by the `cli/commands` or `core` modules to execute specific, granular functionalities, acting as extensions of the core capabilities. |
| `tests` | Contains all unit, integration, property, and verification tests for the codebase. | Organized into subdirectories like `unit/`, `integration/`, `property/`, `contract/`, `doc_query/`, `llm_doc_gen/`, `parity/`, `sdd_next/`, `skills/`, and `verification/`, reflecting different testing methodologies and areas. | `pytest.ini` defines test configuration; `conftest.py` holds shared fixtures. Individual `test_*.py` files are the entry points for test runners. | Ensures code quality, validates functionality, and verifies adherence to specifications. Run via `pytest`. |
| `docs` | Stores project documentation, including architecture, CLI best practices, codebase standards, generated content, guides, and MCP best practices. | Contains Markdown files (`.md`) and JSON files for various documentation aspects, including architectural decision records (ADRs), usage guides, and automatically generated content. | None, these are static documentation files. | Provides comprehensive information about the project's design, usage, and development guidelines. Generated content is produced by `docgen` commands in the CLI. |

### 3. Entry Points

*   **Main Application Entry Point:** The primary entry point for the CLI application is `src/foundry_mcp/cli/__main__.py`, which likely dispatches to `src/foundry_mcp/cli/main.py`.
*   **Additional Entry Points:**
    *   **CLI Commands:** Various `entry_point` functions are defined within `src/foundry_mcp/cli/commands/` (e.g., `cache_info_cmd`, `dev_gendocs_cmd`, `docgen_cmd`, `review_cmd`, `validate_cmd`, `main.cli_entry_point`). These correspond to individual CLI commands.
    *   **Server:** `src/foundry_mcp/server.py` suggests an API server entry point, though its specifics would require further analysis.
*   **Bootstrap Process:** The application likely bootstraps by `src/foundry_mcp/cli/__main__.py` (or `src/foundry_mcp/cli/main.py`) initializing the CLI framework (e.g., Click, argparse), loading configuration from `src/foundry_mcp/config.py` and potentially `src/foundry_mcp/cli/config.py`, and then registering and dispatching commands from `src/foundry_mcp/cli/commands/`.

### 4. File Organization Patterns

*   **Naming Conventions:** Python files and directories largely follow `snake_case` (e.g., `test_cli_verbosity.py`, `core/feature_flags.py`). Classes often use `CamelCase`. CLI commands and entry points are often suffixed with `_cmd` or `_group`.
*   **File Grouping Strategies:** Files are primarily grouped **by layer/module**, as seen with distinct `cli/`, `core/`, `prompts/`, `resources/`, `schemas/`, and `tools/` directories. Within these, further grouping is **by domain/feature** (e.g., `cli/commands/` for specific commands, `core/responses.py` for response handling, `tools/git_integration.py` for Git-related tools).
*   **Module/Package Structure:** The project uses standard Python package structure with `__init__.py` files indicating packages. Sub-packages are created for logical separation (e.g., `foundry_mcp.cli.commands`).
*   **Co-location Patterns:**
    *   Tests are co-located in a top-level `tests/` directory, mirroring the source code structure (e.g., `tests/unit/test_analysis.py` corresponds to `src/foundry_mcp/core/analysis.py` or similar).
    *   Documentation (especially generated docs) are co-located in `docs/generated/`.
    *   Schemas are co-located in `src/foundry_mcp/schemas/`.

### 5. Key File Types

| File Type | Pattern | Purpose | Examples |
| :-------- | :------ | :------ | :------- |
| Source Code | `*.py` | Python modules containing application logic, classes, functions, and command definitions. | `src/foundry_mcp/cli/main.py`, `src/foundry_mcp/core/responses.py`, `src/foundry_mcp/tools/analysis.py` |
| Tests | `test_*.py` | Python test files containing unit, integration, and property tests. | `tests/test_cli_verbosity.py`, `tests/unit/test_llm_provider.py`, `tests/integration/test_mcp_smoke.py` |
| Configuration | `*.toml`, `*.ini`, `*.yml`, `*.json` (for schemas) | Project-level configuration, test configuration, CI/CD workflows, and data schema definitions. | `pyproject.toml`, `pytest.ini`, `.github/workflows/publish.yml`, `src/foundry_mcp/schemas/sdd-spec-schema.json` |
| Markdown | `*.md` | Human-readable documentation, guides, and architectural decision records. | `README.md`, `CHANGELOG.md`, `docs/architecture/adr-001-cli-architecture.md`, `docs/cli_best_practices/01-cli-runtime.md` |

### 6. Configuration Files

*   **Build Configuration:**
    *   `pyproject.toml`: Specifies build system requirements, project metadata, dependencies, and tools like `ruff` (linter) and `pytest` configuration.
*   **Runtime Configuration:**
    *   `src/foundry_mcp/config.py`: Likely contains application-wide runtime settings.
    *   `src/foundry_mcp/cli/config.py`: Specific configuration for the CLI.
    *   `samples/foundry-mcp.toml`: Provides a sample configuration file, indicating user-editable runtime settings.
*   **Development Tools:**
    *   `pytest.ini`: Configuration for the `pytest` testing framework.
    *   `pyproject.toml` (sections for `tool.ruff`): Linter configuration.
*   **CI/CD Configuration:**
    *   `.github/workflows/publish.yml`: Defines GitHub Actions workflow for publishing the project, likely including build, test, and deployment steps.

### 7. Asset Locations

*   **Documentation Files:** The `docs/` directory is the primary location for all project documentation. It contains subdirectories for architecture, best practices, generated content, and guides.
*   **Example/Sample Data:** The `samples/` directory, specifically `samples/foundry-mcp.toml`, provides example configuration.
*   **Schemas:** `src/foundry_mcp/schemas/sdd-spec-schema.json` serves as a structured asset for data validation.

### 8. Development Notes

*   **Core Logic:** For understanding the fundamental operations and shared utilities, explore `src/foundry_mcp/core/`. This directory contains many frequently called functions as highlighted in the analysis.
*   **CLI Commands:** New CLI commands or modifications to existing ones will typically involve `src/foundry_mcp/cli/commands/` and `src/foundry_mcp/cli/main.py`.
*   **AI/LLM Integration:** Look into `src/foundry_mcp/prompts/` for prompt definitions and `src/foundry_mcp/core/llm_config.py`, `src/foundry_mcp/core/llm_provider.py` for how LLMs are configured and interacted with.
*   **Tooling Extensions:** For adding specialized functionalities or external integrations, the `src/foundry_mcp/tools/` directory is the place to extend.
*   **Testing:** All tests reside in the `tests/` directory. When adding new features or fixing bugs, ensure corresponding tests are created or updated, following the existing structure. Refer to `pytest.ini` for test runner configuration.
*   **Schema Definitions:** Data validation logic relies on schemas defined in `src/foundry_mcp/schemas/`.
*   **Configuration:** Project-wide settings are likely managed through `pyproject.toml` and potentially `src/foundry_mcp/config.py` or `src/foundry_mcp/cli/config.py`.
*   **Output Handling:** CLI output is managed in `src/foundry_mcp/cli/output.py`, which is crucial for consistent user experience.
*   **Resilience and Error Handling:** `src/foundry_mcp/cli/resilience.py` and `src/foundry_mcp/core/responses.py` are key for understanding how the application handles errors and maintains stability.
*   **Documentation Generation:** The presence of `docs/generated/` and CLI commands like `docgen.py` indicates an automated documentation generation process.```
### 1. Source Tree Overview

The Foundry MCP codebase is primarily organized by **layer and module**, with a clear separation between CLI components, core logic, prompt definitions, resources, schemas, and tools. The structure is characterized by a single-part application design, rooted in `src/foundry_mcp`, with distinct top-level directories for different functional concerns.

### 2. Critical Directories

| Directory Path | Purpose | Contents Summary | Entry Points | Integration Notes |
| :------------- | :------ | :--------------- | :----------- | :---------------- |
| `src/foundry_mcp/cli` | Handles command-line interface logic, parsing arguments, and dispatching commands. | Contains `__main__.py` for CLI entry, `commands/` for specific CLI commands, and modules for agent integration, configuration, context management, logging, output, registry, and resilience. | `src/foundry_mcp/cli/__main__.py`, individual command files in `src/foundry_mcp/cli/commands/` (e.g., `cache.py`, `dev.py`, `docgen.py`, `main.py`). | Integrates heavily with `core/` for business logic and `tools/` for execution, providing the user-facing interface. |
| `src/foundry_mcp/core` | Encapsulates the core business logic, foundational services, and common utilities. | Includes modules for caching, capabilities, concurrency, discovery, documentation, feature flags, journaling, lifecycle management, LLM configuration and providers, naming conventions, observability, pagination, progress tracking, rate limiting, rendering, resilience, responses, security, specification management, tasks, testing, and validation. | None directly as entry points, but functions like `error_response`, `success_response` (from `responses.py`), `canonical_tool` (from `naming.py`), `find_specs_directory`, `load_spec` (from `spec.py`), and `with_sync_timeout` (from `resilience.py`) are frequently called by other modules. | Provides the underlying functionalities consumed by the CLI and various tools. Acts as the central hub for shared logic. |
| `src/foundry_mcp/prompts` | Stores prompt templates and definitions used for AI/LLM interactions. | Contains `workflows.py` which likely defines structured prompts or sequences for different AI-driven tasks. | None. | Used by `core/llm_provider.py` or `tools/` modules that interact with LLMs to generate responses or perform actions. |
| `src/foundry_mcp/resources` | Houses various static resources or data files required by the application. | Contains `specs.py` which might define paths or access methods for project specifications. | None. | Accessed by other modules (e.g., `core/spec.py`) to locate and load external resources. |
| `src/foundry_mcp/schemas` | Defines data schemas, particularly for validating input and output data structures. | Contains `sdd-spec-schema.json`, likely a JSON schema for validating SDD (Specification-Driven Development) specifications. | None. | Utilized by `core/validation.py` and `tools/validation.py` to enforce data integrity and consistency. |
| `src/foundry_mcp/tools` | Provides a collection of specific tools or utilities that perform distinct operations, often interacting with external systems or specialized logic. | Includes modules for analysis, authoring, context management, discovery, documentation, environment interaction, Git integration, journaling, lifecycle, mutations, planning, pull request workflows, queries, rendering, reporting, review, spec helpers, tasks, testing, and utilities. | None directly, but tools within these modules are called by CLI commands or core logic to perform discrete tasks. | These tools are typically invoked by the `cli/commands` or `core` modules to execute specific, granular functionalities, acting as extensions of the core capabilities. |
| `tests` | Contains all unit, integration, property, and verification tests for the codebase. | Organized into subdirectories like `unit/`, `integration/`, `property/`, `contract/`, `doc_query/`, `llm_doc_gen/`, `parity/`, `sdd_next/`, `skills/`, and `verification/`, reflecting different testing methodologies and areas. | `pytest.ini` defines test configuration; `conftest.py` holds shared fixtures. Individual `test_*.py` files are the entry points for test runners. | Ensures code quality, validates functionality, and verifies adherence to specifications. Run via `pytest`. |
| `docs` | Stores project documentation, including architecture, CLI best practices, codebase standards, generated content, guides, and MCP best practices. | Contains Markdown files (`.md`) and JSON files for various documentation aspects, including architectural decision records (ADRs), usage guides, and automatically generated content. | None, these are static documentation files. | Provides comprehensive information about the project's design, usage, and development guidelines. Generated content is produced by `docgen` commands in the CLI. |

### 3. Entry Points

*   **Main Application Entry Point:** The primary entry point for the CLI application is `src/foundry_mcp/cli/__main__.py`, which likely dispatches to `src/foundry_mcp/cli/main.py`.
*   **Additional Entry Points:**
    *   **CLI Commands:** Various `entry_point` functions are defined within `src/foundry_mcp/cli/commands/` (e.g., `cache_info_cmd`, `dev_gendocs_cmd`, `docgen_cmd`, `review_cmd`, `validate_cmd`, `main.cli_entry_point`). These correspond to individual CLI commands.
    *   **Server:** `src/foundry_mcp/server.py` suggests an API server entry point, though its specifics would require further analysis.
*   **Bootstrap Process:** The application likely bootstraps by `src/foundry_mcp/cli/__main__.py` (or `src/foundry_mcp/cli/main.py`) initializing the CLI framework (e.g., Click, argparse), loading configuration from `src/foundry_mcp/config.py` and potentially `src/foundry_mcp/cli/config.py`, and then registering and dispatching commands from `src/foundry_mcp/cli/commands/`.

### 4. File Organization Patterns

*   **Naming Conventions:** Python files and directories largely follow `snake_case` (e.g., `test_cli_verbosity.py`, `core/feature_flags.py`). Classes often use `CamelCase`. CLI commands and entry points are often suffixed with `_cmd` or `_group`.
*   **File Grouping Strategies:** Files are primarily grouped **by layer/module**, as seen with distinct `cli/`, `core/`, `prompts/`, `resources/`, `schemas/`, and `tools/` directories. Within these, further grouping is **by domain/feature** (e.g., `cli/commands/` for specific commands, `core/responses.py` for response handling, `tools/git_integration.py` for Git-related tools).
*   **Module/Package Structure:** The project uses standard Python package structure with `__init__.py` files indicating packages. Sub-packages are created for logical separation (e.g., `foundry_mcp.cli.commands`).
*   **Co-location Patterns:**
    *   Tests are co-located in a top-level `tests/` directory, mirroring the source code structure (e.g., `tests/unit/test_analysis.py` corresponds to `src/foundry_mcp/core/analysis.py` or similar).
    *   Documentation (especially generated docs) are co-located in `docs/generated/`.
    *   Schemas are co-located in `src/foundry_mcp/schemas/`.

### 5. Key File Types

| File Type | Pattern | Purpose | Examples |
| :-------- | :------ | :------ | :------- |
| Source Code | `*.py` | Python modules containing application logic, classes, functions, and command definitions. | `src/foundry_mcp/cli/main.py`, `src/foundry_mcp/core/responses.py`, `src/foundry_mcp/tools/analysis.py` |
| Tests | `test_*.py` | Python test files containing unit, integration, and property tests. | `tests/test_cli_verbosity.py`, `tests/unit/test_llm_provider.py`, `tests/integration/test_mcp_smoke.py` |
| Configuration | `*.toml`, `*.ini`, `*.yml`, `*.json` (for schemas) | Project-level configuration, test configuration, CI/CD workflows, and data schema definitions. | `pyproject.toml`, `pytest.ini`, `.github/workflows/publish.yml`, `src/foundry_mcp/schemas/sdd-spec-schema.json` |
| Markdown | `*.md` | Human-readable documentation, guides, and architectural decision records. | `README.md`, `CHANGELOG.md`, `docs/architecture/adr-001-cli-architecture.md`, `docs/cli_best_practices/01-cli-runtime.md` |

### 6. Configuration Files

*   **Build Configuration:**
    *   `pyproject.toml`: Specifies build system requirements, project metadata, dependencies, and tools like `ruff` (linter) and `pytest` configuration.
*   **Runtime Configuration:**
    *   `src/foundry_mcp/config.py`: Likely contains application-wide runtime settings.
    *   `src/foundry_mcp/cli/config.py`: Specific configuration for the CLI.
    *   `samples/foundry-mcp.toml`: Provides a sample configuration file, indicating user-editable runtime settings.
*   **Development Tools:**
    *   `pytest.ini`: Configuration for the `pytest` testing framework.
    *   `pyproject.toml` (sections for `tool.ruff`): Linter configuration.
*   **CI/CD Configuration:**
    *   `.github/workflows/publish.yml`: Defines GitHub Actions workflow for publishing the project, likely including build, test, and deployment steps.

### 7. Asset Locations

*   **Documentation Files:** The `docs/` directory is the primary location for all project documentation. It contains subdirectories for architecture, best practices, generated content, and guides.
*   **Example/Sample Data:** The `samples/` directory, specifically `samples/foundry-mcp.toml`, provides example configuration.
*   **Schemas:** `src/foundry_mcp/schemas/sdd-spec-schema.json` serves as a structured asset for data validation.

### 8. Development Notes

*   **Core Logic:** For understanding the fundamental operations and shared utilities, explore `src/foundry_mcp/core/`. This directory contains many frequently called functions as highlighted in the analysis.
*   **CLI Commands:** New CLI commands or modifications to existing ones will typically involve `src/foundry_mcp/cli/commands/` and `src/foundry_mcp/cli/main.py`.
*   **AI/LLM Integration:** Look into `src/foundry_mcp/prompts/` for prompt definitions and `src/foundry_mcp/core/llm_config.py`, `src/foundry_mcp/core/llm_provider.py` for how LLMs are configured and interacted with.
*   **Tooling Extensions:** For adding specialized functionalities or external integrations, the `src/foundry_mcp/tools/` directory is the place to extend.
*   **Testing:** All tests reside in the `tests/` directory. When adding new features or fixing bugs, ensure corresponding tests are created or updated, following the existing structure. Refer to `pytest.ini` for test runner configuration.
*   **Schema Definitions:** Data validation logic relies on schemas defined in `src/foundry_mcp/schemas/`.
*   **Configuration:** Project-wide settings are likely managed through `pyproject.toml` and potentially `src/foundry_mcp/config.py` or `src/foundry_mcp/cli/config.py`.
*   **Output Handling:** CLI output is managed in `src/foundry_mcp/cli/output.py`, which is crucial for consistent user experience.
*   **Resilience and Error Handling:** `src/foundry_mcp/cli/resilience.py` and `src/foundry_mcp/core/responses.py` are key for understanding how the application handles errors and maintains stability.
*   **Documentation Generation:** The presence of `docs/generated/` and CLI commands like `docgen.py` indicates an automated documentation generation process.
```
```


---

## Related Documentation

For additional information, see:

- `index.md` - Master documentation index
- `project-overview.md` - Project overview and summary
- `architecture.md` - Detailed architecture

---

*Generated using LLM-based documentation workflow*