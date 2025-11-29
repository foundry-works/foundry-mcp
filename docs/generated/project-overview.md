# Foundry MCP - Project Overview

**Date:** 2025-11-28
**Type:** Software Project
**Architecture:** monolith

## Project Classification

- **Repository Type:** monolith
- **Project Type:** Software Project
- **Primary Language(s):** python

## Technology Stack Summary

- **Languages:** python

---

### 1. Executive Summary

The `foundry-mcp` project implements an MCP (Model Context Protocol) server and a native command-line interface (CLI) designed for managing Spec-Driven Development (SDD) specifications. It provides a comprehensive suite of tools that enable both AI coding assistants and human developers to query, navigate, and manage specification files, track task progress, validate specs, and analyze codebase documentation.

The project primarily targets AI agents and developers engaged in SDD workflows. It addresses the challenge of reliably integrating AI tools into development processes by offering a structured, JSON-only output for its CLI, which is optimized for machine parsing. This approach ensures consistent and predictable interaction, making it a unique solution for facilitating AI-assisted software development by standardizing the way specifications and development tasks are accessed and managed.

### 2. Key Features

*   **Spec Management**: Provides tools (`list_specs`, `get_spec`, `get_spec_hierarchy`) for listing, finding, and navigating SDD specifications. This enables users to efficiently locate and understand the structure of their project's requirements and designs.
*   **Task Operations**: Offers capabilities (`task_prepare`, `task_next`, `task_update_status`) to manage individual development tasks, including tracking their status, dependencies, and overall progress within a specification's lifecycle. This automates workflow orchestration for developers and AI agents.
*   **Validation & Fixes**: Includes tools (`spec_validate`, `spec_fix`) to validate specifications against predefined schemas and apply automated fixes for common issues. This ensures the integrity and adherence to standards of the SDD artifacts.
*   **Journaling**: Features for adding journal entries (`journal_add`), marking tasks as blocked (`task_block`), and tracking status changes. This creates an auditable record of decisions, impediments, and resolutions throughout the development process.
*   **Code Documentation**: Allows querying of codebase documentation (`code_find_class`, `code_trace_calls`), tracing call graphs, and performing impact analysis. This significantly assists in understanding code structure, dependencies, and the potential effects of changes.

### 3. Architecture Highlights

The `foundry-mcp` project employs a layered architectural pattern, emphasizing a clear separation of concerns.

*   **High-level architecture pattern**: It utilizes a core business logic layer (`foundry_mcp.core`) that is transport-agnostic, meaning it has no direct dependencies on either the MCP server (`foundry_mcp.tools`) or the native CLI (`foundry_mcp.cli`). Both the MCP server and the CLI act as separate interfaces that consume this shared core logic.
*   **Key architectural decisions**:
    *   **JSON-Only CLI Output**: As detailed in `docs/architecture/adr-001-cli-architecture.md`, the CLI exclusively outputs JSON. This design choice is critical for AI coding assistants, which are the primary consumers, ensuring consistent and reliable parsing of structured data without the complexities of human-readable formats or verbosity flags.
    *   **Shared Core Logic**: The `foundry_mcp.core` module encapsulates all business logic, such as spec operations, task management, and validation. This shared foundation (referenced in `docs/architecture/adr-001-cli-architecture.md`) ensures consistent behavior and data handling across both the MCP server and the native CLI, reducing duplication and maintenance overhead.
    *   **CLI as a Sibling to MCP Tools**: The `foundry_mcp.cli` package is structured to be an independent runtime that depends directly on the core logic, rather than wrapping the MCP tools. This allows for independent development and testing of the CLI without coupling it to the asynchronous and decorator-driven nature of the MCP tools.
*   **Notable design patterns**:
    *   **Decorator Pattern**: Extensively used in `src/foundry_mcp/config.py` for cross-cutting concerns like logging (`@log_call`), performance timing (`@timed`), and authentication (`@require_auth`).
    *   **Configuration Management**: The `ServerConfig` class in `src/foundry_mcp/config.py` provides a robust configuration loading mechanism, prioritizing environment variables over TOML files (`foundry-mcp.toml`) and then default values, offering flexibility and explicit control over settings.
    *   **Structured Response Envelope**: A consistent JSON response structure with `success`, `data`, `error`, and optional `meta` fields is employed for all outputs from both the MCP server and the CLI. This standardization, supported by helpers in `src/foundry_mcp/core/responses.py` and `src/foundry_mcp/cli/output.py`, ensures predictable data contracts for consuming applications.

### 4. Development Overview

*   **Prerequisites needed**:
    *   Python 3.10 or higher.
    *   Key Python dependencies as specified in `pyproject.toml` include `fastmcp>=0.1.0`, `mcp>=1.0.0`, `click>=8.0.0`, and `tomli>=2.0.0` (for Python < 3.11). Testing requires additional dependencies like `pytest` and `jsonschema`.
*   **Key setup/installation steps**:
    *   **Using pip**: Install directly via `pip install foundry-mcp`.
    *   **From source**: Clone the GitHub repository (`git clone https://github.com/tylerburleigh/foundry-mcp.git`), navigate into the directory (`cd foundry-mcp`), and install in editable development mode (`pip install -e .`).
*   **Primary development commands**:
    *   **Install in development mode**: `pip install -e .`
    *   **Run tests**: `pytest`
    *   **Run the MCP server**: `foundry-mcp` (as defined in `pyproject.toml`)
    *   **Run the native CLI**: `foundry-cli` (as defined in `pyproject.toml` for `foundry_mcp.cli.main:cli`)

---

## Documentation Map

For detailed information, see:

- `index.md` - Master documentation index
- `architecture.md` - Detailed architecture
- `development-guide.md` - Development workflow

---

*Generated using LLM-based documentation workflow*