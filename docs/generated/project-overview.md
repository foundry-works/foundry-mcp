# foundry-mcp - Project Overview

**Date:** 2025-12-01
**Type:** Software Project
**Architecture:** monolith

## Project Classification

- **Repository Type:** monolith
- **Project Type:** Software Project
- **Primary Language(s):** python

## Technology Stack Summary

- **Languages:** python

---

Here are the research findings for the `foundry-mcp` project overview:

### 1. Executive Summary

`foundry-mcp` is an innovative MCP (Model Context Protocol) server designed to facilitate spec-driven development (SDD) for AI assistants. Its primary purpose is to provide a structured and intelligent framework for managing the entire lifecycle of software specifications, from creation and validation to task tracking and integration with AI-powered analysis tools. The project targets developers and AI agents seeking a methodical approach to feature implementation, offering capabilities like breaking down features into trackable tasks, automated validation, and advanced code analysis.

This project addresses the challenge of unstructured development in AI-assisted coding by offering a robust system for defining, managing, and verifying work against clear specifications. Its unique value lies in deeply integrating AI capabilities, such as AI-powered review suggestions and documentation generation, directly into the SDD workflow, enabling higher quality and more efficient development cycles.

### 2. Key Features

1.  **Spec Management**: Enables the creation, validation (`spec-validate`, `spec-fix`), and lifecycle management (`spec-lifecycle-*`) of specification files. Specifications move through `pending`, `active`, `completed`, and `archived` statuses, ensuring a clear, structured development progression.
2.  **Task Operations**: Provides a suite of tools (`task-next`, `task-prepare`, `task-start`, `task-complete`, `task-block`/`task-unblock`) to track development progress on individual tasks within a specification. This facilitates detailed task management, dependency tracking, and status updates.
3.  **LLM-Powered Analysis**: Integrates Large Language Models (LLMs) to offer intelligent features like `spec-review` for improvement suggestions, `spec-review-fidelity` to verify implementation against specs, `spec-doc-llm` for comprehensive documentation generation, and `pr-create-with-spec` for AI-enhanced pull request descriptions. These features augment developer capabilities and streamline workflows.
4.  **Code Intelligence**: Offers tools (`code-find-class`, `code-find-function`, `code-trace-calls`, `code-impact-analysis`, `doc-stats`) for deep introspection of the codebase. These capabilities help users and AI agents understand code structure, dependencies, and potential impacts of changes.
5.  **Testing Integration**: Facilitates the execution and discovery of tests using `pytest` with configurable options and presets (`test-run`, `test-run-quick`, `test-discover`, `test-presets`). This promotes a strong focus on quality assurance and streamlines the testing process within the SDD framework.

### 3. Architecture Highlights

The `foundry-mcp` project employs a layered architecture with a strong emphasis on separation of concerns, as detailed in `docs/architecture/adr-001-cli-architecture.md`.

*   **MCP Server (`foundry_mcp.server`)**: This is the primary interface for MCP-compatible clients, built on `FastMCP` and exposing over 80 distinct tools and resources.
*   **Core Logic (`foundry_mcp.core.*`)**: This layer contains the foundational business logic for specifications, tasks, journals, lifecycle management, validation, and standardized responses. Crucially, it is designed to be **transport-agnostic**, meaning it has no direct dependencies on either the MCP server or the native CLI, allowing for maximum reusability and independent testing.
*   **MCP Tools (`foundry_mcp.tools.*`)**: These modules implement the specific MCP tools by calling functions within the `foundry_mcp.core` layer.
*   **Native CLI (`foundry_mcp.cli.*`)**: A `Click`-based command-line interface provides a standalone execution surface. It also directly interacts with the `foundry_mcp.core` logic.

A notable architectural decision is the **JSON-only output** for the CLI. This choice, outlined in `ADR-001`, prioritizes machine readability for AI agents, streamlining integration and eliminating the complexities of managing multiple output formats or verbosity levels. The `cli/output.py` module is responsible for adapting core response envelopes into this consistent JSON format. The shared helper strategy ensures consistent behavior across both MCP and CLI surfaces for areas like validation (`core/validation.py`), security (`core/security.py`), and pagination (`core/pagination.py`).

### 5. Development Overview

*   **Prerequisites**:
    *   Python 3.10 or higher.
    *   An MCP-compatible client (e.g., Claude Desktop, Claude Code).
*   **Key Setup/Installation Steps**:
    *   **Installation (recommended for Claude Desktop)**: Use `uvx foundry-mcp`.
    *   **Installation (via pip)**: `pip install foundry-mcp`.
    *   **Installation (from source)**:
        ```bash
        git clone https://github.com/tylerburleigh/foundry-mcp.git
        cd foundry-mcp
        pip install -e .
        ```
    *   **Claude Desktop Setup**: Requires adding `foundry-mcp` to the `claude_desktop_config.json` file, specifying the command (`uvx foundry-mcp` or `foundry-mcp`) and optionally setting the `FOUNDRY_MCP_SPECS_DIR` environment variable.
*   **Configuration**: The project's behavior can be customized through:
    *   **Environment Variables**: (`FOUNDRY_MCP_SPECS_DIR`, `FOUNDRY_MCP_LOG_LEVEL`, `FOUNDRY_MCP_API_KEYS`, etc.), which take the highest precedence.
    *   **TOML Configuration File**: `foundry-mcp.toml` allows for detailed settings across `[workspace]`, `[logging]`, `[workflow]`, and `[llm]` sections.
    *   **LLM Provider Setup**: Supports OpenAI, Anthropic, and local Ollama models, configurable via environment variables (e.g., `OPENAI_API_KEY`) or the TOML file.
*   **Primary Development Commands**:
    *   **Install for development (with test dependencies)**:
        ```bash
        git clone https://github.com/tylerburleigh/foundry-mcp.git
        cd foundry-mcp
        pip install -e ".[test]"
        ```
    *   **Run tests**: `pytest`
    *   **Run the MCP server**: `foundry-mcp`
    *   **Run the native CLI**: `foundry-cli` (as defined in `pyproject.toml` and implemented in `src/foundry_mcp/cli/main.py`)

---

## Documentation Map

For detailed information, see:

- `index.md` - Master documentation index
- `architecture.md` - Detailed architecture
- `development-guide.md` - Development workflow

---

*Generated using LLM-based documentation workflow*