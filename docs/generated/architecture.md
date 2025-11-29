# Foundry MCP - Architecture Documentation

**Date:** 2025-11-28
**Project Type:** Software Project
**Primary Language(s):** python

## Technology Stack Details

### Core Technologies

- **Languages:** python

---

## 1. Executive Summary

Foundry MCP is a software project offering a Model Context Protocol (MCP) server and a native CLI for managing Spec-Driven Development (SDD) specifications. It provides a comprehensive suite of over 40 tools for spec management, task operations, validation, journaling, lifecycle, code documentation, and testing. The primary architectural pattern is a **Layered Architecture** with a strong emphasis on a transport-agnostic core, enabling reuse across the MCP server and a dedicated CLI. A notable characteristic is the **JSON-only output** of the CLI, specifically designed for seamless integration with AI coding assistants.

## 2. Architecture Pattern Identification

### Layered Architecture

*   **Evidence from the codebase:** Explicitly defined in `docs/architecture/adr-001-cli-architecture.md`, which outlines distinct layers: `foundry_mcp.server` (MCP Server), `foundry_mcp.core.*` (Shared Business Logic), `foundry_mcp.tools.*` (MCP Tool Implementations), and `foundry_mcp.cli` (Native CLI). The `pyproject.toml` further reinforces this with separate entry points for `foundry-mcp` (server) and `foundry-cli`.
*   **How the pattern is implemented:**
    *   `foundry_mcp.core`: This layer contains the pure business logic, such as spec file operations, task management, journaling, and validation. It is designed to be transport-agnostic, having no direct dependencies on the MCP protocol or the Click CLI framework.
    *   `foundry_mcp.tools`: This layer acts as the interface for the MCP server. It implements MCP tools, typically by decorating functions that call into the `foundry_mcp.core` logic.
    *   `foundry_mcp.cli`: This layer provides the command-line interface using the `Click` framework. Its commands directly invoke functions within `foundry_mcp.core`, with `foundry_mcp.cli.output` handling CLI-specific JSON formatting.
    *   `foundry_mcp.server`: This is the top layer, orchestrating the `foundry_mcp.tools` to expose functionality via the Model Context Protocol.
*   **Benefits this pattern provides for this project:**
    *   **Clear Separation of Concerns:** Each layer has a well-defined responsibility, enhancing modularity.
    *   **Code Reusability:** The core business logic (`foundry_mcp.core`) is shared and reused by both the MCP server and the native CLI, minimizing duplication.
    *   **Improved Testability:** The `foundry_mcp.core` can be tested in isolation, independent of specific interface implementations (MCP or CLI).
    *   **Maintainability:** Changes within one layer are less likely to ripple through and impact other layers.

### Plugin Architecture

*   **Evidence from the codebase:** The `README.md` lists over 40 specialized tools categorized by function (Spec Management, Task Operations, etc.), indicating an extensible system. The `ADR-001` details how `foundry_mcp.cli.commands` are structured, with commands grouped into separate files and registered with `Click`, implying a pluggable command structure. For the MCP server, `foundry_mcp.tools.*` modules likely use `mcp` decorators for tool registration.
*   **How the pattern is implemented:**
    *   **CLI Commands:** The `foundry_mcp.cli.main.py` entry point sets up the `Click` group, and `foundry_mcp.cli.registry.register_all_commands` dynamically registers commands from `foundry_mcp.cli.commands` modules.
    *   **MCP Tools:** The `foundry_mcp.tools` package contains modules (e.g., `tasks.py`, `lifecycle.py`) where functions are likely decorated with `@mcp.tool` (or similar) to register them as callable MCP operations.
*   **Benefits this pattern provides for this project:**
    *   **Extensibility:** New tools or CLI commands can be added easily by creating new modules and registering them, without modifying the core dispatcher.
    *   **Modularity:** Individual tools/commands are independent units, simplifying development and maintenance.
    *

---

## Related Documentation

For additional information, see:

- `index.md` - Master documentation index
- `project-overview.md` - Project overview and summary
- `../guides/development-guide.md` - Development workflow and setup

---

*Generated using LLM-based documentation workflow*