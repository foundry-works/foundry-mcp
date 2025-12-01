# foundry-mcp - Architecture Documentation

**Date:** 2025-12-01
**Project Type:** Software Project
**Primary Language(s):** python

## Technology Stack Details

### Core Technologies

- **Languages:** python

---

### 1. Executive Summary

Foundry MCP is a Python-based software project designed to facilitate Spec-Driven Development (SDD) through a Microservice Communication Protocol (MCP) server and an accompanying command-line interface (CLI). Its architecture centers around managing and interacting with various "specifications" (specs) using a modular suite of "tools" that cover the entire SDD lifecycle, from environment setup and spec authoring to validation, task planning, and AI/LLM integration. The primary architectural patterns are Client-Server (with the CLI acting as a client to the MCP server) and a Plugin Architecture (for its extensible tools). Key characteristics include robust versioned contracts, standardized JSON-first responses with "envelopes," rigorous input validation, comprehensive observability, resilience mechanisms, and explicit patterns for AI/LLM integration.

### 2. Architecture Pattern Identification

*   **Client-Server:**
    *   **Evidence:** The `pyproject.toml` defines two distinct entry points: `foundry-mcp = "foundry_mcp.server:main"` for the server and `foundry-cli = "foundry_mcp.cli.main:cli"` for the command-line client.
    *   **Implementation:** The `foundry-mcp` component acts as the server, presumably exposing a set of MCP-compliant functionalities. The `foundry-cli` serves as a thin client, invoking these functionalities either directly as local commands or via an underlying communication protocol (implied by `fastmcp`/`mcp` dependencies).
    *   **Benefits:** This separation allows for independent development, deployment, and scaling of the server and client components. The CLI can remain lightweight, offloading complex business logic to the server.
*   **Plugin Architecture:**
    *   **Evidence:** The project extensively uses the concept of "tools" (e.g., `env_verify_toolchain`, `spec_find_related_files`, `plan_format`). The `docs/mcp_best_practices/README.md` and `docs/codebase_standards/naming-conventions.md` detail how these tools are named and discovered. The `src/foundry_mcp/tools` directory houses numerous modules (e.g., `queries.py`, `validation.py`, `rendering.py`, `journal.py`, `docs.py`, `discovery.py`, `lifecycle.py`, `tasks.py`, `testing.py`, `providers.py`), each encapsulating a set of related "canonical tools."
    *   **Implementation:** The system provides a core runtime capable of discovering, loading, and executing these modular "tools." Each tool adheres to strict naming conventions (`prefix-verb` in `kebab-case`) and communicates using a standardized response envelope.
    *   **Benefits:** High extensibility, allowing new SDD functionalities to be added as independent, discoverable units without requiring modifications to the core system. This promotes modularity, reusability, and easier integration with external systems, particularly Large Language Models (LLMs).
*   **Layered Architecture:**
    *   **Evidence:** The core directory structure within `src/foundry_mcp` (`cli`, `core`, `schemas`, `tools`) naturally segregates responsibilities into distinct conceptual layers. The `CLI Best Practices` explicitly references "Shared Services" within `foundry_mcp.core.*` for common functionalities like response handling and validation.
    *   **Implementation:**
        *   **Presentation Layer:** `src/foundry_mcp/cli` is responsible for handling user input, command parsing, and presenting output (JSON or human-readable text).
        *   **Application/Business Logic Layer:** The modules within `src/foundry_mcp/tools` (e.g., `queries.py`, `validation.py`) implement the specific SDD operations and business rules. `src/foundry_mcp/server.py` orchestrates these tools for the MCP server.
        *   **Core Services/Utility Layer:** `src/foundry_mcp/core` provides foundational services such as standardized response generation (`responses.py`), naming conventions (`naming.py`), and spec file management (`spec.py`).
        *   **Data Model/Schema Layer:** `src/foundry_mcp/schemas` likely defines the data structures and validation rules used throughout the system.
    *   **Benefits:** Promotes a clear separation of concerns, which enhances maintainability, testability, and reduces coupling between different parts of the system.

### 3. Key Architectural Decisions

| Decision Category        | Choice Made

---

## Related Documentation

For additional information, see:

- `index.md` - Master documentation index
- `project-overview.md` - Project overview and summary
- `development-guide.md` - Development workflow and setup

---

*Generated using LLM-based documentation workflow*