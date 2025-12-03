"""
Documentation generation prompts for AI consultation workflows.

This module provides prompt templates for two categories of documentation:

1. **Codebase Analysis Prompts** (DOC_GEN_*_V1):
   - DOC_GEN_PROJECT_OVERVIEW_V1: Project overview narrative generation
   - DOC_GEN_ARCHITECTURE_V1: Architecture documentation generation
   - DOC_GEN_COMPONENT_INVENTORY_V1: Component inventory generation

2. **Code Documentation Prompts** (analyze_*, generate_*, summarize_*):
   - analyze_module: Analyze a Python module and describe its purpose
   - analyze_function: Document a specific function with examples
   - analyze_class: Document a class with its methods and attributes
   - generate_overview: Generate architectural overview for a package
   - summarize_changes: Summarize changes between code versions

Each DOC_GEN_* template expects context with project data. Each analyze_*
template expects context with source code content.
"""

from __future__ import annotations

from typing import Any, Dict, List

from foundry_mcp.core.prompts import PromptBuilder, PromptRegistry, PromptTemplate


# =============================================================================
# System Prompts for DOC_GEN Templates
# =============================================================================

_DOC_GEN_SYSTEM_PROMPT = """You are an expert technical documentation writer.
Your task is to analyze codebase information and generate clear, comprehensive documentation.

Guidelines:
- Focus on evidence-based analysis from the provided codebase data
- Use clear, structured markdown formatting
- Reference specific files, classes, and functions when making observations
- Keep explanations concise but informative
- Avoid speculation - only document what can be verified from the data
- You have READ-ONLY access - do not attempt to write files"""

_IGNORE_PATTERNS_SECTION = """
## Files and Directories to Ignore

When analyzing this codebase, ignore the following:
- `specs/` - Project specifications (not source code)
- `.claude/` - Claude AI configuration
- `.agents/` - Agent configuration
- `AGENTS.md`, `CLAUDE.md` - Agent documentation
- Standard build/dependency directories (node_modules, dist, build, etc.)
- Any paths matching .gitignore patterns

Focus only on actual source code and project documentation."""


# =============================================================================
# DOC_GEN_PROJECT_OVERVIEW_V1
# =============================================================================

DOC_GEN_PROJECT_OVERVIEW_V1 = PromptTemplate(
    id="DOC_GEN_PROJECT_OVERVIEW_V1",
    version="1.0",
    system_prompt=_DOC_GEN_SYSTEM_PROMPT,
    user_template="""# Task: Project Overview Research (Read-Only)

**IMPORTANT: You have READ-ONLY access. Do not attempt to write files.**
Analyze this codebase and provide research findings for a project overview.
Your findings will be used to compose the final documentation.
{ignore_patterns}

## Project Context

{project_context}

## Key Files to Analyze

{key_files}

## Research Findings to Provide

### 1. Executive Summary (2-3 paragraphs)
- What does this project do? (Purpose and functionality)
- Who is it for? (Target users/audience)
- What problem does it solve?
- What makes it unique or notable?

### 2. Key Features (3-5 main features)
- List the most important features/capabilities
- Brief description of each feature
- Why each feature matters to users

### 3. Architecture Highlights
- High-level architecture pattern (e.g., layered, microservices, client-server)
- Key architectural decisions you can identify
- Notable design patterns in use

### 4. Development Overview
- Prerequisites needed to work on this project
- Key setup/installation steps
- Primary development commands (install, dev, build, test)

## Output Format

Provide your research findings as structured text with clear section headers.
Use markdown formatting for readability (headers, lists, code references).
Be specific and reference actual code/files when making observations.
Keep the executive summary concise (2-3 paragraphs max).

**DO NOT write documentation files yourself.**
Just return your research findings as text output.""",
    required_context=["project_context", "key_files"],
    optional_context=["ignore_patterns"],
    metadata={
        "author": "foundry-mcp",
        "category": "documentation",
        "workflow": "DOC_GENERATION",
        "description": "Generate project overview narrative from codebase analysis",
    },
)


# =============================================================================
# DOC_GEN_ARCHITECTURE_V1
# =============================================================================

DOC_GEN_ARCHITECTURE_V1 = PromptTemplate(
    id="DOC_GEN_ARCHITECTURE_V1",
    version="1.0",
    system_prompt=_DOC_GEN_SYSTEM_PROMPT,
    user_template="""# Task: Architecture Analysis Research (Read-Only)

**IMPORTANT: You have READ-ONLY access. Do not attempt to write files.**
Analyze this codebase to identify architecture patterns, design decisions, and implementation patterns.
Your findings will be used to compose the final architecture documentation.
{ignore_patterns}

## Project Context

{project_context}

## Key Files to Analyze

{key_files}

## Research Findings to Provide

### 1. Executive Summary
- High-level architecture overview (2-3 sentences)
- Primary architecture pattern (e.g., layered, microservices, event-driven, client-server)
- Key architectural characteristics

### 2. Architecture Pattern Identification

Identify the dominant architecture pattern(s) from:
- **Layered Architecture:** Organized in horizontal layers (presentation, business, data)
- **Microservices:** Independent services with separate deployment
- **Event-Driven:** Asynchronous event processing and pub/sub
- **Client-Server:** Clear separation between client and server components
- **Plugin Architecture:** Core system with extensible plugins
- **Monolith:** Single unified codebase and deployment
- **Other:** Describe if different pattern

For each identified pattern, explain:
- Evidence from the codebase (specific files/directories)
- How the pattern is implemented
- Benefits this pattern provides for this project

### 3. Key Architectural Decisions

Identify major architectural decisions and their rationale:

| Decision Category | Choice Made | Rationale/Evidence |
| --- | --- | --- |
| Database Architecture | ? | Based on code analysis |
| API Pattern | ? | Based on code analysis |
| State Management | ? | Based on code analysis |
| Authentication/Authorization | ? | Based on code analysis |
| Deployment Model | ? | Based on code analysis |

Add additional decision categories as you identify them.

### 4. Project Structure Analysis
- How is the code organized? (by feature, by layer, by module?)
- What are the main components/modules?
- How do components relate to each other?
- Identify any nested sub-projects or workspaces

### 5. Technology Integration Points
- External APIs and services used
- Database connections and ORM/query patterns
- Frontend-backend communication
- Third-party libraries and frameworks
- Authentication/authorization integration

### 6. Implementation Patterns

**Naming Conventions:**
- How are files/classes/functions named?
- Any consistent prefixes/suffixes?

**Code Organization:**
- How are related files grouped?
- Module/package structure patterns

**Error Handling:**
- How are errors handled consistently?
- Custom error types or standard approaches?

**Logging/Monitoring:**
- What logging approach is used?
- Any monitoring or observability patterns?

### 7. Data Architecture
- Data models and their relationships
- Data flow through the system
- Persistence strategies
- Caching approaches (if any)

### 8. Security Architecture
- Authentication mechanisms
- Authorization/access control
- Data protection (encryption, validation)
- API security (rate limiting, CORS, etc.)

### 9. Performance Considerations
- Caching layers
- Async/parallel processing
- Database query optimization
- Resource management

### 10. Novel or Unique Design Patterns
- Custom patterns specific to this project
- Creative solutions to specific problems
- Innovative architecture choices

## Output Format

Provide your research findings as structured text following the sections above.
Use markdown formatting (headers, lists, tables, code references).
Be specific and reference actual files/code when making observations.
Use the decision table format where applicable.
Focus on evidence-based analysis from the actual codebase.

**DO NOT write documentation files yourself.**
Just return your research findings as text output.""",
    required_context=["project_context", "key_files"],
    optional_context=["ignore_patterns"],
    metadata={
        "author": "foundry-mcp",
        "category": "documentation",
        "workflow": "DOC_GENERATION",
        "description": "Generate architecture documentation from codebase analysis",
    },
)


# =============================================================================
# DOC_GEN_COMPONENT_INVENTORY_V1
# =============================================================================

DOC_GEN_COMPONENT_INVENTORY_V1 = PromptTemplate(
    id="DOC_GEN_COMPONENT_INVENTORY_V1",
    version="1.0",
    system_prompt=_DOC_GEN_SYSTEM_PROMPT,
    user_template="""# Task: Component Inventory Analysis (Read-Only)

**IMPORTANT: You have READ-ONLY access. Do not attempt to write files.**
Analyze this codebase to understand component organization, directory purposes, and file patterns.
Your findings will be used to compose the final component inventory documentation.
{ignore_patterns}

## Project Context

{project_context}

## Directory Structure

{directory_structure}

## Directories to Analyze

{directories_to_analyze}

## Research Findings to Provide

### 1. Source Tree Overview
- High-level description of how the codebase is organized (2-3 sentences)
- Primary organizational pattern (by feature, by layer, by module, hybrid)
- Notable characteristics of the directory structure

### 2. Critical Directories

For each critical directory, provide:

| Directory Path | Purpose | Contents Summary | Entry Points | Integration Notes |
| --- | --- | --- | --- | --- |
| (Fill in) | (Why it exists) | (What it contains) | (Main files) | (How it connects) |

Examples of critical directories:
- Main application code directories
- Configuration directories
- Test directories
- Build/deployment directories
- Documentation directories

### 3. Entry Points
- Main application entry point
- Additional entry points (CLI tools, API servers, etc.)
- How the application starts/bootstraps

### 4. File Organization Patterns
- Naming conventions (camelCase, snake_case, kebab-case)
- File grouping strategies (by feature, by type, by domain)
- Module/package structure
- Co-location patterns (tests with code, styles with components, etc.)

### 5. Key File Types

| File Type | Pattern | Purpose | Examples |
| --- | --- | --- | --- |
| Source Code | `*.ext` | What these files do | Specific examples |
| Tests | `*.test.*` | Testing approach | Test file examples |
| Config | `*.config.*` | Configuration purpose | Config examples |

### 6. Configuration Files
- Build configuration (package.json, setup.py, Cargo.toml, etc.)
- Runtime configuration (.env, config files)
- Development tools (linters, formatters, IDE configs)
- CI/CD configuration

### 7. Asset Locations
- Static assets (images, fonts, icons)
- Styles (CSS, SCSS, style files)
- Documentation files
- Example/sample data

### 8. Development Notes
- Where to find specific functionality
- Important directories for common tasks
- Potential confusion points or gotchas
- Recommendations for navigating the codebase

## Output Format

Provide your research findings as structured text following the sections above.
Use markdown formatting (headers, lists, tables, code references).
Be specific and reference actual directories/files when making observations.
Use the table formats provided where applicable.
Focus on evidence-based analysis from the actual directory structure.

**DO NOT write documentation files yourself.**
Just return your research findings as text output.""",
    required_context=["project_context", "directory_structure", "directories_to_analyze"],
    optional_context=["ignore_patterns"],
    metadata={
        "author": "foundry-mcp",
        "category": "documentation",
        "workflow": "DOC_GENERATION",
        "description": "Generate component inventory from directory analysis",
    },
)


# =============================================================================
# Legacy String Templates (for backwards compatibility)
# =============================================================================


ANALYZE_MODULE_TEMPLATE = """Analyze the following Python module and provide a comprehensive documentation summary.

## Module Information
- File Path: {file_path}
- Module Name: {module_name}

## Source Code
```python
{content}
```

## Analysis Requirements
1. **Purpose**: Describe the primary purpose and responsibility of this module
2. **Public API**: List and describe all public functions, classes, and constants
3. **Dependencies**: Note any notable imports and their usage
4. **Design Patterns**: Identify any design patterns or architectural decisions
5. **Usage Examples**: Provide 1-2 concise usage examples if applicable

## Output Format
Provide a structured markdown response with clear sections for each requirement above.
Keep the documentation concise but complete - aim for clarity over verbosity.
"""


ANALYZE_FUNCTION_TEMPLATE = """Document the following function with detailed information suitable for API documentation.

## Function Information
- Function Name: {function_name}
- File Path: {file_path}
- Module: {module_name}

## Source Code
```python
{content}
```

## Context (if available)
{context_info}

## Documentation Requirements
1. **Summary**: One-sentence description of what the function does
2. **Parameters**: Document each parameter with type, description, and constraints
3. **Returns**: Document the return value with type and description
4. **Raises**: List exceptions that may be raised
5. **Examples**: Provide 1-2 usage examples showing common use cases
6. **Notes**: Any important implementation details or caveats

## Output Format
Return the documentation in Google-style docstring format, ready to be inserted into the source code.
"""


ANALYZE_CLASS_TEMPLATE = """Document the following class with comprehensive API documentation.

## Class Information
- Class Name: {class_name}
- File Path: {file_path}
- Module: {module_name}
- Base Classes: {base_classes}

## Source Code
```python
{content}
```

## Documentation Requirements
1. **Summary**: Describe the class purpose and when to use it
2. **Attributes**: Document all public attributes
3. **Methods**: List and describe all public methods with signatures
4. **Constructor**: Document __init__ parameters
5. **Usage Example**: Show how to instantiate and use the class
6. **Related Classes**: Note any important relationships or inheritance

## Output Format
Provide markdown documentation suitable for a developer guide.
Include a class diagram in ASCII if the class has complex relationships.
"""


GENERATE_OVERVIEW_TEMPLATE = """Generate an architectural overview for the following package/module collection.

## Package Information
- Package Path: {package_path}
- Package Name: {package_name}

## Module Summary
{module_summary}

## File Structure
{file_structure}

## Requirements
1. **Architecture Overview**: Describe the high-level architecture and design
2. **Component Relationships**: How do the modules interact?
3. **Entry Points**: What are the main entry points for users?
4. **Extension Points**: How can the package be extended?
5. **Dependencies**: External dependencies and their roles

## Output Format
Provide a markdown document suitable for a README or architecture documentation.
Include ASCII diagrams where helpful for understanding relationships.
"""


SUMMARIZE_CHANGES_TEMPLATE = """Summarize the changes between two versions of the code.

## Change Information
- File Path: {file_path}
- Change Type: {change_type}

## Previous Version
```python
{old_content}
```

## Current Version
```python
{new_content}
```

## Summary Requirements
1. **What Changed**: Describe the key changes made
2. **Why It Matters**: Explain the impact of these changes
3. **Breaking Changes**: Note any breaking changes to the public API
4. **Migration Notes**: If applicable, how to update calling code

## Output Format
Provide a concise changelog-style summary suitable for release notes.
"""


# =============================================================================
# Template Registries
# =============================================================================


# DOC_GEN_* templates registry (PromptTemplate instances)
DOC_GEN_TEMPLATES: Dict[str, PromptTemplate] = {
    "DOC_GEN_PROJECT_OVERVIEW_V1": DOC_GEN_PROJECT_OVERVIEW_V1,
    "DOC_GEN_ARCHITECTURE_V1": DOC_GEN_ARCHITECTURE_V1,
    "DOC_GEN_COMPONENT_INVENTORY_V1": DOC_GEN_COMPONENT_INVENTORY_V1,
}


# Legacy string templates (for backwards compatibility)
TEMPLATES: Dict[str, str] = {
    "analyze_module": ANALYZE_MODULE_TEMPLATE,
    "analyze_function": ANALYZE_FUNCTION_TEMPLATE,
    "analyze_class": ANALYZE_CLASS_TEMPLATE,
    "generate_overview": GENERATE_OVERVIEW_TEMPLATE,
    "summarize_changes": SUMMARIZE_CHANGES_TEMPLATE,
}


# =============================================================================
# Prompt Builder Implementation
# =============================================================================


class DocGenerationPromptBuilder(PromptBuilder):
    """
    Prompt builder for documentation generation workflows.

    Provides access to both DOC_GEN_* templates (PromptTemplate instances)
    and legacy analyze_*/generate_*/summarize_* templates.

    DOC_GEN Templates (recommended):
        - DOC_GEN_PROJECT_OVERVIEW_V1: Project overview narrative
        - DOC_GEN_ARCHITECTURE_V1: Architecture documentation
        - DOC_GEN_COMPONENT_INVENTORY_V1: Component inventory

    Legacy Templates:
        - analyze_module: Python module analysis
        - analyze_function: Function documentation
        - analyze_class: Class documentation
        - generate_overview: Package overview
        - summarize_changes: Code diff summary

    Example:
        builder = DocGenerationPromptBuilder()

        # Using DOC_GEN_* templates
        prompt = builder.build("DOC_GEN_PROJECT_OVERVIEW_V1", {
            "project_context": "Name: MyProject\\nType: Python\\n...",
            "key_files": "- src/main.py\\n- src/config.py\\n...",
        })

        # Using legacy templates
        prompt = builder.build("analyze_module", {
            "file_path": "src/main.py",
            "module_name": "main",
            "content": "def main(): pass",
        })
    """

    def __init__(self) -> None:
        """Initialize the builder with all templates."""
        self._registry = PromptRegistry()
        # Register DOC_GEN_* templates
        for template in DOC_GEN_TEMPLATES.values():
            self._registry.register(template)

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a documentation generation prompt.

        Args:
            prompt_id: Template ID (DOC_GEN_*, analyze_*, generate_*, summarize_*)
            context: Context dict with required variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id not found or required context missing
        """
        # Check if it's a DOC_GEN_* template
        if prompt_id in DOC_GEN_TEMPLATES:
            # Add default ignore patterns if not provided
            render_context = dict(context)
            if "ignore_patterns" not in render_context:
                render_context["ignore_patterns"] = _IGNORE_PATTERNS_SECTION
            return self._registry.render(prompt_id, render_context)

        # Check if it's a legacy template
        if prompt_id in TEMPLATES:
            template = TEMPLATES[prompt_id]
            # Provide safe defaults for optional context keys
            safe_context = {
                "file_path": context.get("file_path", "unknown"),
                "module_name": context.get("module_name", "unknown"),
                "content": context.get("content", ""),
                "function_name": context.get("function_name", "unknown"),
                "class_name": context.get("class_name", "unknown"),
                "base_classes": context.get("base_classes", "object"),
                "context_info": context.get(
                    "context_info", "No additional context provided."
                ),
                "package_path": context.get("package_path", "unknown"),
                "package_name": context.get("package_name", "unknown"),
                "module_summary": context.get("module_summary", "Not provided"),
                "file_structure": context.get("file_structure", "Not provided"),
                "change_type": context.get("change_type", "modification"),
                "old_content": context.get("old_content", ""),
                "new_content": context.get("new_content", ""),
            }
            try:
                return template.format(**safe_context)
            except KeyError as exc:
                raise ValueError(f"Missing required context key: {exc}") from exc

        # Unknown prompt_id
        all_ids = sorted(list(DOC_GEN_TEMPLATES.keys()) + list(TEMPLATES.keys()))
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available: {all_ids}")

    def list_prompts(self) -> List[str]:
        """
        Return all available prompt IDs.

        Returns:
            Sorted list of all prompt IDs (DOC_GEN_* and legacy)
        """
        return sorted(list(DOC_GEN_TEMPLATES.keys()) + list(TEMPLATES.keys()))

    def get_template(self, prompt_id: str) -> PromptTemplate:
        """
        Get a DOC_GEN_* template by ID for inspection.

        Args:
            prompt_id: Template identifier (must be a DOC_GEN_* template)

        Returns:
            The PromptTemplate

        Raises:
            KeyError: If not found or not a DOC_GEN_* template
        """
        if prompt_id not in DOC_GEN_TEMPLATES:
            available = sorted(DOC_GEN_TEMPLATES.keys())
            raise KeyError(
                f"Template '{prompt_id}' not found. "
                f"Only DOC_GEN_* templates can be inspected. Available: {available}"
            )
        return self._registry.get_required(prompt_id)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # DOC_GEN_* templates
    "DOC_GEN_PROJECT_OVERVIEW_V1",
    "DOC_GEN_ARCHITECTURE_V1",
    "DOC_GEN_COMPONENT_INVENTORY_V1",
    "DOC_GEN_TEMPLATES",
    # Legacy templates
    "TEMPLATES",
    "ANALYZE_MODULE_TEMPLATE",
    "ANALYZE_FUNCTION_TEMPLATE",
    "ANALYZE_CLASS_TEMPLATE",
    "GENERATE_OVERVIEW_TEMPLATE",
    "SUMMARIZE_CHANGES_TEMPLATE",
    # Builder
    "DocGenerationPromptBuilder",
]
