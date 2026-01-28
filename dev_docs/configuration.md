# Configuration Reference

> Complete reference for foundry-mcp configuration options.

## Overview

foundry-mcp can be configured through:
1. **TOML file**: `foundry-mcp.toml` in workspace root
2. **Environment variables**: Prefixed with `FOUNDRY_MCP_`
3. **Default values**: Built-in defaults for all options

Priority: Environment variables > TOML file > Defaults

## Configuration File

Create `foundry-mcp.toml` in your workspace root:

```toml
[workspace]
specs_dir = "./specs"

[logging]
level = "INFO"
structured = true

[research]
deep_research_digest_policy = "auto"
deep_research_digest_min_chars = 10000
```

## Research Configuration

### Deep Research Settings

Core settings for the deep research workflow:

| Setting | Default | Description |
|---------|---------|-------------|
| `deep_research_max_iterations` | `3` | Maximum refinement iterations |
| `deep_research_max_sub_queries` | `5` | Max sub-queries to generate |
| `deep_research_max_sources_per_query` | `5` | Max sources per sub-query |
| `deep_research_follow_links` | `true` | Follow and extract linked content |
| `deep_research_timeout` | `600` | Overall research timeout (seconds) |

### Document Digest Settings

Settings for the document digest phase that compresses source content:

| Setting | Type | Default | Valid Values | Description |
|---------|------|---------|--------------|-------------|
| `deep_research_digest_policy` | string | `"auto"` | `"off"`, `"auto"`, `"always"` | Controls when digest is applied |
| `deep_research_digest_min_chars` | int | `10000` | ≥0 | Minimum source chars for auto-policy eligibility |
| `deep_research_digest_max_sources` | int | `8` | ≥1 | Maximum sources to digest per batch |
| `deep_research_digest_timeout` | float | `60.0` | >0 | Timeout per digest operation (seconds) |
| `deep_research_digest_max_concurrent` | int | `3` | ≥1 | Maximum concurrent digest operations |
| `deep_research_digest_include_evidence` | bool | `true` | `true`, `false` | Include evidence snippets in output |
| `deep_research_digest_evidence_max_chars` | int | `400` | 1-500 | Maximum characters per evidence snippet |
| `deep_research_digest_max_evidence_snippets` | int | `5` | 1-10 | Maximum evidence snippets per digest |
| `deep_research_digest_fetch_pdfs` | bool | `false` | `true`, `false` | Fetch and extract PDF content |

Note: Evidence snippet limits are clamped to schema caps (500 chars, 10 snippets) to prevent validation errors.

#### Digest Policy Details

**`off`**: Digest is completely disabled. All sources pass through with original content unchanged. Use when you want maximum fidelity and have sufficient context budget.

**`auto`** (default): Intelligent digestion based on:
- Source must exceed `min_chars` threshold (default 10,000)
- Source quality must be HIGH or MEDIUM
- Low quality and unknown quality sources are skipped
- Recommended for most use cases

**`always`**: Digest all sources with content regardless of size or quality. Use for aggressive compression when context budget is tight.

#### Evidence Snippet Configuration

Evidence snippets preserve query-relevant excerpts with position locators for citation:

```toml
[research]
# Include evidence snippets (recommended for citations)
deep_research_digest_include_evidence = true

# Maximum chars per snippet (truncation applied at render time)
deep_research_digest_evidence_max_chars = 400

# Maximum snippets per digest (top-scoring by relevance)
deep_research_digest_max_evidence_snippets = 5
```

#### Performance Tuning

For large research jobs, tune concurrency and timeouts:

```toml
[research]
# Increase concurrent digests for faster processing
deep_research_digest_max_concurrent = 5

# Increase timeout for complex documents
deep_research_digest_timeout = 90.0

# Process more sources per batch
deep_research_digest_max_sources = 12
```

#### PDF Processing

Enable PDF extraction for research involving PDF documents:

```toml
[research]
# Enable PDF fetching and text extraction
deep_research_digest_fetch_pdfs = true
```

When enabled:
- PDF URLs are fetched and text is extracted
- Page boundaries are tracked for locators
- Evidence locators include page references (e.g., `page:3:char:200-450`)

### Content Archival Settings

Settings for archiving original source content:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `deep_research_archive_content` | bool | `false` | Archive canonical text before digest |
| `deep_research_archive_retention_days` | int | `30` | Days to retain archived content (0 = keep indefinitely) |

When archival is enabled:
- Canonical (normalized) text is stored before compression
- Path: `~/.foundry-mcp/research_archives/{source_id}/{hash}.txt`
- Evidence locators can be verified against archived content

### Provider Settings

Configure LLM providers for research phases:

| Setting | Default | Description |
|---------|---------|-------------|
| `deep_research_analysis_provider` | `"claude"` | Provider for analysis phase |
| `deep_research_synthesis_provider` | `"claude"` | Provider for synthesis/report |
| `summarization_provider` | `"claude"` | Provider for digest summarization |

## Example Configurations

### Minimal (defaults)

```toml
# Use all defaults - digest enabled with auto policy
[research]
```

### High Fidelity (digest off)

```toml
[research]
# Disable digest for maximum source fidelity
deep_research_digest_policy = "off"
```

### Aggressive Compression

```toml
[research]
# Digest everything for tight context budgets
deep_research_digest_policy = "always"
deep_research_digest_min_chars = 1000
deep_research_digest_max_sources = 15
deep_research_digest_max_concurrent = 5
```

### Research with PDFs

```toml
[research]
# Enable PDF processing with archival
deep_research_digest_fetch_pdfs = true
deep_research_archive_content = true
deep_research_archive_retention_days = 60
```

### Citation-Focused

```toml
[research]
# Maximize evidence for citation support
deep_research_digest_include_evidence = true
deep_research_digest_max_evidence_snippets = 10
deep_research_digest_evidence_max_chars = 500
```

## Environment Variables

All settings can be set via environment variables with `FOUNDRY_MCP_` prefix and uppercase:

```bash
# Set digest policy
export FOUNDRY_MCP_DEEP_RESEARCH_DIGEST_POLICY=auto

# Set minimum chars
export FOUNDRY_MCP_DEEP_RESEARCH_DIGEST_MIN_CHARS=10000

# Enable PDF processing
export FOUNDRY_MCP_DEEP_RESEARCH_DIGEST_FETCH_PDFS=true
```

## Validation

Invalid configuration values produce clear error messages:

```
Invalid deep_research_digest_policy: 'invalid'. Must be one of: off, auto, always
Invalid deep_research_digest_min_chars: -100. Must be >= 0
Invalid deep_research_digest_timeout: 0. Must be > 0
```

Configuration is validated at startup. The server will fail to start with invalid configuration.
