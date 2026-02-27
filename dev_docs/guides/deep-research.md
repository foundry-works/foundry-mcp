# Deep Research Workflow

> Multi-phase iterative research with query decomposition, source gathering, document digestion, and synthesized reporting.

## Overview

The Deep Research workflow provides comprehensive research capabilities through:
- Query decomposition into targeted sub-queries
- Multi-provider parallel source gathering
- Intelligent document digestion with evidence extraction
- Context budget management for LLM processing
- Iterative refinement with follow-up queries
- Synthesized markdown report generation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepResearchWorkflow                         │
│  - Background execution via daemon threads                      │
│  - Immediate research_id return                                 │
│  - Status polling while running                                 │
│  - Cancellation support                                         │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Research Phases                             │
├─────────────────────────────────────────────────────────────────┤
│  PLANNING → GATHERING → ANALYSIS → REFINEMENT → SYNTHESIS       │
│                              ↑           │                       │
│                              └───────────┘                       │
│                         (iterative refinement)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Digest Phase

The **digest phase** runs during ANALYSIS to compress large source documents into structured payloads that preserve key information while reducing token usage.

### Digest Pipeline

1. **Content Extraction**: Raw HTML/text normalized to canonical form
2. **PDF Processing**: Optional PDF text extraction with page boundaries
3. **Quality Ranking**: Sources ranked by quality and relevance
4. **Selection**: Top N sources selected for digestion
5. **Compression**: LLM-powered summarization with key points
6. **Evidence Extraction**: Query-relevant snippets with locators

### DigestPayload Structure

When a source is digested, its `content` field is replaced with a JSON DigestPayload:

```json
{
  "version": "1.0",
  "content_type": "digest/v1",
  "query_hash": "ab12cd34",
  "summary": "Condensed summary of the source...",
  "key_points": [
    "First key insight from the document",
    "Second key insight with supporting detail"
  ],
  "evidence_snippets": [
    {
      "text": "Exact quote from the source document...",
      "locator": "char:1500-1650",
      "relevance_score": 0.85
    }
  ],
  "original_chars": 25000,
  "digest_chars": 2500,
  "compression_ratio": 0.10,
  "source_text_hash": "sha256:abc123..."
}
```

### Digest Policy

The digest policy controls when sources are eligible for compression:

| Policy | Behavior |
|--------|----------|
| `off` | Never digest - all sources pass through unchanged |
| `auto` | **Default**. Digest sources above size threshold with HIGH/MEDIUM quality |
| `always` | Digest all sources with content, regardless of size or quality |

Configure via `deep_research_digest_policy` in config.

### Evidence Locators

Evidence snippets include locators that reference positions in the canonical (normalized) text:

**Text/HTML Format:**
```
char:{start}-{end}
```
Example: `char:1500-1800` means characters 1500-1799 (exclusive end).

**PDF Format:**
```
page:{n}:char:{start}-{end}
```
Example: `page:3:char:200-450` means page 3, characters 200-449.

**Locator Semantics:**
- Start and end are 0-based character positions
- End boundary is exclusive (Python slice semantics)
- Page numbers are 1-based (human-readable)
- Offsets reference canonical text (post-normalization)

**Verification:**
```python
# Locators can be verified against archived content
canonical_text[start:end] == snippet.text
```

### Content Archival

When `deep_research_archive_content=true`, canonical source text is archived:

- **Path**: `~/.foundry-mcp/research_archives/{source_id}/{source_text_hash}.txt`
- **Format**: UTF-8 encoded canonical text
- **Retention**: 30 days default (configurable)
- **Linkage**: `source.metadata["_digest_archive_hash"]` tracks archive

Evidence locators reference offsets in archived canonical text, enabling citation verification.

## Caching

### Digest Cache

Digest results are cached to avoid redundant LLM calls:

**Cache Key Components:**
- Implementation version (e.g., "1.0")
- Source ID
- Content hash (SHA256 of canonical text)
- Query hash (8-char hex of research query)
- Config hash (digest configuration parameters)

**Key Format:**
```
digest:{version}:{source_id}:{content_hash}:{query_hash}:{config_hash}
```

**Cache Behavior:**
- Cache entries are keyed by all factors affecting output
- Changing any component invalidates the cache
- Query-conditioned: different queries produce different digests
- Config-aware: changing config settings invalidates cache

**Cache Size:**
- Default maximum: 100 entries
- Eviction: Half-flush strategy (removes oldest 50% when full)

### Research Memory

Research sessions are persisted for resume and crash recovery:

- **Location**: `~/.foundry-mcp/research/deep_research/`
- **Format**: JSON state files per research_id
- **Crash markers**: `.crash` files with traceback on unhandled exceptions

## Cost-Tiered Model Routing

Deep research uses **cost-tiered model defaults** to route high-volume, low-complexity LLM calls to cheaper models automatically. This reduces cost without meaningful quality loss.

### Role Classification

| Tier | Roles | Default Model | Rationale |
|------|-------|---------------|-----------|
| **Cheap** | `summarization`, `compression` | `2.0-flash` | High call volume, structured output, low reasoning complexity |
| **Standard** | `research`, `report`, `reflection`, `supervision`, `clarification` | Provider default | Requires nuanced reasoning, synthesis, or judgment |

### How It Works

When no explicit model is configured for a role, `resolve_model_for_role()` applies the cost-tier default from `ResearchConfig._COST_TIER_MODEL_DEFAULTS`:

1. **Explicit config** (`deep_research_{role}_model`) takes highest priority
2. **Provider spec model** (e.g., `[cli]claude:haiku` embeds `haiku`) is next
3. **Cost-tier default** applies for summarization/compression if nothing above is set
4. **Provider default** (no model override) is used for all other roles

### Example Configurations

**Default (recommended for most users):**
```toml
[research]
# Summarization and compression automatically use 2.0-flash
# No configuration needed — cost-tier defaults are built in
default_provider = "gemini"
```

**Override cheap roles to a different model:**
```toml
[research]
default_provider = "gemini"
# Use Claude Haiku for summarization instead of Gemini Flash
deep_research_summarization_provider = "[cli]claude:haiku"
# Use a specific model for compression
deep_research_compression_model = "pro"
```

**All roles on the same model (opt out of cost tiering):**
```toml
[research]
default_provider = "[cli]claude:sonnet"
# Force summarization and compression to use the same model as research
deep_research_summarization_model = "sonnet"
deep_research_compression_model = "sonnet"
```

### Cost Impact

Summarization and compression together account for the majority of LLM calls in a deep research session (one call per source fetched, plus one per topic compressed). Routing these to a ~10x cheaper model significantly reduces total cost while the higher-quality model is reserved for reasoning-intensive phases like research, supervision, and synthesis.

## Configuration

### Digest Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `deep_research_digest_policy` | `auto` | Digest eligibility policy (off/auto/always) |
| `deep_research_digest_min_chars` | `10000` | Minimum chars for auto-policy eligibility |
| `deep_research_digest_max_sources` | `8` | Max sources to digest per batch |
| `deep_research_digest_timeout` | `120.0` | Timeout per digest operation (seconds) |
| `deep_research_digest_max_concurrent` | `3` | Max concurrent digest operations |
| `deep_research_digest_include_evidence` | `true` | Include evidence snippets in output |
| `deep_research_digest_evidence_max_chars` | `400` | Max chars per evidence snippet |
| `deep_research_digest_max_evidence_snippets` | `5` | Max evidence snippets per digest |
| `deep_research_digest_fetch_pdfs` | `false` | Fetch and extract PDF content |
| `deep_research_digest_provider` | `null` | Primary LLM provider for digest (uses analysis provider if not set) |
| `deep_research_digest_providers` | `[]` | Fallback providers for digest (tried in order if primary fails) |

### Example Configuration

```toml
[research]
deep_research_digest_policy = "auto"
deep_research_digest_min_chars = 10000
deep_research_digest_max_sources = 8
deep_research_digest_timeout = 120.0
deep_research_digest_include_evidence = true
deep_research_digest_evidence_max_chars = 400
deep_research_digest_max_evidence_snippets = 5
# deep_research_digest_provider = "[cli]gemini:flash"
# deep_research_digest_providers = ["[cli]claude:haiku", "[cli]codex:gpt-4.1-mini"]
```

## Circuit Breaker

The digest system includes a circuit breaker to prevent cascade failures:

**Triggering:**
- Tracks a sliding window of recent operations
- Opens when failure ratio exceeds 70% with ≥5 samples
- Emits `digest.circuit_breaker_triggered` audit event

**Behavior When Open:**
- New digest operations are skipped
- Cache reads still allowed (cached results returned)
- Auto-resets after 60 seconds

**Manual Reset:**
- Call `digestor.reset_circuit_breaker()` at iteration start
- Recommended: reset at each research iteration

## Consuming Digests

Downstream consumers should detect and handle digested sources:

```python
# Check if source contains digest
if source.content_type == "digest/v1":
    # Parse as DigestPayload
    payload = DigestPayload.from_json(source.content)

    # Use summary for context
    context = payload.summary

    # Use key_points for highlights
    for point in payload.key_points:
        print(f"• {point}")

    # Use evidence_snippets for citations
    for ev in payload.evidence_snippets:
        print(f'"{ev.text}" [{ev.locator}]')

    # IMPORTANT: Skip further summarization
    # Content is already compressed
else:
    # Process raw content normally
    content = source.content
```

## Observability

### Audit Events

| Event | Description |
|-------|-------------|
| `digest.started` | Digest operation initiated for source |
| `digest.completed` | Digest successfully generated |
| `digest.skipped` | Source skipped (ineligible or policy) |
| `digest.error` | Digest operation failed |
| `digest.circuit_breaker_triggered` | Circuit breaker opened |
| `digest.pdf_extract_error` | PDF extraction failed |

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `digest_sources_processed` | Counter | Total sources processed by outcome |
| `digest_cache_hits` | Counter | Cache hit count |
| `digest_duration_seconds` | Histogram | Digest operation duration |
| `digest_compression_ratio` | Histogram | Compression ratio achieved |
| `digest_evidence_snippets` | Histogram | Evidence snippets per digest |

## Fidelity Tracking

The digest phase records fidelity metadata for each source:

```python
fidelity_record = {
    "source_id": "src-abc123",
    "phase": "digest",
    "original_tokens": 6250,  # original_chars / 4
    "final_tokens": 625,       # digest_chars / 4
    "reason": "digest_compression"
}
```

This enables tracking compression impact on source fidelity throughout the research pipeline.
