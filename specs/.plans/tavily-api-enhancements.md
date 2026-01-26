# Tavily API Enhancements

## Mission

Expose full Tavily API capabilities to power users by implementing missing search parameters, adding the Extract endpoint, and integrating smart defaults based on research mode.

## Objective

Enhance the Tavily integration in foundry-mcp to leverage the full power of Tavily's Search and Extract APIs, giving power users fine-grained control over search behavior while providing intelligent defaults based on research context.

## Success Criteria

- [ ] All 7 missing Tavily search parameters are supported
- [ ] New TavilyExtractProvider implemented for URL content extraction
- [ ] Configuration fields added to ResearchConfig with TOML support
- [ ] `search_depth` defaults to "basic" (current behavior); "advanced" opt-in via config
- [ ] Research mode (academic/technical/general) influences Tavily settings
- [ ] Response envelope and error semantics follow MCP best practices
- [ ] Unit tests cover new parameters and extract provider
- [ ] Integration test verifies end-to-end deep research with new settings
- [ ] Extract provider exposed via research tool with clear invocation path

## Scope

### In Scope
- Add missing search parameters: `topic`, `days`, `include_images`, `include_favicon`, `country`, `chunks_per_source`, `auto_parameters`
- Implement Tavily Extract endpoint (`/extract`) as new provider
- Add 9 Tavily configuration fields to ResearchConfig
- Update TOML parsing for new config fields
- Integrate smart defaults based on `deep_research_mode`
- Pass Tavily kwargs from deep research workflow
- Unit tests for search parameters and extract provider
- Integration test for deep research with Tavily config

### Out of Scope
- Tavily Crawl/Map endpoints (future enhancement)
- UI/CLI for Tavily settings (config-only for now)
- Billing/credit tracking integration
- Other search providers (Google, Perplexity, Semantic Scholar)

## Background Research

### Tavily API Contract Reference

**Source**: https://docs.tavily.com/documentation/api-reference/endpoint/search

#### Search Endpoint (`POST /search`)

| Parameter | Type | Allowed Values | Default | Description |
|-----------|------|----------------|---------|-------------|
| `api_key` | string | Valid API key | Required | Tavily API key |
| `query` | string | Any text | Required | Search query (max 400 chars) |
| `max_results` | int | 1-20 | 10 | Maximum results to return |
| `search_depth` | string | `"basic"`, `"advanced"`, `"fast"`, `"ultra_fast"` | `"basic"` | Search depth (advanced=2x credits) |
| `include_answer` | bool/string | `true`, `false`, `"basic"`, `"advanced"` | `false` | Include AI answer |
| `include_raw_content` | bool/string | `true`, `false`, `"markdown"`, `"text"` | `false` | Include page content |
| `include_images` | bool | `true`, `false` | `false` | Include image results |
| `include_favicon` | bool | `true`, `false` | `false` | Include favicon URLs |
| `include_domains` | list[str] | URLs | `[]` | Limit to domains (max 300) |
| `exclude_domains` | list[str] | URLs | `[]` | Exclude domains (max 150) |
| `topic` | string | `"general"`, `"news"` | `"general"` | Search topic |
| `days` | int | 1-365 | None | Days limit (news topic only) |
| `country` | string | ISO 3166-1 alpha-2 | None | Boost country results |
| `chunks_per_source` | int | 1-5 | 3 | Chunks per source (advanced only) |
| `auto_parameters` | bool | `true`, `false` | `false` | Auto-configure based on query |

#### Extract Endpoint (`POST /extract`)

**Source**: https://docs.tavily.com/documentation/api-reference/endpoint/extract

| Parameter | Type | Allowed Values | Default | Description |
|-----------|------|----------------|---------|-------------|
| `api_key` | string | Valid API key | Required | Tavily API key |
| `urls` | list[str] | Valid HTTP(S) URLs | Required | URLs to extract (max 20) |
| `extract_depth` | string | `"basic"`, `"advanced"` | `"basic"` | Extraction depth |
| `include_images` | bool | `true`, `false` | `false` | Include images |
| `format` | string | `"markdown"`, `"text"` | `"markdown"` | Output format |
| `query` | string | Any text | None | Rerank chunks by relevance |
| `chunks_per_source` | int | 1-5 | 3 | Max chunks per URL |

#### API Limits & Credits

| Operation | Credits | Rate Limit |
|-----------|---------|------------|
| Search (basic) | 1 | ~1 req/sec |
| Search (advanced) | 2 | ~1 req/sec |
| Extract (basic) | 1 per 5 URLs | ~1 req/sec |
| Extract (advanced) | 2 per 5 URLs | ~1 req/sec |

### Current Implementation Files

- `src/foundry_mcp/core/research/providers/tavily.py` - Search provider (384 lines)
- `src/foundry_mcp/config.py` - ResearchConfig class (lines 402-812)
- `src/foundry_mcp/core/research/workflows/deep_research.py` - Consumer (lines 2488-2493)
- `src/foundry_mcp/core/research/providers/__init__.py` - Exports

### Key Finding

The deep research workflow **never passes `search_depth`** to Tavily - it always defaults to "basic", missing the benefit of "advanced" search which provides better relevance (at 2x API credits).

## Phases

### Phase 1: Enhanced Search Parameters

**Purpose**: Add all missing Tavily search parameters to the existing provider

**Tasks**:
1. Update `search()` method signature to document new kwargs
2. Extract new parameters in search method: `topic`, `days`, `include_images`, `include_favicon`, `country`, `chunks_per_source`, `auto_parameters`
3. Update payload construction to conditionally include new parameters
4. Handle `include_raw_content` as string ("markdown"/"text") or bool
5. Add parameter validation (e.g., `chunks_per_source` must be 1-5)

**Files**:
- `src/foundry_mcp/core/research/providers/tavily.py`

**Verification**:
- Unit tests pass for each new parameter
- Payload correctly includes parameters when set

### Phase 2: Tavily Extract Provider

**Purpose**: Implement the Extract endpoint for direct URL content extraction

**Tasks**:
1. Create `TavilyExtractProvider` class following existing provider patterns
2. Implement `extract(urls, **kwargs)` method with all extract params
3. Add retry logic matching `TavilySearchProvider`
4. Create response parsing for extract results
5. Add health check method
6. Export from `providers/__init__.py`

**Files**:
- `src/foundry_mcp/core/research/providers/tavily_extract.py` (NEW)
- `src/foundry_mcp/core/research/providers/__init__.py`

**Verification**:
- Unit tests for extract method
- Mock API responses handled correctly
- Retry logic works for rate limits

### Phase 3: Configuration Support

**Purpose**: Add configurable Tavily settings to ResearchConfig

**Tasks**:
1. Add Tavily search config fields to `ResearchConfig` dataclass:
   - `tavily_search_depth: str = "basic"` (preserve current default; advanced opt-in)
   - `tavily_topic: str = "general"`
   - `tavily_news_days: Optional[int] = None`
   - `tavily_include_images: bool = False`
   - `tavily_country: Optional[str] = None`
   - `tavily_chunks_per_source: int = 3`
   - `tavily_auto_parameters: bool = False`
2. Add Tavily extract config fields:
   - `tavily_extract_depth: str = "basic"`
   - `tavily_extract_include_images: bool = False`
3. Update `from_toml_dict()` to parse new fields
4. Add validation for all fields:
   - `search_depth`: must be one of ["basic", "advanced", "fast", "ultra_fast"]
   - `topic`: must be one of ["general", "news"]
   - `chunks_per_source`: must be 1-5
   - `country`: must be valid ISO 3166-1 alpha-2 code or None
   - `days`: must be positive integer or None
   - `include_raw_content`: accept bool or string ("markdown", "text")
5. Document parameter precedence: explicit kwargs > config values > research-mode defaults > auto_parameters

**Files**:
- `src/foundry_mcp/config.py`

**Verification**:
- Config loads from TOML correctly
- Default values applied when not specified
- Invalid enum values raise validation errors
- Validation tests for all bounds and types

### Phase 4: Deep Research Integration

**Purpose**: Pass Tavily config to the search provider with research mode smart defaults

**Tasks**:
1. Add `_get_tavily_search_kwargs()` method to compute settings based on:
   - Config values as base
   - `deep_research_mode` overrides (academic/technical/general)
   - News topic handling with days
2. Update `provider.search()` call (line ~2488) to pass Tavily kwargs
3. Document research mode behavior in method docstring

**Files**:
- `src/foundry_mcp/core/research/workflows/deep_research.py`

**Verification**:
- Academic mode uses higher `chunks_per_source`
- Technical mode ensures `search_depth="advanced"`
- News topic includes `days` when configured

### Phase 5: Extract Integration

**Purpose**: Expose Extract provider via research tool for user invocation

**Tasks**:
1. Add `extract` action to research router/workflow
2. Define input schema: `urls` (required), optional params
3. Define output mapping: ExtractResult -> ResearchSource conversion
4. Wire to MCP tool registry for direct invocation
5. Add to deep research workflow as optional follow-up step for URL expansion

**Files**:
- `src/foundry_mcp/core/research/workflows/deep_research.py`
- `src/foundry_mcp/routers/research.py` (or appropriate router)

**Verification**:
- Extract can be invoked via `mcp__foundry-mcp__research action="extract"`
- Output conforms to response envelope schema
- Integration test covers extract -> source conversion

### Phase 6: Testing & Fixtures

**Purpose**: Ensure comprehensive test coverage for all changes

**Tasks**:
1. Add unit tests for new search parameters in `test_tavily.py`:
   - Test each parameter is included in payload when set
   - Test parameter validation (bounds, enums)
   - Test default values preserved
   - Test invalid value rejection
2. Create `test_tavily_extract.py` for extract provider:
   - Test extract method with various kwargs
   - Test retry logic with mock 429 responses
   - Test response parsing and ResearchSource mapping
   - Test error handling (auth, rate limit, network)
3. Add config tests for new Tavily fields:
   - Test TOML parsing
   - Test validation errors for invalid values
   - Test precedence rules
4. Add integration test for deep research with Tavily config:
   - Test research-mode smart defaults
   - Test config override behavior
5. Create/update fixtures:
   - Mock Tavily search response with new fields
   - Mock Tavily extract response
   - Regenerate affected fixtures and run freshness checks

**Files**:
- `tests/unit/core/research/providers/test_tavily.py`
- `tests/unit/core/research/providers/test_tavily_extract.py` (NEW)
- `tests/unit/test_config.py`
- `tests/integration/test_deep_research_tavily.py` (NEW or extend existing)
- `tests/fixtures/` (as needed)

**Verification**:
- All tests pass: `pytest tests/unit/core/research/providers/test_tavily*.py -v`
- Integration tests pass: `pytest tests/integration/ -k tavily -v`
- Coverage report shows new code paths covered

### Phase 7: Documentation & Specs

**Purpose**: Update documentation and spec artifacts per repo standards

**Tasks**:
1. Update CHANGELOG.md with new Tavily features
2. Update configuration documentation with new fields
3. Add usage examples for new parameters
4. Document credit cost implications (advanced = 2x)
5. Update API reference if applicable

**Files**:
- `CHANGELOG.md`
- `docs/configuration.md` (or equivalent)
- `README.md` (if user-facing features section exists)

**Verification**:
- Documentation builds without errors
- Examples are accurate and testable

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API credits increase (advanced = 2x) | Medium | Document cost; keep "basic" as default; opt-in for advanced |
| Breaking existing behavior | High | Default values preserve current behavior |
| Tavily API changes | Low | Use kwargs pattern for forward compatibility |
| Extract endpoint rate limits | Medium | Reuse retry logic; respect rate limits |
| Large extract payloads | Medium | Enforce `chunks_per_source` cap (1-5); truncate if needed |

## Technical Specifications

### Research Mode Defaults Matrix

| Parameter | general (default) | academic | technical |
|-----------|-------------------|----------|-----------|
| `search_depth` | "basic" | "advanced" | "advanced" |
| `chunks_per_source` | 3 | 5 | 4 |
| `include_raw_content` | False | "markdown" | "markdown" |
| `topic` | "general" | "general" | "general" |
| `include_images` | False | False | False |

Users can override any mode default via explicit config or kwargs.

### Extract Tool Contract

**Action**: `mcp__foundry-mcp__research action="extract"`

**Input Schema**:
```python
{
    "action": "extract",              # Required
    "urls": ["url1", "url2", ...],    # Required, list of URLs (max 10)
    "extract_depth": "basic",         # Optional, "basic" | "advanced"
    "include_images": false,          # Optional, bool
    "format": "markdown",             # Optional, "markdown" | "text"
    "query": "optional relevance query",  # Optional, for chunk reranking
    "chunks_per_source": 3            # Optional, 1-5
}
```

**URL Security Validation** (SSRF Protection):
```python
# All URLs MUST pass validation before extraction
def validate_extract_url(url: str) -> bool:
    """Validate URL for safe extraction."""
    parsed = urlparse(url)

    # 1. Scheme validation: only http/https allowed
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid scheme: {parsed.scheme}. Only http/https allowed.")

    # 2. Host validation: block private/internal networks
    BLOCKED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
    BLOCKED_PATTERNS = [
        r"^10\.",           # 10.0.0.0/8
        r"^172\.(1[6-9]|2[0-9]|3[0-1])\.",  # 172.16.0.0/12
        r"^192\.168\.",     # 192.168.0.0/16
        r"^169\.254\.",     # Link-local
        r"\.local$",        # mDNS
        r"\.internal$",     # Internal domains
    ]
    if parsed.hostname in BLOCKED_HOSTS:
        raise ValueError(f"Blocked host: {parsed.hostname}")
    for pattern in BLOCKED_PATTERNS:
        if re.match(pattern, parsed.hostname or ""):
            raise ValueError(f"Blocked private network: {parsed.hostname}")

    # 3. URL length limit
    if len(url) > 2048:
        raise ValueError(f"URL too long: {len(url)} chars (max 2048)")

    return True
```

**Request Constraints**:
- Max URLs per request: 10
- Max redirects followed: 3
- Request timeout: 30s per URL
- Max response payload: 5MB per URL
- Concurrency: 3 parallel extractions max

**Output Mapping Strategy**: One `ResearchSource` per URL (aggregated chunks)

```python
# Each extracted URL -> one ResearchSource
ResearchSource(
    url=extracted_url,
    title=extracted_title or domain_name,  # max 500 chars, truncated
    snippet=first_chunk[:500],              # First chunk as snippet
    content="\n\n".join(all_chunks)[:50000], # Aggregated, max 50KB
    source_type=SourceType.WEB,
    metadata={
        "extract_depth": str,       # "basic" | "advanced"
        "chunk_count": int,         # Number of chunks extracted
        "format": str,              # "markdown" | "text"
        "images": Optional[List[str]],  # URLs, max 10 images
        "favicon": Optional[str],   # Single URL or None
        "truncated": bool,          # True if content was truncated
    }
)
```

**Response Envelope** (Canonical MCP Schema - response-v2):

Uses `success_response()` / `error_response()` helpers from `foundry_mcp.core.responses`.

**Full Success** (all URLs extracted):
```python
success_response(
    data={
        "action": "extract",
        "sources": [ResearchSource, ...],
        "stats": {
            "requested": int,
            "succeeded": int,
            "failed": 0,
            "total_chunks": int,
        }
    },
    request_id=context.request_id,
)
# Result: success=True, error=None, meta.version="response-v2"
```

**Partial Success** (some URLs failed - per MCP "Blocked or partial work" pattern):
```python
success_response(
    data={
        "action": "extract",
        "sources": [ResearchSource, ...],  # Successfully extracted
        "stats": {
            "requested": int,
            "succeeded": int,
            "failed": int,
            "total_chunks": int,
        },
        "failed_urls": [  # URLs that failed
            {"url": str, "error_code": str, "error": str}
        ],
    },
    warnings=[f"Failed to extract {n} of {total} URLs"],  # Goes to meta.warnings
    request_id=context.request_id,
)
# Result: success=True, meta.warnings populated, data.failed_urls contains details
```

**Total Failure** (all URLs failed or validation error):
```python
error_response(
    message="Extract failed: all URLs blocked or invalid",
    error_code="EXTRACT_FAILED",
    error_type="validation",  # or "internal" for server errors
    remediation="Check URLs are valid public HTTP(S) addresses",
    details={
        "failed_urls": [{"url": str, "error_code": str, "error": str}]
    },
    request_id=context.request_id,
)
# Result: success=False, data.error_code/error_type/remediation/details populated
```

**Error Codes** (aligned with MCP standard codes):

| Code | error_type | Description |
|------|------------|-------------|
| `VALIDATION_ERROR` | validation | Generic URL validation failure |
| `INVALID_URL` | validation | URL failed scheme/length validation |
| `BLOCKED_HOST` | validation | Private/internal network blocked (SSRF) |
| `RATE_LIMIT_EXCEEDED` | rate_limit | Tavily rate limit exceeded |
| `TIMEOUT` | unavailable | Request timed out |
| `EXTRACT_FAILED` | internal | Tavily could not extract content |
| `PAYLOAD_TOO_LARGE` | validation | Response exceeded 5MB limit |

### Response Envelope Contract

All Tavily provider methods MUST return responses conforming to the MCP response schema:

```python
# Search results -> ResearchSource mapping (existing pattern)
ResearchSource(
    url=result.get("url"),
    title=result.get("title"),
    snippet=result.get("content"),
    content=result.get("raw_content"),
    source_type=SourceType.WEB,
    metadata={"tavily_score": result.get("score"), ...}
)

# Extract results -> similar mapping
ResearchSource(
    url=extracted_url,
    title=extracted_title,
    content=extracted_content,  # markdown or text based on format
    source_type=SourceType.WEB,
    metadata={"extract_depth": depth, "chunks": chunks, ...}
)
```

### Error Semantics

Follow existing provider error hierarchy:
- `AuthenticationError`: 401 responses (invalid API key) - not retryable
- `RateLimitError`: 429 responses - retryable with backoff
- `SearchProviderError`: Other errors - retryable for 5xx, not for 4xx

### Retry & Resilience Strategy

Reuse existing retry logic from `TavilySearchProvider._execute_with_retry()`:
- Max retries: 3 (configurable via constructor)
- Backoff: Exponential (2^attempt seconds)
- Retryable: 429 (rate limit), 5xx (server errors), timeout, network errors
- Not retryable: 401 (auth), 4xx (client errors)
- Rate limit: Parse `Retry-After` header when available
- Timeout: 30s default (configurable)

### Parameter Precedence Matrix

When building Tavily request kwargs, apply in this order (highest to lowest priority):

| Priority | Source | Example | Wins Over |
|----------|--------|---------|-----------|
| 1 (highest) | Explicit kwargs | `provider.search(..., search_depth="fast")` | All below |
| 2 | ResearchConfig | `tavily_search_depth = "advanced"` in TOML | Mode defaults, auto |
| 3 | Research-mode defaults | `academic` mode → `search_depth="advanced"` | auto_parameters |
| 4 (lowest) | Tavily auto_parameters | Tavily API decides based on query | Nothing |

**Resolution Algorithm**:
```python
def build_tavily_kwargs(explicit_kwargs, config, research_mode):
    # Start with mode defaults
    kwargs = MODE_DEFAULTS[research_mode].copy()

    # Override with config values (if explicitly set, not None)
    for key in TAVILY_CONFIG_KEYS:
        config_val = getattr(config, f"tavily_{key}", None)
        if config_val is not None:
            kwargs[key] = config_val

    # Override with explicit kwargs (highest priority)
    kwargs.update({k: v for k, v in explicit_kwargs.items() if v is not None})

    # auto_parameters only affects unset fields (handled by Tavily API)
    if config.tavily_auto_parameters:
        kwargs["auto_parameters"] = True

    return kwargs
```

**Example Scenarios**:

| Scenario | Mode | Config | Explicit | Result |
|----------|------|--------|----------|--------|
| Default academic | academic | (none) | (none) | `search_depth="advanced"` |
| Config overrides mode | academic | `search_depth="basic"` | (none) | `search_depth="basic"` |
| Explicit overrides all | academic | `search_depth="basic"` | `search_depth="fast"` | `search_depth="fast"` |

### include_raw_content Mapping

The `include_raw_content` parameter accepts multiple types with canonical mapping:

| Input Value | Mapped To | Behavior |
|-------------|-----------|----------|
| `True` | `"markdown"` | Returns content as markdown (default for bool true) |
| `False` | `false` | No raw content returned |
| `"markdown"` | `"markdown"` | Returns content as markdown |
| `"text"` | `"text"` | Returns content as plain text |

**Normalization**:
```python
def normalize_include_raw_content(value: Union[bool, str]) -> Union[bool, str]:
    if value is True:
        return "markdown"  # Default mapping for True
    if value is False:
        return False
    if value in ("markdown", "text"):
        return value
    raise ValueError(f"Invalid include_raw_content: {value}. Use bool or 'markdown'/'text'.")
```

### Health Check Contract

Both Search and Extract providers implement `health_check() -> bool`:
- Perform minimal API call to verify connectivity
- Return `True` if API responds successfully
- Return `False` on auth errors, log warning
- Catch and log other exceptions, return `False`

## Dependencies

- `httpx` (already used)
- Tavily API key in environment (`TAVILY_API_KEY`)
- No new external dependencies required

### Validation & Compatibility

**Validation Rules** (raise `ValueError` on violation):
- `search_depth`: must be in `["basic", "advanced", "fast", "ultra_fast"]`
- `topic`: must be in `["general", "news"]`
- `chunks_per_source`: must be integer 1-5
- `days`: must be positive integer if set
- `country`: must match `/^[A-Z]{2}$/` (ISO 3166-1 alpha-2) if set
- `include_raw_content`: accept `bool`, `"markdown"`, or `"text"`

**Migration Policy**:
- Invalid config values: Raise error at config load time with clear message
- New fields with defaults: Old configs work unchanged (fields default to current behavior)
- No silent fallbacks: Invalid values fail loudly rather than coercing

**Testing Strategy**:
- Unit tests: Mock `httpx` responses; no live API calls
- Integration tests: Use recorded fixtures with replay mode
- Smoke tests (optional, CI-skippable): Live API call to verify connectivity

### Auto-Parameters Interaction

When `tavily_auto_parameters=True`:
1. Tavily API auto-configures parameters based on query intent
2. Explicit config values **override** auto-parameters
3. Research-mode defaults are applied **before** sending to Tavily
4. User kwargs have highest precedence

Effective precedence: `explicit kwargs > config values > mode defaults > auto_parameters`

## Assumptions

1. Tavily API endpoints remain stable
2. Users have valid Tavily API keys configured
3. Default `search_depth="basic"` preserves current behavior (advanced opt-in)
4. Extract endpoint uses same API key as search
5. Extract endpoint has same rate limits as search
