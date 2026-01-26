# Tavily Configuration Examples

This guide demonstrates how to configure and use the enhanced Tavily search and extract features in deep research workflows.

## Basic Configuration

### Minimal Setup

```toml
[research]
# Just need API key - all other settings use sensible defaults
# Set via environment: export TAVILY_API_KEY="tvly-..."
```

### Standard Configuration

```toml
[research]
# Search parameters
tavily_search_depth = "basic"      # 1x credits
tavily_topic = "general"
tavily_include_images = false

# Extract disabled by default
tavily_extract_in_deep_research = false
```

## Search Depth Examples

### Basic Search (Default)
Standard web search with snippet extraction.

```toml
[research]
tavily_search_depth = "basic"  # 1x credits
```

### Advanced Search
Deeper analysis with raw content, chunks, and more comprehensive results.

```toml
[research]
tavily_search_depth = "advanced"   # 2x credits
tavily_chunks_per_source = 5       # More content chunks (1-5)
```

**When to use advanced:**
- Academic or technical research requiring full article content
- Complex topics needing deeper source analysis
- When `deep_research_mode = "academic"` or `"technical"`

### Fast/Ultra-Fast Search
Reduced latency for quick lookups.

```toml
[research]
tavily_search_depth = "fast"       # Faster response
# OR
tavily_search_depth = "ultra_fast" # Minimal latency
```

## News Search Configuration

Search recent news articles on a topic.

```toml
[research]
tavily_topic = "news"
tavily_news_days = 7               # Last 7 days (1-365)
tavily_country = "US"              # Boost US news sources
```

### Example MCP Call

```json
{
  "action": "deep-research",
  "query": "Latest developments in quantum computing",
  "max_iterations": 2
}
```

With config above, this will search news from the last 7 days, prioritizing US sources.

## Geographic Targeting

Boost results from a specific country.

```toml
[research]
tavily_country = "DE"  # ISO 3166-1 alpha-2 code
```

Common country codes: `US`, `GB`, `DE`, `FR`, `JP`, `AU`, `CA`

## Extract Integration

Enable URL content extraction as a follow-up step in deep research.

### Basic Extract

```toml
[research]
tavily_extract_in_deep_research = true
tavily_extract_max_urls = 5        # Extract top 5 URLs per run
tavily_extract_depth = "basic"
```

### Advanced Extract

```toml
[research]
tavily_extract_in_deep_research = true
tavily_extract_max_urls = 10
tavily_extract_depth = "advanced"  # More comprehensive extraction
tavily_extract_include_images = true
```

### Standalone Extract Action

Use extract independently via MCP:

```json
{
  "action": "extract",
  "urls": [
    "https://arxiv.org/abs/2401.12345",
    "https://docs.example.com/api-reference"
  ],
  "extract_depth": "advanced",
  "include_images": false,
  "format": "markdown"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "sources": [
      {
        "url": "https://arxiv.org/abs/2401.12345",
        "title": "Paper Title",
        "content": "Full extracted content in markdown...",
        "snippet": "First 500 chars..."
      }
    ],
    "failed_urls": [],
    "partial_success": false
  }
}
```

## Research Mode Smart Defaults

The `deep_research_mode` setting automatically adjusts Tavily parameters.

### General Mode (Default)

```toml
[research]
deep_research_mode = "general"
# Tavily uses: search_depth="basic", no domain preferences
```

### Academic Mode

```toml
[research]
deep_research_mode = "academic"
# Tavily uses: search_depth="advanced" (auto-upgraded)
# Prioritizes: journals, publishers, preprints, .edu domains
```

### Technical Mode

```toml
[research]
deep_research_mode = "technical"
# Tavily uses: search_depth="advanced" (auto-upgraded)
# Prioritizes: official docs, arxiv, Stack Overflow, GitHub
```

## Complete Configuration Example

Full configuration for technical research with extract follow-up:

```toml
[research]
# Enable research tools
enabled = true

# Deep research settings
deep_research_mode = "technical"
deep_research_max_iterations = 3
deep_research_providers = ["tavily", "semantic_scholar"]

# Tavily search configuration
tavily_search_depth = "advanced"   # Will be auto-upgraded anyway for technical mode
tavily_topic = "general"
tavily_chunks_per_source = 4
tavily_auto_parameters = false     # We want explicit control

# Tavily extract configuration
tavily_extract_in_deep_research = true
tavily_extract_max_urls = 8
tavily_extract_depth = "advanced"
tavily_extract_include_images = false

# Rate limiting
[research.per_provider_rate_limits]
tavily = 60  # requests per minute
```

## CLI Usage

### Start Deep Research with Tavily

```bash
# Basic research (uses config file settings)
foundry research deep-research \
  --query "Transformer architectures for computer vision"

# Override mode for this run
foundry research deep-research \
  --query "Latest AI safety research" \
  --mode academic
```

### Check Status

```bash
foundry research deep-research-status --research-id deepres-abc123
```

### Get Report

```bash
foundry research deep-research-report --research-id deepres-abc123
```

## Credit Cost Optimization

| Depth | Cost | Best For |
|-------|------|----------|
| `basic` | 1x | General searches, quick lookups |
| `advanced` | 2x | In-depth research, academic work |
| `fast` | 1x | Time-sensitive queries |
| `ultra_fast` | 1x | Real-time applications |

**Tips:**
- Use `basic` for initial exploration, `advanced` for targeted deep dives
- Set `tavily_auto_parameters = true` to let Tavily optimize based on query
- Academic/technical modes auto-upgrade to `advanced` when beneficial

## Security Notes

The Tavily Extract provider includes SSRF protection:
- Blocks localhost, private IPs (10.x, 172.x, 192.168.x)
- Blocks dangerous schemes (file://, gopher://, data://)
- Validates URLs before extraction
- Max URL length: 2048 characters

Blocked URL patterns will return errors with `BLOCKED_HOST` error code.
