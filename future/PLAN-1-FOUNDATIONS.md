# PLAN-1: Foundations — Profiles, Provenance & Academic Output

> **Goal**: Establish the structural foundations that all subsequent plans build on — research profiles (replacing the monolithic `research_mode`), a provenance audit trail (addressing reproducibility), and the core academic output improvements (literature review synthesis, APA citations, structured output).
>
> **Estimated scope**: ~1000-1400 LOC implementation + ~400-550 LOC tests across 10-14 files
>
> **Dependencies**: PLAN-0 (ResearchExtensions container model, supervision.py refactoring)
>
> **Note on state model**: All new state fields (`research_profile`, `provenance`, `structured_output`) are stored on `state.extensions` (introduced in PLAN-0 item 2), not directly on `DeepResearchState`. Convenience property accessors on the state object keep downstream code clean.

---

## Design Principles

1. **Composability over completeness.** Deep research should produce outputs that downstream tools (Zotero MCP, Scite MCP, visualization tools) can consume — not just human-readable markdown.
2. **Profiles over modes.** Instead of a single `research_mode` enum, research profiles compose capabilities declaratively. Named profiles provide sensible defaults; per-request overrides provide flexibility.
3. **Provenance by default.** Every research session should produce a machine-readable audit trail. This isn't optional — it's what makes the MCP approach reproducible.
4. **No regression.** All changes must leave GENERAL-mode behavior completely unchanged.

---

## 1. Research Profiles

### Problem

`research_mode` (`ResearchMode` enum in `models/sources.py`) is a monolithic switch with three values: `GENERAL`, `ACADEMIC`, `TECHNICAL`. This forces all academic research through identical settings — a systematic review in clinical medicine gets the same configuration as a theoretical literature review in philosophy or a bibliometric analysis.

The existing enum also only controls source quality heuristics (domain tier scoring in `DOMAIN_TIERS`). It doesn't influence provider selection, synthesis templates, citation style, or tool availability — those are all hardcoded or globally configured.

### Design

A **ResearchProfile** is a named bundle of settings that configures the deep research pipeline per-session:

```python
class ResearchProfile(BaseModel):
    """Composable research configuration applied per-session."""

    name: str = "general"

    # Provider chain (ordered by preference)
    providers: list[str] = Field(
        default_factory=lambda: ["tavily", "semantic_scholar"],
        description="Search providers to use, in priority order",
    )

    # Source quality mode (existing ResearchMode behavior)
    source_quality_mode: ResearchMode = ResearchMode.GENERAL

    # Citation & output
    citation_style: str = "default"          # "default" | "apa" | "ieee" | "chicago"
    export_formats: list[str] = Field(default_factory=lambda: ["bibtex"])
    synthesis_template: Optional[str] = None  # None = auto-detect | "literature_review" | ...

    # Academic tools (citation graph, related papers, etc.)
    enable_citation_tools: bool = False
    enable_methodology_assessment: bool = False
    enable_citation_network: bool = False
    enable_pdf_extraction: bool = False

    # Source preferences (injected into brief phase)
    source_type_hierarchy: Optional[list[str]] = None  # e.g. ["peer_reviewed", "meta_analysis", ...]
    disciplinary_scope: Optional[list[str]] = None     # e.g. ["education", "psychology"]
    time_period: Optional[str] = None                  # e.g. "last_10_years"
    methodology_preferences: Optional[list[str]] = None # e.g. ["rct", "meta_analysis"]
```

**Built-in profiles** (registered at startup, overridable via config):

| Profile Name | Providers | Citation Style | Tools | Notes |
|-------------|-----------|---------------|-------|-------|
| `general` | tavily, semantic_scholar | default | none | Current default behavior |
| `academic` | semantic_scholar, openalex, tavily | apa | citation_tools | General academic research |
| `systematic-review` | semantic_scholar, openalex, pubmed, tavily | apa | citation_tools, methodology_assessment, pdf_extraction | Rigorous lit review |
| `bibliometric` | openalex, semantic_scholar | apa | citation_tools, citation_network | Citation analysis focus |
| `technical` | tavily, google | default | none | Technical docs / code |

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 1a. Add ResearchProfile model

Add `ResearchProfile` as defined above, after the existing `DeepResearchPhase` enum.

#### 1b. Add profile field to ResearchExtensions

```python
# In ResearchExtensions (PLAN-0 item 2):
research_profile: Optional[ResearchProfile] = None

# Convenience accessor on DeepResearchState:
@property
def research_profile(self) -> ResearchProfile:
    return self.extensions.research_profile or ResearchProfile()
```

**File: `src/foundry_mcp/config/research.py`**

#### 1c. Add profile registry to config

```python
# New config fields:
deep_research_profiles: dict[str, dict] = Field(
    default_factory=dict,
    description="Custom research profiles (name -> profile config dict)",
)
deep_research_default_profile: str = "general"
```

#### 1d. Profile resolution logic

Add `resolve_profile()` function that:
1. If request specifies `profile="academic"`, look up built-in profile
2. If request specifies `profile="my-custom-profile"`, look up in config
3. If request specifies `research_mode="academic"` (legacy), map to built-in profile for backward compat
4. If neither specified, use `deep_research_default_profile` from config
5. Per-request overrides (e.g. `citation_style="ieee"`) are applied on top of the resolved profile

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 1e. Add profile parameter to deep-research handler

```python
def _handle_deep_research(
    *,
    query: str,
    research_mode: Optional[str] = None,   # LEGACY — maps to profile
    research_profile: Optional[str] = None, # NEW — profile name
    profile_overrides: Optional[dict] = None, # NEW — per-request overrides
    ...
) -> dict:
```

Resolution order: `research_profile` > `research_mode` (legacy mapping) > config default.

**File: `src/foundry_mcp/tools/unified/research.py`**

#### 1f. Pass profile through action router

Ensure `research_profile` and `profile_overrides` flow from MCP tool arguments through to the handler.

### Backward Compatibility

- `research_mode="academic"` still works — it maps to the `academic` built-in profile
- `research_mode="general"` maps to `general` profile
- `research_mode="technical"` maps to `technical` profile
- If both `research_mode` and `research_profile` are specified, `research_profile` wins and a deprecation warning is emitted
- Omitting both uses the config default (which defaults to `general`)

### Testing

- Unit test: profile resolution with built-in names
- Unit test: profile resolution with legacy `research_mode` mapping
- Unit test: per-request overrides applied on top of profile
- Unit test: unknown profile name raises validation error
- Unit test: backward compat — `research_mode="academic"` produces same behavior as `research_profile="academic"`
- Unit test: config-defined custom profiles load correctly

---

## 2. Research Provenance Audit Trail

### Problem

Every deep research session's methodology is opaque. When a literature review misses an obvious paper, there's no way to diagnose why — was it not in the provider's index? Was it found but scored low? Was the sub-query decomposition too narrow? Did the coverage assessment prematurely signal completion?

This is also the strongest criticism of the MCP-based approach to academic research: every user's environment is different, making results unreproducible. A provenance log addresses this by making the methodology inspectable.

### Design

A **ProvenanceLog** is a structured, append-only record of every decision and action taken during a deep research session. It is:
- **Automatic**: populated by the pipeline, not by the user
- **Machine-readable**: JSON-serializable for programmatic analysis
- **Human-inspectable**: includes textual summaries alongside structured data
- **Persisted with the session**: stored alongside `DeepResearchState`

```python
class ProvenanceEntry(BaseModel):
    """Single entry in the provenance log."""
    timestamp: str                    # ISO 8601
    phase: str                        # brief | supervision | synthesis
    event_type: str                   # See event types below
    summary: str                      # Human-readable summary
    details: dict = Field(default_factory=dict)  # Structured event data

class ProvenanceLog(BaseModel):
    """Complete provenance record for a research session."""
    session_id: str
    query: str
    profile: str                      # Profile name used
    profile_config: dict              # Full profile settings (frozen at session start)
    started_at: str
    completed_at: Optional[str] = None
    entries: list[ProvenanceEntry] = Field(default_factory=list)

    def append(self, phase: str, event_type: str, summary: str, **details) -> None:
        """Append a provenance entry with current timestamp."""
```

**Event types**:

| Event Type | Phase | Details |
|-----------|-------|---------|
| `brief_generated` | brief | `{brief_text, scope_boundaries, source_preferences}` |
| `decomposition` | supervision | `{directives: [{topic, perspective, priority}], rationale}` |
| `provider_query` | supervision | `{provider, query, result_count, source_ids}` |
| `source_discovered` | supervision | `{source_id, title, provider, source_type, url}` |
| `source_deduplicated` | supervision | `{source_id, duplicate_of, reason}` |
| `coverage_assessment` | supervision | `{scores: {source_adequacy, domain_diversity, ...}, decision}` |
| `gap_identified` | supervision | `{gap_id, description, priority, suggested_queries}` |
| `gap_resolved` | supervision | `{gap_id, resolution_notes}` |
| `iteration_complete` | supervision | `{iteration, round, total_sources, total_findings}` |
| `synthesis_query_type` | synthesis | `{query_type, detection_reason}` |
| `synthesis_completed` | synthesis | `{report_length, source_count, citation_count}` |

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 2a. Add provenance models

Add `ProvenanceEntry` and `ProvenanceLog` models as defined above.

#### 2b. Add provenance field to ResearchExtensions

```python
# In ResearchExtensions (PLAN-0 item 2):
provenance: Optional[ProvenanceLog] = None

# Convenience accessor on DeepResearchState:
@property
def provenance(self) -> Optional[ProvenanceLog]:
    return self.extensions.provenance
```

Provenance is persisted separately (item 2g) and excluded from main state serialization via the extensions container's `exclude_none` behavior.

#### 2c. Initialize provenance at session creation

In the state factory / initialization logic, populate `provenance.session_id`, `provenance.query`, `provenance.profile`, `provenance.profile_config`, and `provenance.started_at`.

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`**

#### 2d. Log brief generation

After brief is generated, append:
```python
state.provenance.append(
    phase="brief",
    event_type="brief_generated",
    summary=f"Research brief generated: {brief_text[:100]}...",
    brief_text=brief_text,
    scope_boundaries=scope_boundaries,
    source_preferences=source_preferences,
)
```

**Files: `phases/supervision_delegation.py`, `phases/supervision_coverage.py`** (refactored in PLAN-0)

#### 2e. Log supervision events

Instrument the following points (in refactored modules from PLAN-0 item 1):
1. In `supervision_delegation.py`: after delegation response — log `decomposition` event with directives
2. In topic researcher after each provider search — log `provider_query` event
3. When sources are added to state — log `source_discovered` event
4. When deduplication occurs — log `source_deduplicated` event
5. In `supervision_coverage.py`: after `_assess_coverage_heuristic()` — log `coverage_assessment` with scores and decision
6. When gaps are added — log `gap_identified`
7. When gaps are resolved — log `gap_resolved`
8. At end of each supervision round — log `iteration_complete`

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 2f. Log synthesis events

1. After `_classify_query_type()` — log `synthesis_query_type`
2. After synthesis completes — log `synthesis_completed`

**File: `src/foundry_mcp/core/research/memory.py`**

#### 2g. Persist provenance separately

Save provenance log as a separate file alongside the state:
```
~/.foundry-mcp/research/deep_research/
├── deepres-abc123.json           # State (compact)
├── deepres-abc123.provenance.json # Provenance log (append-only)
```

This keeps the main state file compact while allowing the provenance log to grow without impacting state load/save performance.

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 2h. Expose provenance in report response

Add provenance to the `deep-research-report` response:
```python
if include_provenance:  # Default: True
    response["provenance"] = state.provenance.model_dump()
```

Also add a dedicated action:
```python
def _handle_deep_research_provenance(research_id: str) -> dict:
    """Get provenance log for a research session."""
```

### Testing

- Unit test: `ProvenanceLog.append()` creates timestamped entries
- Unit test: provenance is populated after brief phase
- Unit test: provenance is populated after supervision round
- Unit test: provenance is persisted and loadable
- Unit test: provenance is included in report response
- Unit test: provenance serialization roundtrip

---

## 3. `literature_review` Query Type

### Problem

The synthesis phase (`phases/synthesis.py`) classifies queries into four types: `comparison`, `enumeration`, `howto`, and `explanation`. Academic literature review queries fall through to `explanation`, producing a generic "Key Findings / Conclusions" structure instead of the thematic, chronological, methodological organization researchers expect.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 3a. Add detection pattern (after existing patterns)

```python
_LITERATURE_REVIEW_PATTERNS = re.compile(
    r"\b(literature review|systematic review|meta.?analysis|survey of|"
    r"state of the art|research landscape|body of (research|literature|work)|"
    r"prior (work|research|studies)|existing (research|literature|studies)|"
    r"background (research|review)|review of (the )?(literature|research|studies)|"
    r"what (does|has) (the )?research (say|show|suggest|indicate|find)|"
    r"research (on|into|about|regarding) .{5,60} (in|for|across))\b",
    re.IGNORECASE,
)
```

#### 3b. Add classification check

In `_classify_query_type()`, insert a check for `_LITERATURE_REVIEW_PATTERNS` **before** the existing comparison/enumeration/howto checks.

Also check the profile: if `research_profile.synthesis_template == "literature_review"`, use it directly. If `research_profile.source_quality_mode == ResearchMode.ACADEMIC` and the query is ambiguous, bias toward `literature_review`.

#### 3c. Add structure guidance

Add `"literature_review"` entry to `_STRUCTURE_GUIDANCE` dict:

```python
"literature_review": """\
For **literature review** queries, use this structure:
# Literature Review: [Topic]
## Executive Summary
Brief overview of the research landscape, key themes, and scope of the review.
## Introduction & Scope
Define the topic, its significance, and the boundaries of this review.
## Theoretical Foundations
Seminal works and foundational theories that define the field.
## Thematic Analysis
### [Theme 1]
### [Theme 2]
### [Theme N]
Organize findings by major themes or research questions. Within each theme,
discuss studies chronologically or by methodology as appropriate.
## Methodological Approaches
Overview of research methods used across the literature (quantitative,
qualitative, mixed methods, case studies, etc.) with strengths and limitations.
## Key Debates & Contradictions
Where studies disagree, present both sides with evidence and possible explanations.
## Research Gaps & Future Directions
Explicitly identify what remains unstudied or underexplored.
## Conclusions
Synthesize the overall state of knowledge and its implications.
## References
Full bibliographic entries (format determined by citation_style setting).""",
```

#### 3d. Add academic synthesis instructions

When `query_type == "literature_review"`, inject additional instructions into `_build_synthesis_system_prompt()`:

```
Additional instructions for literature review synthesis:
- Organize studies thematically, not just as a list of summaries.
- For each study cited, include author(s), year, and a brief methodological note
  (e.g., "Smith & Jones (2021) conducted a randomized controlled trial with N=450...").
- Identify seminal/foundational works explicitly (high citation count, historical significance).
- Note methodological trends across the literature (shift from qualitative to mixed methods, etc.).
- When studies conflict, present both findings with context rather than choosing one.
- The References section MUST use the configured citation style format using available metadata
  (authors, year, title, venue/journal, DOI).
```

### Testing

- Unit test: `_classify_query_type("literature review on X")` → `"literature_review"`
- Unit test: `_classify_query_type("what does the research say about X")` → `"literature_review"`
- Unit test: `_classify_query_type("survey of prior work on X")` → `"literature_review"`
- Unit test: profile with `synthesis_template="literature_review"` forces the type
- Unit test: generic query still returns `"explanation"` (no regression)
- Unit test: comparison query still returns `"comparison"` (no regression)
- Unit test: structure guidance is correctly injected

---

## 4. APA Citation Formatting

### Problem

The citation postprocessing pipeline (`phases/_citation_postprocess.py`) formats sources as `[N] [Title](URL)`, discarding rich metadata already present in `ResearchSource.metadata` (authors, year, venue, DOI, citation_count). Academic researchers need proper bibliographic entries.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py`**

#### 4a. Add APA formatting function

```python
def format_source_apa(source: ResearchSource) -> str:
    """Format a ResearchSource as an APA 7th edition reference entry.

    Falls back gracefully when metadata is incomplete:
    - Full: Smith, J., & Jones, K. (2023). Title. *Journal*, 30(2), 145-168. https://doi.org/...
    - Minimal: Title. Retrieved from https://...
    """
```

Logic:
1. Extract `authors`, `year`, `venue` from `source.metadata`
2. If `source.source_type == SourceType.ACADEMIC` and metadata is rich:
   - Format as: `Authors (Year). Title. *Venue*. DOI_URL`
   - Handle "et al." for >5 authors
3. If web source with limited metadata:
   - Format as: `Author/Organization (Year). Title. *Site Name*. URL`
4. Fallback: `Title. URL`

#### 4b. Add `format_style` parameter to `build_sources_section()`

When `format_style == "apa"`:
- Use `format_source_apa()` instead of current `[{cn}] [{title}]({url})` format
- Use section header "## References" instead of "## Sources"

When `format_style == "default"`:
- Preserve existing behavior exactly

#### 4c. Connect to profile

In `postprocess_citations()`, read `state.research_profile.citation_style` to determine format_style. Also use `format_style="apa"` when query_type is `literature_review` regardless of profile setting.

### Source Metadata Availability

The Semantic Scholar provider (`providers/semantic_scholar.py`) already stores:
- `metadata.authors` — comma-separated names
- `metadata.year` — publication year
- `metadata.venue` — journal/conference name
- `metadata.doi` — DOI string
- `metadata.citation_count` — for importance ranking
- `metadata.fields_of_study` — disciplinary tags

For web sources, Tavily provides `title` and `url` at minimum.

### Testing

- Unit test: `format_source_apa()` with full academic metadata (all fields present)
- Unit test: `format_source_apa()` with partial metadata (missing venue, missing DOI)
- Unit test: `format_source_apa()` with web source (no academic metadata)
- Unit test: `format_source_apa()` with >5 authors ("et al." handling)
- Unit test: `format_source_apa()` with missing year (use "n.d." per APA convention)
- Unit test: `build_sources_section()` with `format_style="apa"` produces "## References" header
- Unit test: `build_sources_section()` with `format_style="default"` preserves existing format
- Integration test: synthesis in `academic` profile produces APA references

---

## 5. Academic Brief Enrichment

### Problem

The brief phase system prompt (`phases/brief.py`) is generic. When an academic profile is active, the brief should probe for discipline, education level, time period, and methodology preferences — dimensions that fundamentally shape a literature review.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`**

#### 5a. Profile-aware brief system prompt

Modify `_build_brief_system_prompt()` to accept the research profile and append profile-specific instructions:

When profile has `source_quality_mode == ACADEMIC`:

```
Additional instructions for ACADEMIC research:
Your brief MUST additionally address these dimensions:
6. **Disciplinary scope**: Identify the primary discipline(s) and any interdisciplinary
   angles (e.g., "education + psychology + computer science").
7. **Time period**: Specify the temporal scope — foundational/seminal works (any era)
   plus recent literature (suggest last 5-10 years unless the user specifies otherwise).
8. **Methodology preferences**: Note whether the user seeks quantitative studies,
   qualitative research, mixed methods, meta-analyses, theoretical frameworks, or all types.
9. **Education level / population**: If applicable, specify the target population
   (K-12, higher education, professional development, specific age groups).
10. **Source type hierarchy**: For academic research, prioritize:
    (a) Peer-reviewed journal articles and conference papers
    (b) Systematic reviews and meta-analyses
    (c) Books and book chapters from academic publishers
    (d) Preprints from recognized repositories (arXiv, SSRN, EdArXiv)
    (e) Technical reports from research organizations
    Deprioritize: blog posts, news articles, Wikipedia, social media.
```

If the profile specifies `source_type_hierarchy`, `disciplinary_scope`, `time_period`, or `methodology_preferences`, inject those as pre-filled constraints so the brief incorporates them.

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`**

#### 5b. Profile-aware decomposition

In `_build_first_round_delegation_system_prompt()`, append when academic:

```
Additional decomposition guidelines for ACADEMIC research:
- Include a directive specifically targeting foundational/seminal works (sorted by citation count)
- Include a directive for recent empirical studies (last 3-5 years)
- If the topic spans disciplines, create separate directives per discipline
- Prefer directives that map to literature review sections: theoretical foundations,
  methodological approaches, empirical findings, and research gaps
- Each directive's "evidence_needed" should specify: peer-reviewed articles,
  sample sizes, effect sizes, or theoretical frameworks as appropriate
```

### Testing

- Unit test: brief prompt includes academic dimensions when profile is academic
- Unit test: brief prompt is unchanged when profile is general
- Unit test: profile-specified constraints are injected into brief prompt
- Unit test: supervision prompt includes academic guidelines when profile is academic
- Unit test: supervision prompt is unchanged when profile is general

---

## 6. Structured Output Mode

### Problem

Deep research produces only a markdown report. This is great for humans but opaque to downstream tools. A Zotero MCP server can't consume the sources. A visualization tool can't render the citation relationships. The report is a terminal artifact rather than a composable intermediate.

### Design

Add a `structured_output` field to the deep research result that contains machine-readable data alongside the prose report. This makes foundry-mcp an orchestration layer that other tools can plug into.

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 6a. Add structured output model

```python
class StructuredResearchOutput(BaseModel):
    """Machine-readable research output for downstream tool consumption."""

    # Source catalog (every source with full metadata)
    sources: list[dict] = Field(
        default_factory=list,
        description="All discovered sources with full metadata, suitable for import into reference managers",
    )

    # Findings with source linkage
    findings: list[dict] = Field(
        default_factory=list,
        description="Extracted findings with confidence levels and source IDs",
    )

    # Identified gaps (unresolved)
    gaps: list[dict] = Field(
        default_factory=list,
        description="Research gaps identified but not resolved during the session",
    )

    # Contradictions found
    contradictions: list[dict] = Field(
        default_factory=list,
        description="Contradicting findings across sources",
    )

    # Query classification
    query_type: str = "explanation"

    # Profile used
    profile: str = "general"
```

#### 6b. Build structured output after synthesis

Add `_build_structured_output()` method that transforms state into the structured format:
- Sources: `[{id, title, url, source_type, authors, year, venue, doi, citation_count, ...}]` — flat, denormalized, ready for consumption
- Findings: `[{id, content, confidence, source_ids, category}]`
- Gaps: `[{id, description, priority, suggested_queries}]` (unresolved only)
- Contradictions: `[{id, description, source_ids}]`

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 6c. Include structured output in report response

```python
structured = state.extensions.structured_output
response["structured"] = structured.model_dump() if structured else None
```

This is always present in the response. Consumers that want just the report ignore it; consumers that want structured data use it.

### Testing

- Unit test: `_build_structured_output()` with diverse sources
- Unit test: sources include full denormalized metadata
- Unit test: only unresolved gaps are included
- Unit test: structured output appears in report response
- Unit test: structured output serialization roundtrip

---

## Testing Budget

| Item | Impl LOC | Test LOC | Test Focus |
|------|----------|----------|------------|
| 1. Research Profiles | ~200-250 | ~120-150 | Profile resolution, backward compat, config loading |
| 2. Provenance Audit Trail | ~300-400 | ~100-130 | Append, serialize, persist, load |
| 3. Literature Review Query Type | ~100-150 | ~60-80 | Classification patterns, no regression |
| 4. APA Citations | ~150-200 | ~80-100 | Format variants, metadata completeness |
| 5. Academic Brief Enrichment | ~100-150 | ~40-50 | Prompt injection, profile gating |
| 6. Structured Output | ~150-200 | ~50-70 | Transform, serialize, response inclusion |
| **Total** | **~1000-1350** | **~450-580** | |

## File Impact Summary

| File | Change Type | Items |
|------|-------------|-------|
| `models/deep_research.py` | Modify | 1 (ResearchProfile), 2 (ProvenanceLog), 6 (StructuredResearchOutput) — all via ResearchExtensions |
| `config/research.py` | Modify | 1 (profile registry, default profile) |
| `phases/synthesis.py` | Modify | 3 (literature_review type), 6 (structured output) |
| `phases/_citation_postprocess.py` | Modify | 4 (APA formatting) |
| `phases/brief.py` | Modify | 5 (profile-aware brief) |
| `phases/supervision_delegation.py` | Modify | 2 (provenance logging) — refactored in PLAN-0 |
| `phases/supervision_coverage.py` | Modify | 2 (provenance logging) — refactored in PLAN-0 |
| `phases/supervision_first_round.py` | Modify | 5 (profile-aware decomposition) — refactored in PLAN-0 |
| `phases/topic_research.py` | Modify | 2 (provenance logging for provider queries) |
| `memory.py` | Modify | 2 (provenance persistence) |
| `handlers_deep_research.py` | Modify | 1 (profile parameter), 2 (provenance endpoint), 6 (structured output in response) |
| `research.py` (action router) | Modify | 1 (pass profile through) |

## Dependency Graph

```
[PLAN-0: Prerequisites]
       │
       ├──▶ [1. Research Profiles]
       │           │
       │           ├──▶ [3. Literature Review Query Type] (uses profile.synthesis_template)
       │           ├──▶ [4. APA Citations] (uses profile.citation_style)
       │           ├──▶ [5. Academic Brief Enrichment] (uses profile fields)
       │
       ├──▶ [2. Provenance Audit Trail] (independent, can be done in parallel with 1)
       │
       └──▶ [6. Structured Output Mode] (independent, can be done in parallel with 1-5)
```

Items 1, 2, and 6 are independent foundations (all require PLAN-0). Items 3-5 depend on item 1 (profiles) for configuration.
