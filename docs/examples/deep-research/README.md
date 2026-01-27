# Deep Research Examples

This directory contains example outputs from the `deep-research` workflow, demonstrating how foundry-mcp conducts automated, multi-phase research on topics.

## Available Examples

| Example | Report | Audit | Description |
|---------|--------|-------|-------------|
| **LLM Judges** | `llm-judges-report.md` | `llm-judges-audit.jsonl` | Techniques, architectures, and evaluation methods for LLM-as-a-Judge *(earlier iteration)* |
| **Conversation-Based Assessment** | `cba-report.md` | `cba-audit.jsonl` | Methodologies, frameworks, and AI applications in educational/professional assessment |

---

# Example 1: LLM Judges

> **Note:** This example is from an earlier iteration of the deep research workflow (v0.8.0). The current workflow has additional phases, improved source gathering, and enhanced synthesis capabilities.

This section documents the LLM Judges research output.

## Research Query

> "LLM Judges: techniques, architectures, evaluation methods, and applications for using large language models as automated evaluators and judges"

## Workflow Overview

The deep research workflow executes in distinct phases:

### Phase 1: Planning
The system analyzes the query and generates targeted sub-queries to explore different facets of the topic. For this research, it generated 12 sub-queries covering:
- Core architectures (pairwise comparison, direct scoring)
- Known biases (positional, verbosity, self-preference)
- Mitigation techniques (Chain-of-Thought, position swapping)
- Advanced approaches (Judge Assembly, hybrid frameworks)

### Phase 2: Gathering
Each sub-query is executed against multiple search providers in parallel:
- **Tavily** - 12 queries
- **Perplexity** - 12 queries
- **Google** - 12 queries
- **Semantic Scholar** - 12 queries

Total: 48 search queries across 4 providers, yielding 156 unique sources from 64 distinct domains.

### Phase 3: Analysis
Findings are synthesized, conflicts are identified, and knowledge gaps are noted for refinement iterations.

### Phase 4: Synthesis
A final report is generated with executive summary, key findings organized by theme, analysis of supporting/conflicting evidence, limitations, and actionable conclusions.

### Phase 5: Refinement
The workflow iterates up to 3 times, identifying gaps and generating additional sub-queries to fill them.

## Statistics

| Metric | Value |
|--------|-------|
| Total Iterations | 3 |
| Sub-queries Generated | 12 |
| Search Queries Executed | 48 |
| Sources Examined | 156 |
| Unique Source Domains | 64 |
| Key Findings | 12 |
| Knowledge Gaps | 6 |
| Total Tokens Used | 129,685 |
| Duration | ~74 seconds |

## Files in This Directory

| File | Description |
|------|-------------|
| `llm-judges-report.md` | The final synthesized research report |
| `llm-judges-audit.jsonl` | Detailed audit trail of every operation (JSONL format) |

## Audit Trail Structure

The audit file (`llm-judges-audit.jsonl`) contains one JSON object per line, recording:

```json
{
  "timestamp": "2026-01-01T01:18:35.518082Z",
  "event_id": "94c477f3916948558059faefd5a6d856",
  "event_type": "workflow_complete",
  "research_id": "deepres-906a9d34c7b2",
  "phase": "synthesis",
  "iteration": 3,
  "level": "info",
  "data": {
    "source_count": 156,
    "finding_count": 12,
    "total_tokens_used": 129685,
    "search_provider_stats": {
      "tavily": 12,
      "perplexity": 12,
      "google": 12,
      "semantic_scholar": 12
    }
  }
}
```

Event types include:
- `workflow_start` / `workflow_complete` - Session lifecycle
- `phase_start` / `phase_complete` - Phase transitions with timing
- `planning_result` - Sub-queries generated
- `gathering_provider_result` - Per-provider search results
- `analysis_result` - Findings and gaps extracted
- `synthesis_result` - Report generation
- `refinement_result` - Gap-filling iterations

## Usage

To run your own deep research:

```bash
# Start research (runs in background)
foundry research deep-research \
  --query "Your research topic here" \
  --max-iterations 3

# Check progress
foundry research deep-research-status --research-id <id>

# Get final report
foundry research deep-research-report --research-id <id>
```

Or via MCP tool calls:

```python
# Start
{"action": "deep-research", "query": "...", "max_iterations": 3}

# Status (shows live progress)
{"action": "deep-research-status", "research_id": "..."}

# Report
{"action": "deep-research-report", "research_id": "..."}
```

## Key Takeaways from This Research

The research revealed that LLM-as-a-Judge is a powerful but systematically biased paradigm:

1. **Human-level agreement** - GPT-4 achieves >80% agreement with human annotators, matching inter-rater reliability
2. **Three critical biases** require active mitigation:
   - **Position bias** - First option favored in pairwise comparisons
   - **Verbosity bias** - Longer responses rated higher regardless of accuracy
   - **Self-preference bias** - Models favor outputs from their own family
3. **Mandatory mitigations** - Position swapping and Chain-of-Thought prompting are essential
4. **Domain-specific validation** - For technical tasks like code evaluation, use "Judge Assembly" patterns combining LLM reasoning with deterministic checks (execution, linting)
5. **Hybrid frameworks** - Co-Eval approaches augment LLM judgment with objective metrics to reduce hallucinated scoring

## Source Diversity

The research drew from 64 unique domains including:
- Academic sources: arxiv.org, neurips.cc, aclanthology.org, openreview.net
- Industry blogs: cameronrwolfe.substack.com, eugeneyan.com, wandb.ai
- Documentation: docs.ragas.io, langchain-opentutorial.gitbook.io
- Research tools: semantic scholar, google scholar references

---

# Example 2: Conversation-Based Assessment

This section documents the Conversation-Based Assessment (CBA) research output.

## Research Query

> "Conversation based assessment: methodologies, frameworks, applications in education and professional settings, AI-powered conversational assessment tools, validity and reliability considerations, best practices for design and implementation"

## Workflow Overview

The research explored conversation-based assessment across multiple dimensions:

### Phase 1: Planning
The system generated 4 targeted sub-queries covering:
- Core methodologies and frameworks (ORID, Caring Assessments)
- AI applications in recruitment and healthcare
- Educational efficacy and validity considerations
- Best practices for implementation

### Phase 2: Gathering
Sub-queries executed across search providers, yielding 27 unique sources.

### Phase 3-5: Analysis, Synthesis, Refinement
Findings synthesized across healthcare, education, and professional domains with gap analysis.

## Statistics

| Metric | Value |
|--------|-------|
| Total Iterations | 2 |
| Sub-queries Generated | 4 |
| Sources Examined | 44 |
| Key Findings | 4 |
| Knowledge Gaps | 2 |
| Total Tokens Used | ~275,000 |

## Files

| File | Description |
|------|-------------|
| `cba-report.md` | The final synthesized research report |
| `cba-audit.jsonl` | Detailed audit trail of every operation |
| `cba-session.json` | Full session state including all sources and findings |

## Key Takeaways

1. **Structured Frameworks Matter**: ORID (Objective, Reflective, Interpretive, Decisional) ensures cognitive depth beyond simple recall
2. **AI Validity Varies by Domain**:
   - **Healthcare**: High validity for screening (depression scales, medical Q&A)
   - **Recruitment**: Strong market validation for technical skill assessment
   - **Education**: Engagement ≠ Learning - positive feedback doesn't guarantee improved outcomes
3. **Critical Biases**: Insufficient data on linguistic diversity and neurodiverse populations
4. **Hybrid Approaches Recommended**: AI for initial screening; human oversight for complex pedagogical goals

## Source Diversity

The research drew from diverse domains:
- Healthcare: JAMA Network, ScienceDirect, PubMed Central
- Education: SAGE Journals, ETS Research, ResearchGate
- Professional: Gartner, iMocha, Testlify, Metaview
- Frameworks: Better Evaluation, SFJ Awards
