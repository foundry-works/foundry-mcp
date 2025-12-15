# AI Consultation Layer

> Unified interface for LLM-powered workflows: documentation generation, plan review, and fidelity review.

## Overview

The AI Consultation Layer provides a centralized orchestration system for invoking external LLM providers. It abstracts provider discovery, request formatting, response parsing, and caching into a single `ConsultationOrchestrator` class.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ConsultationOrchestrator                  │
│  - Provider discovery & fallback                            │
│  - Request routing                                          │
│  - Response caching                                         │
│  - Timeout management                                       │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  DOC_GENERATION │  │   PLAN_REVIEW   │  │ FIDELITY_REVIEW │
│    Workflow     │  │    Workflow     │  │    Workflow     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Core Components

### ConsultationOrchestrator

The main entry point for all AI consultations:

```python
from foundry_mcp.core.ai_consultation import ConsultationOrchestrator

orchestrator = ConsultationOrchestrator(
    default_timeout=300
)

# Check availability
if orchestrator.is_available():
    result = await orchestrator.consult(request)
```

### ConsultationWorkflow

Enum defining supported workflow types:

- `DOC_GENERATION` - Generate documentation from code
- `PLAN_REVIEW` - Review specification plans
- `FIDELITY_REVIEW` - Compare implementation against spec

### ConsultationRequest

Request structure for consultations:

```python
from foundry_mcp.core.ai_consultation import (
    ConsultationRequest,
    ConsultationWorkflow
)

request = ConsultationRequest(
    workflow=ConsultationWorkflow.PLAN_REVIEW,
    prompt_id="PLAN_REVIEW_FULL_V1",
    context={
        "spec_content": spec_json,
        "review_focus": "completeness,feasibility"
    },
    provider_id="gemini",  # Optional: specific provider
    timeout=300
)
```

### ConsultationResult

Response structure from consultations:

```python
result = await orchestrator.consult(request)

if result.error:
    print(f"Error: {result.error}")
else:
    print(f"Provider: {result.provider_id}")
    print(f"Model: {result.model_used}")
    print(f"Content: {result.content}")
    print(f"Cached: {result.cache_hit}")
```

## Workflows

### Documentation Generation

Generate documentation from code analysis:

```python
request = ConsultationRequest(
    workflow=ConsultationWorkflow.DOC_GENERATION,
    prompt_id="DOC_GENERATION_V1",
    context={
        "code_content": source_code,
        "file_path": "src/module.py",
        "output_format": "markdown"
    }
)
```

### Plan Review

Review specification plans for quality:

```python
request = ConsultationRequest(
    workflow=ConsultationWorkflow.PLAN_REVIEW,
    prompt_id="PLAN_REVIEW_FULL_V1",
    context={
        "spec_content": json.dumps(spec_data),
        "review_type": "full",
        "dimensions": "completeness,feasibility,clarity,security"
    }
)
```

Response includes:
- `verdict`: pass/partial/fail
- `dimensions`: Scored assessments (completeness, feasibility, etc.)
- `critical_issues`: Blocking problems
- `suggestions`: Improvement recommendations

### Fidelity Review

Compare implementation against specification:

```python
request = ConsultationRequest(
    workflow=ConsultationWorkflow.FIDELITY_REVIEW,
    prompt_id="FIDELITY_REVIEW_V1",
    context={
        "spec_requirements": requirements_text,
        "implementation_artifacts": code_content,
        "test_results": test_output,
        "journal_entries": journal_text
    }
)
```

Response includes:
- `verdict`: pass/partial/fail
- `deviations`: List of spec-to-code mismatches
- `compliance`: Per-category compliance status
- `recommendations`: Suggested fixes

## CLI Integration

### Plan Review Command

```bash
sdd plan-review <spec-id> --type full
sdd plan-review <spec-id> --type quick --provider gemini
```

### Fidelity Review Command

```bash
sdd fidelity-review <spec-id>
sdd fidelity-review <spec-id> --task task-2-1
sdd fidelity-review <spec-id> --phase phase-1
```

## MCP Tool Integration

### spec-review

```python
# MCP tool for plan review
result = await spec_review(
    spec_id="my-spec-001",
    review_type="full",
    dry_run=False
)
```

### spec-review-fidelity

```python
# MCP tool for fidelity review
result = await spec_review_fidelity(
    spec_id="my-spec-001",
    task_id="task-2-1",  # Optional
    phase_id=None,
    use_ai=True,
    consensus_threshold=2
)
```

## Provider Configuration

Providers are discovered automatically. Configure via environment:

```bash
# Gemini
export GEMINI_API_KEY="your-key"

# Claude/Anthropic
export ANTHROPIC_API_KEY="your-key"

# OpenAI-compatible
export OPENAI_API_KEY="your-key"
```

## Error Handling

The consultation layer returns structured errors:

| Error Code | Description |
|------------|-------------|
| `AI_NO_PROVIDER` | No LLM providers available |
| `AI_NOT_AVAILABLE` | Requested provider unavailable |
| `AI_TIMEOUT` | Request timed out |
| `AI_PARSE_ERROR` | Failed to parse LLM response |

## Caching

Consultation results are cached by default:

- Cache key: workflow + prompt_id + context hash
- Default TTL: 1 hour
- Disable: Set `use_cache=False` in request

## Testing

Test fixtures are provided in `tests/fixtures/ai_responses/`:

- `doc_gen_response.json` - Sample doc generation
- `plan_review_response.json` - Sample plan review
- `fidelity_review_response.json` - Sample fidelity review

## Related Documentation

- [Provider Management](../mcp_best_practices/11-ai-llm-integration.md)
- [Prompt Templates](../codebase_standards/prompt-templates.md)
- [MCP Response Schema](../codebase_standards/mcp_response_schema.md)

---

**Navigation:** [Testing Guide](testing.md) | [Index](../mcp_best_practices/README.md)
