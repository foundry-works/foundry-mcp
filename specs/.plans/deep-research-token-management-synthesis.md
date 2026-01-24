# Synthesis

## Overall Assessment
- **Consensus Level**: Moderate (both reviews identify critical architectural issues, but focus on different aspects)

Both reviewers agree the specification is comprehensive and well-structured, but identify several critical blockers that must be addressed before implementation. The reviews complement each other: cursor-agent focuses heavily on contract consistency, schema completeness, and data integrity, while gemini emphasizes runtime architecture and configuration clarity.

## Critical Blockers
Issues that must be fixed before implementation (identified by multiple models):

- **[Architecture]** Runtime overhead inference logic conflates provider ID with runtime environment - flagged by: gemini
  - **Impact**: If a user runs `foundry-mcp` inside Gemini CLI (~40k overhead) but configures `claude:sonnet` as the model, the code will incorrectly apply Claude Code overhead (~60k) instead of the actual environment overhead. This leads to incorrect budget calculations (wasting tokens or overflowing context).
  - **Recommended fix**: Remove inference logic from `get_effective_context`. Define `runtime_overhead` (or `system_overhead`) as a top-level configuration setting in `foundry-mcp.toml` or detect it via an environment variable set by the host CLI. The `provider_id` should only determine Model Limits, not System Overhead.

- **[Completeness]** Response contract placement conflicts for warning fields - flagged by: cursor-agent
  - **Impact**: Plan states `warning_details` is in `data` but also says warnings live in `meta.warnings` and `meta.warnings` may be omitted when empty. It also says "Always include content_fidelity_schema_version… warning_details empty list" but then says omit `meta.warnings` when empty. The contract example shows `warning_details` inside `data`, but schema fragment doesn't show where it lives and fixtures guidance conflicts (always include vs omit).
  - **Recommended fix**: Explicitly define location and presence rules per field (e.g., `data.warning_details` always present, `meta.warnings` optional). Update schema fragments to include full path and required/optional status and align fixtures guidance with that.

- **[Architecture]** State vs response warning duplication is underspecified - flagged by: cursor-agent
  - **Impact**: The plan introduces `content_fidelity[item].phases[phase].warnings`, `data.warning_details`, and `meta.warnings` without a canonical source of truth or precedence rules. Warning code drift and duplication will cause inconsistent reporting and tests that are brittle.
  - **Recommended fix**: Define a single generation pipeline: per-item fidelity warnings → aggregated `data.warning_details` → de-duplicated `meta.warnings`. Specify de-dupe rules and which layer is authoritative.

- **[Clarity]** Ambiguity between `model_limits.json` and Python constants - flagged by: gemini
  - **Impact**: The plan lists `src/foundry_mcp/core/research/model_limits.json` as a file to create, but Phase 1 implementation details show a hardcoded Python dictionary `DEFAULT_MODEL_LIMITS`. Hardcoding limits in Python requires code changes/releases to update model definitions, whereas a JSON registry allows for easier updates or external overrides.
  - **Recommended fix**: Clarify that `model_limits.json` is the source of truth loaded at runtime, and the Python dictionary is either a fallback or strictly for type definitions. Ensure the implementation loads this JSON file.

- **[Feasibility]** Model limits and overhead values treated as static facts - flagged by: cursor-agent
  - **Impact**: The plan hardcodes token limits and overhead estimates for several models/CLIs as if authoritative, but also says "limited public data" and "estimated." There is no gating to prevent unsafe defaults when these are wrong. Incorrect limits risk overflow errors and degraded outputs on production workloads.
  - **Recommended fix**: Make defaults explicitly conservative and require runtime validation before using aggressive caps; add a "unknown/unsafe" mode that enforces smaller budgets and emits `LIMITS_DEFAULTED` until verified.

- **[Risk]** Archive implementation ignores sensitive data handling and error paths - flagged by: cursor-agent
  - **Impact**: The archive writes full content with minimal safeguards (only private permissions); TTL cleanup reads JSON and unlinks without handling parse errors or partial writes. Also no mention of file locking or concurrent writes. Potential data leakage, corruption, and runtime crashes, plus inconsistent cleanup.
  - **Recommended fix**: Add atomic writes (write temp + rename), JSON decode error handling, and a clear "skip on corruption" policy with warnings. Explicitly note that archive is best-effort and never blocks the workflow.

## Major Suggestions
Significant improvements that enhance quality, maintainability, or design:

- **[Completeness]** Missing specification for `TokenUsageObservation` persistence schema - flagged by: cursor-agent
  - **Description**: Plan references `TokenUsageObservation` stored in `DeepResearchState` and surfaced in `meta.telemetry` but doesn't define its schema or retention policy.
  - **Recommended fix**: Define the exact schema (fields, types, required), retention limits, and where it appears in the response envelope.

- **[Architecture]** Model limits registry file format not specified - flagged by: cursor-agent
  - **Description**: `model_limits.json` is introduced but no schema, versioning, or provenance fields are defined.
  - **Recommended fix**: Define a JSON schema with `version`, `last_verified`, `source_url`, per-model `context_window`, `max_output_tokens`, `budgeting_mode`, and an optional `notes` field.

- **[Architecture]** Map-Reduce Partial Failure Strategy - flagged by: gemini
  - **Description**: The plan states "Provider fallback exhausted: warning if partial data exists" but doesn't explicitly define the behavior if *one* chunk of a multi-chunk map-reduce summary fails while others succeed.
  - **Recommended fix**: Explicitly define a "Best Effort" strategy for map-reduce: if > X% of chunks succeed, assemble the partial summary and append a specific warning (e.g., `PARTIAL_SUMMARY_INCOMPLETE`) indicating which sections are missing.

- **[Feasibility]** Dynamic `runtime_id` Detection - flagged by: gemini
  - **Description**: The plan references `runtime_id` in `ModelContextRegistry.record_token_observation` for calibration tracking, but provides no mechanism for how this ID is established at runtime.
  - **Recommended fix**: Add a mechanism to establish the `runtime_id` at startup (e.g., via config `runtime_id = "gemini-cli-v2"` or env var `FOUNDRY_RUNTIME_ID`), ensuring token calibration is bucketed correctly per environment.

- **[Sequencing]** Tests/fixtures update ordering unclear - flagged by: cursor-agent
  - **Description**: Phase 0 says contract and fixture updates are a prerequisite to implementation, but later phases also modify schema-bearing code and tests. The plan doesn't specify whether fixtures are updated once or iteratively.
  - **Recommended fix**: Specify a sequence: update schema + docs + golden fixtures first, then implement runtime changes, then update tests and fixtures again if necessary.

- **[Feasibility]** Summarization output schema + prompt coupling lacks validation strategy - flagged by: cursor-agent
  - **Description**: JSON output schema is declared but no explicit parser/validator selection is listed (e.g., strict JSON, JSON5, or model-assisted).
  - **Recommended fix**: Define a strict parser with a permissive fallback, and specify exact failure handling (e.g., one retry → downgrade → failure).

- **[Risk]** Protected content handling is underspecified for mixed-content items - flagged by: cursor-agent
  - **Description**: "Never drop protected content" is stated, but items may contain mixed protected/unprotected sections, and map-reduce chunk IDs are not clearly tied back to "protected" status.
  - **Recommended fix**: Define protection at the chunk or sub-item level and ensure allocation respects those units.

- **[Clarity]** Conflicting guidance on defaults vs omission - flagged by: cursor-agent
  - **Description**: The plan says "always include empty defaults" but also says "omit meta.warnings when none." It also says "warning_details empty list when no warnings" but example shows warnings in data.
  - **Recommended fix**: Choose a single approach and document it with explicit required/optional flags per field.

## Questions for Author
Clarifications needed (common questions across models):

- **[Clarity]** How should `warning_details` be structured in the response contract? - flagged by: cursor-agent
  - **Context**: The plan implies `warning_details` in `data` and `meta.warnings` in `meta`, but schema doesn't show location.
  - **Needed**: Exact schema path and required/optional status for `warning_details`.

- **[Architecture]** What is the canonical source of truth for warnings? - flagged by: cursor-agent
  - **Context**: Warnings appear in per-item fidelity, `data.warning_details`, and `meta.warnings`.
  - **Needed**: Precedence and dedupe rules, plus which layer drives tests.

- **[Feasibility]** Tokenizer Availability - flagged by: gemini
  - **Context**: The plan mentions using "provider-native token counts -> tokenizer library".
  - **Needed**: Does `foundry-mcp` already include heavy tokenizer libraries (like `tiktoken` or `tokenizers`)? If not, adding them increases the package size significantly. Are we assuming these are present or adding them as dependencies?

- **[Completeness]** How does `DeepResearchState` versioning interact with persisted telemetry? - flagged by: cursor-agent
  - **Context**: Migrations are defined for fidelity fields, but telemetry fields like token observations are not detailed.
  - **Needed**: Migration steps and defaulting behavior for telemetry.

- **[Feasibility]** What is the exact parser/validator used for structured summarization output? - flagged by: cursor-agent
  - **Context**: The plan assumes structured JSON but doesn't specify parsing and failure handling rules.
  - **Needed**: Parser choice, strictness, and retry/degrade sequence.

- **[Risk]** What is the behavior when archive JSON is corrupt or partial? - flagged by: cursor-agent
  - **Context**: Cleanup and retrieve assume valid JSON.
  - **Needed**: Corruption handling policy (skip with warning, delete, or quarantine).

## Design Strengths
What the spec does well (areas of agreement):

- **[Architecture]** Strong separation of concerns across phases and modules - noted by: cursor-agent
  - **Why**: Token management, summarization, allocation, and integration are isolated, which reduces coupling and simplifies testing.

- **[Feasibility]** Calibration Strategy - noted by: cursor-agent, gemini
  - **Why**: The use of an Exponential Moving Average (EMA) to calibrate token estimation against observed usage is a sophisticated and robust way to handle the inherent fuzziness of token counting without requiring perfect tokenizers for every model.

- **[Completeness]** Fidelity Tracking - noted by: cursor-agent, gemini
  - **Why**: Explicitly tracking `dropped_content_ids` and `content_fidelity` per phase is excellent. It ensures the system is transparent about what it "forgot" or compressed, which is critical for trust in AI research tools.

- **[Completeness]** Good contract and fixture awareness - noted by: cursor-agent
  - **Why**: The plan explicitly ties schema changes to fixtures, docs, and changelog, aligning with spec-driven development.

- **[Risk]** Thoughtful degradation strategy - noted by: cursor-agent
  - **Why**: The summarization → headline → truncate → drop chain with guardrails provides predictable behavior under pressure.

- **[Testing]** Integration Approach - noted by: gemini
  - **Why**: The strategy of running integration tests with "artificially low model limits" to force degradation paths is a smart, high-leverage way to verify complex fallback logic without spending massive amounts of tokens.

- **[Clarity]** Detailed configuration surface - noted by: cursor-agent
  - **Why**: Config options are enumerated with sensible defaults and clear intent, making the system tunable without code changes.

## Points of Agreement
- Both reviewers praise the calibration strategy using EMA for token estimation drift correction
- Both reviewers praise the explicit fidelity tracking (`dropped_content_ids`, `content_fidelity`)
- Both reviewers identify concerns about warning location/structure (though cursor-agent is more detailed)
- Both reviewers identify concerns about model limits configuration (cursor-agent focuses on validation/safety, gemini on JSON vs Python)
- Both reviewers appreciate the thoughtful degradation strategy and integration testing approach
- Both reviewers note the specification is comprehensive and well-structured overall

## Points of Disagreement
- **Focus areas**: cursor-agent provides more granular feedback on contract consistency, schema completeness, and data integrity (6 critical blockers, 6 major suggestions), while gemini focuses more on runtime architecture and configuration clarity (2 critical blockers, 2 major suggestions). This is not a true disagreement but rather complementary perspectives.

- **Criticality assessment**: cursor-agent flags more issues as "critical blockers" (4 vs 2), suggesting a more conservative approach to contract/schema issues. gemini prioritizes runtime correctness (overhead inference) and configuration clarity as the most critical issues.

- **No fundamental disagreements**: The reviews don't contradict each other on core design decisions. They identify different aspects of the same underlying concerns (e.g., both worry about model limits, but cursor-agent focuses on validation/safety while gemini focuses on JSON vs Python representation).

## Synthesis Notes
- **Overall themes**: The reviews reveal three major themes:
  1. **Contract consistency**: Multiple warning fields and response schema locations need explicit definition and precedence rules
  2. **Runtime architecture**: Overhead calculation and model limits configuration need clearer separation between runtime environment and model provider
  3. **Data integrity**: Archive implementation, telemetry persistence, and state migrations need more robust error handling and schema definitions

- **Actionable next steps**:
  1. **Immediate**: Resolve the runtime overhead vs provider ID confusion (gemini's critical blocker) - this affects core budget calculations
  2. **Immediate**: Clarify warning field locations and presence rules (cursor-agent's critical blocker) - this affects contract compliance
  3. **High priority**: Define canonical warning generation pipeline and deduplication rules
  4. **High priority**: Specify `model_limits.json` schema and clarify JSON vs Python constants relationship
  5. **High priority**: Add archive corruption handling and atomic write patterns
  6. **Medium priority**: Define `TokenUsageObservation` schema and telemetry migration behavior
  7. **Medium priority**: Specify summarization parser/validator and partial failure strategy for map-reduce
  8. **Medium priority**: Clarify protected content handling at chunk/sub-item level
  9. **Low priority**: Address minor suggestions around priority weights, safety margins, and cache invalidation

- **Review quality**: Both reviews are thorough and identify legitimate issues. cursor-agent's review is more exhaustive on contract/schema details, while gemini's review is more focused on runtime correctness. Together they provide comprehensive coverage of potential issues.
