# Research Router Implementation

## Objective

Create a new "research" router in foundry-mcp that re-implements core workflows from claude-model-chorus, providing multi-model orchestration capabilities through CHAT, CONSENSUS, THINKDEEP, and IDEATE workflows.

## Mission

Enable intelligent research workflows with conversation threading, multi-model consensus, hypothesis-driven investigation, and creative brainstorming through a unified MCP tool interface.

## Scope

### In Scope
- CHAT workflow: Single-model conversation with thread persistence
- CONSENSUS workflow: Multi-model parallel consultation with synthesis strategies
- THINKDEEP workflow: Systematic investigation with hypothesis tracking
- IDEATE workflow: Creative brainstorming with idea clustering and scoring
- ROUTE action: Intelligent workflow selection based on prompt analysis
- Thread management: List, get, delete conversation threads
- File-based storage for conversation/investigation state
- ResearchConfig dataclass for configuration
- Integration with existing provider router for LLM calls
- Comprehensive test suite

### Out of Scope
- ARGUMENT workflow (dialectical reasoning) - future phase
- STUDY workflow (persona-based research) - future phase
- SQLite storage backend - future enhancement
- External CLI provider support (claude/gemini CLIs)
- Real-time streaming responses

## Phases

### Phase 1: Foundation - Data Models & Memory

**Purpose**: Establish the core data structures and persistence layer that all workflows depend on.

**Tasks**:
1. Create `src/foundry_mcp/core/research/__init__.py` - Package exports
2. Create `src/foundry_mcp/core/research/models.py` - Pydantic models:
   - Enums: `WorkflowType`, `ConfidenceLevel`, `ConsensusStrategy`, `ThreadStatus`
   - Conversation: `ConversationMessage`, `ConversationThread`
   - THINKDEEP: `Hypothesis`, `InvestigationStep`, `ThinkDeepState`
   - IDEATE: `Idea`, `IdeaCluster`, `IdeationState`
   - CONSENSUS: `ModelResponse`, `ConsensusConfig`, `ConsensusState`
3. Create `src/foundry_mcp/core/research/memory.py` - File-based storage:
   - `FileStorageBackend` class with save/load/delete/list operations
   - `ResearchMemory` class with thread/investigation/ideation CRUD
   - File locking for thread safety
   - TTL-based cleanup

**Verification**:
- Unit tests pass for all Pydantic models (serialization/validation)
- Memory operations work correctly (create, read, update, delete)
- File locking prevents race conditions

### Phase 2: Configuration Extension

**Purpose**: Add ResearchConfig to ServerConfig for workflow settings.

**Tasks**:
1. Add `ResearchConfig` dataclass to `src/foundry_mcp/config.py`:
   - `enabled: bool = True`
   - `storage_path: str = ""` (default: ~/.foundry-mcp/research)
   - `storage_backend: str = "file"`
   - `ttl_hours: int = 24`
   - `max_messages_per_thread: int = 100`
   - `default_provider: str = "gemini"`
   - `consensus_providers: List[str]`
   - `thinkdeep_max_depth: int = 5`
   - `ideate_perspectives: List[str]`
2. Add `from_toml_dict()` classmethod for config loading
3. Add `research` field to `ServerConfig` dataclass
4. Update TOML parsing to handle `[research]` section

**Verification**:
- Config loads from TOML correctly
- Default values work when section is missing
- Environment variable overrides function

### Phase 3: Workflow Base & CHAT Implementation

**Purpose**: Create the base workflow infrastructure and implement the simplest workflow first.

**Tasks**:
1. Create `src/foundry_mcp/core/research/workflows/__init__.py` - Exports
2. Create `src/foundry_mcp/core/research/workflows/base.py`:
   - `ResearchWorkflowBase` class with provider integration
   - `_execute_provider()` method using existing provider infrastructure
   - `_resolve_provider()` helper
3. Create `src/foundry_mcp/core/research/workflows/chat.py`:
   - `ChatWorkflow` class extending base
   - Thread creation and continuation
   - Conversation context building with token budgeting
   - Message persistence

**Verification**:
- CHAT workflow creates new threads
- Conversation continuation works with history
- Provider calls execute successfully

### Phase 4: CONSENSUS Workflow

**Purpose**: Implement multi-model parallel execution with synthesis strategies.

**Tasks**:
1. Create `src/foundry_mcp/core/research/workflows/consensus.py`:
   - `ConsensusWorkflow` class
   - `_execute_parallel()` async method using asyncio.gather()
   - Concurrency limiting with asyncio.Semaphore
   - Synthesis strategies: all_responses, synthesize, majority, first_valid
   - Partial failure handling (continue on some provider errors)

**Verification**:
- Parallel execution works with multiple providers
- Each synthesis strategy produces correct output
- Partial failures are handled gracefully
- Concurrency limits are respected

### Phase 5: THINKDEEP Workflow

**Purpose**: Implement systematic investigation with hypothesis tracking.

**Tasks**:
1. Create `src/foundry_mcp/core/research/workflows/thinkdeep.py`:
   - `ThinkDeepWorkflow` class
   - Investigation step execution
   - Hypothesis creation and tracking
   - Confidence level progression
   - Convergence detection
   - State persistence across turns

**Verification**:
- Investigation creates and updates hypotheses
- Confidence levels progress based on evidence
- State persists correctly across invocations
- Max depth is respected

### Phase 6: IDEATE Workflow

**Purpose**: Implement creative brainstorming with phased execution.

**Tasks**:
1. Create `src/foundry_mcp/core/research/workflows/ideate.py`:
   - `IdeateWorkflow` class
   - Divergent phase: Multi-perspective idea generation
   - Convergent phase: Idea clustering and scoring
   - Selection phase: Mark clusters for elaboration
   - Elaboration phase: Develop selected clusters

**Verification**:
- Divergent phase generates diverse ideas
- Convergent phase clusters and scores correctly
- Phase transitions work properly
- Elaboration produces detailed plans

### Phase 7: Router Implementation

**Purpose**: Create the unified research tool with all action handlers.

**Tasks**:
1. Create `src/foundry_mcp/tools/unified/research.py`:
   - Register `research_tools` feature flag (BETA)
   - Define `_ACTION_SUMMARY` dict
   - Implement handlers: `_handle_chat`, `_handle_consensus`, `_handle_thinkdeep`, `_handle_ideate`, `_handle_route`
   - Implement thread handlers: `_handle_thread_list`, `_handle_thread_get`, `_handle_thread_delete`
   - Create `_RESEARCH_ROUTER` ActionRouter
   - Implement `_dispatch_research_action()`
   - Implement `register_unified_research_tool()`
2. Update `src/foundry_mcp/tools/unified/__init__.py`:
   - Import and call `register_unified_research_tool()`

**Verification**:
- All 8 actions route correctly
- Feature flag gates access
- Response envelopes match v2 schema
- Validation errors return proper error codes

### Phase 8: Testing & Documentation

**Purpose**: Comprehensive test coverage and usage documentation.

**Tasks**:
1. Create `tests/core/research/test_models.py`:
   - Model validation tests
   - Serialization/deserialization tests
   - Enum value tests
2. Create `tests/core/research/test_memory.py`:
   - Storage backend tests
   - Thread CRUD tests
   - TTL cleanup tests
   - File locking tests
3. Create `tests/tools/unified/test_research.py`:
   - Router dispatch tests
   - Action handler tests with mocked providers
   - Error condition tests
   - Response envelope validation

**Verification**:
- All tests pass
- Coverage > 80% for new code
- Integration tests work with real providers (manual)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Provider unavailability during consensus | Medium | Graceful degradation - continue with available providers |
| File storage race conditions | High | Use filelock library for thread-safe operations |
| Context window overflow in long conversations | Medium | Token budgeting in context building |
| Async complexity in consensus | Medium | Use established asyncio patterns from best practices |
| Feature flag not checked | High | Copy pattern from provider.py exactly |

## Success Criteria

- [ ] All 8 actions (chat, consensus, thinkdeep, ideate, route, thread-list, thread-get, thread-delete) work correctly
- [ ] Response envelopes match v2 schema with proper error codes
- [ ] Conversation threading persists and continues across invocations
- [ ] Multi-model consensus executes in parallel with synthesis
- [ ] Hypothesis tracking maintains state across investigation steps
- [ ] Idea generation progresses through all 4 phases
- [ ] Feature flag gates access to research tools
- [ ] Test coverage > 80% for new code
- [ ] Best practices compliance (§02-§15) verified

## Dependencies

- `foundry_mcp.core.providers` - For LLM execution
- `foundry_mcp.core.responses` - For response envelopes
- `foundry_mcp.core.feature_flags` - For feature gating
- `foundry_mcp.tools.unified.router` - For ActionRouter pattern
- `pydantic` - For data models
- `filelock` - For thread-safe file operations
- `asyncio` - For parallel execution
