# MCP Tool Industry Best Practices

> A comprehensive guide to building reliable, secure, and maintainable MCP tools.

This documentation follows [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119) conventions:
- **MUST** / **REQUIRED** - Absolute requirements
- **SHOULD** / **RECOMMENDED** - Best practice with valid exceptions
- **MAY** / **OPTIONAL** - Truly optional

## Quick Links

| Category | Documents | Description |
|----------|-----------|-------------|
| **Design** | [Versioned Contracts](./01-versioned-contracts.md), [Tool Discovery](./13-tool-discovery.md) | Schema design & API contracts |
| **Implementation** | [Envelopes](./02-envelopes-metadata.md), [Helpers](./03-serialization-helpers.md), [Validation](./04-validation-input-hygiene.md) | Core implementation patterns |
| **Operations** | [Observability](./05-observability-telemetry.md), [Resilience](./12-timeout-resilience.md) | Production concerns |
| **Data Flow** | [Pagination](./06-pagination-streaming.md), [Errors](./07-error-semantics.md) | Request/response handling |
| **Security** | [Trust Boundaries](./08-security-trust-boundaries.md) | Security patterns |
| **Process** | [Spec-Driven Dev](./09-spec-driven-development.md), [Testing](./10-testing-fixtures.md) | Development workflow |
| **Integration** | [AI Integration](./11-ai-llm-integration.md), [Resilience](./12-timeout-resilience.md), [Discovery](./13-tool-discovery.md) | LLM patterns & resilience |
| **Advanced** | [Feature Flags](./14-feature-flags.md), [Concurrency](./15-concurrency-patterns.md) | Rollouts & async patterns |

## Document Index

### Core Practices

1. [Stable, Versioned Contracts](./01-versioned-contracts.md) - Schema versioning & backward compatibility
2. [Consistent Envelopes & Metadata](./02-envelopes-metadata.md) - Response structure standards
3. [Shared Serialization Helpers](./03-serialization-helpers.md) - Centralized response creation
4. [Validation, Typing & Input Hygiene](./04-validation-input-hygiene.md) - Input handling & security
5. [Tool Naming Conventions](../codebase_standards/naming-conventions.md) - Domain-prefixed, LLM-friendly tool names

### Operational Excellence

5. [Observability & Telemetry](./05-observability-telemetry.md) - Logging, tracing, metrics
6. [Pagination, Streaming & Idempotency](./06-pagination-streaming.md) - Data transfer patterns
7. [Graceful Degradation & Error Semantics](./07-error-semantics.md) - Error handling philosophy

### Security & Trust

8. [Security & Trust Boundaries](./08-security-trust-boundaries.md) - Security patterns for MCP tools

### Development Process

9. [Spec-Driven Development](./09-spec-driven-development.md) - Documentation-first approach
10. [Testing & Fixtures](./10-testing-fixtures.md) - Test strategy & maintenance

### Integration & Discovery

11. [AI/LLM Integration Patterns](./11-ai-llm-integration.md) - LLM-specific design considerations
12. [Timeout & Resilience](./12-timeout-resilience.md) - Circuit breakers, retries, timeouts
13. [Tool Metadata & Discovery](./13-tool-discovery.md) - Tool registration & capability negotiation

### Advanced Patterns

14. [Feature Flags & Gradual Rollouts](./14-feature-flags.md) - Controlled feature deployment
15. [Concurrency & Async Patterns](./15-concurrency-patterns.md) - Async/await and parallelism

## How to Use This Guide

### For New Tool Development

1. Start with [Tool Discovery](./13-tool-discovery.md) to design your tool's interface
2. Review [Versioned Contracts](./01-versioned-contracts.md) for schema design
3. Implement using [Envelopes](./02-envelopes-metadata.md) and [Helpers](./03-serialization-helpers.md)
4. Add [Validation](./04-validation-input-hygiene.md) and [Security](./08-security-trust-boundaries.md)
5. Set up [Observability](./05-observability-telemetry.md) and [Testing](./10-testing-fixtures.md)
6. Align verbs/prefixes with [Tool Naming Conventions](../codebase_standards/naming-conventions.md)

### For Code Review

Use this checklist:
- [ ] Uses standardized envelope via helpers ([§2](./02-envelopes-metadata.md), [§3](./03-serialization-helpers.md))
- [ ] Validates inputs early ([§4](./04-validation-input-hygiene.md))
- [ ] Handles errors gracefully ([§7](./07-error-semantics.md))
- [ ] Includes appropriate logging ([§5](./05-observability-telemetry.md))
- [ ] Respects security boundaries ([§8](./08-security-trust-boundaries.md))
- [ ] Has adequate test coverage ([§10](./10-testing-fixtures.md))
- [ ] Handles timeouts appropriately ([§12](./12-timeout-resilience.md))
- [ ] Uses async correctly if I/O-bound ([§15](./15-concurrency-patterns.md))

### For Debugging Production Issues

1. Check [Observability](./05-observability-telemetry.md) for logging patterns
2. Review [Error Semantics](./07-error-semantics.md) for error classification
3. See [Resilience](./12-timeout-resilience.md) for timeout/retry issues

## Related Documentation

- [Response Schema Reference](../codebase_standards/mcp_response_schema.md) - Canonical response contract
- [Response Helpers](../../src/foundry_mcp/core/responses.py) - Implementation code
- [Capabilities Manifest](../../mcp/capabilities_manifest.json) - Tool discovery and feature flags

## Environment & Setup Tools

The following environment tools support SDD workflow initialization and verification:

| Tool | Description | Feature Flag |
|------|-------------|--------------|
| `env_verify_toolchain` | Verify local CLI/toolchain availability (git, python, node, SDD CLI) | `environment_tools` |
| `env_init_workspace` | Bootstrap working directory with specs folders and config | `environment_tools` |
| `env_detect_topology` | Auto-detect repository layout for specs/docs directories | `environment_tools` |
| `env_verify_environment` | Validate OS packages, runtimes, and credentials | `environment_tools` |

These tools are gated by the `environment_tools` feature flag (beta, 100% rollout). The `env_auto_fix` flag (experimental) enables automatic fix capabilities for environment issues.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.2.0 | 2025-11-27 | Added environment tools section, capabilities manifest reference |
| 2.1.0 | 2025-11-26 | Added feature flags, concurrency patterns, multi-tenancy |
| 2.0.0 | 2025-11-26 | Sharded structure; added AI/LLM patterns, resilience, tool discovery |
| 1.0.0 | 2025-11-26 | Initial consolidated document |

---

*Last updated: 2025-11-27*
