# Spec-Driven Development Philosophy

## What is SDD?

Spec-Driven Development (SDD) is a documentation-first methodology where machine-readable specifications serve as the single source of truth for software development. Unlike traditional approaches where documentation follows (or never arrives), SDD requires structured specifications *before* implementation begins.

The core insight: **specifications are contracts designed for AI coding assistants**, not just human readers. By encoding requirements, tasks, dependencies, and decisions in structured JSON, SDD enables AI tools to autonomously discover work, track progress, and maintain alignment between intention and implementation.

Think of SDD as "documentation-first for the AI age."

---

## The Problem SDD Solves

Traditional development suffers from several persistent problems:

**Code first, document later.** Documentation is often an afterthought—written hastily before a release, or not at all. The result: codebases where intent must be reverse-engineered from implementation.

**AI assistants lack context.** When you ask an AI coding assistant to "implement the next feature," it has no structured way to know what "next" means, what dependencies exist, or what decisions have already been made.

**Drift between intentions and code.** Requirements live in wikis, designs in Figma, tasks in Jira, and code in Git. Over time, these drift apart. Which source tells the truth?

**Lost decision rationale.** Six months later, no one remembers *why* a particular approach was chosen. The code shows *what* was built, but not the alternatives considered or trade-offs accepted.

SDD addresses these problems by making specifications the authoritative source—structured, versioned, and designed for both human understanding and machine consumption.

---

## Core Principles

### 1. Specifications are the source of truth

Not code. Not comments. Not wikis. The spec defines what should be built, how it should behave, and what completion looks like. Implementation should reflect the spec; when they diverge, the spec wins (or must be updated first).

### 2. AI-native by design

Specs are JSON-first, structured for LLM consumption. This means:
- Concise, predictable formats that fit in context windows
- Machine-parseable task definitions with clear status
- Explicit dependencies that enable automated task ordering
- Structured metadata that AI tools can query and update

### 3. Progressive task discovery

Rather than presenting an AI assistant with a massive backlog, SDD enables *progressive discovery*: "What's the next actionable task given the current state?" The spec answers this question automatically based on task status, dependencies, and phase order.

### 4. Decision traceability

Every significant decision gets journaled—not buried in commit messages or Slack threads. Journals capture:
- What was decided
- Why that approach was chosen
- What alternatives were considered
- Who made the decision and when

This creates an audit trail that future developers (human or AI) can consult.

### 5. Verification as first-class

Every phase has explicit verification criteria. "Done" isn't a feeling—it's a checklist of automated tests, manual checks, or review sign-offs that must pass before marking work complete.

---

## How SDD Differs from Traditional Approaches

| Aspect | Traditional | SDD |
|--------|-------------|-----|
| Primary consumer | Humans | AI coding assistants |
| Format | Prose, markdown, wikis | Machine-readable JSON |
| When written | After implementation | Before implementation |
| Task discovery | Manual triage | Automated next-task |
| Decision tracking | Ad-hoc (comments, PRs) | Structured journals |
| Completion criteria | Subjective | Explicit verification steps |
| Spec-code sync | Best effort | Enforced by tooling |

---

## The SDD Workflow

SDD follows a lifecycle that moves specifications through distinct phases:

### 1. Plan
Create a specification with phases, tasks, subtasks, and assumptions. Define dependencies between tasks. Capture constraints and requirements. The spec lives in `specs/pending/` until ready.

### 2. Activate
Move the spec from pending to active. This signals that planning is complete and implementation can begin. Active specs live in `specs/active/`.

### 3. Execute
AI assistants (or humans) query for the next actionable task. Each task includes context about what to do, which files are involved, and what dependencies must be met. As work progresses, journal entries capture decisions and blockers.

### 4. Verify
Run verification steps defined for each phase. Record results. If verification fails, the phase remains open until issues are resolved.

### 5. Complete
Mark tasks as done with completion notes. Generate PR descriptions from spec context and journal entries. The spec provides rich context for code review.

### 6. Archive
Move completed specs to `specs/completed/` or `specs/archived/` for historical reference. The decision trail remains available for future consultation.

---

## Why AI-Native Matters

SDD's AI-native design unlocks capabilities that prose documentation cannot provide:

**Token efficiency.** Context windows are limited. Structured JSON specs convey maximum information in minimum tokens, leaving room for code and conversation.

**Autonomous task discovery.** AI assistants can ask "What should I work on next?" and receive a concrete, actionable answer without human intervention.

**Tool chaining.** Specs enable multi-step autonomous workflows: discover task → read dependencies → implement → verify → mark complete → discover next task.

**Fidelity reviews.** Tooling can automatically compare implementation against spec requirements, identifying drift before it becomes technical debt.

**PR generation.** AI can synthesize journal entries, completed tasks, and phase summaries into meaningful PR descriptions—no more "various fixes and improvements."

---

## When to Use SDD

SDD adds value when:

- **Multi-step features** require coordination across files, systems, or time
- **Design-first thinking** would prevent costly rework
- **Decision traceability** matters for compliance, audits, or future maintainers
- **AI-assisted development** is part of your workflow
- **Team collaboration** benefits from a shared, unambiguous source of truth

---

## When NOT to Use SDD

SDD adds overhead that isn't always justified:

- **Trivial changes** — A typo fix doesn't need a spec
- **Exploratory prototyping** — Spec *after* you've learned what works, not before
- **Emergency hotfixes** — Fix first, document retroactively if needed

The goal is pragmatic adoption, not dogmatic process. Use SDD where it adds value; skip it where it doesn't.

---

## Learn More

For detailed technical guidance:
- [Spec-Driven Development Requirements](../mcp_best_practices/09-spec-driven-development.md) — MUST/SHOULD/MAY rules for specs
- [AI/LLM Integration Patterns](../mcp_best_practices/11-ai-llm-integration.md) — Designing for AI consumption
- [CLI Architecture Decision Record](../architecture/adr-001-cli-architecture.md) — JSON-first rationale
