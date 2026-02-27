"""Coverage assessment helpers for the supervision phase.

Extracted from ``supervision.py`` to isolate coverage-analysis utility code
from the LLM orchestration loop.  Every function here is either a pure
function or a simple state-mutating helper that takes explicit parameters
(no ``self``).
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import urlparse

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
)

# ======================================================================
# Verdict / issue parsing patterns
# ======================================================================

# Pre-compiled patterns for verdict/issue parsing (used by critique_has_issues)
VERDICT_NO_ISSUES_RE = re.compile(r"VERDICT\s*:\s*NO[_\s]?ISSUES", re.IGNORECASE)
VERDICT_REVISION_RE = re.compile(r"VERDICT\s*:\s*REVISION[_\s]?NEEDED", re.IGNORECASE)
# Match "ISSUE:" as a structured marker — either at line start, after numbering,
# or after a label prefix (e.g. "Redundancy: ISSUE:"). Avoids false positives
# on conversational uses like "this is not an issue" by requiring the colon.
ISSUE_MARKER_RE = re.compile(r"(?:^|\.\s+|:\s*)ISSUE\s*:", re.IGNORECASE | re.MULTILINE)


def critique_has_issues(critique_text: str) -> bool:
    """Check whether the critique indicates issues that need revision.

    Looks for the ``VERDICT:`` line with flexible whitespace/formatting.
    Falls back to checking for ``ISSUE:`` markers at line starts if no
    verdict is found.

    Args:
        critique_text: Raw text from the critique LLM call

    Returns:
        True if revision is needed, False if all criteria passed
    """
    if VERDICT_NO_ISSUES_RE.search(critique_text):
        return False
    if VERDICT_REVISION_RE.search(critique_text):
        return True
    # Fallback: check for ISSUE markers at line starts to reduce false positives
    return bool(ISSUE_MARKER_RE.search(critique_text))


# ======================================================================
# Per-query coverage data
# ======================================================================


def build_per_query_coverage(
    state: DeepResearchState,
) -> list[dict[str, Any]]:
    """Build per-sub-query coverage data for supervision assessment.

    For each completed or failed sub-query, computes:
    - Source count (from source_ids on the sub-query)
    - Quality distribution (HIGH/MEDIUM/LOW/UNKNOWN counts)
    - Unique domain count (from source URLs)
    - Findings summary from topic research results
    - Compressed findings excerpt (when inline compression is available)

    Args:
        state: Current research state

    Returns:
        List of coverage dicts, one per non-pending sub-query
    """
    coverage: list[dict[str, Any]] = []

    # Build lookup for topic research results by sub_query_id
    topic_results_by_sq: dict[str, Any] = {}
    for tr in state.topic_research_results:
        topic_results_by_sq[tr.sub_query_id] = tr

    # Build sub_query_id → list[source] lookup to avoid O(n×m) filtering
    sources_by_sq: dict[str, list[Any]] = {}
    for s in state.sources:
        if s.sub_query_id is not None:
            sources_by_sq.setdefault(s.sub_query_id, []).append(s)

    for sq in state.sub_queries:
        if sq.status == "pending":
            continue

        # Look up sources for this sub-query via pre-built index
        sq_sources = sources_by_sq.get(sq.id, [])
        source_count = len(sq_sources)

        # Quality distribution
        quality_dist: dict[str, int] = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "UNKNOWN": 0,
        }
        for s in sq_sources:
            quality_key = s.quality.value.upper() if s.quality else "UNKNOWN"
            if quality_key in quality_dist:
                quality_dist[quality_key] += 1
            else:
                quality_dist["UNKNOWN"] += 1

        # Unique domains
        domains: set[str] = set()
        for s in sq_sources:
            if s.url:
                try:
                    parsed_url = urlparse(s.url)
                    if parsed_url.netloc:
                        domains.add(parsed_url.netloc)
                except Exception:
                    pass

        # Findings summary from topic research
        topic_result = topic_results_by_sq.get(sq.id)
        findings_summary = None
        compressed_findings_excerpt = None
        if topic_result:
            if topic_result.per_topic_summary:
                findings_summary = topic_result.per_topic_summary[:500]
            # Prefer supervisor_summary (structured for gap analysis) over
            # raw compressed_findings truncation when available.
            if topic_result.supervisor_summary:
                compressed_findings_excerpt = topic_result.supervisor_summary
            elif topic_result.compressed_findings:
                compressed_findings_excerpt = topic_result.compressed_findings[:2000]

        coverage.append(
            {
                "sub_query_id": sq.id,
                "query": sq.query,
                "status": sq.status,
                "source_count": source_count,
                "quality_distribution": quality_dist,
                "unique_domains": len(domains),
                "domain_list": sorted(domains),
                "findings_summary": findings_summary,
                "compressed_findings_excerpt": compressed_findings_excerpt,
            }
        )

    return coverage


# ======================================================================
# Coverage snapshots and deltas
# ======================================================================


def store_coverage_snapshot(
    state: DeepResearchState,
    coverage_data: list[dict[str, Any]],
    suffix: Optional[str] = None,
) -> None:
    """Store a coverage snapshot for the current supervision round.

    Snapshots are keyed by ``"{round}_{suffix}"`` (when *suffix* is given) or
    ``"{round}"`` (legacy / backward-compatible) in
    ``state.metadata["coverage_snapshots"]`` and used by
    ``compute_coverage_delta`` to produce round-over-round deltas.

    Use ``suffix="pre"`` before directive execution and ``suffix="post"``
    after execution so that the pre-directive snapshot is not silently
    overwritten.

    Each snapshot entry stores the lightweight fields needed for delta
    comparison: source_count, unique_domains, and status.

    Args:
        state: Current research state
        coverage_data: Per-sub-query coverage from ``build_per_query_coverage``
        suffix: Optional label (e.g. ``"pre"``, ``"post"``) appended to the
            round key.  When ``None``, the key is the bare round number
            (backward-compatible).
    """
    snapshots = state.metadata.setdefault("coverage_snapshots", {})
    snapshot: dict[str, dict[str, Any]] = {}
    for entry in coverage_data:
        snapshot[entry["sub_query_id"]] = {
            "query": entry["query"],
            "source_count": entry["source_count"],
            "unique_domains": entry["unique_domains"],
            "status": entry["status"],
        }
    # Store with string key (JSON-safe)
    key = f"{state.supervision_round}_{suffix}" if suffix else str(state.supervision_round)
    snapshots[key] = snapshot


def compute_coverage_delta(
    state: DeepResearchState,
    coverage_data: list[dict[str, Any]],
    min_sources: int = 3,
) -> Optional[str]:
    """Compute a coverage delta between the current and previous supervision round.

    Compares per-query source counts, domain counts, and status against the
    snapshot from the previous round.  Produces a compact summary like::

        Coverage delta (round 0 → 1):
        - query_1: +2 sources, +1 domain (now: 4 sources, 3 domains) — SUFFICIENT
        - query_2: +0 sources — STILL INSUFFICIENT
        - query_3 [NEW]: 1 source from this round's directives

    Returns ``None`` if there is no previous snapshot (round 0) or if
    coverage_snapshots metadata is missing.

    Args:
        state: Current research state
        coverage_data: Current per-sub-query coverage
        min_sources: Minimum sources per query for "SUFFICIENT" label

    Returns:
        Compact delta summary string, or ``None`` if no previous snapshot exists
    """
    snapshots = state.metadata.get("coverage_snapshots", {})
    prev_round = state.supervision_round - 1

    # Prefer suffixed keys ("{round}_post" / "{round}_pre") written by the
    # updated store_coverage_snapshot; fall back to bare round keys for
    # backward compatibility with snapshots written before the suffix change.
    prev_snapshot = snapshots.get(f"{prev_round}_post") or snapshots.get(str(prev_round))
    if prev_snapshot is None:
        return None

    # Build current lookup
    current_by_id: dict[str, dict[str, Any]] = {}
    for entry in coverage_data:
        current_by_id[entry["sub_query_id"]] = entry

    lines: list[str] = [
        f"Coverage delta (round {prev_round} → {state.supervision_round}):",
    ]

    # Track IDs we've seen to detect new queries
    prev_ids = set(prev_snapshot.keys())
    current_ids = set(current_by_id.keys())

    # Process queries that existed in previous round
    for sq_id in sorted(prev_ids & current_ids):
        prev = prev_snapshot[sq_id]
        curr = current_by_id[sq_id]
        src_delta = curr["source_count"] - prev["source_count"]
        dom_delta = curr["unique_domains"] - prev["unique_domains"]

        # Determine sufficiency label
        if curr["source_count"] >= min_sources:
            if prev["source_count"] < min_sources:
                status_label = "NEWLY SUFFICIENT"
            else:
                status_label = "SUFFICIENT"
        else:
            status_label = "STILL INSUFFICIENT"

        src_sign = f"+{src_delta}" if src_delta >= 0 else str(src_delta)
        dom_sign = f"+{dom_delta}" if dom_delta >= 0 else str(dom_delta)

        query_text = curr.get("query", prev.get("query", sq_id))[:80]
        lines.append(
            f"- {query_text}: {src_sign} sources, {dom_sign} domains "
            f"(now: {curr['source_count']} sources, {curr['unique_domains']} domains) "
            f"— {status_label}"
        )

    # Process new queries (not in previous round)
    for sq_id in sorted(current_ids - prev_ids):
        curr = current_by_id[sq_id]
        query_text = curr.get("query", sq_id)[:80]
        lines.append(f"- {query_text} [NEW]: {curr['source_count']} sources, {curr['unique_domains']} domains")

    # Process removed queries (in previous but not current — rare)
    for sq_id in sorted(prev_ids - current_ids):
        prev = prev_snapshot[sq_id]
        query_text = prev.get("query", sq_id)[:80]
        lines.append(f"- {query_text} [REMOVED]")

    return "\n".join(lines)


# ======================================================================
# Heuristic coverage assessment
# ======================================================================


def assess_coverage_heuristic(
    state: DeepResearchState,
    min_sources: int,
    config: Any,
) -> dict[str, Any]:
    """Assess coverage using multi-dimensional confidence scoring.

    Computes a confidence score from three dimensions:

    - **Source adequacy**: minimum of ``min(1.0, sources / min_sources)``
      across all completed sub-queries — uses ``min()`` so coverage is
      only sufficient when *every* sub-query meets its threshold.
    - **Domain diversity**: ``unique_domains / (query_count * 2)`` capped
      at 1.0 — rewards breadth of sourcing.
    - **Query completion rate**: ``completed / total`` sub-queries.

    The overall confidence is a weighted mean (configurable via
    ``deep_research_coverage_confidence_weights``).  When
    ``confidence >= threshold`` (default 0.75), the heuristic declares
    coverage sufficient and sets ``should_continue_gathering=False``.

    Args:
        state: Current research state
        min_sources: Minimum sources per sub-query for "sufficient" coverage
        config: Research config (for weights and threshold)

    Returns:
        Dict with coverage assessment, confidence breakdown, and
        should_continue_gathering flag
    """
    completed = state.completed_sub_queries()
    total_queries = len(state.sub_queries)

    if not completed:
        return {
            "overall_coverage": "insufficient",
            "should_continue_gathering": False,
            "queries_assessed": 0,
            "queries_sufficient": 0,
            "confidence": 0.0,
            "confidence_dimensions": {
                "source_adequacy": 0.0,
                "domain_diversity": 0.0,
                "query_completion_rate": 0.0,
            },
            "dominant_factors": [],
            "weak_factors": ["source_adequacy", "domain_diversity", "query_completion_rate"],
        }

    # Build sub_query_id → list[source] lookup to avoid O(n×m) filtering
    sources_by_sq: dict[str, list[Any]] = {}
    for s in state.sources:
        if s.sub_query_id is not None:
            sources_by_sq.setdefault(s.sub_query_id, []).append(s)

    # --- Dimension 1: Source adequacy ---
    source_ratios: list[float] = []
    sufficient_count = 0
    for sq in completed:
        sq_sources = sources_by_sq.get(sq.id, [])
        count = len(sq_sources)
        ratio = min(1.0, count / min_sources) if min_sources > 0 else 1.0
        source_ratios.append(ratio)
        if count >= min_sources:
            sufficient_count += 1
    source_adequacy = min(source_ratios)

    # --- Dimension 2: Domain diversity ---
    all_domains: set[str] = set()
    for s in state.sources:
        if s.url:
            try:
                parsed = urlparse(s.url)
                if parsed.netloc:
                    all_domains.add(parsed.netloc)
            except Exception:
                pass
    query_count = len(completed)
    domain_diversity = min(1.0, len(all_domains) / (query_count * 2)) if query_count > 0 else 0.0

    # --- Dimension 3: Query completion rate ---
    query_completion_rate = len(completed) / total_queries if total_queries > 0 else 0.0

    # --- Weighted confidence ---
    weights = getattr(
        config,
        "deep_research_coverage_confidence_weights",
        None,
    ) or {"source_adequacy": 0.5, "domain_diversity": 0.2, "query_completion_rate": 0.3}
    total_weight = sum(weights.values())
    dimensions = {
        "source_adequacy": source_adequacy,
        "domain_diversity": domain_diversity,
        "query_completion_rate": query_completion_rate,
    }
    confidence = (
        sum(dimensions[k] * weights.get(k, 0.0) for k in dimensions) / total_weight if total_weight > 0 else 0.0
    )

    # --- Factor classification ---
    strong_threshold = 0.7
    weak_threshold = 0.5
    dominant_factors = [k for k, v in dimensions.items() if v >= strong_threshold]
    weak_factors = [k for k, v in dimensions.items() if v < weak_threshold]

    # --- Overall coverage label (backward-compatible) ---
    if sufficient_count == query_count:
        overall = "sufficient"
    elif sufficient_count > 0:
        overall = "partial"
    else:
        overall = "insufficient"

    # --- Confidence-based decision ---
    threshold = getattr(
        config,
        "deep_research_coverage_confidence_threshold",
        0.75,
    )
    should_continue = confidence < threshold

    return {
        "overall_coverage": overall,
        "should_continue_gathering": should_continue,
        "queries_assessed": query_count,
        "queries_sufficient": sufficient_count,
        "confidence": round(confidence, 4),
        "confidence_threshold": threshold,
        "confidence_dimensions": dimensions,
        "dominant_factors": dominant_factors,
        "weak_factors": weak_factors,
    }
