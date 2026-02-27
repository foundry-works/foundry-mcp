"""SSRF protection, prompt injection sanitization, and context builders."""

from __future__ import annotations

import html as html_module
import ipaddress
import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from foundry_mcp.core.research.workflows.deep_research._content_dedup import (
    NoveltyTag,
)

if TYPE_CHECKING:
    from foundry_mcp.core.research.models.deep_research import DeepResearchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSRF protection for extract URLs
# ---------------------------------------------------------------------------

# Private/reserved IPv4/IPv6 networks that must be blocked.
_BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),  # link-local + cloud metadata
    ipaddress.IPv4Network("0.0.0.0/8"),  # "this" network (RFC 1122)
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),  # unique local
    ipaddress.IPv6Network("fe80::/10"),  # link-local
    ipaddress.IPv6Network("ff00::/8"),  # multicast
]

# Hostname suffixes that resolve to internal/local services.
_BLOCKED_HOSTNAME_SUFFIXES: tuple[str, ...] = (
    ".local",
    ".internal",
    ".localhost",
)


def validate_extract_url(url: str, *, resolve_dns: bool = False) -> bool:
    """Validate a URL is safe for server-side extraction (SSRF protection).

    Rejects:
    - Non-HTTP(S) schemes
    - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
    - Loopback (127.x, localhost, ::1)
    - Cloud metadata endpoint (169.254.169.254)
    - Link-local addresses (169.254.x)
    - IPv6 unique-local and link-local ranges

    **TOCTOU note**: Without ``resolve_dns=True`` this function only validates
    URL syntax and blocks obvious IP-literal attacks.  DNS rebinding attacks
    (a hostname that resolves to a private IP) are NOT blocked.  When the
    fetch layer (e.g. Tavily API) operates in its own network context the
    risk is limited, but for URLs sourced from untrusted input (LLM output)
    callers should pass ``resolve_dns=True`` for an additional
    ``socket.getaddrinfo`` check.  Even with DNS resolution there is an
    inherent TOCTOU gap — the IP may change between validation and fetch.

    Args:
        url: URL string to validate
        resolve_dns: When True, resolve the hostname via ``socket.getaddrinfo``
            and reject if any resolved address falls within ``_BLOCKED_NETWORKS``.

    Returns:
        True if the URL is safe for extraction, False otherwise
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False

    # Require HTTP(S) scheme
    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # Block localhost by name
    if hostname in ("localhost", "localhost.localdomain"):
        return False

    # Block internal hostname suffixes (.local, .internal, .localhost)
    lower_host = hostname.lower()
    for suffix in _BLOCKED_HOSTNAME_SUFFIXES:
        if lower_host.endswith(suffix):
            return False

    # Try to parse as IP address and check against blocked networks
    try:
        addr = ipaddress.ip_address(hostname)
        # Check IPv4-mapped IPv6 addresses (e.g. ::ffff:169.254.169.254)
        # to prevent SSRF bypass of IPv4 blocked networks via IPv6 encoding.
        mapped = getattr(addr, "ipv4_mapped", None)
        if mapped is not None:
            for network in _BLOCKED_NETWORKS:
                if mapped in network:
                    return False
        for network in _BLOCKED_NETWORKS:
            if addr in network:
                return False
    except ValueError:
        # Not an IP literal — optionally resolve DNS to catch rebinding
        if resolve_dns:
            try:
                import socket

                infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
                for *_, sockaddr in infos:
                    resolved_ip = ipaddress.ip_address(sockaddr[0])
                    resolved_mapped = getattr(resolved_ip, "ipv4_mapped", None)
                    if resolved_mapped is not None:
                        for network in _BLOCKED_NETWORKS:
                            if resolved_mapped in network:
                                return False
                    for network in _BLOCKED_NETWORKS:
                        if resolved_ip in network:
                            return False
            except (socket.gaierror, OSError, ValueError):
                # DNS resolution failed — treat as unsafe
                return False

    return True


# ---------------------------------------------------------------------------
# Prompt injection surface reduction
# ---------------------------------------------------------------------------

# XML-like tags that could override LLM instructions when injected via
# web-scraped content.  Matched case-insensitively.
_INJECTION_TAG_PATTERN: re.Pattern[str] = re.compile(
    r"<\s*/?\s*(?:system|instructions|tool_use|tool_result|human|assistant"
    r"|function_calls|(?:antml:)?invoke|prompt"
    r"|message|messages|context|document|thinking|reflection"
    r"|example|result|output|user|role|artifact|search_results"
    r"|function_declaration|function_response)(?=[\s/>_]|$)[^>]*>",
    re.IGNORECASE,
)

# OpenAI-family special tokens (e.g. <|im_start|>, <|im_end|>, <|endoftext|>).
_SPECIAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"<\|.*?\|>",
)

# Markdown headings that mimic system-level sections.
_INJECTION_HEADING_PATTERN: re.Pattern[str] = re.compile(
    r"^#{1,3}\s+(?:SYSTEM|INSTRUCTIONS|TOOL[_ ]USE|HUMAN|ASSISTANT)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def sanitize_external_content(text: str) -> str:
    """Strip prompt-injection vectors from web-scraped content.

    Removes XML-like tags, special tokens, and markdown headings that could
    override LLM instructions when external content is interpolated into
    supervision prompts.  Normal content (citations, formatting, data) is
    preserved.

    Pre-processing steps handle common obfuscation techniques:
    - Zero-width Unicode characters (U+200B, U+200C, U+200D, U+FEFF) are
      stripped so they can't break up tag names.
    - HTML entities (``&lt;`` → ``<``) are decoded so entity-encoded tags
      are caught by the regex patterns.

    Args:
        text: Raw external content (source title, snippet, etc.)

    Returns:
        Sanitized text with injection vectors removed.
    """
    if not text:
        return text
    # Strip zero-width and invisible characters that could obfuscate injection tags
    sanitized = text.translate(
        {
            0x00AD: None,  # Soft Hyphen
            0x034F: None,  # Combining Grapheme Joiner
            0x180E: None,  # Mongolian Vowel Separator
            0x200B: None,  # Zero Width Space
            0x200C: None,  # Zero Width Non-Joiner
            0x200D: None,  # Zero Width Joiner
            0x2060: None,  # Word Joiner
            0x2061: None,  # Function Application
            0x2062: None,  # Invisible Times
            0x2063: None,  # Invisible Separator
            0x2064: None,  # Invisible Plus
            0xFEFF: None,  # Zero Width No-Break Space (BOM)
        }
    )
    # Decode HTML entities so entity-encoded tags are caught.
    # Loop until stable to defeat multi-layer encoding (e.g. &amp;lt; → &lt; → <).
    _MAX_UNESCAPE_ROUNDS = 5
    for _ in range(_MAX_UNESCAPE_ROUNDS):
        unescaped = html_module.unescape(sanitized)
        if unescaped == sanitized:
            break
        sanitized = unescaped
    # Strip XML-like instruction/override tags
    sanitized = _INJECTION_TAG_PATTERN.sub("", sanitized)
    # Strip OpenAI-family special tokens
    sanitized = _SPECIAL_TOKEN_PATTERN.sub("", sanitized)
    # Strip markdown heading injection patterns
    sanitized = _INJECTION_HEADING_PATTERN.sub("", sanitized)
    return sanitized


def build_sanitized_context(state: "DeepResearchState") -> dict[str, str]:
    """Return pre-sanitized versions of common state fields for prompt building.

    Centralises the ``sanitize_external_content`` calls for the four fields
    that are interpolated into almost every LLM prompt across brief, planning,
    and supervision phases.  Callers can destructure the dict instead of
    repeating inline sanitisation.

    Args:
        state: Current research state

    Returns:
        Dict with keys ``original_query``, ``system_prompt``, ``constraints``,
        and ``research_brief`` — all sanitised and safe for prompt interpolation.
    """
    return {
        "original_query": sanitize_external_content(state.original_query or ""),
        "system_prompt": sanitize_external_content(state.system_prompt or ""),
        "constraints": sanitize_external_content(
            str(state.clarification_constraints) if state.clarification_constraints else ""
        ),
        "research_brief": sanitize_external_content(state.research_brief or ""),
    }


def build_novelty_summary(
    tags: list[NoveltyTag],
) -> str:
    """Build a one-line novelty summary for the search results header.

    Args:
        tags: List of NoveltyTag results for all sources in a search batch.

    Returns:
        Summary string like ``"Novelty: 3 new, 1 related, 1 duplicate out of 5 results"``
    """
    new_count = sum(1 for t in tags if t.category == "new")
    related_count = sum(1 for t in tags if t.category == "related")
    dup_count = sum(1 for t in tags if t.category == "duplicate")
    total = len(tags)
    return f"Novelty: {new_count} new, {related_count} related, {dup_count} duplicate out of {total} results"
