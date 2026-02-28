# PLAN: DOCX Extraction Support for Deep Research

## Problem

When deep research encounters a `.docx` URL (e.g., from Tavily search/extract), the raw
binary ZIP content flows unfiltered into the summarization pipeline. The LLM receives
garbled bytes and returns an error like:

> Unable to process content. The provided input appears to be binary or compressed data
> from a Microsoft Word document (.docx file format).

There is no detection, extraction, or graceful fallback for Office document formats.

## Goal

Add `.docx` text extraction that mirrors the existing PDF extractor architecture, with
binary content detection as a safety net for all unrecognized binary formats.

## Architecture

### Existing Content Flow (where .docx breaks)

```
Tavily Search → raw_content (binary .docx bytes as string)
       ↓
SourceSummarizer.summarize_source(content)  ← garbled content sent to LLM
       ↓
LLM returns error / garbage summary
       ↓
Stored as source.content — pollutes downstream synthesis
```

### Proposed Content Flow

```
Tavily Search → raw_content
       ↓
content_classifier.classify(content, url)   ← NEW: detect binary/docx
       ↓
  ├── TEXT/HTML → pass through (existing path)
  ├── PDF       → PDFExtractor (existing)
  ├── DOCX      → DocxExtractor.extract(bytes) ← NEW
  └── BINARY    → skip with warning            ← NEW graceful fallback
       ↓
SourceSummarizer.summarize_source(extracted_text)
```

### Key Design Decisions

1. **Separate `docx_extractor.py`** — mirrors `pdf_extractor.py` structure (same SSRF
   guards, result dataclass, async API, optional metrics). Keeps modules focused.

2. **`content_classifier.py`** — lightweight module that detects content type via
   magic bytes + URL extension + Content-Type header. Returns an enum
   (`TEXT | HTML | PDF | DOCX | BINARY_UNKNOWN`). Centralizes detection so both
   extractors and the summarizer can use it.

3. **Integration at SourceSummarizer level** — the binary guard goes into
   `SourceSummarizer.summarize_source()` (in `providers/shared.py`) since that's
   the last chokepoint before content hits the LLM. Also integrate in
   `TavilySearchProvider._apply_source_summarization` for the Tavily search path.

4. **`python-docx` as optional dependency** — follows the `pdf = [...]` pattern.
   Graceful degradation when not installed (log warning, skip extraction).

---

## Phases

### Phase 1: Content Classifier (`content_classifier.py`)

**Files:**
- `src/foundry_mcp/core/research/content_classifier.py` (NEW)
- `tests/core/research/test_content_classifier.py` (NEW)

**Details:**

Create a `ContentType` enum and `classify_content()` function:

```python
class ContentType(Enum):
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    BINARY_UNKNOWN = "binary_unknown"

def classify_content(
    content: str | bytes,
    *,
    url: str | None = None,
    content_type_header: str | None = None,
) -> ContentType:
    """Classify content type using magic bytes, URL extension, and Content-Type header."""
```

Detection strategy (ordered by confidence):
1. **Magic bytes** (highest confidence):
   - `%PDF-` → PDF
   - `PK\x03\x04` (ZIP header) → probe for `word/document.xml` entry → DOCX
   - `PK\x03\x04` without `word/` → BINARY_UNKNOWN (generic ZIP)
2. **Content-Type header** (if provided):
   - `application/vnd.openxmlformats-officedocument.wordprocessingml.document` → DOCX
   - `application/pdf` → PDF
3. **URL extension** (fallback):
   - `.docx` → DOCX, `.pdf` → PDF
4. **Binary heuristic** — if content is `bytes` or contains high ratio of
   non-printable chars → BINARY_UNKNOWN
5. **Default** — check for HTML tags → HTML, else TEXT

Also provide `is_binary_content(content: str) -> bool` as a fast check for the
summarizer guard — detects non-printable byte sequences commonly seen when binary
data is decoded as a string.

**Tests:** Magic bytes detection, URL extension fallback, Content-Type header parsing,
binary heuristic, edge cases (empty content, mixed signals).

---

### Phase 2: DOCX Extractor (`docx_extractor.py`)

**Files:**
- `src/foundry_mcp/core/research/docx_extractor.py` (NEW)
- `tests/core/research/test_docx_extractor.py` (NEW)
- `pyproject.toml` (MODIFY — add `docx` optional dependency group)
- `src/foundry_mcp/core/errors/research.py` (MODIFY — add DOCX error classes)

**Details:**

Mirror `PDFExtractor` architecture:

```python
@dataclass
class DocxExtractionResult:
    text: str
    warnings: list[str]
    paragraph_count: int
    table_count: int
    # No page_offsets — DOCX doesn't have fixed pages

    @property
    def success(self) -> bool: ...
    @property
    def has_warnings(self) -> bool: ...

class DocxExtractor:
    def __init__(
        self,
        max_size: int = DEFAULT_MAX_DOCX_SIZE,  # 10 MB
        fetch_timeout: float = DEFAULT_FETCH_TIMEOUT,  # 30s
    ): ...

    async def extract(
        self,
        source: bytes | io.BytesIO,
        *,
        validate_magic: bool = True,
    ) -> DocxExtractionResult: ...

    async def extract_from_url(self, url: str) -> DocxExtractionResult: ...
```

Implementation notes:
- Uses `python-docx` (lazy import, same pattern as `pdfminer.six`) for extraction
- Extracts paragraph text + table cell text (tables concatenated row-by-row)
- SSRF protection: reuse `validate_url_for_ssrf()` from `pdf_extractor.py`
  (consider extracting to shared `_url_security.py` if duplication is excessive,
  or just import from `pdf_extractor`)
- Magic byte validation: `PK\x03\x04` header + probe for `word/document.xml`
  entry via `zipfile`
- Content-Type validation: accept
  `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
  and `application/octet-stream`
- Runs CPU-bound `python-docx` operations in thread pool (`asyncio.to_thread`)
- Optional Prometheus metrics (same pattern as PDF extractor)

**Error classes** (add to `core/errors/research.py`):

```python
class DocxSecurityError(Exception):
    """Base exception for DOCX security violations."""

class InvalidDocxError(DocxSecurityError):
    """Raised when DOCX validation fails (magic bytes, content-type)."""

class DocxSizeError(DocxSecurityError):
    """Raised when DOCX exceeds size limits."""
```

**Dependency** (add to `pyproject.toml`):

```toml
docx = [
    "python-docx>=1.1.0",
]
```

**Tests:** Valid .docx extraction, magic byte validation, SSRF protection (reuse
PDF test patterns), size limits, table extraction, empty document, corrupted file
handling.

---

### Phase 3: Integration — Binary Guard + Extraction Wiring

**Files:**
- `src/foundry_mcp/core/research/providers/shared.py` (MODIFY)
- `src/foundry_mcp/core/research/providers/tavily.py` (MODIFY)
- `src/foundry_mcp/core/research/providers/tavily_extract.py` (MODIFY)
- `src/foundry_mcp/core/research/document_digest/digestor.py` (MODIFY)
- `tests/core/research/test_binary_content_guard.py` (NEW)

**Details:**

#### 3a. Binary content guard in SourceSummarizer

In `providers/shared.py`, add a guard at the top of `summarize_source()`:

```python
async def summarize_source(self, content: str) -> SourceSummarizationResult:
    # Binary content guard — prevent sending garbled bytes to LLM
    from foundry_mcp.core.research.content_classifier import is_binary_content
    if is_binary_content(content):
        logger.warning("Skipping summarization: binary content detected")
        return SourceSummarizationResult(
            executive_summary="[Content skipped: binary/non-text document detected]",
            key_excerpts=[],
            input_tokens=0,
            output_tokens=0,
        )
    # ... existing code
```

#### 3b. Content type detection + extraction in Tavily search path

In `providers/tavily.py`, in `_apply_source_summarization()`, before summarizing:

```python
# Detect binary/docx content and extract text before summarization
from foundry_mcp.core.research.content_classifier import classify_content, ContentType

for source in sources:
    if source.content:
        content_type = classify_content(source.content, url=source.url)
        if content_type == ContentType.DOCX:
            extracted = await self._extract_docx_content(source.content)
            if extracted:
                source.content = extracted
        elif content_type == ContentType.BINARY_UNKNOWN:
            logger.warning("Binary content detected for %s, skipping", source.url)
            source.content = None  # Will be skipped by summarizer
```

#### 3c. Content type detection in Tavily Extract path

In `providers/tavily_extract.py`, add similar detection after content retrieval.

#### 3d. DocumentDigestor awareness

Add `docx_extractor: DocxExtractor` to `DocumentDigestor.__init__` alongside
`pdf_extractor`, following the same pattern. This enables the digest pipeline
to handle .docx URLs passed directly.

**Tests:** Binary guard triggers for garbled content, DOCX content gets extracted
before summarization, unknown binary formats are skipped gracefully, normal
text/HTML content passes through unchanged.

---

### Phase 4: Deep Research Workflow Wiring

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/__init__.py` (MODIFY)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/analysis.py` (MODIFY)

**Details:**

- Re-export `DocxExtractor` from the deep research `__init__.py` (same pattern as
  `PDFExtractor` re-export)
- Add `DocxExtractor` re-export in `phases/analysis.py` for test patch targets
- The main integration happens in Phase 3 at the provider level, so this phase is
  primarily about ensuring re-exports and test patch targets are correct

---

### Phase 5: Testing & Verification

**Files:**
- All test files from prior phases
- Integration test verifying end-to-end: `.docx` URL → extracted text → summarization

**Verification steps:**
1. `pytest tests/core/research/test_content_classifier.py` — classifier tests pass
2. `pytest tests/core/research/test_docx_extractor.py` — extractor tests pass
3. `pytest tests/core/research/test_binary_content_guard.py` — guard tests pass
4. `pytest tests/core/research/` — no regressions in existing PDF/research tests
5. `ruff check src/foundry_mcp/core/research/` — no lint issues
6. `pyright src/foundry_mcp/core/research/` — no type errors

---

## Dependency Graph

```
Phase 1 (content_classifier)
    ↓                         Phase 2 (docx_extractor) — can run in parallel
    ↓                              ↓
Phase 3 (integration) ←── depends on both Phase 1 and Phase 2
    ↓
Phase 4 (workflow wiring) ←── depends on Phase 3
    ↓
Phase 5 (verification) ←── depends on all
```

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `python-docx` not installed | Medium | Optional dep with lazy import + graceful skip |
| Binary detection false positives | Low | Conservative heuristic (high non-printable threshold) |
| Large .docx files causing OOM | Low | Size limit (10 MB default, same as PDF) |
| Malicious .docx (zip bombs) | Low | Size limit + `python-docx` doesn't decompress aggressively |
| SSRF via .docx URLs | Low | Reuse existing `validate_url_for_ssrf()` |

## Out of Scope

- `.doc` (legacy binary Word format) — would need `olefile` or similar, rare in web sources
- `.pptx` / `.xlsx` — could follow same pattern later but not needed now
- OCR for scanned documents — separate concern
- Page boundary tracking for DOCX — DOCX is flow-layout, no fixed pages
