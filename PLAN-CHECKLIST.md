# DOCX Extraction Support — Checklist

## Phase 1: Content Classifier

### `src/foundry_mcp/core/research/content_classifier.py` (NEW)

- [x] Create `ContentType` enum (`TEXT`, `HTML`, `PDF`, `DOCX`, `BINARY_UNKNOWN`)
- [x] Implement `classify_content(content, *, url=None, content_type_header=None)` function
- [x] Magic bytes detection: `%PDF-` → PDF
- [x] Magic bytes detection: `PK\x03\x04` + `word/document.xml` probe → DOCX
- [x] Magic bytes detection: `PK\x03\x04` without `word/` → BINARY_UNKNOWN
- [x] Content-Type header parsing (DOCX MIME type, PDF MIME type)
- [x] URL extension fallback (`.docx`, `.pdf`)
- [x] Binary heuristic (non-printable character ratio for string content)
- [x] HTML detection (presence of common HTML tags)
- [x] Default to TEXT for clean string content
- [x] Implement `is_binary_content(content: str) -> bool` fast check
- [x] Handle edge cases: empty content, None, very short content
- [x] Add module docstring with usage examples

### `tests/core/research/test_content_classifier.py` (NEW)

- [x] Test magic bytes: valid PDF header → PDF
- [x] Test magic bytes: valid DOCX (PK + word/document.xml) → DOCX
- [x] Test magic bytes: generic ZIP (PK without word/) → BINARY_UNKNOWN
- [x] Test Content-Type header: DOCX MIME type → DOCX
- [x] Test Content-Type header: PDF MIME type → PDF
- [x] Test URL extension: `.docx` → DOCX
- [x] Test URL extension: `.pdf` → PDF
- [x] Test URL extension with query params: `.docx?v=1` → DOCX
- [x] Test binary heuristic: high non-printable ratio → BINARY_UNKNOWN
- [x] Test HTML detection: content with `<html>` tags → HTML
- [x] Test plain text → TEXT
- [x] Test `is_binary_content()`: garbled string → True
- [x] Test `is_binary_content()`: normal text → False
- [x] Test empty/None content edge cases
- [x] Test conflicting signals (e.g., .docx URL but text content) — magic bytes wins

---

## Phase 2: DOCX Extractor

### `pyproject.toml` (MODIFY)

- [x] Add `docx = ["python-docx>=1.1.0"]` to `[project.optional-dependencies]`
- [x] Add `"foundry-mcp[docx]"` to the `dev` extras (alongside `pdf`)

### `src/foundry_mcp/core/errors/research.py` (MODIFY)

- [x] Add `DocxSecurityError(Exception)` base class
- [x] Add `InvalidDocxError(DocxSecurityError)` for magic bytes / content-type failures
- [x] Add `DocxSizeError(DocxSecurityError)` for size limit violations
- [x] Add section header comment `# DOCX Extraction Errors` (following PDF pattern)

### `src/foundry_mcp/core/research/docx_extractor.py` (NEW)

- [x] Module docstring mirroring `pdf_extractor.py` style
- [x] Lazy import for `python-docx` (same pattern as `pdfminer.six`)
- [x] Constants: `DOCX_MAGIC_BYTES`, `VALID_DOCX_CONTENT_TYPES`, `DEFAULT_MAX_DOCX_SIZE`, `DEFAULT_FETCH_TIMEOUT`
- [x] Import error classes from `core.errors.research`
- [x] Import SSRF validation: `validate_url_for_ssrf` from `pdf_extractor`
- [x] `DocxExtractionResult` dataclass:
  - [x] `text: str`
  - [x] `warnings: list[str]`
  - [x] `paragraph_count: int`
  - [x] `table_count: int`
  - [x] `success` property
  - [x] `has_warnings` property
- [x] `validate_docx_magic_bytes(data: bytes)` function
- [x] `validate_docx_content_type(content_type: str | None)` function
- [x] `DocxExtractor` class:
  - [x] `__init__(max_size, fetch_timeout)` with defaults
  - [x] `async extract(source: bytes | BytesIO, *, validate_magic=True)` method
  - [x] Run `python-docx` parsing in `asyncio.to_thread` (CPU-bound)
  - [x] Extract paragraph text
  - [x] Extract table cell text (row-by-row concatenation)
  - [x] Handle warnings (empty paragraphs, extraction issues)
  - [x] `async extract_from_url(url: str)` method
  - [x] SSRF validation before fetch
  - [x] Content-Type validation from HTTP response
  - [x] Size limit enforcement during streaming
  - [x] Redirect handling (max 5, re-validate after each)
  - [x] User-Agent header: `foundry-mcp/1.0 DocxExtractor`
- [x] Optional Prometheus metrics (lazy init, same pattern as PDF)
- [x] Logging via `logging.getLogger(__name__)`

### `tests/core/research/test_docx_extractor.py` (NEW)

- [x] Fixture: `simple_docx_bytes()` — create minimal .docx with python-docx
- [x] Fixture: `docx_with_tables_bytes()` — .docx with table content
- [x] Test: extract valid .docx → text extracted, paragraph_count > 0
- [x] Test: extract .docx with tables → table text included
- [x] Test: magic byte validation — invalid header raises `InvalidDocxError`
- [x] Test: magic byte validation — too short data raises `InvalidDocxError`
- [x] Test: size limit — oversized data raises `DocxSizeError`
- [x] Test: SSRF protection — localhost URL raises `SSRFError`
- [x] Test: SSRF protection — private IP raises `SSRFError`
- [x] Test: content-type validation — wrong type raises `InvalidDocxError`
- [x] Test: empty document → success with empty text, warnings
- [x] Test: corrupted .docx → appropriate error handling
- [x] Test: `python-docx` not installed → graceful degradation
- [x] Test: extract runs in thread pool (CPU-bound work off event loop)

---

## Phase 3: Integration

### `src/foundry_mcp/core/research/providers/shared.py` (MODIFY)

- [x] Add binary content guard at top of `SourceSummarizer.summarize_source()`
- [x] Import `is_binary_content` from `content_classifier`
- [x] Return skip result with `[Content skipped: binary/non-text document detected]`
- [x] Log warning when binary content is detected

### `src/foundry_mcp/core/research/providers/tavily.py` (MODIFY)

- [x] Add content type detection in `_apply_source_summarization()`
- [x] Import `classify_content`, `ContentType` from `content_classifier`
- [x] For DOCX content: extract text via `DocxExtractor` before summarization
- [x] For BINARY_UNKNOWN: set `source.content = None` with warning log
- [x] Add `_extract_docx_content()` helper method on `TavilySearchProvider`
- [x] Handle `python-docx` not installed gracefully (skip extraction, log warning)

### `src/foundry_mcp/core/research/providers/tavily_extract.py` (MODIFY)

- [x] Add content type detection after URL content retrieval
- [x] For DOCX content: extract text before returning `ResearchSource`
- [x] For BINARY_UNKNOWN: skip source with warning

### `src/foundry_mcp/core/research/document_digest/digestor.py` (MODIFY)

- [x] Add `docx_extractor: DocxExtractor` parameter to `__init__`
- [x] Store as `self.docx_extractor`
- [x] Update `create()` factory method if it exists
- [x] Update docstrings to mention DOCX support
- [x] Import `DocxExtractor` from `docx_extractor`

### `tests/core/research/test_binary_content_guard.py` (NEW)

- [x] Test: binary content in `summarize_source()` → skip result returned
- [x] Test: normal text content → passes through to summarization
- [x] Test: DOCX content in Tavily search path → extracted before summarization
- [x] Test: BINARY_UNKNOWN in Tavily search path → source.content set to None
- [x] Test: HTML content → passes through unchanged
- [x] Test: `python-docx` not installed → binary content skipped gracefully

---

## Phase 4: Workflow Wiring

### `src/foundry_mcp/core/research/workflows/deep_research/__init__.py` (MODIFY)

- [x] Add `DocxExtractor` to imports from `phases/analysis.py`
- [x] Add `DocxExtractor` to `__all__` or re-export block

### `src/foundry_mcp/core/research/workflows/deep_research/phases/analysis.py` (MODIFY)

- [x] Add `from foundry_mcp.core.research.docx_extractor import DocxExtractor  # noqa: F401`
- [x] Comment: `# re-export for test patch targets` (matching PDF pattern)

---

## Phase 5: Verification

### Automated checks

- [x] `pytest tests/core/research/test_content_classifier.py -v` — all pass (46/46)
- [x] `pytest tests/core/research/test_docx_extractor.py -v` — all pass (34/34)
- [x] `pytest tests/core/research/test_binary_content_guard.py -v` — all pass (16/16)
- [x] `pytest tests/core/research/ -v` — no regressions (2760 passed, 6 skipped)
- [x] `ruff check src/foundry_mcp/core/research/` — clean (fixed 4 pre-existing I001/F401 issues)
- [x] `pyright src/foundry_mcp/core/research/` — no new errors (10 pre-existing errors on DeepResearchState attrs, 2 pre-existing warnings for optional deps)

### Manual verification

- [x] `pip install python-docx` succeeds (v1.2.0 installed)
- [x] Confirm `python-docx` lazy import works when not installed (unit test `test_extract_without_python_docx` passes)
- [x] Confirm `python-docx` lazy import works when installed (DocxExtractor instantiates correctly)
- [x] Review: no existing tests broken by new imports or signatures (2760 passed, 0 failed)
- [x] Review: error classes follow existing naming conventions in `research.py` (DocxSecurityError/InvalidDocxError/DocxSizeError mirror PDF pattern)
- [x] Review: SSRF protection reuse doesn't introduce circular imports (all cross-module imports verified)
