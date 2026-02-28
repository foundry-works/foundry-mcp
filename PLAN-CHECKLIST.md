# DOCX Extraction Support — Checklist

## Phase 1: Content Classifier

### `src/foundry_mcp/core/research/content_classifier.py` (NEW)

- [ ] Create `ContentType` enum (`TEXT`, `HTML`, `PDF`, `DOCX`, `BINARY_UNKNOWN`)
- [ ] Implement `classify_content(content, *, url=None, content_type_header=None)` function
- [ ] Magic bytes detection: `%PDF-` → PDF
- [ ] Magic bytes detection: `PK\x03\x04` + `word/document.xml` probe → DOCX
- [ ] Magic bytes detection: `PK\x03\x04` without `word/` → BINARY_UNKNOWN
- [ ] Content-Type header parsing (DOCX MIME type, PDF MIME type)
- [ ] URL extension fallback (`.docx`, `.pdf`)
- [ ] Binary heuristic (non-printable character ratio for string content)
- [ ] HTML detection (presence of common HTML tags)
- [ ] Default to TEXT for clean string content
- [ ] Implement `is_binary_content(content: str) -> bool` fast check
- [ ] Handle edge cases: empty content, None, very short content
- [ ] Add module docstring with usage examples

### `tests/core/research/test_content_classifier.py` (NEW)

- [ ] Test magic bytes: valid PDF header → PDF
- [ ] Test magic bytes: valid DOCX (PK + word/document.xml) → DOCX
- [ ] Test magic bytes: generic ZIP (PK without word/) → BINARY_UNKNOWN
- [ ] Test Content-Type header: DOCX MIME type → DOCX
- [ ] Test Content-Type header: PDF MIME type → PDF
- [ ] Test URL extension: `.docx` → DOCX
- [ ] Test URL extension: `.pdf` → PDF
- [ ] Test URL extension with query params: `.docx?v=1` → DOCX
- [ ] Test binary heuristic: high non-printable ratio → BINARY_UNKNOWN
- [ ] Test HTML detection: content with `<html>` tags → HTML
- [ ] Test plain text → TEXT
- [ ] Test `is_binary_content()`: garbled string → True
- [ ] Test `is_binary_content()`: normal text → False
- [ ] Test empty/None content edge cases
- [ ] Test conflicting signals (e.g., .docx URL but text content) — magic bytes wins

---

## Phase 2: DOCX Extractor

### `pyproject.toml` (MODIFY)

- [ ] Add `docx = ["python-docx>=1.1.0"]` to `[project.optional-dependencies]`
- [ ] Add `"foundry-mcp[docx]"` to the `dev` extras (alongside `pdf`)

### `src/foundry_mcp/core/errors/research.py` (MODIFY)

- [ ] Add `DocxSecurityError(Exception)` base class
- [ ] Add `InvalidDocxError(DocxSecurityError)` for magic bytes / content-type failures
- [ ] Add `DocxSizeError(DocxSecurityError)` for size limit violations
- [ ] Add section header comment `# DOCX Extraction Errors` (following PDF pattern)

### `src/foundry_mcp/core/research/docx_extractor.py` (NEW)

- [ ] Module docstring mirroring `pdf_extractor.py` style
- [ ] Lazy import for `python-docx` (same pattern as `pdfminer.six`)
- [ ] Constants: `DOCX_MAGIC_BYTES`, `VALID_DOCX_CONTENT_TYPES`, `DEFAULT_MAX_DOCX_SIZE`, `DEFAULT_FETCH_TIMEOUT`
- [ ] Import error classes from `core.errors.research`
- [ ] Import SSRF validation: `validate_url_for_ssrf` from `pdf_extractor`
- [ ] `DocxExtractionResult` dataclass:
  - [ ] `text: str`
  - [ ] `warnings: list[str]`
  - [ ] `paragraph_count: int`
  - [ ] `table_count: int`
  - [ ] `success` property
  - [ ] `has_warnings` property
- [ ] `validate_docx_magic_bytes(data: bytes)` function
- [ ] `validate_docx_content_type(content_type: str | None)` function
- [ ] `DocxExtractor` class:
  - [ ] `__init__(max_size, fetch_timeout)` with defaults
  - [ ] `async extract(source: bytes | BytesIO, *, validate_magic=True)` method
  - [ ] Run `python-docx` parsing in `asyncio.to_thread` (CPU-bound)
  - [ ] Extract paragraph text
  - [ ] Extract table cell text (row-by-row concatenation)
  - [ ] Handle warnings (empty paragraphs, extraction issues)
  - [ ] `async extract_from_url(url: str)` method
  - [ ] SSRF validation before fetch
  - [ ] Content-Type validation from HTTP response
  - [ ] Size limit enforcement during streaming
  - [ ] Redirect handling (max 5, re-validate after each)
  - [ ] User-Agent header: `foundry-mcp/1.0 DocxExtractor`
- [ ] Optional Prometheus metrics (lazy init, same pattern as PDF)
- [ ] Logging via `logging.getLogger(__name__)`

### `tests/core/research/test_docx_extractor.py` (NEW)

- [ ] Fixture: `simple_docx_bytes()` — create minimal .docx with python-docx
- [ ] Fixture: `docx_with_tables_bytes()` — .docx with table content
- [ ] Test: extract valid .docx → text extracted, paragraph_count > 0
- [ ] Test: extract .docx with tables → table text included
- [ ] Test: magic byte validation — invalid header raises `InvalidDocxError`
- [ ] Test: magic byte validation — too short data raises `InvalidDocxError`
- [ ] Test: size limit — oversized data raises `DocxSizeError`
- [ ] Test: SSRF protection — localhost URL raises `SSRFError`
- [ ] Test: SSRF protection — private IP raises `SSRFError`
- [ ] Test: content-type validation — wrong type raises `InvalidDocxError`
- [ ] Test: empty document → success with empty text, warnings
- [ ] Test: corrupted .docx → appropriate error handling
- [ ] Test: `python-docx` not installed → graceful degradation
- [ ] Test: extract runs in thread pool (CPU-bound work off event loop)

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

- [ ] `pytest tests/core/research/test_content_classifier.py -v` — all pass
- [ ] `pytest tests/core/research/test_docx_extractor.py -v` — all pass
- [ ] `pytest tests/core/research/test_binary_content_guard.py -v` — all pass
- [ ] `pytest tests/core/research/ -v` — no regressions
- [ ] `ruff check src/foundry_mcp/core/research/` — clean
- [ ] `pyright src/foundry_mcp/core/research/` — clean (or no new errors)

### Manual verification

- [ ] `pip install python-docx` succeeds
- [ ] Confirm `python-docx` lazy import works when not installed
- [ ] Confirm `python-docx` lazy import works when installed
- [ ] Review: no existing tests broken by new imports or signatures
- [ ] Review: error classes follow existing naming conventions in `research.py`
- [ ] Review: SSRF protection reuse doesn't introduce circular imports
