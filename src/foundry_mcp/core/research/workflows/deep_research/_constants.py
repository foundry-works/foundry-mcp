"""Budget allocation and validation constants for deep research."""

# Input bounds validation constants
MAX_ITERATIONS = 10  # Maximum refinement iterations
MAX_SUB_QUERIES = 20  # Maximum sub-queries per research session
MAX_SOURCES_PER_QUERY = 50  # Maximum sources per sub-query
MAX_CONCURRENT_PROVIDERS = 10  # Maximum concurrent provider operations

# Budget allocation constants
ANALYSIS_PHASE_BUDGET_FRACTION = 0.80  # 80% of effective context for analysis
ANALYSIS_OUTPUT_RESERVED = 4000  # Reserve tokens for findings/gaps JSON output
SYNTHESIS_PHASE_BUDGET_FRACTION = 0.85  # 85% of effective context for synthesis
SYNTHESIS_OUTPUT_RESERVED = 8000  # Reserve tokens for comprehensive markdown report
REFINEMENT_PHASE_BUDGET_FRACTION = 0.70  # 70% of effective context for refinement
REFINEMENT_OUTPUT_RESERVED = 2000  # Reserve tokens for follow-up queries JSON
REFINEMENT_REPORT_BUDGET_FRACTION = 0.50  # 50% of phase budget for report summary

# Final-fit validation constants
FINAL_FIT_MAX_ITERATIONS = 2  # Max attempts to fit payload within budget
FINAL_FIT_COMPRESSION_FACTOR = 0.85  # Reduce budget target by 15% on retry
FINAL_FIT_SAFETY_MARGIN = 0.10  # 10% safety margin for token estimation uncertainty

# Claim verification constants
VERIFICATION_MAX_CLAIMS_DEFAULT = 50  # Max claims to verify per report
VERIFICATION_MAX_CONCURRENT_DEFAULT = 10  # Max parallel verification LLM calls
VERIFICATION_MAX_CORRECTIONS_DEFAULT = 5  # Max correction LLM calls per report
VERIFICATION_SOURCE_MAX_CHARS = 8000  # Per-source content truncation in verification prompts
VERIFICATION_OUTPUT_RESERVED = 2000  # Reserve tokens for verification output
VERIFICATION_MAX_INPUT_TOKENS_DEFAULT = 200_000  # Total token budget escape hatch
VERIFICATION_SOURCE_DEEPEN_MAX_CHARS = 24_000  # 3x normal window for expanded re-verification
