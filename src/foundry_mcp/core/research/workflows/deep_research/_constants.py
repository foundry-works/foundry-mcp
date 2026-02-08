"""Budget allocation and validation constants for deep research."""

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
