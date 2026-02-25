# PLAN-CHECKLIST: Deep Research Prompt Alignment

## Phase 1: Research brief — domain-specific source guidance

- [ ] Edit `_build_brief_system_prompt()` in `phases/brief.py` (~line 215) to add domain-specific source examples after the generic "Specify source preferences" instruction
- [ ] Verify the prompt renders correctly by reading the updated method
- [ ] Run tests: `pytest tests/ -k brief` (if applicable)

## Phase 2: Supervisor directive isolation + no-acronym instruction

- [ ] Edit `_build_first_round_delegation_user_prompt()` in `phases/supervision.py` (~line 1842) to add rationale for self-contained directives
- [ ] Add no-acronym instruction to the decomposition guidelines in `_build_first_round_delegation_system_prompt()` (~line 1788)
- [ ] Add same no-acronym instruction to the follow-up delegation prompt in `_build_followup_delegation_system_prompt()` (~line 1410 area)
- [ ] Verify both prompts render correctly
- [ ] Run tests: `pytest tests/ -k supervision` (if applicable)

## Phase 3: Language preservation in synthesis and brief

- [ ] Edit `_build_synthesis_system_prompt()` in `phases/synthesis.py` (~line 606) to add language matching instruction before the closing IMPORTANT line
- [ ] Edit `_build_brief_system_prompt()` in `phases/brief.py` (~line 218) to add language-specific source guidance
- [ ] Verify both prompts render correctly
- [ ] Run tests: `pytest tests/ -k "synthesis or brief"` (if applicable)

## Final Validation

- [ ] Run full deep research test suite
- [ ] Verify no prompt regressions by spot-checking key prompt strings
