# Model Token Limits Reference

Definitive reference for LLM context window sizes and output limits, as of February 2026.
Used to maintain `src/foundry_mcp/config/model_token_limits.json` and the fallback dict
in `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`.

> **Maintenance**: When a new model ships, add it to both the JSON file and the
> `_FALLBACK_MODEL_TOKEN_LIMITS` dict, then run `pytest tests/unit/test_config_phase4.py -x -q`.

---

## Anthropic (Claude)

All Claude models share a 200K standard context window (1M in beta for select models).

| Model ID | Context Window | Max Output | Status | Notes |
|---|---|---|---|---|
| `claude-opus-4-6` | 200,000 | 128,000 | **Current flagship** | 1M beta available |
| `claude-sonnet-4-6` | 200,000 | 64,000 | **Current default** | 1M beta available |
| `claude-haiku-4-5` | 200,000 | 64,000 | Current budget | |
| `claude-opus-4-5` | 200,000 | 64,000 | Previous gen | |
| `claude-sonnet-4-5` | 200,000 | 64,000 | Previous gen | 1M beta available |
| `claude-sonnet-4` | 200,000 | 64,000 | Previous gen | 1M beta available |
| `claude-opus-4` | 200,000 | 64,000 | Legacy | |
| `claude-sonnet-3-7` | 200,000 | 64,000 | Deprecated | |
| `claude-haiku-3-5` | 200,000 | 8,192 | Legacy budget | |

**CLI access**: Claude Code. Default model: Sonnet 4.6.
Environment overrides: `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`.

---

## Google (Gemini)

| Model ID | Context Window | Max Output | Status | Notes |
|---|---|---|---|---|
| `gemini-3.1-pro` | 1,048,576 | 65,536 | **Current flagship** (Preview) | |
| `gemini-3-flash` | 1,048,576 | 65,536 | Preview | |
| `gemini-3-pro` | 1,048,576 | 65,536 | Discontinuing | Superseded by 3.1 |
| `gemini-2.5-pro` | 1,048,576 | 65,536 | Stable | Retires June 2026 |
| `gemini-2.5-flash` | 1,048,576 | 65,536 | Stable | Retires June 2026 |
| `gemini-2.5-flash-lite` | 1,048,576 | 65,536 | Stable | Retires July 2026 |
| `gemini-2.0-flash` | 1,048,576 | 8,192 | **Deprecated** | Retires June 1, 2026 |
| `gemini-2.0-flash-lite` | 1,048,576 | 8,192 | **Deprecated** | Retires June 1, 2026 |
| `gemini-1.5-pro` | 2,097,152 | 8,192 | Legacy | |

**CLI access**: Gemini CLI (`@google/gemini-cli`). Default: auto-routes between 2.5 models.

---

## OpenAI (GPT / Codex)

### GPT-5.x Family (reasoning models)

All GPT-5 family share 400K context and 128K max output. Reasoning tokens consume output budget.

| Model ID | Context Window | Max Output | Status | Notes |
|---|---|---|---|---|
| `gpt-5.3-codex` | 400,000 | 128,000 | **Codex CLI default** | Feb 2026 |
| `gpt-5.3-codex-spark` | 128,000 | ~128,000 | Research preview | Cerebras; 1K+ tok/sec |
| `gpt-5.3` | 400,000 | 128,000 | Current | Feb 2026 |
| `gpt-5.2-codex` | 400,000 | 128,000 | Current | Dec 2025 |
| `gpt-5.2` | 400,000 | 128,000 | Current | Dec 2025 |
| `gpt-5.2-pro` | 400,000 | 128,000 | Premium | $21/$168 per MTok |
| `gpt-5.1-codex` | 400,000 | 128,000 | Current | Nov 2025 |
| `gpt-5.1-codex-mini` | 400,000 | 128,000 | Current budget | Nov 2025 |
| `gpt-5.1-codex-max` | 400,000 | 128,000 | Current | Multi-context-window |
| `gpt-5.1` | 400,000 | 128,000 | Superseded | Nov 2025 |
| `gpt-5` | 400,000 | 128,000 | Superseded | Aug 2025 |
| `gpt-5-mini` | 400,000 | 128,000 | Budget | Aug 2025 |

### GPT-4.1 Family (non-reasoning, ~1M context)

| Model ID | Context Window | Max Output | Status | Notes |
|---|---|---|---|---|
| `gpt-4.1` | 1,048,576 | 32,768 | Current | Apr 2025 |
| `gpt-4.1-mini` | 1,048,576 | 32,768 | Current | Apr 2025 |
| `gpt-4.1-nano` | 1,048,576 | 32,768 | Current | Apr 2025 |

### GPT-4o Family (legacy, 128K context)

| Model ID | Context Window | Max Output | Status | Notes |
|---|---|---|---|---|
| `gpt-4o` | 128,000 | 16,384 | Legacy | Retired from ChatGPT |
| `gpt-4o-mini` | 128,000 | 16,384 | Legacy | API still active |

**CLI access**: Codex CLI. Default: `gpt-5.3-codex`. Select with `codex -m <id>`.

---

## Z.ai (Zhipu AI / GLM)

### GLM-5 / GLM-4.x Family

| Model ID | Context Window | Max Output | Status | Notes |
|---|---|---|---|---|
| `glm-5` | 204,800 | 131,072 | **Flagship** | Feb 2026; MIT; 744B/40B active |
| `glm-5-code` | 204,800 | 131,072 | Current | Coding variant |
| `glm-4.7` | 202,752 | 131,072 | Current | Dec 2025; 358B/32B active |
| `glm-4.7-flash` | 202,752 | 131,072 | Free tier | Jan 2026; 31B/3B active |
| `glm-4.7-flashx` | 202,752 | 131,072 | Current | Premium flash |
| `glm-4.6` | 202,752 | 131,072 | Current | Sep 2025 |
| `glm-4.5` | 131,072 | 98,304 | Current | Aug 2025 |
| `glm-4.5-flash` | 131,072 | 98,304 | Free tier | Budget |
| `glm-4.5-air` | 131,072 | 98,304 | Current | Lightweight |

**API access**: OpenAI-compatible at `https://open.z.ai/api/paas/v4/`.
Works with Claude Code, Kilo Code, Roo Code, Cline, Zed via OpenAI compat layer.

---

## Token Limits JSON Mapping

The `model_token_limits.json` file uses **context window** (input) as its values, since
that's what the token budget system needs for planning. Ordering matters: more-specific
substrings must precede less-specific ones because `estimate_token_limit_for_model()`
returns the first match.

When adding new models, ensure:
1. Specific model IDs (`gemini-2.5-flash`) come before generic prefixes (`gemini-2`)
2. Both `model_token_limits.json` and `_FALLBACK_MODEL_TOKEN_LIMITS` are updated
3. Run `pytest tests/unit/test_config_phase4.py tests/unit/test_config_supervision.py -x -q`
