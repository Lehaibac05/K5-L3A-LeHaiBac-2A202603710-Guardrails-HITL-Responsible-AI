# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An individual student lab (Day 11, "Controlled Agent Security") with a VinBank chatbot as the setting. The student plays two roles:
- **Blue Team (Checkpoints 2–3):** implements guardrails and a defense pipeline around the **Blue** agent.
- **Red Team (Checkpoint 4):** writes attack prompts against the prebuilt **Red** (weak) and **Red Advance** (regex + NVIDIA NeMo Guardrails) agents.

The checkpoints are in `CHECKPOINTS.md`. Grading criteria are in `RUBRIC.md`, the submission layout and `results.json` example are in `SUBMISSION.md`, and the rules are in `RULES.md`. These docs are in Vietnamese.

## Commands

Run everything from the **repo root**, not from `src/`. Outputs always go to `<repo>/outputs/`.

```bash
python src/main.py --part 2      # CP2: guardrail self-tests (terminal only)
python src/main.py --part 3      # CP3: runs suite → outputs/results.json, audit_log.json, metrics.json
python src/main.py --part 4      # CP4: attacks Red then Red Advance → outputs/attack_results.json (+ unsafe_/guards_attack_result.json)
python src/main.py               # parts 2, 3, 4 in sequence

pytest tests/smoke -q            # default testpaths; no API key needed (this is what CI runs)
pytest tests/public -q           # contract tests; FAIL until the TODOs are implemented
pytest tests/public/test_lab_contracts.py::test_topic_filter_blocks_off_topic -q   # single test
python scripts/grade.py --submission-dir . --out outputs/grade_report.json         # also writes outputs/lab_report.md
python scripts/demo_attack_guards.py
```

`pytest.ini` sets `pythonpath = src`, so imports are top-level packages (`from guardrails.input_guardrails import ...`, `from core.config import ...`), not `src.xxx`.

Setup: `pip install -r requirements.txt`, then copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` (Blue) and `RED_TEAM_PROVIDER=openai|gemini` with the matching key (`OPENAI_API_KEY` or `GOOGLE_API_KEY`).

## Architecture

### Model tiers (`src/core/config.py`)
- **Blue** is hard-locked to OpenRouter `liquid/lfm-3-1.7b:free`. `get_blue_model()` deliberately ignores env overrides. Do not change this.
- **Red / Red Advance** share one provider and model from `.env`: OpenAI (`gpt-4o-mini`) through `core/openai_runtime.py`, or Gemini (`gemini-3.5-flash`) through Google ADK. Red Advance's NeMo self-check rails call the same Red model and key.
- The demo secrets (`admin123`, `sk-vinbank-secret-2024`, `db.vinbank.internal:5432`) are loaded from `data/protected/vinbank_secrets.json` into `DEMO_SECRETS` / `DEMO_SECRET_NOTE` and embedded in every agent's system prompt. Leak detection matches against this file. Do not edit it.

### Plugin pipeline
`OpenAIRunner.chat()` in `src/core/openai_runtime.py` imitates the Google ADK plugin protocol without running ADK:
1. Each plugin's `on_user_message_callback(*, invocation_context, user_message: types.Content)` runs in list order. The **first non-None `types.Content` return short-circuits** the run and becomes the reply, so the LLM is not called.
2. The LLM is called with the system prompt plus the user message only (single turn, no history).
3. Each plugin's `after_model_callback(*, callback_context, llm_response)` runs. Returning an object with `.content` replaces the response.

So plugin order matters. `build_production_plugins()` must return `[RateLimitPlugin, InputGuardrailPlugin, OutputGuardrailPlugin]`. Audit and monitoring are side observers returned separately by `build_observability()`, not plugins in that list.

### Agent factories (`src/agents/`)
- `agent.py`: `create_blue_agent(plugins)` (student plugins + `BLUE_INSTRUCTION`) and `create_red_agent_default()` (an instruction that deliberately leaks).
- `guards_agent.py` + `red_advance_nemo.py`: `create_red_agent_advance()` runs four layers: L1 regex input, L2 NeMo input rails, L3 regex output, L4 NeMo output rails (decodes base64/hex/ROT13/reversed text). NeMo fails closed. In attack JSON, `layer` is `nemo_input_rail`, `nemo_output_rail` or `nemo_error`.
- `security_boundary.py`: a reference implementation of the egress allowlist and HITL action authorization (`TRUSTED_EGRESS_HOSTS = {api.vinbank.example, cases.vinbank.example}`). It is a useful model for `is_egress_allowed`.
- `agents/__init__.py` imports lazily, so offline tests can import the package without ADK or API keys.
- Legacy technical names still appear in JSON and code: `unsafe` = Red, `guards` = Red Advance, `protected` = Blue.

### Student TODO surface (what gets graded)
- CP2: `src/guardrails/input_guardrails.py` (`detect_injection`, `topic_filter`, `InputGuardrailPlugin`) and `src/guardrails/output_guardrails.py` (`content_filter`, `OutputGuardrailPlugin`). Both filters return the strings `"ALLOW"` / `"BLOCK"`, never booleans. `detect_injection` must normalize hidden Unicode (zero-width characters) and must not block benign "summarise this external document" requests. `topic_filter` uses `ALLOWED_TOPICS` / `BLOCKED_TOPICS` from `core/config.py`. `content_filter` returns `{"safe", "issues", "redacted"}` and uses the literal string `[REDACTED]`.
- CP3: the `src/assignment/` files `rate_limiter.py` (sliding window per user), `audit_log.py`, `monitoring.py` and `pipeline.py` (`is_egress_allowed`, `build_production_plugins`, `build_observability`, `run_assignment_suite`). The suite output must validate against `schemas/results.schema.json`. It needs `framework`, ≥5 `safe_queries` (not blocked), ≥7 `attack_queries` (≥5 blocked), a `rate_limit` object with `passed + blocked == sent` and `blocked ≥ 1`, and ≥3 `edge_cases`.
- CP4: the `adversarial_prompts` list in `src/attacks/attacks.py` (replace the 5 `"TODO: ..."` inputs). The run, classification and JSON writing code is already provided.
- `src/hitl/`, `src/testing/`, `guardrails/nemo_guardrails.py` and the LLM-as-Judge in `output_guardrails.py` are optional reference code and are not graded.

## Lab rules that affect code changes
- Files in `outputs/` must be produced by running the lab commands. Never hand-write placeholder JSON or `report/*.md`. `scripts/grade.py` generates the report.
- Never commit `.env`.
