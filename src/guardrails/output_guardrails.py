"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

REDACTED = "[REDACTED]"

# Public contact points from data/pii_hallucination_samples.json (ground_truth):
# these are not customer PII and must stay readable in banking answers.
PUBLIC_ALLOWLIST = {
    "support@vinbank.example",
}

# Words that commonly follow "password is ..." in benign answers
# ("your password is required / case-sensitive / incorrect").
_BENIGN_AFTER_PASSWORD = (
    r"required|incorrect|wrong|invalid|expired|correct|valid|case|not|never|"
    r"too|the|your|a|an|reset|changed|sent|strong|weak|secret|private|"
    r"confidential|stored|encrypted|hashed|only|missing|empty|needed"
)

_PASSWORD_LABEL = r"\b(?:password|passwd|pwd|passcode|m[aậ]t\s+kh[aẩ]u)"
_VALUE = r"[\"']?[^\s,;\"']*[^\s,;\"'.]"   # value without trailing "." / ","

# Order matters: typed secrets first, then PII, then the literal fallback.
# Each entry is (issue_type, compiled pattern, label_group) — when
# label_group is set, only the VALUE is replaced and the label
# ("password: ") is kept readable.
_PATTERNS: list[tuple[str, re.Pattern, int | None]] = [
    # API keys: OpenAI-style "sk-...", AWS access keys, "api key is/: XXX".
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{4,}"), None),
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), None),
    (
        "api_key",
        re.compile(
            r"(\bapi[\s_-]?key\s*(?:is|:|=)\s*)"
            r"(?!\[REDACTED\])[\"']?[^\s,;\"']*[^\s,;\"'.]",
            re.IGNORECASE,
        ),
        1,
    ),
    # Passwords, explicit assignment: "password=x", "password: x".
    (
        "password",
        re.compile(rf"({_PASSWORD_LABEL}\s*[:=]\s*)(?!\[REDACTED\]){_VALUE}",
                   re.IGNORECASE),
        1,
    ),
    # Passwords in prose: "password is x", "mật khẩu là x" — but not
    # "your password is required / case-sensitive".
    (
        "password",
        re.compile(
            rf"({_PASSWORD_LABEL}\s+(?:is|l[aà])\s+)"
            rf"(?!\[REDACTED\]|(?:{_BENIGN_AFTER_PASSWORD})\b){_VALUE}",
            re.IGNORECASE,
        ),
        1,
    ),
    # Internal hostnames / connection strings (db.*.internal:5432, ...).
    (
        "internal_host",
        re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:internal|local|corp)(?::\d+)?\b",
                   re.IGNORECASE),
        None,
    ),
    # Email (public support address is allow-listed in _redact()).
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"), None),
    # VN phone: 0xxxxxxxxx (10 digits) / 0xxxxxxxxxx (11, e.g. 028 landline)
    # or +84..., optional space/dot/dash separators. The public hotline
    # "1900 545 467" starts with 1 and is not matched.
    (
        "phone",
        re.compile(r"(?<![\w+])(?:\+84|0)(?:[\s.-]?\d){9,10}(?!\d)"),
        None,
    ),
    # CMND (9 digits) / CCCD (12 digits) — not a money amount ("... VND").
    (
        "national_id",
        re.compile(
            r"(?<![\d.,])(?:\d{12}|\d{9})(?![\d.,]*\d)"
            r"(?!\s*(?:vnd|vnđ|đ|dong|đồng|usd)\b)",
            re.IGNORECASE,
        ),
        None,
    ),
    # Fallback: literal internal values from data/protected/vinbank_secrets.json
    # (admin123, sk-vinbank-secret-2024, db.vinbank.internal...) in any
    # phrasing the typed patterns above missed ("the admin one is admin123").
    *[
        ("internal_secret", re.compile(re.escape(s), re.IGNORECASE), None)
        for s in sorted(DEMO_SECRETS, key=len, reverse=True) if s
    ],
]


def _redact(pattern: re.Pattern, label_group: int | None, text: str) -> tuple[str, int]:
    """Replace matches of ``pattern`` with [REDACTED]; return (text, count)."""
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        if m.group(0).lower() in PUBLIC_ALLOWLIST:
            return m.group(0)
        count += 1
        prefix = m.group(label_group) if label_group else ""
        return f"{prefix}{REDACTED}"

    return pattern.sub(_sub, text), count


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys.
        ``issues`` items look like ``"phone: 1 found"`` (type first, so the
        type list can be compared with data/pii_hallucination_samples.json).

    Limits: regex only catches values written out verbatim. Encoded /
    split secrets (base64, "a-d-m-i-n-1-2-3", reversed) need a semantic
    layer (LLM judge / NeMo output rail like Red Advance L4).
    """
    if not isinstance(response, str) or not response:
        return {"safe": True, "issues": [], "redacted": response or ""}

    counts: dict[str, int] = {}
    redacted = response

    # Patterns run on the progressively redacted text so each value is
    # reported once, under its most specific type.
    for issue_type, pattern, label_group in _PATTERNS:
        redacted, n = _redact(pattern, label_group, redacted)
        if n:
            counts[issue_type] = counts.get(issue_type, 0) + n

    issues = [f"{name}: {n} found" for name, n in counts.items()]

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# Issue types that mean the MODEL itself is leaking system internals —
# the whole reply is withheld instead of just masked (the rest of that reply
# may carry partial / encoded hints). Customer PII only gets redacted.
CREDENTIAL_ISSUES = ("internal_secret", "api_key", "password", "internal_host")

# Contains the marker attacks.py uses to classify layer = "output_filter".
OUTPUT_BLOCK_MESSAGE = (
    "I cannot share internal system details. "
    "Xin lỗi, tôi không thể cung cấp thông tin nội bộ của hệ thống. "
    "(Response withheld by output guardrail.)"
)


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_issues: list[str] = []

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # Step 1: regex content filter (PII + secrets)
        result = content_filter(response_text)
        self.last_issues = result["issues"]

        if not result["safe"]:
            leaked_credentials = any(
                issue.startswith(CREDENTIAL_ISSUES) for issue in result["issues"]
            )
            if leaked_credentials:
                # System internals leaked -> withhold the whole reply.
                self.blocked_count += 1
                self._set_text(llm_response, OUTPUT_BLOCK_MESSAGE)
                return llm_response

            # Customer PII only -> keep the answer, mask the values.
            self.redacted_count += 1
            self._set_text(llm_response, result["redacted"])
            response_text = result["redacted"]

        # Step 2 (optional, not graded): LLM-as-Judge on the cleaned text
        if self.use_llm_judge:
            judge = await llm_safety_check(response_text)
            if not judge["safe"]:
                self.blocked_count += 1
                self._set_text(llm_response, OUTPUT_BLOCK_MESSAGE)

        # Step 3: return the (possibly modified) response
        return llm_response

    @staticmethod
    def _set_text(llm_response, text: str) -> None:
        """Replace the response content with a single text part."""
        llm_response.content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
