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

from core.utils import chat_with_agent
from guardrails.text_security import find_protected_kinds, plain_secret_spans, secret_groups


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

# Email chính thức của VinBank (data/pii_hallucination_samples.json -> ground_truth)
# không phải thông tin cá nhân nên không che.
_OFFICIAL_EMAILS = {"support@vinbank.example"}

_PHONE_RE = re.compile(r"(?<![\d.,])(?:\+84|0)(?:[\s.-]?\d){9,10}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NATIONAL_ID_RE = re.compile(
    r"(?<![\d.,+-])(?:\d{12}|\d{9})(?!\d)(?!\s*(?:vnd|vnđ|đ|₫|dong|đồng|usd|\$))", re.IGNORECASE
)
_API_KEY_RE = re.compile(
    r"\b(?:sk|pk|rk)[-_][A-Za-z0-9_-]{6,}|\bAIza[0-9A-Za-z_-]{30,}|\bAKIA[0-9A-Z]{16}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[abp]-[A-Za-z0-9-]{10,}|\bBearer\s+[A-Za-z0-9._~+/-]{16,}"
)
_PASSWORD_RE = re.compile(
    r"(?P<label>\b(?:pass(?:word|wd|code|phrase)?|pwd|mat\s*khau|mật\s*khẩu)\b\s*(?:is|are|là|:|=|-)\s*)"
    r"[\"'`]?(?P<value>[^\s\"'`,;]+)",
    re.IGNORECASE,
)
_INTERNAL_HOST_RE = re.compile(
    r"\b[\w-]+(?:\.[\w-]+)*\.internal(?::\d+)?\b|\b(?:postgres(?:ql)?|mysql|mongodb|redis|amqp)://\S+",
    re.IGNORECASE,
)

_SECRET_KINDS = ("password", "api_key", "internal_host")


def _has_digit_or_symbol(token: str) -> bool:
    return any(not c.isalpha() for c in token)


def _sub_count(pattern: re.Pattern, text: str, repl) -> tuple[str, int]:
    count = 0

    def _wrap(m: re.Match) -> str:
        nonlocal count
        out = repl(m)
        if out != m.group(0):
            count += 1
        return out

    return pattern.sub(_wrap, text), count


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Che: SĐT VN, email, CCCD/CMND, API key, mật khẩu, host nội bộ, và mọi giá trị
    trong data/protected (kể cả khi bị mã hóa base64/hex/ROT13/đảo chuỗi/tách ký tự).

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
        (thêm 'secret_leak': True nếu có bí mật hệ thống — plugin sẽ chặn cả câu)
    """
    response = response or ""
    redacted = response
    found: dict[str, int] = {}

    def note(kind: str, n: int) -> None:
        if n:
            found[kind] = found.get(kind, 0) + n

    # 1. Bí mật dạng cấu trúc (key, password=..., host nội bộ)
    redacted, n = _sub_count(_API_KEY_RE, redacted, lambda m: REDACTED)
    note("api_key", n)
    redacted, n = _sub_count(
        _PASSWORD_RE,
        redacted,
        lambda m: m.group("label") + REDACTED
        if _has_digit_or_symbol(m.group("value")) and m.group("value") != REDACTED
        else m.group(0),
    )
    note("password", n)
    redacted, n = _sub_count(_INTERNAL_HOST_RE, redacted, lambda m: REDACTED)
    note("internal_host", n)

    # 2. Giá trị bảo vệ nguyên văn (đối chiếu với data/protected/vinbank_secrets.json)
    kinds_by_value = {v.lower(): kind for kind, vals in secret_groups().items() for v in vals if len(v) >= 6}
    for m in sorted(plain_secret_spans(redacted), key=lambda x: -x.start()):
        kind = kinds_by_value.get(m.group(0).lower(), "password")
        note(kind, 1)
        redacted = redacted[: m.start()] + REDACTED + redacted[m.end() :]

    # 3. Giá trị bảo vệ bị che giấu (base64/hex/ROT13/đảo chuỗi/tách ký tự/chữ số bằng chữ)
    hidden = find_protected_kinds(redacted)
    if hidden:
        for kind in sorted(hidden):
            note("encoded_" + kind, 1)
        redacted = REDACTED  # không thể che chính xác chỗ -> che cả câu

    # 4. PII
    redacted, n = _sub_count(_PHONE_RE, redacted, lambda m: REDACTED)
    note("phone", n)
    redacted, n = _sub_count(
        _EMAIL_RE, redacted, lambda m: m.group(0) if m.group(0).lower() in _OFFICIAL_EMAILS else REDACTED
    )
    note("email", n)
    redacted, n = _sub_count(_NATIONAL_ID_RE, redacted, lambda m: REDACTED)
    note("national_id", n)

    issues = [f"{kind}: {count} found" for kind, count in found.items()]
    secret_leak = any(k in found or ("encoded_" + k) in found for k in _SECRET_KINDS)
    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
        "secret_leak": secret_leak,
    }


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

    SAFE_MESSAGE = (
        "I'm sorry, I can't share that information. "
        "How else can I help with your VinBank account or banking needs?"
    )

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_action: str | None = None  # "blocked" | "redacted" | None (cho audit)
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

        self.last_action = None
        self.last_issues = []

        # 1. Regex + đối chiếu giá trị bảo vệ
        result = content_filter(response_text)
        if not result["safe"]:
            self.last_issues = result["issues"]
            if result.get("secret_leak"):
                # Bí mật hệ thống: không cố vá từng chỗ, thay cả câu trả lời.
                self.blocked_count += 1
                self.last_action = "blocked"
                self._replace(llm_response, self.SAFE_MESSAGE)
                return llm_response
            # PII thường (SĐT, email, CCCD...): che rồi cho đi tiếp.
            self.redacted_count += 1
            self.last_action = "redacted"
            self._replace(llm_response, result["redacted"])

        # 2. LLM-as-Judge (tuỳ chọn, mặc định tắt)
        if self.use_llm_judge:
            verdict = await llm_safety_check(self._extract_text(llm_response))
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_action = "blocked"
                self.last_issues = self.last_issues + ["judge: unsafe"]
                self._replace(llm_response, self.SAFE_MESSAGE)

        return llm_response

    @staticmethod
    def _replace(llm_response, text: str) -> None:
        llm_response.content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
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
