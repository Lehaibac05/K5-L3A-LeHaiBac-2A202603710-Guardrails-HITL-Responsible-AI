"""Interactive prompt tester for the Blue defense pipeline.

Gõ prompt, xem nó đi qua từng lớp phòng thủ:
  RateLimit → InputGuardrail (detect_injection + topic_filter) → Blue LLM → OutputGuardrail

Chạy từ gốc repo:
    python scripts/try_defense.py            # gọi Blue LLM thật (cần OPENROUTER_API_KEY)
    python scripts/try_defense.py --no-llm   # chỉ test lớp input, không gọi LLM (offline)

Lệnh trong prompt:
    :out <text>   test riêng content_filter (output guardrail) trên <text>
    :reset        reset rate limiter
    :q            thoát
Không ghi file nào vào outputs/.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows console: allow Vietnamese / emoji output
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

from assignment.pipeline import (  # noqa: E402
    _ensure_pipeline, _run_request, build_observability, build_production_plugins,
)
from guardrails.input_guardrails import detect_injection, topic_filter  # noqa: E402
from guardrails.output_guardrails import content_filter  # noqa: E402


def _show_filters(text: str) -> None:
    print(f"  detect_injection : {detect_injection(text)}")
    print(f"  topic_filter     : {topic_filter(text)}")


def _show_content_filter(text: str) -> None:
    r = content_filter(text)
    print(f"  safe     : {r['safe']}")
    print(f"  issues   : {r['issues']}")
    print(f"  redacted : {r['redacted']}")


async def main(call_llm: bool) -> None:
    if call_llm:
        from core.config import setup_api_key
        setup_api_key()

    # High limit so manual testing isn't throttled; use :reset if needed.
    plugins = build_production_plugins(max_requests=1000, window_seconds=60)
    if call_llm:
        pipeline = _ensure_pipeline({"plugins": plugins})
    else:
        audit, monitor = build_observability()
        pipeline = {"plugins": plugins, "audit": audit, "monitor": monitor}

    mode = "Blue LLM thật" if call_llm else "offline (không gọi LLM)"
    print(f"=== Test phòng thủ Blue — {mode} ===")
    print("Nhập prompt (:out <text> test output filter, :reset, :q thoát)\n")

    n = 0
    while True:
        try:
            text = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = text.strip()
        if cmd in (":q", ":quit", ":exit"):
            break
        if cmd == ":reset":
            pipeline["plugins"][0] = build_production_plugins(max_requests=1000)[0]
            print("  rate limiter reset\n")
            continue
        if cmd.startswith(":out "):
            _show_content_filter(text[5:])
            print()
            continue

        n += 1
        _show_filters(text)
        row = await _run_request(
            pipeline, text, user_id="manual_tester",
            request_id=f"manual-{n}", call_llm=call_llm,
        )
        status = "BLOCKED" if row["blocked"] else "PASSED"
        print(f"  => {status}  layer={row['layer']}  reason={row['reason']}")
        print(f"  response: {row['response_preview']}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="chỉ chạy lớp input, không gọi Blue LLM")
    args = ap.parse_args()
    asyncio.run(main(call_llm=not args.no_llm))
