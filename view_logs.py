import json
import os
from collections import defaultdict
from datetime import datetime

LOG_FILE = "logs/events.jsonl"


def load_events():
    if not os.path.exists(LOG_FILE):
        print("No logs found yet. Run the agent first.")
        return []
    events = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def print_summary(events):
    llm_calls = [e for e in events if e["event_type"] == "llm_call"]
    tool_calls = [e for e in events if e["event_type"] == "tool_call"]

    print("=" * 70)
    print("OBSERVABILITY SUMMARY")
    print("=" * 70)
    print(f"Total events logged: {len(events)}")
    print(f"LLM calls: {len(llm_calls)}  |  Tool calls: {len(tool_calls)}")
    print()

    # ── LLM Calls breakdown ──────────────────────────────
    print("-" * 70)
    print("LLM CALLS BY MODEL")
    print("-" * 70)
    by_model = defaultdict(lambda: {"success": 0, "fail": 0, "durations": []})
    for e in llm_calls:
        model = e.get("model", "unknown")
        if e.get("success"):
            by_model[model]["success"] += 1
        else:
            by_model[model]["fail"] += 1
        by_model[model]["durations"].append(e.get("duration_sec", 0))

    print(f"{'Model':<30} {'Success':<10} {'Fail':<8} {'Avg Duration':<12}")
    for model, stats in by_model.items():
        avg_dur = sum(stats["durations"]) / len(stats["durations"]) if stats["durations"] else 0
        print(f"{model:<30} {stats['success']:<10} {stats['fail']:<8} {avg_dur:.2f}s")
    print()

    # ── Tool Calls breakdown ─────────────────────────────
    print("-" * 70)
    print("TOOL CALLS BY TOOL")
    print("-" * 70)
    by_tool = defaultdict(lambda: {"success": 0, "fail": 0, "durations": []})
    for e in tool_calls:
        tool = e.get("tool", "unknown")
        if e.get("success"):
            by_tool[tool]["success"] += 1
        else:
            by_tool[tool]["fail"] += 1
        by_tool[tool]["durations"].append(e.get("duration_sec", 0))

    print(f"{'Tool':<25} {'Success':<10} {'Fail':<8} {'Avg Duration':<12}")
    for tool, stats in by_tool.items():
        avg_dur = sum(stats["durations"]) / len(stats["durations"]) if stats["durations"] else 0
        print(f"{tool:<25} {stats['success']:<10} {stats['fail']:<8} {avg_dur:.2f}s")
    print()

    # ── Recent errors ─────────────────────────────────────
    errors = [e for e in events if not e.get("success", True) and e.get("error")]
    if errors:
        print("-" * 70)
        print(f"RECENT ERRORS (last {min(5, len(errors))})")
        print("-" * 70)
        for e in errors[-5:]:
            ts = e.get("timestamp", "")[:19]
            src = e.get("model") or e.get("tool", "unknown")
            print(f"[{ts}] {src}: {e['error'][:100]}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    events = load_events()
    if events:
        print_summary(events)