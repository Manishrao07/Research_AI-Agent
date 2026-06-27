import json
import time
import os
from datetime import datetime
from functools import wraps

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "events.jsonl")


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_event(event_type: str, **fields):
    """
    Ek single event ko JSON-lines file mein likhta hai.
    event_type: 'llm_call', 'tool_call', 'pipeline_stage', etc.
    fields: koi bhi extra data (duration, model, success, error, etc.)
    """
    _ensure_log_dir()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        **fields
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def track_llm_call(model_name: str):
    """
    Decorator — kisi bhi function (jo LLM call karta hai) ko wrap karke
    automatically duration, success/failure, aur model name log karta hai.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = round(time.time() - start, 3)
                log_event(
                    "llm_call",
                    model=model_name,
                    function=func.__name__,
                    duration_sec=duration,
                    success=True
                )
                return result
            except Exception as e:
                duration = round(time.time() - start, 3)
                log_event(
                    "llm_call",
                    model=model_name,
                    function=func.__name__,
                    duration_sec=duration,
                    success=False,
                    error=str(e)[:200]
                )
                raise
        return wrapper
    return decorator


def track_tool_call(tool_name: str, duration_sec: float, success: bool = True, error: str = None):
    """Tool execution ko manually log karne ke liye (decorator use nahi kar sakte
    kyunki tools LangChain @tool decorator se already wrapped hain)."""
    log_event(
        "tool_call",
        tool=tool_name,
        duration_sec=round(duration_sec, 3),
        success=success,
        error=error[:200] if error else None
    )


def log_pipeline_stage(stage_name: str, **fields):
    """Multi-agent pipeline ke stages track karne ke liye (planner, researcher, etc.)"""
    log_event("pipeline_stage", stage=stage_name, **fields)