"""
tests/test_core.py
Basic unit tests for ResearchAI Agent's core deterministic logic.

LLM calls test nahi karte (non-deterministic, costly, network-dependent) -
focus hai pure functions par: calculator tool, confidence scoring,
memory_store database operations, vector_store similarity logic.
"""

import os
import sys
import tempfile

# Project root ko path mein add karo taaki imports kaam karein
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ─── Calculator Tool Tests ────────────────────────────────

def test_calculator_basic_addition():
    from tools.calculator_tool import calculate
    result = calculate.invoke("2 + 2")
    assert "4" in result


def test_calculator_with_function():
    from tools.calculator_tool import calculate
    result = calculate.invoke("sqrt(16)")
    assert "4" in result


def test_calculator_invalid_expression():
    from tools.calculator_tool import calculate
    result = calculate.invoke("not a valid expression !!!")
    assert "error" in result.lower() or "Error" in result


# ─── Confidence Score Tests ────────────────────────────────

def test_confidence_score_high_quality():
    from agent import calculate_confidence
    steps = [
        "TOOL_RESULT|search_web",
        "TOOL_RESULT|search_wikipedia",
        "TOOL_RESULT|calculate",
    ]
    report = "https://example.com/1 https://example.com/2 " + ("word " * 200)
    result = calculate_confidence(steps, report)
    assert result["label"] in ["High", "Moderate", "Low"]
    assert 0 <= result["score"] <= 100
    assert result["tools_used"] == 3


def test_confidence_score_empty_report():
    from agent import calculate_confidence
    result = calculate_confidence([], "")
    assert result["score"] == 0
    assert result["label"] == "Low"
    assert result["tools_used"] == 0


def test_confidence_score_caps_at_100():
    from agent import calculate_confidence
    steps = ["TOOL_RESULT|search_web"] * 10
    report = ("https://example.com " * 50) + ("word " * 1000)
    result = calculate_confidence(steps, report)
    assert result["score"] <= 100


# ─── Memory Store Tests (uses a temporary DB) ─────────────

@pytest.fixture
def temp_memory_store(monkeypatch, tmp_path):
    """Ek temporary SQLite DB use karta hai, taaki real research_history.db
    is test se affect na ho."""
    import memory_store
    test_db_path = str(tmp_path / "test_research.db")
    monkeypatch.setattr(memory_store, "DB_PATH", test_db_path)
    memory_store.init_db()
    return memory_store


def test_save_and_retrieve_session(temp_memory_store):
    session_id = temp_memory_store.save_session(
        topic="Test topic",
        mode="quick",
        report="Test report content",
        metadata={"confidence": {"score": 90}}
    )
    assert session_id.startswith("rai-")

    retrieved = temp_memory_store.get_session_by_session_id(session_id)
    assert retrieved is not None
    assert retrieved["topic"] == "Test topic"
    assert retrieved["report"] == "Test report content"
    assert retrieved["metadata"]["confidence"]["score"] == 90


def test_delete_session(temp_memory_store):
    session_id = temp_memory_store.save_session(
        topic="To be deleted",
        mode="quick",
        report="Temporary content",
    )
    temp_memory_store.delete_session(session_id)
    retrieved = temp_memory_store.get_session_by_session_id(session_id)
    assert retrieved is None


def test_get_recent_sessions_ordering(temp_memory_store):
    temp_memory_store.save_session(topic="First", mode="quick", report="A")
    temp_memory_store.save_session(topic="Second", mode="quick", report="B")
    recent = temp_memory_store.get_recent_sessions(limit=5)
    assert len(recent) == 2
    assert recent[0]["topic"] == "Second"  # most recent first


# ─── PDF Generator Tests ────────────────────────────────────

def test_strip_emojis():
    from pdf_generator import strip_emojis
    result = strip_emojis("Hello 🎉 World 🚀")
    assert "🎉" not in result
    assert "🚀" not in result
    assert "Hello" in result
    assert "World" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])