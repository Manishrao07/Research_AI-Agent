"""
multi_agent.py
Multi-agent research pipeline: Planner -> Researcher -> Synthesizer -> Critic

Yeh existing agent.py ke tools aur run_agent() logic ko reuse karta hai,
lekin ek orchestration layer add karta hai jo:
1. Topic ko sub-questions mein todta hai (Planner)
2. Har sub-question ko research karta hai (Researcher)
3. Findings ko ek report mein combine karta hai (Synthesizer)
4. Report ko verify karta hai (Critic) - agar gaps hain to 1 baar retry

agent.py ko import karte hain - ALL_TOOLS aur model setup reuse hota hai,
duplicate nahi karte.
"""

from dotenv import load_dotenv
load_dotenv()

import re
import json
import time
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"


def make_llm(model_name, temperature=0, with_tools=False):
    llm = ChatGroq(model=model_name, temperature=temperature)
    if with_tools:
        llm = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
    return llm


# ─── PLANNER ──────────────────────────────────────────────
# Planner ka kaam: ek broad topic le kar 2-4 focused sub-questions banana,
# taaki research zyada targeted aur thorough ho, ek hi generic query ki jagah.

PLANNER_PROMPT = """You are a research planning agent. Your job is to break down a broad \
research topic into 2-4 specific, focused sub-questions that together give comprehensive \
coverage of the topic.

Rules:
- Sub-questions should be specific and searchable (good for web search / Wikipedia)
- Cover different angles: current state, causes/background, impact, future trends
- Do NOT answer the questions, only generate them
- Output ONLY valid JSON, nothing else, no markdown fences, no preamble

Output format:
{"sub_questions": ["question 1", "question 2", "question 3"]}

Topic:"""


def planner_node(topic: str) -> list[str]:
    """Topic ko sub-questions mein todta hai. Failure par single fallback question deta hai."""
    planner_llm = make_llm(PRIMARY_MODEL, temperature=0.2)

    try:
        response = planner_llm.invoke([
            SystemMessage(content="You output only valid JSON, no other text."),
            HumanMessage(content=PLANNER_PROMPT + topic)
        ])
        raw = response.content.strip()
        # Kabhi kabhi model markdown fences laga deta hai, strip karo
        raw = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
        data = json.loads(raw)
        sub_questions = data.get("sub_questions", [])
        sub_questions = [q.strip() for q in sub_questions if q.strip()]
        if not sub_questions:
            raise ValueError("Empty sub_questions list")
        return sub_questions[:4]  # max 4, cost control ke liye
    except Exception as e:
        print(f"⚠️ Planner failed ({e}), falling back to single direct question")
        return [topic]

# ─── RESEARCHER ───────────────────────────────────────────
# Researcher ka kaam: har sub-question ko agent.py ke existing run_agent()
# logic se research karna (tools call karna), aur saare findings collect karna.

from agent import run_agent as research_single_question


def researcher_node(sub_questions: list[str], callback=None) -> list[dict]:
    """
    Har sub-question ko independently research karta hai.
    Returns list of dicts: [{"question": ..., "report": ..., "steps": ...}, ...]
    """
    findings = []
    for i, question in enumerate(sub_questions, 1):
        if callback:
            callback(f"RESEARCH_PHASE|Sub-question {i}/{len(sub_questions)}: {question}")

        result = research_single_question(question, callback=callback)
        findings.append({
            "question": question,
            "report": result["report"],
            "steps": result["steps"],
        })
    return findings
# ─── SYNTHESIZER ──────────────────────────────────────────
# Synthesizer ka kaam: saare sub-question findings ko ek cohesive,
# well-structured final report mein combine karna (existing report format).

SYNTHESIZER_PROMPT = """You are a research synthesis agent. Below are research findings \
for several sub-questions about a broader topic. Combine them into ONE cohesive, \
well-structured report.

ORIGINAL TOPIC: {topic}

SUB-QUESTION FINDINGS:
{findings_text}

Write a single combined report using this EXACT format:

## 📋 Research Report: {topic}

### 🔍 Executive Summary
[2-3 line overview synthesizing all findings]

### 📊 Key Findings
[Important points pulled from all sub-question findings]

### 🌐 Latest Developments
[Recent news/trends across all findings]

### 📚 Background & History
[Relevant background context]

### 💡 Analysis & Insights
[Your synthesis connecting the different sub-questions]

### 🔮 Future Outlook
[Forward-looking trends across findings]

### 📎 Sources
[List ALL unique markdown links found across all sub-question findings, deduplicated]

Do not invent new information — only synthesize what's in the findings above."""


def synthesizer_node(topic: str, findings: list[dict]) -> str:
    """Saare findings ko ek combined report mein synthesize karta hai."""
    findings_text = ""
    for f in findings:
        findings_text += f"\n--- Sub-question: {f['question']} ---\n{f['report']}\n"

    synth_llm = ChatGroq(model=PRIMARY_MODEL, temperature=0.2)

    try:
        prompt = SYNTHESIZER_PROMPT.format(topic=topic, findings_text=findings_text)
        response = synth_llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"⚠️ Synthesizer failed ({e}), falling back to fallback model")
        synth_llm_fallback = ChatGroq(model=FALLBACK_MODEL, temperature=0.2)
        prompt = SYNTHESIZER_PROMPT.format(topic=topic, findings_text=findings_text)
        response = synth_llm_fallback.invoke([HumanMessage(content=prompt)])
        return response.content
# ─── CRITIC ───────────────────────────────────────────────
# Critic ka kaam: final report ko verify karna - kya saare sub-questions
# answer hue, kya claims findings se backed hain, kya sources present hain.
# Agar gaps milein, to specific missing points return karta hai (for 1 retry).

CRITIC_PROMPT = """You are a quality-control critic agent reviewing a research report.

ORIGINAL SUB-QUESTIONS THAT SHOULD BE ANSWERED:
{questions_list}

FINAL REPORT TO REVIEW:
{report}

Evaluate the report against these criteria:
1. Does the report address ALL the sub-questions above (at least partially)?
2. Are there specific, concrete claims (not vague generalities)?
3. Does the report cite sources?
4. Is there any obvious unsupported claim or contradiction?

Output ONLY valid JSON in this exact format, nothing else:
{{"verdict": "PASS" or "NEEDS_REVISION", "missing_points": ["point 1", "point 2"], "reasoning": "1-2 sentence explanation"}}

If verdict is PASS, missing_points should be an empty list."""


def critic_node(sub_questions: list[str], report: str) -> dict:
    """
    Report ko verify karta hai. Returns dict with verdict, missing_points, reasoning.
    Failure par default PASS deta hai (taaki pipeline block na ho).
    """
    critic_llm = ChatGroq(model=PRIMARY_MODEL, temperature=0)

    questions_list = "\n".join(f"- {q}" for q in sub_questions)

    try:
        prompt = CRITIC_PROMPT.format(questions_list=questions_list, report=report)
        response = critic_llm.invoke([
            SystemMessage(content="You output only valid JSON, no other text."),
            HumanMessage(content=prompt)
        ])
        raw = response.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
        data = json.loads(raw)
        return {
            "verdict": data.get("verdict", "PASS"),
            "missing_points": data.get("missing_points", []),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception as e:
        print(f"⚠️ Critic failed ({e}), defaulting to PASS")
        return {"verdict": "PASS", "missing_points": [], "reasoning": "Critic check skipped due to error."}
# ─── ORCHESTRATOR ─────────────────────────────────────────
# Pura pipeline chalata hai: Planner -> Researcher -> Synthesizer -> Critic
# Agar Critic NEEDS_REVISION de, to missing points ke saath ek baar retry karta hai.

def run_multi_agent_pipeline(topic: str, callback=None):
    """
    Full multi-agent research pipeline.
    Returns dict with: report, sub_questions, findings, critic_verdict, steps
    """
    all_steps = []

    def track(step):
        all_steps.append(step)
        if callback:
            callback(step)

    # 1. PLANNER
    track(f"PLANNER_PHASE|Breaking down topic into sub-questions")
    sub_questions = planner_node(topic)
    track(f"PLANNER_DONE|Generated {len(sub_questions)} sub-questions")

    # 2. RESEARCHER
    findings = researcher_node(sub_questions, callback=track)

    # 3. SYNTHESIZER
    track(f"SYNTHESIS_PHASE|Combining findings into final report")
    report = synthesizer_node(topic, findings)
    track(f"SYNTHESIS_DONE|Report generated")

    # 4. CRITIC
    track(f"CRITIC_PHASE|Verifying report quality")
    verdict = critic_node(sub_questions, report)
    track(f"CRITIC_DONE|Verdict: {verdict['verdict']}")

    # 5. RETRY (max 1 baar) agar Critic ne gaps pakde
    if verdict["verdict"] == "NEEDS_REVISION" and verdict["missing_points"]:
        track(f"RETRY_PHASE|Addressing gaps: {', '.join(verdict['missing_points'][:2])}")

        # Missing points ko extra sub-questions ki tarah research karo
        gap_findings = researcher_node(verdict["missing_points"], callback=track)
        findings.extend(gap_findings)

        track(f"RESYNTHESIS_PHASE|Regenerating report with additional findings")
        report = synthesizer_node(topic, findings)

        # Critic ko dobara verify karne do (lekin is baar result accept karo regardless)
        verdict = critic_node(sub_questions, report)
        track(f"FINAL_VERDICT|{verdict['verdict']}")

    return {
        "report": report,
        "sub_questions": sub_questions,
        "findings": findings,
        "critic_verdict": verdict,
        "steps": all_steps,
    }
if __name__ == "__main__":
    test_topic = "Impact of Artificial Intelligence on jobs in 2025"
    print(f"Topic: {test_topic}\n")
    print("=" * 60)

    def print_step(step):
        print(f"  {step}")

    result = run_multi_agent_pipeline(test_topic, callback=print_step)

    print("\n" + "=" * 60)
    print("📄 FINAL REPORT:")
    print("=" * 60)
    print(result["report"])
    print("\n" + "=" * 60)
    print(f"Critic verdict: {result['critic_verdict']['verdict']}")
    print(f"Total steps: {len(result['steps'])}")