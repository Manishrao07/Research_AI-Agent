from dotenv import load_dotenv
import re
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from observability import log_event, track_tool_call
from typing import Annotated
from typing_extensions import TypedDict
from tools import ALL_TOOLS
import json
import time

# ─── LLM Setup ───────────────────────────────────────────
import os

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

QUICK_API_KEY = os.getenv("GROQ_API_KEY")
DEEP_API_KEY = os.getenv("GROQ_API_KEY_DEEP")

def make_llm(model_name, api_key=None):
    key = api_key or QUICK_API_KEY
    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=key
    ).bind_tools(ALL_TOOLS, parallel_tool_calls=False)

llm = make_llm(PRIMARY_MODEL)
fallback_llm = make_llm(FALLBACK_MODEL)

llm_deep = make_llm(PRIMARY_MODEL, api_key=DEEP_API_KEY)
fallback_llm_deep = make_llm(FALLBACK_MODEL, api_key=DEEP_API_KEY)
SYSTEM_PROMPT = """You are ResearchAI — an expert research assistant with access to tools.

You MUST use your tools to research before writing any report. Never make up information.

TOOL SELECTION RULES:
- Use search_web to find general web information, latest news, and current events.
- Use search_wikipedia to find historical context, background, and established general knowledge.
- ONLY call search_arxiv if the research topic is unambiguously a highly technical, scientific, AI/ML, physics, or math query (e.g., neural network architectures, quantum algorithms).
  * DO NOT call search_arxiv for queries related to sports (IPL, cricket, football, match stats), pop culture, entertainment, celebrities, politics, business, general news, or history. If in doubt, skip search_arxiv.
- Use calculate only if mathematical calculations or arithmetic are needed.

Report writing process:
1. Analyze the research query to identify its domain.
2. Select and call only the tools that are appropriate for that domain. Do not run tools sequentially as a checklist if they are not relevant.
3. Write the final report using ONLY information from the tool execution results.

REPORT FORMAT:
## 📋 Research Report: [Topic]

### 🔍 Executive Summary
[2-3 line overview based on research]

### 📊 Key Findings
[Important points from search results]

### 🌐 Latest Developments
[Recent news from web search]

### 📚 Background & History
[Info from Wikipedia]

### 💡 Analysis & Insights
[Your synthesis of the research]

### 🔮 Future Outlook
[Trends based on research]

### 📎 Sources
[List each source as a markdown link: - [Source Name](URL). Use the exact URLs returned by your tools.]"""

# ─── State ───────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ─── Agent Node ──────────────────────────────────────────
def agent_node(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    max_retries = 2
    last_error = None

    # Pehle primary model (70B) try karo
    for attempt in range(max_retries):
        start = time.time()
        try:
            response = llm.invoke(messages)
            log_event("llm_call", model=PRIMARY_MODEL, attempt=attempt + 1,
                       duration_sec=round(time.time() - start, 3), success=True)
            return {"messages": [response]}
        except Exception as e:
            last_error = e
            error_str = str(e)
            log_event("llm_call", model=PRIMARY_MODEL, attempt=attempt + 1,
                       duration_sec=round(time.time() - start, 3), success=False,
                       error=error_str[:200])
            if "tool_use_failed" in error_str or "tool call validation failed" in error_str:
                print(f"⚠️ Primary model tool call error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(1)
                continue
            elif "rate_limit" in error_str.lower() or "429" in error_str:
                print(f"⚠️ Primary model rate-limited, switching to fallback immediately...")
                break
            else:
                raise

    # Primary fail ho gaya - fallback model try karo
    print(f"🔄 Primary model ({PRIMARY_MODEL}) failed, switching to fallback ({FALLBACK_MODEL})...")
    for attempt in range(max_retries):
        start = time.time()
        try:
            response = fallback_llm.invoke(messages)
            log_event("llm_call", model=FALLBACK_MODEL, attempt=attempt + 1,
                       duration_sec=round(time.time() - start, 3), success=True)
            return {"messages": [response]}
        except Exception as e:
            last_error = e
            error_str = str(e)
            log_event("llm_call", model=FALLBACK_MODEL, attempt=attempt + 1,
                       duration_sec=round(time.time() - start, 3), success=False,
                       error=error_str[:200])
            if "tool_use_failed" in error_str or "tool call validation failed" in error_str:
                print(f"⚠️ Fallback model tool call error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(1)
                continue
            elif "rate_limit" in error_str.lower() or "429" in error_str:
                print(f"⚠️ Fallback model also rate-limited.")
                break
            else:
                raise

    raise RuntimeError(f"Agent failed on both primary and fallback models: {last_error}")

# ─── Router ──────────────────────────────────────────────
def should_continue(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

# ─── Graph ───────────────────────────────────────────────
tool_node = ToolNode(ALL_TOOLS)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

agent = graph.compile()

# ─── Confidence Score ────────────────────────────────────
def calculate_confidence(steps, report):
    """
    Research confidence score calculate karo based on:
    - Kitne unique tools use hue
    - Kitne sources cite hue report mein
    - Report ki length/depth
    """
    tool_results = [s for s in steps if s.startswith("TOOL_RESULT|")]
    unique_tools = set(s.split("|")[1] for s in tool_results)

    source_count = len(re.findall(r'https?://\S+', report))

    tool_score = min(len(unique_tools) * 20, 40)
    source_score = min(source_count * 8, 35)
    depth_score = min(len(report.split()) // 30, 25)

    total = min(tool_score + source_score + depth_score, 100)

    if total >= 80:
        label = "High"
    elif total >= 55:
        label = "Moderate"
    else:
        label = "Low"

    return {
        "score": total,
        "label": label,
        "tools_used": len(unique_tools),
        "sources_found": source_count
    }
def run_agent_with_models(topic: str, primary_model: str, fallback_model: str, api_key: str, callback=None):
    """
    run_agent jaisa hi hai, lekin custom model names aur API key ke saath chalata hai.
    Multi-agent pipeline (Deep mode) isko use karta hai, alag Groq account se.
    """
    custom_primary = ChatGroq(model=primary_model, temperature=0, api_key=api_key).bind_tools(
        ALL_TOOLS, parallel_tool_calls=False
    )
    custom_fallback = ChatGroq(model=fallback_model, temperature=0, api_key=api_key).bind_tools(
        ALL_TOOLS, parallel_tool_calls=False
    )

    def custom_agent_node(state: AgentState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        max_retries = 1
        last_error = None

        for attempt in range(max_retries):
            try:
                response = custom_primary.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "tool_use_failed" in error_str or "tool call validation failed" in error_str:
                    time.sleep(1)
                    continue
                elif "rate_limit" in error_str.lower() or "429" in error_str:
                    break
                else:
                    raise

        for attempt in range(max_retries):
            try:
                response = custom_fallback.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "tool_use_failed" in error_str or "tool call validation failed" in error_str:
                    time.sleep(1)
                    continue
                elif "rate_limit" in error_str.lower() or "429" in error_str:
                    break
                else:
                    raise

        raise RuntimeError(f"Agent failed on both models (deep mode): {last_error}")

    custom_tool_node = ToolNode(ALL_TOOLS)
    custom_graph = StateGraph(AgentState)
    custom_graph.add_node("agent", custom_agent_node)
    custom_graph.add_node("tools", custom_tool_node)
    custom_graph.set_entry_point("agent")
    custom_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    custom_graph.add_edge("tools", "agent")
    custom_agent = custom_graph.compile()

    messages = [HumanMessage(content=f"Research this topic and write a detailed report: {topic}")]
    steps = []
    final_report = ""
    tool_call_start_times = {}

    for chunk in custom_agent.stream({"messages": messages}, stream_mode="values"):
        last_msg = chunk["messages"][-1]

        if isinstance(last_msg, AIMessage):
            if isinstance(last_msg.content, str) and last_msg.content.strip():
                if len(last_msg.content) < 300:
                    step = f"THINKING|{last_msg.content.strip()}"
                    steps.append(step)
                    if callback:
                        callback(step)

            if last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("args", {})
                    query_preview = list(tool_args.values())[0] if tool_args else ""
                    step = f"TOOL_CALL|{tool_name}|{query_preview}"
                    steps.append(step)
                    if callback:
                        callback(step)
                    tool_call_start_times[tool_name] = time.time()

        if isinstance(last_msg, ToolMessage):
            step = f"TOOL_RESULT|{last_msg.name}"
            steps.append(step)
            if callback:
                callback(step)
            tool_name = last_msg.name
            start_time = tool_call_start_times.get(tool_name, time.time())
            duration = round(time.time() - start_time, 3)
            tool_success = not (isinstance(last_msg.content, str) and "error" in last_msg.content.lower()[:50])
            log_event("tool_call", tool=tool_name, duration_sec=duration, success=tool_success)

        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            if isinstance(last_msg.content, str) and len(last_msg.content) > 200:
                final_report = last_msg.content

    confidence = calculate_confidence(steps, final_report)

    return {
        "steps": steps,
        "report": final_report,
        "confidence": confidence
    }
# ─── Run Function ────────────────────────────────────────
def run_agent(topic: str, callback=None):
    messages = [HumanMessage(content=f"Research this topic and write a detailed report: {topic}")]

    steps = []
    final_report = ""
    tool_call_start_times = {}

    for chunk in agent.stream(
        {"messages": messages},
        stream_mode="values"
    ):
        last_msg = chunk["messages"][-1]

        if isinstance(last_msg, AIMessage):
            if isinstance(last_msg.content, str) and last_msg.content.strip():
                if len(last_msg.content) < 300:
                    step = f"THINKING|{last_msg.content.strip()}"
                    steps.append(step)
                    if callback:
                        callback(step)

            if last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("args", {})
                    query_preview = list(tool_args.values())[0] if tool_args else ""
                    step = f"TOOL_CALL|{tool_name}|{query_preview}"
                    steps.append(step)
                    if callback:
                        callback(step)
                    tool_call_start_times[tool_name] = time.time()

        if isinstance(last_msg, ToolMessage):
            step = f"TOOL_RESULT|{last_msg.name}"
            steps.append(step)
            if callback:
                callback(step)

            tool_name = last_msg.name
            start_time = tool_call_start_times.get(tool_name, time.time())
            duration = round(time.time() - start_time, 3)
            tool_success = not (isinstance(last_msg.content, str) and "error" in last_msg.content.lower()[:50])
            log_event("tool_call", tool=tool_name, duration_sec=duration, success=tool_success)

        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            if isinstance(last_msg.content, str) and len(last_msg.content) > 200:
                content = last_msg.content
                # Agar AI ne report se pehle koi "thinking out loud" text likha ho
                # (jaise tool-error ke baad self-correction), sirf actual report
                # se shuru karte hain - report hamesha "## " heading se start hota hai
                report_start = content.find("## ")
                if report_start > 0:
                    content = content[report_start:]
                final_report = content

    confidence = calculate_confidence(steps, final_report)

    return {
        "steps": steps,
        "report": final_report,
        "confidence": confidence
    }

# ─── Comparison Mode ─────────────────────────────────────
def run_comparison(topic_a: str, topic_b: str, callback=None):
    """
    Do topics ko independently research karo, phir comparison table banao.
    """
    if callback:
        callback(f"COMPARE_START|{topic_a}|{topic_b}")

    if callback:
        callback(f"COMPARE_PHASE|Researching: {topic_a}")
    result_a = run_agent(topic_a, callback=callback)

    if callback:
        callback(f"COMPARE_PHASE|Researching: {topic_b}")
    result_b = run_agent(topic_b, callback=callback)

    if callback:
        callback(f"COMPARE_PHASE|Generating comparison")

    comparison_prompt = f"""You have two research reports below. Create a detailed comparison.

REPORT A — {topic_a}:
{result_a['report']}

REPORT B — {topic_b}:
{result_b['report']}

Create a comparison with this EXACT format:

## Comparison: {topic_a} vs {topic_b}

### Side-by-Side Summary
| Aspect | {topic_a} | {topic_b} |
|--------|-----------|-----------|
| [Create 5-6 rows comparing key aspects from both reports] |

### Key Differences
[Bullet points of the most important differences]

### Key Similarities
[Bullet points of what they have in common, if any]

### Verdict
[A brief 2-3 line conclusion on how they compare]"""

    comparison_llm = ChatGroq(model=PRIMARY_MODEL, temperature=0.3)
    comparison_response = comparison_llm.invoke([HumanMessage(content=comparison_prompt)])

    if callback:
        callback("COMPARE_DONE|Comparison ready")

    return {
        "report_a": result_a['report'],
        "report_b": result_b['report'],
        "comparison": comparison_response.content,
        "steps_a": result_a['steps'],
        "steps_b": result_b['steps']
    }

# ─── Follow-up Q&A ────────────────────────────────────────
def ask_followup(report: str, original_topic: str, question: str, chat_history: list = None):
    """
    Generated report ke context mein follow-up sawaal ka jawab do.
    chat_history: pehle ke Q&A pairs, taaki conversation context bana rahe.
    """
    followup_llm = ChatGroq(model=PRIMARY_MODEL, temperature=0.4)

    context_msgs = [
        SystemMessage(content=f"""You are answering follow-up questions about a research report.

ORIGINAL TOPIC: {original_topic}

REPORT CONTENT:
{report}

Answer the user's question using ONLY information from this report. If the report doesn't contain 
the answer, say so clearly and suggest what additional research might be needed. Keep answers 
concise — 2-4 sentences unless the question needs more detail.""")
    ]

    if chat_history:
        for q, a in chat_history:
            context_msgs.append(HumanMessage(content=q))
            context_msgs.append(AIMessage(content=a))

    context_msgs.append(HumanMessage(content=question))

    response = followup_llm.invoke(context_msgs)
    return response.content

# ─── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 ResearchAI Agent starting...\n")
    print("=" * 60)

    topic = "Impact of Artificial Intelligence on jobs in 2025"
    print(f"📝 Research topic: {topic}\n")
    print("=" * 60)

    def print_step(step):
        print(f"\n{step}")

    result = run_agent(topic, callback=print_step)

    print("\n" + "=" * 60)
    print("📄 FINAL REPORT:")
    print("=" * 60)
    print(result["report"])
    print("\n" + "=" * 60)
    print(f"✅ Done! Used {len(result['steps'])} tools.")
    print(f"\n📊 Confidence Score: {result['confidence']['score']}/100 ({result['confidence']['label']})")
    print(f"   Tools used: {result['confidence']['tools_used']}  |  Sources found: {result['confidence']['sources_found']}")