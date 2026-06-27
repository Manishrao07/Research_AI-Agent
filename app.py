# pyrefly: ignore [missing-import]
import streamlit as st
import os
from agent import run_agent, run_comparison, ask_followup
from multi_agent import run_multi_agent_pipeline
from memory_store import save_session, get_recent_sessions, get_session_by_session_id, delete_session
from vector_store import add_research, find_similar_research, remove_research
from evaluation import judge_report
from pdf_generator import create_pdf_report
import time as time_module

def stream_text(text: str, delay: float = 0.015):
    """Text ko word-by-word generator ki tarah yield karta hai, typewriter effect ke liye."""
    words = text.split(" ")
    for word in words:
        yield word + " "
        time_module.sleep(delay)
# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="ResearchAI Agent",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)


def format_step(step):
    """Step ko readable text mein convert karo, type detect karke"""
    if step.startswith("THINKING|"):
        text = step.replace("THINKING|", "")
        return ("thinking", f"💭 {text}")
    elif step.startswith("TOOL_CALL|"):
        parts = step.split("|")
        tool_name = parts[1]
        query = parts[2] if len(parts) > 2 else ""
        return ("tool_call", f"🔧 Calling **{tool_name}** — \"{query}\"")
    elif step.startswith("TOOL_RESULT|"):
        tool_name = step.split("|")[1]
        return ("tool_result", f"✅ **{tool_name}** returned results")
    elif step.startswith("COMPARE_START|"):
        parts = step.split("|")
        return ("compare", f"⚔️ Comparing **{parts[1]}** vs **{parts[2]}**")
    elif step.startswith("COMPARE_PHASE|"):
        text = step.replace("COMPARE_PHASE|", "")
        return ("compare_phase", f"📍 {text}")
    elif step.startswith("COMPARE_DONE|"):
        text = step.replace("COMPARE_DONE|", "")
        return ("tool_result", f"✅ {text}")
    elif step.startswith("PLANNER_PHASE|"):
        text = step.replace("PLANNER_PHASE|", "")
        return ("planner", f"🧭 {text}")
    elif step.startswith("PLANNER_DONE|"):
        text = step.replace("PLANNER_DONE|", "")
        return ("tool_result", f"✅ {text}")
    elif step.startswith("RESEARCH_PHASE|"):
        text = step.replace("RESEARCH_PHASE|", "")
        return ("research_phase", f"🔍 {text}")
    elif step.startswith("SYNTHESIS_PHASE|"):
        text = step.replace("SYNTHESIS_PHASE|", "")
        return ("planner", f"🧩 {text}")
    elif step.startswith("SYNTHESIS_DONE|"):
        text = step.replace("SYNTHESIS_DONE|", "")
        return ("tool_result", f"✅ {text}")
    elif step.startswith("CRITIC_PHASE|"):
        text = step.replace("CRITIC_PHASE|", "")
        return ("critic", f"🧐 {text}")
    elif step.startswith("CRITIC_DONE|"):
        text = step.replace("CRITIC_DONE|", "")
        return ("critic", f"⚖️ {text}")
    elif step.startswith("RETRY_PHASE|"):
        text = step.replace("RETRY_PHASE|", "")
        return ("retry", f"🔁 {text}")
    elif step.startswith("RESYNTHESIS_PHASE|"):
        text = step.replace("RESYNTHESIS_PHASE|", "")
        return ("planner", f"🧩 {text}")
    elif step.startswith("FINAL_VERDICT|"):
        text = step.replace("FINAL_VERDICT|", "")
        return ("critic", f"⚖️ Final verdict: {text}")
    else:
        if "Error" in step or "❌" in step:
            return ("error", step)
        return ("other", step)


def render_step(kind, text):
    if kind == "error":
        st.error(text)
    elif kind == "thinking":
        st.warning(text)
    elif kind == "tool_call":
        st.info(text)
    elif kind == "tool_result":
        st.success(text)
    elif kind == "compare":
        st.warning(text)
    elif kind == "compare_phase":
        st.info(text)
    elif kind == "planner":
        st.info(text)
    elif kind == "research_phase":
        st.warning(text)
    elif kind == "critic":
        st.success(text)
    elif kind == "retry":
        st.warning(text)
    else:
        st.info(text)


# ── Header ───────────────────────────────────────────────
st.title("ResearchAI Agent")
st.write("Autonomous research agent — searches the web, reads Wikipedia, computes, and writes the report")
st.write("---")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("Capabilities")
    st.info(
        "• Searches the live web for current information\n"
        "• Reads Wikipedia for background context\n"
        "• Performs calculations when numbers matter\n"
        "• Writes a structured, sectioned report\n"
        "• Exports everything as a polished PDF\n"
        "• Compares two topics side-by-side"
    )

    st.subheader("Stack")
    st.write("LangGraph • Groq LLaMA 3.3 • Tavily • Streamlit • ReportLab • FastAPI")
    with st.expander("📂 Past Research", expanded=False):
        recent_sessions = get_recent_sessions(limit=8)
        if recent_sessions:
            for s in recent_sessions:
                label = f"{s['session_id']} — {s['topic'][:25]}{'...' if len(s['topic']) > 25 else ''}"
                row_col1, row_col2 = st.columns([4, 1])
                with row_col1:
                    if st.button(label, key=f"load_{s['session_id']}"):
                        st.session_state.load_session_id = s['session_id']
                with row_col2:
                    if st.button("🗑️", key=f"delete_{s['session_id']}"):
                        delete_session(s['session_id'])
                        remove_research(s['session_id'])
                        st.rerun()
        else:
            st.caption("No past research yet.")

    st.subheader("Try a topic")
    example_topics = [
        "Impact of AI on jobs in 2025",
        "Climate change latest developments",
        "Bitcoin and cryptocurrency trends",
        "Space exploration in 2025",
        "Electric vehicles market growth",
    ]
    for topic in example_topics:
        if st.button(topic, key=topic):
            st.session_state.selected_topic = topic
# ── Load past session if requested from sidebar ──────────
if st.session_state.get("load_session_id"):
    loaded = get_session_by_session_id(st.session_state.load_session_id)
    if loaded:
        st.session_state.steps = []
        st.session_state.report = loaded["report"]
        st.session_state.last_topic = loaded["topic"]
        st.session_state.is_comparison = False
        st.session_state.chat_history = []
        st.session_state.current_session_id = loaded["session_id"]

        meta = loaded.get("metadata") or {}
        st.session_state.confidence = meta.get("confidence")
        st.session_state.critic_verdict = meta.get("critic_verdict")
        st.session_state.sub_questions = meta.get("sub_questions")

        st.session_state.pdf_path = None  # purana PDF path valid nahi rahega, regenerate nahi karte abhi
    st.session_state.load_session_id = None  # ek baar load hone ke baad clear karo

# ── Mode toggle ──────────────────────────────────────────
compare_mode = st.checkbox("⚔️ Compare two topics")


research_mode = st.radio(
    "Research depth",
    ["⚡ Quick Research", "🔬 Deep Research (Multi-Agent)"],
    horizontal=True,
    help="Quick: single-pass research. Deep: breaks topic into sub-questions, researches each, then verifies the final report with a Critic agent."
)
is_deep_mode = research_mode.startswith("🔬")

st.write("")

# ── Main Layout ──────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    if compare_mode:
        st.subheader("Topics to compare")
        topic_a = st.text_input("Topic A", placeholder="e.g. Tesla")
        topic_b = st.text_input("Topic B", placeholder="e.g. Tata Motors")
        research_clicked = st.button("Start comparison", type="primary")
        topic = None
    else:
        st.subheader("Research topic")
        default_topic = st.session_state.get("selected_topic", "")
        topic = st.text_area(
            "What do you want to research?",
            value=default_topic,
            placeholder="e.g. Impact of AI on healthcare in 2025...",
            height=120,
            label_visibility="collapsed"
        )
        research_clicked = st.button("Start research", type="primary")
        topic_a = topic_b = None

        if topic and topic.strip():
            similar = find_similar_research(topic, top_k=2, similarity_threshold=0.68)
            if similar:
                st.info("📌 Similar research found in your past history:")
                for m in similar:
                    sim_col1, sim_col2 = st.columns([4, 1])
                    with sim_col1:
                        st.write(f"**{m['session_id']}** — {m['topic']} (similarity: {int(m['similarity']*100)}%)")
                    with sim_col2:
                        if st.button("Use this", key=f"use_similar_{m['session_id']}"):
                            st.session_state.load_session_id = m['session_id']
                            st.rerun()

    if "steps" in st.session_state and st.session_state.steps:
        st.subheader("Agent trace")
        for s in st.session_state.steps:
            kind, text = format_step(s)
            render_step(kind, text)

        if "report" in st.session_state and not st.session_state.get("is_comparison"):
            st.subheader("Stats")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Tools used", len(st.session_state.steps))
            with m2:
                word_count = len(st.session_state.report.split())
                st.metric("Words", word_count)
            with m3:
                st.metric("Status", "Done")

with col2:
    st.subheader("Report")

    if "report" not in st.session_state:
        st.info("Awaiting a topic. Enter a topic on the left and press the button — the agent handles the rest automatically.")
    else:
        if st.session_state.get("confidence") and not st.session_state.get("is_comparison"):
            conf = st.session_state.confidence
            label_color = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}.get(conf["label"], "⚪")
            st.markdown(
                f"**{label_color} Research Confidence: {conf['score']}/100 ({conf['label']})** "
                f"— {conf['tools_used']} tools used, {conf['sources_found']} sources found"
            )
            st.write("")

        if st.session_state.get("critic_verdict") and not st.session_state.get("is_comparison"):
            verdict = st.session_state.critic_verdict
            sub_qs = st.session_state.get("sub_questions") or []
            badge = "🟢" if verdict["verdict"] == "PASS" else "🟡"
            st.markdown(f"**{badge} Critic Verdict: {verdict['verdict']}** — {verdict.get('reasoning', '')}")
            with st.expander(f"🧭 {len(sub_qs)} sub-questions researched"):
                for i, q in enumerate(sub_qs, 1):
                    st.write(f"{i}. {q}")
            st.write("")

        if st.session_state.get("just_completed") and not st.session_state.get("is_comparison"):
            st.write_stream(stream_text(st.session_state.report))
            st.session_state.just_completed = False
        else:
            st.markdown(st.session_state.report)

        if st.session_state.get("pdf_path") and os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, "rb") as f:
                pdf_bytes = f.read()
            filename = os.path.basename(st.session_state.pdf_path)
            st.download_button(
                label="📥 Download PDF report",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf"
            )

        if not st.session_state.get("is_comparison"):
            st.write("---")
            st.subheader("🧪 Evaluate this report")
            if st.button("Run quality evaluation", key="evaluate_report_btn"):
                with st.spinner("Judging report quality..."):
                    topic_for_eval = st.session_state.get("last_topic", "")
                    eval_scores = judge_report(topic_for_eval, st.session_state.report)
                    st.session_state.eval_scores = eval_scores

            if st.session_state.get("eval_scores"):
                scores = st.session_state.eval_scores
                avg = (scores["factual_specificity"] + scores["source_quality"] +
                       scores["completeness"] + scores["clarity"]) / 4
                st.markdown(f"**Overall Score: {avg:.1f}/10**")
                ec1, ec2, ec3, ec4 = st.columns(4)
                with ec1:
                    st.metric("Factual Specificity", f"{scores['factual_specificity']}/10")
                with ec2:
                    st.metric("Source Quality", f"{scores['source_quality']}/10")
                with ec3:
                    st.metric("Completeness", f"{scores['completeness']}/10")
                with ec4:
                    st.metric("Clarity", f"{scores['clarity']}/10")
                st.caption(f"💬 {scores['comments']}")

        if not st.session_state.get("is_comparison"):
            st.write("---")
            st.subheader("💬 Ask a follow-up question")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**ResearchAI:** {a}")
                st.write("")

            followup_q = st.text_input("Ask something about this report...", key="followup_input")
            if st.button("Ask", key="ask_followup_btn"):
                if followup_q.strip():
                    with st.spinner("Thinking..."):
                        topic_for_context = st.session_state.get("last_topic", "")
                        answer = ask_followup(
                            st.session_state.report,
                            topic_for_context,
                            followup_q,
                            chat_history=st.session_state.chat_history
                        )
                        st.session_state.chat_history.append((followup_q, answer))
                    st.rerun()


# ── Single Topic Research Logic ──────────────────────────
if research_clicked and not compare_mode and topic and topic.strip():
    st.session_state.steps = []
    st.session_state.pop("report", None)
    st.session_state.pop("pdf_path", None)
    st.session_state.is_comparison = False
    st.session_state.is_deep_research = is_deep_mode

    with col1:
        st.subheader("Agent trace")
        steps_container = st.empty()

    with col2:
        progress_placeholder = st.empty()
        if is_deep_mode:
            progress_placeholder.info("⏳ Deep researching… Planning sub-questions, researching each, then verifying")
        else:
            progress_placeholder.info("⏳ Researching… Searching the web and checking Wikipedia")

    steps_so_far = []

    def update_steps(step):
        steps_so_far.append(step)
        st.session_state.steps = steps_so_far.copy()
        with steps_container.container():
            for s in steps_so_far:
                kind, text = format_step(s)
                render_step(kind, text)

    if is_deep_mode:
        with st.spinner("Multi-agent pipeline working..."):
            try:
                result = run_multi_agent_pipeline(topic, callback=update_steps)
            except Exception as e:
                st.error(
                    "⚠️ Research failed — both the primary and backup AI models hit an "
                    "issue (likely a temporary rate limit or provider hiccup). Please "
                    "wait a moment and try again."
                )
                st.caption(f"Technical detail: {str(e)[:150]}")
                st.stop()
        st.session_state.steps = result["steps"]
        st.session_state.report = result["report"]
        st.session_state.confidence = None  # deep mode mein confidence score nahi, Critic verdict hai
        st.session_state.critic_verdict = result.get("critic_verdict")
        st.session_state.sub_questions = result.get("sub_questions")
        st.session_state.last_topic = topic
        st.session_state.chat_history = []

        saved_id = save_session(
            topic=topic,
            mode="deep",
            report=result["report"],
            metadata={
                "critic_verdict": result.get("critic_verdict"),
                "sub_questions": result.get("sub_questions")
            }
        )
        st.session_state.current_session_id = saved_id
        add_research(saved_id, topic)
        try:
            pdf_path = create_pdf_report(
                topic=topic,
                report_text=result["report"],
                steps=result["steps"]
            )
            st.session_state.pdf_path = pdf_path
        except Exception:
            st.session_state.pdf_path = None
    else:
        with st.spinner("Agent working..."):
            try:
                result = run_agent(topic, callback=update_steps)
            except Exception as e:
                st.error(
                    "⚠️ Research failed — both the primary and backup AI models hit an "
                    "issue (likely a temporary rate limit or provider hiccup). Please "
                    "wait a moment and try again."
                )
                st.caption(f"Technical detail: {str(e)[:150]}")
                st.stop()
        st.session_state.steps = result["steps"]
        st.session_state.report = result["report"]
        st.session_state.confidence = result.get("confidence")
        st.session_state.critic_verdict = None
        st.session_state.sub_questions = None
        st.session_state.last_topic = topic
        st.session_state.chat_history = []

        saved_id = save_session(
            topic=topic,
            mode="quick",
            report=result["report"],
            metadata={"confidence": result.get("confidence")}
        )
        st.session_state.current_session_id = saved_id
        add_research(saved_id, topic)
        st.session_state.just_completed = True
        if result.get("success", True):
            try:
                pdf_path = create_pdf_report(
                    topic=topic,
                    report_text=result["report"],
                    steps=result["steps"]
                )
                st.session_state.pdf_path = pdf_path
            except Exception:
                st.session_state.pdf_path = None
        else:
            st.session_state.pdf_path = None

    st.rerun()

elif research_clicked and not compare_mode and (not topic or not topic.strip()):
    st.warning("Please enter a research topic first!")

# ── Comparison Logic ──────────────────────────────────────
if research_clicked and compare_mode:
    if not topic_a or not topic_a.strip() or not topic_b or not topic_b.strip():
        st.warning("Please enter both topics to compare!")
    else:
        st.session_state.steps = []
        st.session_state.pop("report", None)
        st.session_state.pop("pdf_path", None)
        st.session_state.is_comparison = True

        with col1:
            st.subheader("Agent trace")
            steps_container = st.empty()

        with col2:
            progress_placeholder = st.empty()
            progress_placeholder.info("⏳ Researching both topics, then comparing…")

        steps_so_far = []

        def update_steps(step):
            steps_so_far.append(step)
            st.session_state.steps = steps_so_far.copy()
            with steps_container.container():
                for s in steps_so_far:
                    kind, text = format_step(s)
                    render_step(kind, text)

        with st.spinner("Agent comparing topics..."):
            result = run_comparison(topic_a, topic_b, callback=update_steps)

        st.session_state.steps = steps_so_far
        st.session_state.report = result["comparison"]

        try:
            pdf_path = create_pdf_report(
                topic=f"{topic_a} vs {topic_b}",
                report_text=result["comparison"],
                steps=steps_so_far
            )
            st.session_state.pdf_path = pdf_path
        except Exception:
            st.session_state.pdf_path = None

        st.rerun()