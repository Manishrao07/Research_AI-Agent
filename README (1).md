# ResearchAI Agent

Autonomous multi-agent AI research system that takes a topic and produces a sourced, structured report — searching the live web, Wikipedia, and academic papers, then verifying its own output before handing it back.

**[Live demo →](https://researchai-agent-kdqsqhkfs8nw2erjzwmgnw.streamlit.app)**

---

## What this is

Most "AI research tools" are a single prompt to a chatbot. This is different: it's an engineered system where the reasoning steps, tool selection, and quality checks are explicit and inspectable — not a black box.

Two modes are available depending on how thorough the research needs to be:

- **Quick Research** — a single agentic loop. The model decides which tools to call and when, based on the topic. Fast, cheap, good for straightforward questions.
- **Deep Research (Multi-Agent)** — a four-stage pipeline: a *Planner* breaks the topic into sub-questions, a *Researcher* investigates each one independently, a *Synthesizer* combines the findings into one report, and a *Critic* checks the result for gaps before it's returned. If the Critic finds gaps, the system automatically does one corrective pass.

Every research session is saved, can be recalled later, and is checked against past research for relevant overlaps before a new one starts from scratch.

---

## Why it's built this way

A few decisions here were deliberate trade-offs, not defaults:

- **The multi-agent pipeline is plain Python, not a LangGraph graph.** Planner → Researcher → Synthesizer → Critic is a linear dependency chain — each stage needs the previous stage's output and nothing more. A graph framework would add indirection without adding capability. LangGraph *is* used, but only inside the Researcher, where the actual tool-calling loop benefits from explicit state and conditional routing.
- **Vector search suggests, it doesn't auto-inject.** Past research is matched by embedding similarity and offered to the user as "did you mean this?" — it is never silently pulled into a new report. Research topics are often time-sensitive (prices, scores, current events), and silently blending in stale context would quietly corrupt otherwise-correct answers.
- **Two separate Groq API keys.** Quick mode and Deep mode draw from different accounts so that a token-heavy Deep Research session can't exhaust the quota Quick mode needs to stay responsive — a basic form of resource isolation.
- **A custom logger instead of a managed observability platform for the first pass.** Every LLM call and tool call is timestamped and logged to a flat JSON-lines file, with a small script to summarize them (success rate, latency, error breakdown by model). It doesn't require an external account, and it's how the root-cause analysis below was actually done. LangSmith tracing is also wired in alongside it for deeper trace inspection.
- **Display-level streaming, not token-level.** The Quick mode report appears word-by-word, but this is a post-generation typewriter effect, not real token streaming. True streaming through a tool-calling loop means not knowing whether a given chunk is a tool call or the final answer until it's complete — solvable, but the complexity wasn't worth it for a cosmetic feature.

---

## Architecture

```
                         ┌─────────────────┐
                         │      Topic       │
                         └────────┬────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                                │
            Quick Research                   Deep Research
                  │                                │
          ┌───────▼───────┐                ┌───────▼───────┐
          │  Agent loop    │                │   Planner      │
          │  (LangGraph)   │                │ → sub-questions│
          └───────┬───────┘                └───────┬───────┘
                  │                                │
        ┌─────────┴─────────┐              ┌───────▼────────┐
        │  Tools (as needed) │              │   Researcher    │
        │  web · wiki · arxiv │              │ (runs N agent   │
        │  · calculator       │              │  loops above)   │
        └─────────────────────┘              └───────┬────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │   Synthesizer    │
                                              │ → combined report│
                                              └────────┬────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │     Critic        │
                                              │ PASS / revise once │
                                              └────────┬────────┘
                                                       │
                              ┌────────────────────────▼────────────────────────┐
                              │      Confidence score · PDF · saved to SQLite      │
                              │       checked against ChromaDB for similar past     │
                              └─────────────────────────────────────────────────┘
```

---

## Core features

| Feature | What it does |
|---|---|
| **Live reasoning trace** | Every tool call and result streams into the UI as it happens — not just the final answer. |
| **Multi-agent deep research** | Planner/Researcher/Synthesizer/Critic pipeline with one automatic self-correction pass. |
| **Persistent memory** | Every session is saved to SQLite (`rai-1`, `rai-2`, ...) and can be reloaded from the sidebar. |
| **Semantic research recall** | ChromaDB + sentence-transformer embeddings surface similar past research before a new search runs. |
| **Confidence scoring** | A heuristic score (tools used, sources found, report depth) — the system rates its own output. |
| **Side-by-side comparison** | Two topics researched independently, then diffed into a structured comparison table. |
| **Follow-up Q&A** | Ask questions about a generated report without re-running research — answered from the report's own context. |
| **Self-evaluation** | An "Evaluate this report" button scores any report on factual specificity, source quality, completeness, and clarity via an LLM-as-judge call. |
| **REST API** | The same agent is exposed over FastAPI (`/research`, `/compare`, `/followup`, `/download/{filename}`) — usable outside the Streamlit UI. |
| **Observability** | Every LLM/tool call is logged with latency and outcome; `view_logs.py` summarizes the log into a readable report. |

---

## A real finding from the observability data

Early on, the primary model (`llama-3.3-70b-versatile`) was failing far more often than expected, and it wasn't obvious why. Pulling the logged events apart showed two genuinely separate causes:



The retry-then-fallback logic already in place (switch to `llama-3.1-8b-instant` after two failed attempts) handles both cases and gets to a 100% eventual success rate — but knowing *which* failure mode is dominant is what told me the fix was a fallback model, not a prompt rewrite.

---

## Tech stack

**Agent orchestration** — LangGraph (StateGraph + ToolNode), LangChain
**LLM inference** — Groq (`llama-3.3-70b-versatile` primary, `llama-3.1-8b-instant` fallback)
**Tools** — Tavily (web search), Wikipedia REST API, arXiv API, a sandboxed calculator
**Memory** — SQLite (session history), ChromaDB + `all-MiniLM-L6-v2` (semantic recall)
**Interfaces** — Streamlit (UI), FastAPI + Uvicorn (REST API)
**Output** — ReportLab (PDF generation)
**Observability** — custom JSON-lines logger, LangSmith tracing
**Testing** — pytest (deterministic unit tests — calculator, confidence scoring, memory store)

---

## Project structure

```
agent.py             Quick-mode agent: LangGraph tool-calling loop, fallback logic, confidence scoring
multi_agent.py        Deep-mode pipeline: Planner, Researcher, Synthesizer, Critic
tools/                search_web, search_wikipedia, search_arxiv, calculate
memory_store.py        SQLite session persistence
vector_store.py        ChromaDB similarity search over past research
observability.py        Custom event logger (logs/events.jsonl)
view_logs.py           Reads the log file and prints a summary report
evaluation.py           LLM-as-judge scoring suite
pdf_generator.py        ReportLab PDF report builder
app.py                  Streamlit UI
api.py                  FastAPI backend
tests/                  Pytest suite (deterministic logic only — no live LLM calls)
```

---

## Running it locally

```bash
git clone https://github.com/Manishrao07/Research_AI-Agent.git
cd Research_AI-Agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key
GROQ_API_KEY_DEEP=a_second_key_for_deep_mode
TAVILY_API_KEY=your_key
LANGSMITH_API_KEY=your_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=researchai-agent
```

Run the UI:

```bash
streamlit run app.py
```

Run the API:

```bash
python3 api.py
# docs at localhost:8000/docs
```

Run tests:

```bash
python3 -m pytest tests/ -v
```

---

## Known limitations

- **No persistence on the deployed version.** Streamlit Community Cloud uses an ephemeral filesystem, so SQLite history and the ChromaDB vector store reset on every redeploy/restart. Locally, both persist normally. A production deployment would move these to a managed Postgres + hosted vector DB instead of file-based storage.
- **Free-tier rate limits.** Both Groq accounts are on the free tier (100k tokens/day each). Heavy use of Deep Research mode can exhaust this; the app degrades gracefully (a clear error message, not a crash) rather than hanging.
- **Streaming is cosmetic, not architectural** — see the trade-offs section above.

---

## License

MIT
