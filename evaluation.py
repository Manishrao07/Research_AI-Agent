"""
evaluation.py
Evaluation framework for ResearchAI Agent.

Yeh ek standalone script hai jo:
1. Agent ko 8-10 diverse test topics par chalata hai
2. Har generated report ko ek independent "Judge" LLM se score karwata hai
   (4 criteria par: Factual Specificity, Source Quality, Completeness, Clarity)
3. Results ko aggregate karke summary table + JSON file mein save karta hai

Yeh agent.py ke run_agent() ko reuse karta hai - duplicate logic nahi likhte.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import re
import time
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agent import run_agent

JUDGE_MODEL = "llama-3.3-70b-versatile"

# ─── Test Topics ──────────────────────────────────────────
# Diverse topics: different domains, different complexity, taaki evaluation
# ek balanced picture de, sirf ek tarah ke topic pe bias na ho.

TEST_TOPICS = [
    "Latest developments in renewable energy 2026",
    "Impact of remote work on company productivity",
    "Current state of electric vehicle adoption in India",
    "How does inflation affect stock markets",
    "Recent advances in cancer treatment research",
    "Effects of social media on mental health",
    "Global semiconductor supply chain trends",
    "Benefits and risks of intermittent fasting",
]

# ─── Judge Prompt ─────────────────────────────────────────

JUDGE_PROMPT = """You are an evaluation judge assessing the quality of an AI-generated research report.

TOPIC: {topic}

REPORT TO EVALUATE:
{report}

Score this report on these 4 criteria, each from 1-10 (10 = excellent, 1 = very poor):

1. FACTUAL_SPECIFICITY: Does it contain concrete facts, numbers, dates, named entities? (vs vague generalities)
2. SOURCE_QUALITY: Are sources cited? Do they look credible (real URLs, named publications)?
3. COMPLETENESS: Does it cover the topic thoroughly across multiple angles (not just one narrow aspect)?
4. CLARITY: Is it well-organized, readable, and free of contradictions?

Output ONLY valid JSON in this exact format, nothing else, no markdown fences:
{{"factual_specificity": <1-10>, "source_quality": <1-10>, "completeness": <1-10>, "clarity": <1-10>, "comments": "1-2 sentence summary of strengths/weaknesses"}}"""


def judge_report(topic: str, report: str) -> dict:
    """Ek report ko judge LLM se score karwata hai. Failure par default low-confidence score deta hai."""
    judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0)

    try:
        prompt = JUDGE_PROMPT.format(topic=topic, report=report)
        response = judge_llm.invoke([
            SystemMessage(content="You output only valid JSON, no other text."),
            HumanMessage(content=prompt)
        ])
        raw = response.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
        data = json.loads(raw)
        return {
            "factual_specificity": data.get("factual_specificity", 0),
            "source_quality": data.get("source_quality", 0),
            "completeness": data.get("completeness", 0),
            "clarity": data.get("clarity", 0),
            "comments": data.get("comments", ""),
        }
    except Exception as e:
        print(f"⚠️ Judge failed for topic '{topic}': {e}")
        return {
            "factual_specificity": 0,
            "source_quality": 0,
            "completeness": 0,
            "clarity": 0,
            "comments": f"Judge evaluation failed: {e}",
        }


def run_evaluation(topics: list[str] = None, verbose: bool = True) -> dict:
    """
    Poora evaluation suite chalata hai: har topic ko research karta hai, judge karta hai,
    aur aggregate results return karta hai.
    """
    topics = topics or TEST_TOPICS
    results = []

    for i, topic in enumerate(topics, 1):
        if verbose:
            print(f"\n[{i}/{len(topics)}] Researching: {topic}")

        try:
            agent_result = run_agent(topic)
            report = agent_result["report"]
        except Exception as e:
            print(f"  ⚠️ Research failed: {e}")
            results.append({
                "topic": topic,
                "scores": {"factual_specificity": 0, "source_quality": 0, "completeness": 0, "clarity": 0},
                "comments": f"Research failed: {e}",
                "report_length": 0,
            })
            continue

        if verbose:
            print(f"  Report generated ({len(report)} chars). Judging...")

        scores = judge_report(topic, report)

        results.append({
            "topic": topic,
            "scores": {
                "factual_specificity": scores["factual_specificity"],
                "source_quality": scores["source_quality"],
                "completeness": scores["completeness"],
                "clarity": scores["clarity"],
            },
            "comments": scores["comments"],
            "report_length": len(report),
        })

        if verbose:
            avg = sum(scores[k] for k in ["factual_specificity", "source_quality", "completeness", "clarity"]) / 4
            print(f"  Scores: factual={scores['factual_specificity']}, sources={scores['source_quality']}, "
                  f"completeness={scores['completeness']}, clarity={scores['clarity']} (avg: {avg:.1f})")

        time.sleep(1)  # rate limit ke liye thoda gap

    # ─── Aggregate ──────────────────────────────────────
    criteria = ["factual_specificity", "source_quality", "completeness", "clarity"]
    aggregate = {}
    for c in criteria:
        valid_scores = [r["scores"][c] for r in results if r["scores"][c] > 0]
        aggregate[c] = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0

    overall_avg = round(sum(aggregate.values()) / len(aggregate), 2) if aggregate else 0

    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "total_topics": len(topics),
        "successful_topics": len([r for r in results if r["scores"]["factual_specificity"] > 0]),
        "aggregate_scores": aggregate,
        "overall_average": overall_avg,
        "per_topic_results": results,
    }

    return summary


def save_results(summary: dict, path: str = "evaluation_results.json"):
    """Evaluation results ko JSON file mein save karta hai."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to {path}")


def print_summary_table(summary: dict):
    """Evaluation summary ko readable table format mein print karta hai."""
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Topics evaluated: {summary['successful_topics']}/{summary['total_topics']}")
    print(f"Overall average score: {summary['overall_average']}/10")
    print("\nPer-criterion averages:")
    for criterion, score in summary["aggregate_scores"].items():
        bar = "█" * int(score) + "░" * (10 - int(score))
        print(f"  {criterion:22s} {score:5.2f}/10  {bar}")

    print("\nPer-topic breakdown:")
    for r in summary["per_topic_results"]:
        avg = sum(r["scores"].values()) / 4
        print(f"  [{avg:.1f}/10] {r['topic']}")
    print("=" * 70)


if __name__ == "__main__":
    print("🧪 Starting ResearchAI Agent Evaluation\n")
    print(f"Testing {len(TEST_TOPICS)} topics...")

    summary = run_evaluation(TEST_TOPICS, verbose=True)
    print_summary_table(summary)
    save_results(summary)