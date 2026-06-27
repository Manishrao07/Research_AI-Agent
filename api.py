from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from agent import run_agent, run_comparison, ask_followup
from pdf_generator import create_pdf_report

app = FastAPI(
    title="ResearchAI Agent API",
    description="Autonomous AI research agent — web search, Wikipedia, calculations, and report generation",
    version="1.0.0"
)

# CORS allow karo taaki koi bhi frontend isse call kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────
class ResearchRequest(BaseModel):
    topic: str

class CompareRequest(BaseModel):
    topic_a: str
    topic_b: str

class FollowupRequest(BaseModel):
    report: str
    original_topic: str
    question: str
    chat_history: Optional[list] = None


# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "ResearchAI Agent API",
        "status": "running",
        "endpoints": ["/research", "/compare", "/followup", "/health"]
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/research")
def research(request: ResearchRequest):
    """Single topic research karo aur report return karo"""
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    try:
        result = run_agent(request.topic)

        pdf_filename = None
        try:
            pdf_path = create_pdf_report(
                topic=request.topic,
                report_text=result["report"],
                steps=result["steps"]
            )
            pdf_filename = os.path.basename(pdf_path)
        except Exception:
            pdf_filename = None

        return {
            "topic": request.topic,
            "report": result["report"],
            "steps": result["steps"],
            "confidence": result.get("confidence"),
            "pdf_filename": pdf_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.post("/compare")
def compare(request: CompareRequest):
    """Do topics compare karo"""
    if not request.topic_a.strip() or not request.topic_b.strip():
        raise HTTPException(status_code=400, detail="Both topics are required")

    try:
        result = run_comparison(request.topic_a, request.topic_b)
        return {
            "topic_a": request.topic_a,
            "topic_b": request.topic_b,
            "comparison": result["comparison"],
            "report_a": result["report_a"],
            "report_b": result["report_b"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/followup")
def followup(request: FollowupRequest):
    """Report ke context mein follow-up sawaal poocho"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        history_pairs = [(h["q"], h["a"]) for h in request.chat_history] if request.chat_history else None
        answer = ask_followup(
            request.report,
            request.original_topic,
            request.question,
            chat_history=history_pairs
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Follow-up failed: {str(e)}")

@app.get("/download/{filename}")
def download_pdf(filename: str):
    """PDF report download karo by filename"""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join("reports", safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=safe_filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)