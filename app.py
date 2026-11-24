"""StudyBuddy: upload lecture PDFs, ask questions with page citations, generate quizzes."""
import io
import json
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from search import BM25, chunk_pages

MODEL = "claude-opus-5-5"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
app = FastAPI(title="StudyBuddy")
DOCS = {}  # ponytail: in-memory store, lost on restart; add SQLite when persistence matters

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "answer_index": {"type": "integer"},
            "explanation": {"type": "string"},
            "page": {"type": "integer"},
        },
        "required": ["question", "choices", "answer_index", "explanation", "page"],
        "additionalProperties": False,
    }}},
    "required": ["questions"],
    "additionalProperties": False,
}


def claude(system, prompt, schema=None):
    kwargs = {"output_config": {"format": {"type": "json_schema", "schema": schema}}} if schema else {}
    try:
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": prompt}],
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            **kwargs,
        )
    except (anthropic.AuthenticationError, TypeError):  # TypeError = no credentials configured at all
        raise HTTPException(500, "Missing or invalid ANTHROPIC_API_KEY")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate limited, try again in a minute")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the Claude API")
    if resp.stop_reason == "refusal":
        raise HTTPException(400, "Claude declined this request")
    return "".join(b.text for b in resp.content if b.type == "text")


def context(chunks):
    return "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c in chunks)


def get_doc(doc_id):
    if doc_id not in DOCS:
        raise HTTPException(404, "Unknown document, upload it again")
    return DOCS[doc_id]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/upload")
async def upload(file: UploadFile):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(400, "PDF too large (30 MB max)")
    try:
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
    except Exception:
        raise HTTPException(400, "Could not read that PDF")
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(400, "No text found (scanned PDFs aren't supported yet)")
    doc_id = str(len(DOCS) + 1)
    DOCS[doc_id] = {"name": file.filename, "pages": len(pages), "chunks": chunks, "index": BM25(chunks)}
    return {"id": doc_id, "name": file.filename, "pages": len(pages), "chunks": len(chunks)}


class Ask(BaseModel):
    doc_id: str
    question: str


@app.post("/ask")
def ask(body: Ask):
    doc = get_doc(body.doc_id)
    hits = doc["index"].search(body.question) or doc["chunks"][:6]
    answer = claude(
        "You are a study assistant. Answer ONLY from the lecture excerpts given. "
        "Cite pages inline like (p. 3). If the excerpts don't contain the answer, say so.",
        f"Lecture excerpts:\n{context(hits)}\n\nQuestion: {body.question}",
    )
    return {"answer": answer, "pages": sorted({c["page"] for c in hits})}


class Quiz(BaseModel):
    doc_id: str
    topic: str = ""
    count: int = 5


@app.post("/quiz")
def quiz(body: Quiz):
    doc = get_doc(body.doc_id)
    n = max(1, min(body.count, 15))
    chunks = doc["index"].search(body.topic, k=10) if body.topic else doc["chunks"][:: max(1, len(doc["chunks"]) // 10)][:10]
    raw = claude(
        "You write exam-style multiple-choice questions for university CS students. "
        "Each has 4 choices, exactly one correct, based only on the excerpts. Include the source page.",
        f"Write {n} questions.\n\nLecture excerpts:\n{context(chunks)}",
        schema=QUIZ_SCHEMA,
    )
    return json.loads(raw)
