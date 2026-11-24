"""StudyBuddy: upload lecture PDFs, ask questions with page citations, take Moodle-style quizzes
that remember your mistakes and drill them next time."""
import hashlib
import io
import json
import random
import time
import uuid
from collections import Counter
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from search import BM25, chunk_pages

MODEL = "claude-opus-5-5"
ROOT = Path(__file__).parent
PROGRESS_FILE = ROOT / "data" / "progress.json"
SIZES = {"quiz": 10, "drill": 5, "mock": 30}

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
app = FastAPI(title="StudyBuddy")
DOCS = {}     # doc_id -> parsed PDF + search index (rebuilt on upload)
QUIZZES = {}  # quiz_id -> full questions incl. answers (never sent to the browser before submit)


# ---------- progress store: survives restarts, keyed by the PDF's content hash ----------
# ponytail: one JSON file, no locking; move to SQLite if several people use one server
def load_progress():
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


PROGRESS = load_progress()


def save_progress():
    PROGRESS_FILE.parent.mkdir(exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(PROGRESS, indent=1), encoding="utf-8")


def progress_for(doc_id):
    return PROGRESS.setdefault(doc_id, {"asked": [], "mistakes": [], "history": [], "topics": {}})


def weak_topics(p, limit=6):
    """Topics answered correctly < 70% of the time, worst first."""
    rows = [(r / t, t, name) for name, (r, t) in p["topics"].items() if t and r / t < 0.7]
    return [{"topic": n, "right": round(acc * t), "total": t} for acc, t, n in sorted(rows)[:limit]]


# ---------- Claude ----------
QUESTION = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["mcq", "tf"]},
        "question": {"type": "string"},
        "choices": {"type": "array", "items": {"type": "string"}},
        "answer_index": {"type": "integer"},
        "why": {"type": "string"},
        "hook": {"type": "string"},
        "topic": {"type": "string"},
        "page": {"type": "integer"},
    },
    "required": ["type", "question", "choices", "answer_index", "why", "hook", "topic", "page"],
    "additionalProperties": False,
}
QUIZ_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": QUESTION}},
    "required": ["questions"],
    "additionalProperties": False,
}

EXAMINER = """You write university exam questions in the style of a Moodle midterm.
Rules:
- Mix multiple choice (exactly 4 options) and True/False (about 1 in 3 questions). For True/False use choices ["True", "False"].
- Exams copy slide sentences, so reuse the exact wording of the excerpts.
- A false True/False statement flips ONE key word of a real sentence (likely/unlikely, first/final, more/less).
- Wrong options must be plausible: similar-sounding terms from the same material, so only a student who knows the slide gets it right.
- No calculations unless the excerpts are about calculations.
- "why": one line quoting the fact that makes the answer right.
- "hook": a short memory trick (under 15 words) for the concept.
- "topic": the concept tested, 2-4 words, reuse the same name for the same concept.
- "page": the page the fact comes from.
- Never repeat, or lightly reword, any question listed under "Already asked"."""


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


def clean(q):
    """Normalise one generated question and shuffle MCQ options so the answer letter is random."""
    if q["type"] == "tf" or len(q["choices"]) < 2:
        right = q["choices"][q["answer_index"]] if 0 <= q["answer_index"] < len(q["choices"]) else "True"
        q["type"], q["choices"] = "tf", ["True", "False"]
        q["answer_index"] = 0 if str(right).strip().lower().startswith("t") else 1
        return q
    right = q["choices"][q["answer_index"]]
    random.shuffle(q["choices"])
    q["answer_index"] = q["choices"].index(right)
    return q


# ---------- routes ----------
@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


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
    doc_id = hashlib.sha1(data).hexdigest()[:12]  # same PDF -> same id -> your progress comes back
    DOCS[doc_id] = {"name": file.filename, "pages": len(pages), "chunks": chunks, "index": BM25(chunks)}
    progress_for(doc_id)["name"] = file.filename
    save_progress()
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


class QuizReq(BaseModel):
    doc_id: str
    mode: str = "quiz"  # quiz (10) | drill (5, targets your mistakes) | mock (30)
    topic: str = ""


@app.post("/quiz")
def quiz(body: QuizReq):
    doc = get_doc(body.doc_id)
    if body.mode not in SIZES:
        raise HTTPException(400, "mode must be quiz, drill or mock")
    p = progress_for(body.doc_id)
    n, extra, targets = SIZES[body.mode], "", []
    if body.mode == "drill":
        targets = [i for i, m in enumerate(p["mistakes"]) if not m.get("cleared")][-8:]
        recent = [p["mistakes"][i] for i in targets]
        weak = [w["topic"] for w in weak_topics(p)]
        if not recent:
            raise HTTPException(400, "No open mistakes. Take a quiz first, then drill what you got wrong.")
        extra = "\n\nThe student got these wrong. Re-test each concept from a DIFFERENT angle (new wording, " \
                "the opposite True/False, or the confusing term as the correct answer):\n" + \
                "\n".join(f"- [{m['topic']}] {m['question']} (picked \"{m['picked']}\", correct \"{m['correct']}\")" for m in recent)
        if weak:
            extra += "\nWeakest topics overall: " + ", ".join(weak)
        chunks = doc["index"].search(" ".join(m["question"] + " " + m["correct"] for m in recent), k=12)
    elif body.topic:
        chunks = doc["index"].search(body.topic, k=12)
    else:
        step = max(1, len(doc["chunks"]) // (20 if body.mode == "mock" else 10))
        chunks = doc["chunks"][::step][:20]
    chunks = chunks or doc["chunks"][:10]
    asked = "\n".join(f"- {s}" for s in p["asked"][-80:]) or "(none)"
    raw = claude(EXAMINER, f"Write {n} questions.{extra}\n\nAlready asked:\n{asked}\n\nLecture excerpts:\n{context(chunks)}",
                 schema=QUIZ_SCHEMA)
    questions = [clean(q) for q in json.loads(raw)["questions"]][:n]
    quiz_id = uuid.uuid4().hex[:10]
    QUIZZES[quiz_id] = {"doc_id": body.doc_id, "mode": body.mode, "questions": questions, "targets": targets}
    p["asked"] += [q["question"] for q in questions]
    save_progress()
    title = {"quiz": "Quiz", "drill": "Mistakes drill", "mock": "Mock exam"}[body.mode]
    return {"quiz_id": quiz_id, "title": f"{title} {sum(h['mode'] == body.mode for h in p['history']) + 1}",
            "questions": [{k: q[k] for k in ("type", "question", "choices", "topic", "page")} for q in questions]}


class Submit(BaseModel):
    answers: list[int | None]


@app.post("/quiz/{quiz_id}/submit")
def submit(quiz_id: str, body: Submit):
    qz = QUIZZES.get(quiz_id)
    if not qz:
        raise HTTPException(404, "Quiz expired (server restarted). Start a new one.")
    qs, answers = qz["questions"], (body.answers + [None] * len(qz["questions"]))[: len(qz["questions"])]
    p = progress_for(qz["doc_id"])
    score, wrong_letters, results = 0, [], []
    for q, a in zip(qs, answers):
        ok = a == q["answer_index"]
        score += ok
        stats = p["topics"].setdefault(q["topic"], [0, 0])
        stats[0] += ok
        stats[1] += 1
        if not ok:
            picked = q["choices"][a] if a is not None and 0 <= a < len(q["choices"]) else "(blank)"
            p["mistakes"].append({"question": q["question"], "picked": picked, "correct": q["choices"][q["answer_index"]],
                                  "topic": q["topic"], "page": q["page"], "at": int(time.time())})
            if q["type"] == "mcq" and a is not None:
                wrong_letters.append("abcd"[a])
        results.append({**q, "picked": a})
    if qz["targets"] and score >= 0.8 * len(qs):  # drill passed -> those old mistakes are fixed
        for i in qz["targets"]:
            p["mistakes"][i]["cleared"] = True
    p["history"].append({"mode": qz["mode"], "score": score, "total": len(qs), "at": int(time.time())})
    save_progress()
    pattern = ""
    if len(wrong_letters) >= 3:
        letter, count = Counter(wrong_letters).most_common(1)[0]
        if count / len(wrong_letters) >= 0.5:
            pattern = f"You picked \"{letter}\" on {count} of your {len(wrong_letters)} wrong multiple-choice answers. That looks like guessing: cross out the options from other topics first."
    return {"score": score, "total": len(qs), "results": results, "pattern": pattern, "weak_topics": weak_topics(p)}


@app.get("/progress/{doc_id}")
def progress(doc_id: str):
    p = progress_for(doc_id)
    return {"history": p["history"][-10:], "weak_topics": weak_topics(p),
            "open_mistakes": sum(not m.get("cleared") for m in p["mistakes"])}
