"""StudyBuddy: upload lecture PDFs, ask questions with page citations, take Moodle-style quizzes
that remember your mistakes and drill them next time."""
import hashlib
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from search import BM25, chunk_pages, tokenize

# Local AI through Ollama (https://ollama.com): free, no account, no API key, works offline.
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ROOT = Path(__file__).parent
PROGRESS_FILE = ROOT / "data" / "progress.json"
SIZES = {"quiz": 10, "drill": 5, "mock": 30}

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


# ---------- local AI ----------
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
- Write like a professor setting an exam about the SUBJECT. Never mention "the excerpt", "the text", "the chart", "the candidate" or page numbers inside a question.
- No calculations unless the excerpts are about calculations.
- "why": one line quoting the fact that makes the answer right.
- "hook": a short memory trick (under 15 words) for the concept.
- "topic": the concept tested, 2-4 words, reuse the same name for the same concept.
- "page": the page the fact comes from.
- Every question must test a different fact."""


def llm(system, prompt, schema=None, temperature=0.4):
    """One chat call to the local Ollama server. `schema` forces JSON output in that shape."""
    body = {"model": MODEL, "stream": False, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "num_ctx": 8192}}
    if schema:
        body["format"] = schema
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]["content"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(500, f"Model not downloaded yet. Run: ollama pull {MODEL}")
        raise HTTPException(502, f"Ollama error: {e.read().decode(errors='replace')[:200]}")
    except (urllib.error.URLError, ConnectionError):
        raise HTTPException(503, "Ollama isn't running. Install it from ollama.com, then open the Ollama app.")
    except TimeoutError:
        raise HTTPException(504, "The local model took too long. Try a smaller quiz.")


def context(chunks):
    return "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c in chunks)


def get_doc(doc_id):
    if doc_id not in DOCS:
        raise HTTPException(404, "Unknown document, upload it again")
    return DOCS[doc_id]


SOURCE_TALK = re.compile(r",?\s*(?:as )?(?:exemplified |discussed |described |shown |stated |mentioned |presented |given )?(?:in|from|according to|by) the (?:provided |given |lecture )?(?:excerpt|text|passage|notes|slides?|lecture)"
                         r"(?: (?:on|from) page \d+)?|^\s*(?:on|from) page \d+,?\s*|,?\s*\((?:on )?p(?:age|\.) ?\d+\)", re.I)


def clean(q):
    """Normalise one generated question and shuffle MCQ options so the answer letter is random."""
    q["question"] = SOURCE_TALK.sub("", q["question"]).strip()
    q["question"] = q["question"][:1].upper() + q["question"][1:]
    if q["type"] == "tf" or len(q["choices"]) < 2:
        right = q["choices"][q["answer_index"]] if 0 <= q["answer_index"] < len(q["choices"]) else "True"
        q["type"], q["choices"] = "tf", ["True", "False"]
        q["answer_index"] = 0 if str(right).strip().lower().startswith("t") else 1
        return q
    q["choices"] = list(dict.fromkeys(str(c) for c in q["choices"]))[:4]  # small models sometimes repeat options
    right = q["choices"][min(max(q["answer_index"], 0), len(q["choices"]) - 1)]
    random.shuffle(q["choices"])
    q["answer_index"] = q["choices"].index(right)
    return q


def words(text):
    return set(tokenize(SOURCE_TALK.sub("", text))) - {"what", "which", "who", "does", "did", "do", "s", "following", "mentioned"}


def is_repeat(text, seen):
    """Near-duplicate if 75%+ of the shorter question's key words appear in an earlier one."""
    w = words(text)
    return any(len(w & s) >= 0.75 * min(len(w), len(s)) for s in seen if s and w)


def generate(n, extra, chunks, asked):
    """Ask the model for n questions and drop repeats in code.
    Past questions are NOT put in the prompt: small models copy any question they're shown."""
    seen, out = [words(s) for s in asked], []
    for attempt in range(3):  # ponytail: up to 2 retries with shuffled notes + more randomness
        notes = chunks if attempt == 0 else random.sample(chunks, len(chunks))
        raw = llm(EXAMINER, f"Write {n - len(out)} questions.{extra}\n\nLecture notes:\n{context(notes)}",
                  schema=QUIZ_SCHEMA, temperature=0.4 + 0.3 * attempt)
        try:
            batch = [clean(q) for q in json.loads(raw)["questions"] if q.get("choices") and q.get("question")]
        except (json.JSONDecodeError, KeyError, TypeError):
            batch = []
        for q in batch:
            if not is_repeat(q["question"], seen):
                seen.append(words(q["question"]))
                out.append(q)
        if len(out) >= n:
            break
    return out[:n]


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
    doc_id = hashlib.sha1(data).hexdigest()[:12]  # same PDF -> same id -> your progress comes back
    try:
        pages, ocr_pages = await run_in_threadpool(read_pages, data, doc_id)
    except Exception:
        raise HTTPException(400, "Could not read that PDF")
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(400, "No readable text found, even with OCR")
    DOCS[doc_id] = {"name": file.filename, "pages": len(pages), "chunks": chunks, "index": BM25(chunks)}
    progress_for(doc_id)["name"] = file.filename
    save_progress()
    return {"id": doc_id, "name": file.filename, "pages": len(pages), "chunks": len(chunks), "ocr_pages": ocr_pages}


# ---------- OCR for image-only slides ----------
OCR_MIN_WORDS = 15      # pages with fewer selectable words than this get their images OCR'd
OCR_MIN_CONF = 0.8      # drop OCR lines below this confidence (photos, handwriting -> junk)
_ocr = None


def ocr_engine():
    global _ocr
    if _ocr is None:
        try:
            import logging
            from rapidocr import RapidOCR
            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            _ocr = RapidOCR()
        except ImportError:  # OCR is optional: without it, image-only slides are just skipped
            _ocr = False
    return _ocr


def ocr_page(page):
    engine = ocr_engine()
    if not engine:
        return ""
    import numpy as np
    lines = []
    for im in page.images:
        try:
            img = im.image.convert("RGB")
        except Exception:
            continue
        if min(img.size) < 100:  # logos and icons
            continue
        res = engine(np.array(img))
        lines += [t for t, s in zip(res.txts or [], res.scores or []) if s >= OCR_MIN_CONF]
    text = " ".join(lines)
    return text if len(text.split()) >= 3 else ""


def read_pages(data, doc_id):
    """Page texts, with OCR for image-only pages. Cached per PDF because OCR takes ~1-2 s per image."""
    cache = ROOT / "data" / "pages" / f"{doc_id}.json"
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        return c["pages"], c["ocr_pages"]
    pages, ocr_pages = [], 0
    for p in PdfReader(io.BytesIO(data)).pages:
        text = p.extract_text() or ""
        if len(text.split()) < OCR_MIN_WORDS and (extra := ocr_page(p)):
            text, ocr_pages = f"{text}\n{extra}", ocr_pages + 1
        pages.append(text)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"pages": pages, "ocr_pages": ocr_pages}), encoding="utf-8")
    return pages, ocr_pages


class Ask(BaseModel):
    doc_id: str
    question: str


@app.post("/ask")
def ask(body: Ask):
    doc = get_doc(body.doc_id)
    hits = doc["index"].search(body.question) or doc["chunks"][:6]
    answer = llm(
        "You are a study assistant. Answer ONLY from the lecture excerpts given. "
        "End every sentence with the page it came from, like this: Sprints last 2-4 weeks (p. 3). "
        "If the excerpts don't contain the answer, say so.",
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
        # only the concept + fact, never the old question text: small models copy whatever question they're shown
        extra = "\n\nThe student got these concepts wrong. Write one brand-new question per concept, testing the same fact " \
                "from a different angle (e.g. ask it in reverse, or as True/False):\n" + \
                "\n".join(f"- Concept: {m['topic']}. Correct fact: {m.get('why') or m['correct']}. "
                          f"The student confused it with: {m['picked']}." for m in recent)
        if weak:
            extra += "\nWeakest topics overall: " + ", ".join(weak)
        chunks = doc["index"].search(" ".join(m["question"] + " " + m["correct"] for m in recent), k=8)
    elif body.topic:
        chunks = doc["index"].search(body.topic, k=12)
    else:
        step = max(1, len(doc["chunks"]) // (20 if body.mode == "mock" else 10))
        chunks = doc["chunks"][::step][:20]
    chunks = chunks or doc["chunks"][:10]
    questions = generate(n, extra, chunks, p["asked"])
    if not questions:
        raise HTTPException(502, "The local model returned no usable questions. Try again.")
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
                                  "topic": q["topic"], "page": q["page"], "why": q["why"], "at": int(time.time())})
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
