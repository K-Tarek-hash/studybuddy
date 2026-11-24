"""End-to-end check of the quiz -> mistakes -> drill loop with Claude mocked out. Run: python test_app.py"""
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import app

prompts = []


def fake_claude(system, prompt, schema=None):
    prompts.append(prompt)
    qs = [{"type": "mcq", "question": f"Q{i} who owns the product backlog?", "choices": ["Product Owner", "Scrum Master", "Team", "Sponsor"],
           "answer_index": 0, "why": "PO maintains it.", "hook": "Product backlog -> Product Owner", "topic": "Scrum roles", "page": 1}
          for i in range(len(prompts) * 100, len(prompts) * 100 + 10)]
    qs[0] = {"type": "tf", "question": "Spiral produces software late.", "choices": ["True", "False"], "answer_index": 1,
             "why": "Software is produced EARLY.", "hook": "Spiral = early", "topic": "Spiral model", "page": 2}
    return json.dumps({"questions": qs})


app.claude = fake_claude
app.PROGRESS_FILE = Path(tempfile.mkdtemp()) / "progress.json"
app.PROGRESS.clear()
c = TestClient(app.app)

# a real PDF needs text; reuse the extractor by faking the doc directly
app.DOCS["doc1"] = {"name": "lec.pdf", "pages": 2, "chunks": app.chunk_pages(["Product Owner backlog", "Spiral model early software"]),
                    "index": None}
app.DOCS["doc1"]["index"] = app.BM25(app.DOCS["doc1"]["chunks"])

assert c.post("/quiz", json={"doc_id": "doc1", "mode": "drill"}).status_code == 400  # nothing to drill yet

r = c.post("/quiz", json={"doc_id": "doc1"}).json()
assert len(r["questions"]) == 10 and "answer_index" not in r["questions"][0]  # answers hidden until submit
assert r["questions"][0]["choices"] == ["True", "False"]
full = app.QUIZZES[r["quiz_id"]]["questions"]
right = [q["answer_index"] for q in full]
picks = right[:]
picks[0] = 1 - right[0]                       # wrong T/F
picks[1] = (right[1] + 1) % 4                 # wrong MCQ
picks[2] = None                               # blank
res = c.post(f"/quiz/{r['quiz_id']}/submit", json={"answers": picks}).json()
assert res["score"] == 7 and res["total"] == 10, res["score"]

p = c.get("/progress/doc1").json()
assert p["open_mistakes"] == 3 and p["history"][0]["score"] == 7
assert p["weak_topics"][0]["topic"] == "Spiral model"  # 0/1 is the weakest

d = c.post("/quiz", json={"doc_id": "doc1", "mode": "drill"}).json()
assert len(d["questions"]) == 5
assert "got these wrong" in prompts[-1] and "Spiral produces software late." in prompts[-1]  # drill knows the mistakes
assert "Q105 who owns" in prompts[-1]  # previously asked questions are sent so they are not repeated
ans = [q["answer_index"] for q in app.QUIZZES[d["quiz_id"]]["questions"]]
c.post(f"/quiz/{d['quiz_id']}/submit", json={"answers": ans})
assert c.get("/progress/doc1").json()["open_mistakes"] == 0  # passing the drill clears them

# MCQ shuffle keeps the right answer attached
q = app.clean({"type": "mcq", "question": "x", "choices": ["A", "B", "C", "D"], "answer_index": 2, "why": "", "hook": "", "topic": "", "page": 1})
assert q["choices"][q["answer_index"]] == "C"
assert json.loads(app.PROGRESS_FILE.read_text())["doc1"]["history"]  # persisted to disk
print("test_app.py OK")
