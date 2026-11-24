# StudyBuddy

An AI study assistant for lecture slides. Upload a PDF, ask questions and get answers that cite page numbers, then generate multiple-choice quizzes from the material.

## How it works

1. **Ingest:** the app extracts text from the PDF with `pypdf` and splits each page into overlapping 180-word chunks. Each chunk keeps its page number.
2. **Retrieve:** a **BM25 search engine written from scratch** (`search.py`, no vector database) ranks the chunks against your question.
3. **Generate:** only the top-ranked chunks go to a **local LLM** (Qwen 2.5 3B through Ollama, free and offline, no API key). This keeps answers grounded in the lecture and makes page citations possible. This is retrieval-augmented generation (RAG).
4. **Quiz:** The local model writes Moodle-style exam questions (multiple choice with a–d, plus True/False) in a fixed JSON format (JSON schema-constrained output). False True/False statements flip one key word, and wrong options are similar-sounding terms. The server shuffles the options so the right letter is random, and keeps the answers hidden until you submit.
5. **Remembers your mistakes:** grading happens on the server. Every wrong answer and per-topic accuracy is saved under the PDF's hash, so progress comes back when you re-upload the same file. **Drill my mistakes** generates 5 new questions that re-test those exact concepts from a different angle. Passing the drill (80%+) clears them. Questions are never repeated, and the app flags guessing patterns such as always picking "a".

Modes: **New quiz** (10), **Drill my mistakes** (5), **Mock exam** (30).

```
PDF ─► pypdf ─► chunker ─► BM25 index ─► top-k chunks ─► LLM ─► answer + page refs
                                                     └─► LLM (JSON schema) ─► quiz
```

## Stack

Python, FastAPI, pypdf, [Ollama](https://ollama.com) for the local model (called through its HTTP API with only the standard library), and a plain HTML/JS frontend with no build step.

## Run it

```bash
# 1. Install Ollama from https://ollama.com, then download the model (~2 GB, one time):
ollama pull qwen2.5:3b
# 2. Install and run the app:
pip install -r requirements.txt
uvicorn app:app
```

To use a different model, set `OLLAMA_MODEL` (for example `llama3.2:3b`, or `qwen2.5:7b` if you have more GPU memory).

Then open http://localhost:8000.

## Tests

```bash
python search.py    # self-check for chunking and BM25 ranking
python test_app.py  # quiz -> mistakes -> drill loop, with the model mocked
```

## Roadmap

- Save documents in SQLite so they survive a restart
- OCR for scanned PDFs
- Ask questions across several PDFs
- Spaced-repetition review of quiz questions you got wrong
