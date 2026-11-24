# StudyBuddy

An AI study assistant for lecture slides. Upload a PDF, ask questions and get answers that cite page numbers, then generate multiple-choice quizzes from the material.

## How it works

1. **Ingest:** the app extracts text from the PDF with `pypdf` and splits each page into overlapping 180-word chunks. Each chunk keeps its page number.
2. **Retrieve:** a **BM25 search engine written from scratch** (`search.py`, no vector database) ranks the chunks against your question.
3. **Generate:** only the top-ranked chunks go to Claude (`claude-opus-5-5`). This keeps answers grounded in the lecture and makes page citations possible. This is retrieval-augmented generation (RAG).
4. **Quiz:** Claude returns questions in a fixed JSON format (structured outputs). The frontend renders them as a self-grading quiz.

```
PDF ─► pypdf ─► chunker ─► BM25 index ─► top-k chunks ─► Claude ─► answer + page refs
                                                     └─► Claude (JSON schema) ─► quiz
```

## Stack

Python, FastAPI, pypdf, the Anthropic SDK, and a plain HTML/JS frontend with no build step.

## Run it

```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your-key        # PowerShell: $env:ANTHROPIC_API_KEY="your-key"
uvicorn app:app --reload
```

Then open http://localhost:8000.

## Tests

```bash
python search.py    # self-check for chunking and BM25 ranking
```

## Roadmap

- Save documents in SQLite so they survive a restart
- OCR for scanned PDFs
- Ask questions across several PDFs
- Spaced-repetition review of quiz questions you got wrong
