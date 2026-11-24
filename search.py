"""PDF chunking + BM25 retrieval, written from scratch (no vector DB needed)."""
import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an and or of to in is are was were be for on with as by it this that at from".split())


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def chunk_pages(pages, size=180, overlap=40):
    """Split page texts into overlapping word windows, keeping the page number for citations."""
    chunks = []
    for page_no, text in enumerate(pages, 1):
        words = text.split()
        step = size - overlap
        for start in range(0, max(len(words) - overlap, 1), step):
            piece = " ".join(words[start:start + size])
            if piece.strip():
                chunks.append({"page": page_no, "text": piece})
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=6):
        q = tokenize(query)
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]


if __name__ == "__main__":  # self-check: python search.py
    pages = ["Dijkstra's algorithm finds shortest paths in graphs with non-negative weights.",
             "A hash table maps keys to values using a hash function. Collisions use chaining.",
             "Merge sort is a divide and conquer sorting algorithm running in O(n log n)."]
    idx = BM25(chunk_pages(pages))
    assert idx.search("shortest path graph")[0]["page"] == 1
    assert idx.search("hash collisions")[0]["page"] == 2
    assert idx.search("divide and conquer sort")[0]["page"] == 3
    assert idx.search("quantum chromodynamics") == []
    long = chunk_pages([" ".join(f"w{i}" for i in range(400))])
    assert len(long) == 3 and all(c["page"] == 1 for c in long)
    print("search.py OK")
