"""
rag.py — RAG system using:
  - OpenAI text-embedding-ada-002 for embeddings
  - FAISS as the vector store
  - Topic checkpoints (cosine similarity drop = new topic)
  - Every-100-message checkpoints
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TOPIC_SHIFT_THRESHOLD = 0.80   # ada-002 similarities are high, tune if needed
CHUNK_SIZE = 5                  # messages per chunk for topic detection


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Single embedding via OpenAI ada-002."""
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text[:8000]
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def gpt_summarise(messages: list[dict], label: str) -> str:
    """GPT-3.5-turbo summary of a message segment."""
    convo = "\n".join(f"{m['sender']}: {m['text']}" for m in messages[:40])
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a concise conversation summariser."},
            {"role": "user",   "content": f"Summarise this in 2-3 sentences:\n\n{convo}"}
        ],
        max_tokens=120,
        temperature=0.3,
    )
    return f"[{label}] " + resp.choices[0].message.content.strip()


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_messages(raw_text: str) -> list[dict]:
    messages = []
    for i, line in enumerate(raw_text.strip().split("\n")):
        line = line.strip()
        if line.startswith("User 1:"):
            messages.append({"index": i, "sender": "User1", "text": line[7:].strip()})
        elif line.startswith("User 2:"):
            messages.append({"index": i, "sender": "User2", "text": line[7:].strip()})
    return messages


# ── Topic Detection ───────────────────────────────────────────────────────────

def detect_topic_checkpoints(messages: list[dict]) -> list[dict]:
    """
    Chunk messages → embed each chunk → find similarity drops → new topic.
    Returns list of {topic_num, messages}.
    """
    if len(messages) < CHUNK_SIZE:
        return [{"topic_num": 1, "messages": messages}]

    # Build chunks
    chunks = []
    for i in range(0, len(messages), CHUNK_SIZE):
        batch = messages[i: i + CHUNK_SIZE]
        chunks.append({
            "text":     " ".join(m["text"] for m in batch),
            "messages": batch,
        })

    # Batch embed all chunks in one API call
    texts = [c["text"][:8000] for c in chunks]
    resp  = client.embeddings.create(model="text-embedding-ada-002", input=texts)
    embs  = [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # Detect boundaries
    boundaries = [0]
    for i in range(1, len(embs)):
        if cosine(embs[i - 1], embs[i]) < TOPIC_SHIFT_THRESHOLD:
            boundaries.append(i)
    boundaries.append(len(chunks))

    # Build topic segments
    topics = []
    for t in range(len(boundaries) - 1):
        seg_msgs = []
        for c in chunks[boundaries[t]: boundaries[t + 1]]:
            seg_msgs.extend(c["messages"])
        topics.append({"topic_num": t + 1, "messages": seg_msgs})

    return topics


def make_100msg_checkpoints(messages: list[dict]) -> list[dict]:
    return [
        {"start": i, "messages": messages[i: i + 100]}
        for i in range(0, len(messages), 100)
    ]


# ── FAISS Index ───────────────────────────────────────────────────────────────

class RAGIndex:
    """
    FAISS IndexFlatIP (inner product) on L2-normalised vectors = cosine similarity.
    metadata list is 1-to-1 with FAISS rows.
    """

    DIM = 1536   # text-embedding-ada-002 output dimension

    def __init__(self):
        self.index    = faiss.IndexFlatIP(self.DIM)
        self.metadata = []   # list of {label, summary}

    def _add(self, label: str, summary: str):
        emb = get_embedding(summary)
        emb /= np.linalg.norm(emb) + 1e-9   # normalise
        self.index.add(np.array([emb]))
        self.metadata.append({"label": label, "summary": summary})

    def build_from_conversation(self, raw_text: str):
        messages = parse_messages(raw_text)
        if not messages:
            return

        # Topic checkpoints
        for t in detect_topic_checkpoints(messages):
            label = f"Topic-{t['topic_num']}"
            self._add(label, gpt_summarise(t["messages"], label))

        # 100-message checkpoints
        for cp in make_100msg_checkpoints(messages):
            label = f"Msgs-{cp['start']}-{cp['start'] + len(cp['messages']) - 1}"
            self._add(label, gpt_summarise(cp["messages"], label))

    def query(self, question: str, top_k: int = 3) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q = get_embedding(question)
        q /= np.linalg.norm(q) + 1e-9
        scores, idxs = self.index.search(np.array([q]), min(top_k, self.index.ntotal))
        return [
            {**self.metadata[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0]) if i != -1
        ]

    def save(self, prefix: str = "faiss_store"):
        faiss.write_index(self.index, f"{prefix}.index")
        with open(f"{prefix}.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved → {prefix}.index + {prefix}.json")

    def load(self, prefix: str = "faiss_store"):
        self.index    = faiss.read_index(f"{prefix}.index")
        with open(f"{prefix}.json") as f:
            self.metadata = json.load(f)
        print(f"Loaded {self.index.ntotal} vectors from {prefix}")