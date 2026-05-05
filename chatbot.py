import os
import json
import pandas as pd
from openai import OpenAI

from rag     import RAGIndex
from persona import extract_persona, merge_personas

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

FAISS_PREFIX = "faiss_store"
PERSONA_FILE = "persona.json"
CSV_PATH     = "conversations.csv"
NUM_CONVOS   = 50   


# Step 1 : Load data

def build_and_save(conversations: list[str]):
    index    = RAGIndex()
    personas = []

    print(f"Building index from {len(conversations)} conversations...")
    for i, convo in enumerate(conversations):
        index.build_from_conversation(convo)
        personas.append(extract_persona(convo))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(conversations)} processed")

    index.save(FAISS_PREFIX)

    merged = merge_personas(personas)
    with open(PERSONA_FILE, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Persona saved → {PERSONA_FILE}")

    return index, merged


def load_existing():
    index = RAGIndex()
    index.load(FAISS_PREFIX)
    with open(PERSONA_FILE) as f:
        persona = json.load(f)
    return index, persona


# Answer questions using RAG techniues from documents + persona contextt

def answer(question: str, index: RAGIndex, persona: dict) -> str:
    # Retrieve top-3 relevant summaries from FAISS
    rag_results = index.query(question, top_k=3)
    rag_context = "\n".join(
        f"- [{r['label']} | score {r['score']:.2f}]: {r['summary']}"
        for r in rag_results
    )

    persona_str = json.dumps(persona, indent=2)

    system_prompt = """You are a helpful assistant that answers questions about a chat user.
You have access to:
1. A structured persona extracted from their conversations
2. Relevant conversation summaries retrieved by semantic search

Answer clearly and concisely. Base your answer only on the provided context."""

    user_prompt = f"""Question: {question}

--- Persona ---
{persona_str}

--- Relevant Conversation Summaries (RAG) ---
{rag_context if rag_context else "No relevant summaries found."}
"""

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ── Main 

if __name__ == "__main__":
    # Load or build index
    if os.path.exists(f"{FAISS_PREFIX}.index") and os.path.exists(PERSONA_FILE):
        print("Found existing FAISS index and persona. Loading...")
        index, persona = load_existing()
    else:
        df = pd.read_csv(CSV_PATH, header=None, names=["conversation"])
        convos = df["conversation"].dropna().tolist()[:NUM_CONVOS]
        index, persona = build_and_save(convos)

    print("\n── Persona ─────────────────────────────────────────")
    print(json.dumps(persona, indent=2))
    print("────────────────────────────────────────────────────\n")

    print("Chatbot ready. Try:")
    print("  What kind of person is this user?")
    print("  What are their habits?")
    print("  How do they talk?")
    print("  Type 'quit' to exit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        print(f"\nBot: {answer(q, index, persona)}\n")